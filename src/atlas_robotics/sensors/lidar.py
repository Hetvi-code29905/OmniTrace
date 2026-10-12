"""2D Ray-Casting LiDAR sensor simulation."""

import math
from dataclasses import dataclass, field
from typing import List, Set, Tuple
from atlas_robotics.world import GridWorld


@dataclass
class LidarScan:
    """Stores the telemetry of a single LiDAR scan revolution."""

    origin: Tuple[float, float]
    angles: List[float] = field(default_factory=list)
    ranges: List[float] = field(default_factory=list)
    hit_points: List[Tuple[float, float]] = field(default_factory=list)
    free_cells: Set[Tuple[int, int]] = field(default_factory=set)
    occupied_cells: Set[Tuple[int, int]] = field(default_factory=set)


class LidarSensor:
    """Simulates a 2D planar LiDAR range finder.

    Casts rays at uniform angular intervals and computes range intersections
    with grid obstacles using ray-marching.
    """

    def __init__(
        self,
        num_rays: int = 36,
        max_range: float = 6.0,
        ray_step: float = 0.05,
    ) -> None:
        if num_rays <= 0:
            raise ValueError("num_rays must be positive.")
        if max_range <= 0.0:
            raise ValueError("max_range must be positive.")

        self.num_rays: int = num_rays
        self.max_range: float = max_range
        self.ray_step: float = ray_step

    def scan(self, world: GridWorld, origin: Tuple[float, float]) -> LidarScan:
        """Cast rays in 360 degrees from origin and return sensor measurements.

        Args:
            world: GridWorld ground truth environment.
            origin: (x, y) robot continuous position in grid units.

        Returns:
            LidarScan with ray endpoints, ranges, free cells, and detected obstacles.
        """
        ox, oy = origin
        scan_result = LidarScan(origin=origin)

        angle_increment = (2.0 * math.pi) / self.num_rays

        for i in range(self.num_rays):
            angle = i * angle_increment
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            distance = 0.0
            hit_obstacle = False
            last_valid_point = (ox, oy)

            while distance < self.max_range:
                distance += self.ray_step
                rx = ox + cos_a * distance
                ry = oy + sin_a * distance
                cell_x, cell_y = int(math.floor(rx)), int(math.floor(ry))

                if not world.in_bounds(cell_x, cell_y):
                    # Hit map boundary
                    hit_obstacle = True
                    last_valid_point = (rx, ry)
                    break

                if world.is_blocked(cell_x, cell_y):
                    # Hit an obstacle
                    hit_obstacle = True
                    last_valid_point = (rx, ry)
                    scan_result.occupied_cells.add((cell_x, cell_y))
                    break

                # Ray passed through free space
                scan_result.free_cells.add((cell_x, cell_y))
                last_valid_point = (rx, ry)

            actual_range = min(distance, self.max_range)
            scan_result.angles.append(angle)
            scan_result.ranges.append(actual_range)
            scan_result.hit_points.append(last_valid_point)

        # Make sure current robot cell is always in free cells
        robot_cell = (int(math.floor(ox)), int(math.floor(oy)))
        scan_result.free_cells.add(robot_cell)
        scan_result.occupied_cells.discard(robot_cell)

        return scan_result
