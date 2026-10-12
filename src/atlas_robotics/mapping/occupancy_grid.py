"""Robot-side belief map representing unknown, free, and occupied cells."""

from enum import IntEnum
from typing import List, Tuple
from atlas_robotics.sensors.lidar import LidarScan


class CellState(IntEnum):
    """Discrete belief states for an occupancy grid cell."""

    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


class OccupancyGrid:
    """Maintains the robot's internal spatial map constructed from sensor observations.

    Keeps belief separate from the simulator ground truth.
    """

    def __init__(self, width: int = 20, height: int = 15) -> None:
        self.width: int = width
        self.height: int = height
        self.grid: List[List[int]] = [
            [int(CellState.UNKNOWN) for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if cell is within map bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_state(self, x: int, y: int) -> int:
        """Get the belief state of a cell. Out-of-bounds returns OCCUPIED."""
        if not self.in_bounds(x, y):
            return int(CellState.OCCUPIED)
        return self.grid[y][x]

    def is_blocked(self, x: int, y: int) -> bool:
        """Check if a cell is known to be occupied or out-of-bounds."""
        if not self.in_bounds(x, y):
            return True
        return self.grid[y][x] == int(CellState.OCCUPIED)

    def set_obstacle(self, x: int, y: int, blocked: bool = True) -> None:
        """Explicitly mark a cell as OCCUPIED or FREE."""
        if self.in_bounds(x, y):
            self.grid[y][x] = int(CellState.OCCUPIED if blocked else CellState.FREE)

    def is_known(self, x: int, y: int) -> bool:
        """Check if a cell has been observed (either FREE or OCCUPIED)."""
        if not self.in_bounds(x, y):
            return False
        return self.grid[y][x] != int(CellState.UNKNOWN)

    def update_from_scan(self, scan: LidarScan) -> None:
        """Update occupancy grid cells using a LiDAR sensor scan.

        Performs ray-clearing: if a laser beam penetrates a cell, it is updated to FREE,
        allowing dynamic obstacles to clear when they move away.
        """
        # 1. Update free cells (ray passing through clears any previous ghost obstacle)
        for fx, fy in scan.free_cells:
            if self.in_bounds(fx, fy) and (fx, fy) not in scan.occupied_cells:
                self.grid[fy][fx] = int(CellState.FREE)

        # 2. Update detected occupied obstacle cells
        for ox, oy in scan.occupied_cells:
            if self.in_bounds(ox, oy):
                self.grid[oy][ox] = int(CellState.OCCUPIED)

    def get_explored_ratio(self) -> float:
        """Calculate the ratio of discovered cells (Free + Occupied) vs Total cells."""
        total_cells = self.width * self.height
        if total_cells == 0:
            return 0.0
        known_count = sum(
            1
            for row in self.grid
            for cell in row
            if cell != int(CellState.UNKNOWN)
        )
        return known_count / total_cells

    def reset(self) -> None:
        """Reset the occupancy map back to completely UNKNOWN state."""
        self.grid = [
            [int(CellState.UNKNOWN) for _ in range(self.width)]
            for _ in range(self.height)
        ]
