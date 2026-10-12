"""Spatio-temporal reservation table and multi-robot fleet coordination."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from atlas_robotics.world import GridWorld
from atlas_robotics.robot import Robot
from atlas_robotics.planning.dijkstra import DijkstraPlanner


@dataclass
class FleetRobot:
    """Represents an autonomous mobile robot (AMR) in a coordinated fleet."""

    robot_id: str
    robot: Robot
    goal: Tuple[int, int]
    color: Tuple[int, int, int]
    path: List[Tuple[int, int]] = field(default_factory=list)
    path_index: int = 0
    status: str = "IDLE"  # "IDLE", "NAVIGATING", "YIELDING", "GOAL_REACHED"
    waiting_steps: int = 0

    @property
    def position(self) -> Tuple[int, int]:
        return self.robot.x, self.robot.y

    @property
    def is_done(self) -> bool:
        return self.position == self.goal


class ReservationTable:
    """Time-space reservation table preventing vertex and edge collisions.

    Keyed by (x, y, timestep) -> robot_id.
    """

    def __init__(self) -> None:
        # (x, y, timestep) -> robot_id
        self._reservations: Dict[Tuple[int, int, int], str] = {}

    def is_cell_reserved(
        self, x: int, y: int, timestep: int, requesting_robot_id: str
    ) -> bool:
        """Check if cell (x, y) is booked by ANOTHER robot at the given timestep."""
        owner = self._reservations.get((x, y, timestep))
        return owner is not None and owner != requesting_robot_id

    def reserve(self, robot_id: str, x: int, y: int, timestep: int) -> bool:
        """Book a specific cell at a specific timestep."""
        if self.is_cell_reserved(x, y, timestep, robot_id):
            return False
        self._reservations[(x, y, timestep)] = robot_id
        return True

    def reserve_path(
        self, robot_id: str, path: List[Tuple[int, int]], start_timestep: int
    ) -> bool:
        """Reserve an entire sequence of waypoints across progressive timesteps."""
        # 1. Check if all cells are free
        for i, (px, py) in enumerate(path):
            t = start_timestep + i
            if self.is_cell_reserved(px, py, t, robot_id):
                return False

        # 2. Book reservations
        for i, (px, py) in enumerate(path):
            t = start_timestep + i
            self._reservations[(px, py, t)] = robot_id

        return True

    def clear_robot_reservations(self, robot_id: str) -> None:
        """Clear all future time-space reservations owned by a robot."""
        keys_to_delete = [
            k for k, v in self._reservations.items() if v == robot_id
        ]
        for k in keys_to_delete:
            del self._reservations[k]

    def clear_all(self) -> None:
        """Clear the entire reservation table."""
        self._reservations.clear()

    @property
    def total_reservations(self) -> int:
        return len(self._reservations)


class FleetCoordinator:
    """Central traffic manager executing synchronized multi-robot navigation."""

    def __init__(self, world: GridWorld) -> None:
        self.world: GridWorld = world
        self.planner: DijkstraPlanner = DijkstraPlanner()
        self.reservation_table: ReservationTable = ReservationTable()
        self.robots: List[FleetRobot] = []
        self.current_timestep: int = 0

    def add_robot(
        self,
        robot_id: str,
        start_x: int,
        start_y: int,
        goal_x: int,
        goal_y: int,
        color: Tuple[int, int, int] = (56, 189, 248),
    ) -> FleetRobot:
        """Add a new robot to the fleet and register its initial state."""
        robot_model = Robot(start_x=start_x, start_y=start_y)
        fleet_robot = FleetRobot(
            robot_id=robot_id,
            robot=robot_model,
            goal=(goal_x, goal_y),
            color=color,
        )
        self.robots.append(fleet_robot)
        return fleet_robot

    def plan_fleet_routes(self) -> None:
        """Plan paths for all robots using prioritized reservation booking."""
        self.reservation_table.clear_all()
        self.current_timestep = 0

        # Prioritized planning: higher-priority robots reserve time slots first
        for fleet_robot in self.robots:
            res = self.planner.plan(
                self.world,
                start=fleet_robot.position,
                goal=fleet_robot.goal,
            )
            if res.success:
                fleet_robot.path = res.path
                fleet_robot.path_index = 0
                fleet_robot.status = "NAVIGATING"
                # Book reservation slots
                self.reservation_table.reserve_path(
                    fleet_robot.robot_id, fleet_robot.path, start_timestep=0
                )
            else:
                fleet_robot.path = []
                fleet_robot.path_index = 0
                fleet_robot.status = "NO_PATH"

    def step_fleet(self) -> None:
        """Execute one synchronized step for all robots with dynamic conflict resolution."""
        self.current_timestep += 1
        occupied_next_cells: Set[Tuple[int, int]] = set()

        for fleet_robot in self.robots:
            if fleet_robot.is_done:
                fleet_robot.status = "GOAL_REACHED"
                continue

            if not fleet_robot.path or fleet_robot.path_index >= len(fleet_robot.path) - 1:
                fleet_robot.status = "GOAL_REACHED"
                continue

            next_x, next_y = fleet_robot.path[fleet_robot.path_index + 1]
            curr_x, curr_y = fleet_robot.position

            # 1. Conflict Check: Is the next cell reserved by another robot this second,
            #    or already occupied by another robot in this same simultaneous step?
            is_blocked_by_reservation = self.reservation_table.is_cell_reserved(
                next_x, next_y, self.current_timestep, fleet_robot.robot_id
            )
            is_same_step_conflict = (next_x, next_y) in occupied_next_cells

            if is_blocked_by_reservation or is_same_step_conflict:
                # YIELD / WAIT: Remain in current cell for 1 time step
                fleet_robot.status = "YIELDING (Aisle Reserved)"
                fleet_robot.waiting_steps += 1
                occupied_next_cells.add(fleet_robot.position)
                # Re-book current cell for this step
                self.reservation_table.reserve(
                    fleet_robot.robot_id, curr_x, curr_y, self.current_timestep
                )
            else:
                # Safe to advance!
                dx = next_x - curr_x
                dy = next_y - curr_y
                success = fleet_robot.robot.move(dx, dy, self.world)
                if success:
                    fleet_robot.path_index += 1
                    fleet_robot.status = "NAVIGATING"
                    occupied_next_cells.add(fleet_robot.position)
                else:
                    fleet_robot.status = "COLLISION_REJECTED"

    def reset_fleet(self) -> None:
        """Reset all fleet robots back to their starting positions."""
        for fleet_robot in self.robots:
            fleet_robot.robot.reset()
            fleet_robot.waiting_steps = 0
            fleet_robot.status = "IDLE"
        self.plan_fleet_routes()
