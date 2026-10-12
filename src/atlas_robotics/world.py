"""World model representing a discrete 2D occupancy grid."""

from typing import List, Tuple, Optional


class GridWorld:
    """Represents a 2D discrete grid world for robot navigation.

    Coordinate system convention:
      - Coordinate (x, y) corresponds to column x and row y.
      - Grid storage: self.grid[y][x], where 0 is free space and 1 is an obstacle.
      - Bounds: 0 <= x < width and 0 <= y < height.
      - Out-of-bounds queries are treated as blocked.
    """

    def __init__(self, width: int = 20, height: int = 15) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid width and height must be positive integers.")
        self.width: int = width
        self.height: int = height
        self.grid: List[List[int]] = [
            [0 for _ in range(self.width)] for _ in range(self.height)
        ]

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates (x, y) fall within the grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x: int, y: int) -> bool:
        """Check if a cell (x, y) is blocked or out-of-bounds.

        Out-of-bounds coordinates are always treated as blocked.
        """
        if not self.in_bounds(x, y):
            return True
        return self.grid[y][x] == 1

    def set_obstacle(self, x: int, y: int, blocked: bool = True) -> bool:
        """Set or clear an obstacle at (x, y).

        Returns True if the cell was within bounds and updated, False otherwise.
        """
        if not self.in_bounds(x, y):
            return False
        self.grid[y][x] = 1 if blocked else 0
        return True

    def set_obstacles(self, coordinates: List[Tuple[int, int]]) -> None:
        """Set multiple obstacle coordinates at once."""
        for x, y in coordinates:
            self.set_obstacle(x, y, blocked=True)

    def clear(self) -> None:
        """Clear all obstacles from the grid."""
        self.grid = [
            [0 for _ in range(self.width)] for _ in range(self.height)
        ]

    def load_default_obstacles(
        self,
        start: Optional[Tuple[int, int]] = (1, 1),
        goal: Optional[Tuple[int, int]] = (18, 13),
    ) -> None:
        """Load a structured warehouse / rack layout of obstacles.

        Guarantees that start and goal cells are not blocked.
        """
        self.clear()
        obstacles: List[Tuple[int, int]] = []

        # Outer perimeter border (excluding open room)
        # Vertical shelving units / obstacles
        for y in range(3, 12):
            if y not in (6, 7):  # Aisle crosswalks
                obstacles.append((4, y))
                obstacles.append((5, y))
                obstacles.append((9, y))
                obstacles.append((10, y))
                obstacles.append((14, y))
                obstacles.append((15, y))

        # Horizontal barrier walls
        for x in range(6, 14):
            if x not in (9, 10):
                obstacles.append((x, 1))

        # Ensure start and goal are free
        if start in obstacles:
            obstacles.remove(start)
        if goal in obstacles:
            obstacles.remove(goal)

        self.set_obstacles(obstacles)
