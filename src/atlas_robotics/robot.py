"""Robot model representing discrete state and movement constraints."""

from typing import Tuple
from atlas_robotics.world import GridWorld


class Robot:
    """Represents a mobile robot operating in a discrete GridWorld.

    Enforces 4-connected discrete orthogonal movements and maintains movement
    statistics and reset capabilities.
    """

    def __init__(self, start_x: int = 1, start_y: int = 1) -> None:
        self.start_x: int = start_x
        self.start_y: int = start_y
        self.x: int = start_x
        self.y: int = start_y
        self.moves_attempted: int = 0
        self.blocked_moves: int = 0

    @property
    def position(self) -> Tuple[int, int]:
        """Current (x, y) discrete coordinate."""
        return self.x, self.y

    def move(self, dx: int, dy: int, world: GridWorld) -> bool:
        """Attempt to move the robot by offset (dx, dy).

        Movement contract:
          - The movement vector must be exactly one orthogonal cell (abs(dx) + abs(dy) == 1).
          - Diagonal, stationary, or multi-cell steps raise ValueError.
          - If the destination cell is blocked or out-of-bounds according to the world model,
            the robot position remains unchanged and blocked_moves increments.
          - If valid, x and y are updated.

        Returns:
          True if move succeeded, False if blocked.
        """
        if abs(dx) + abs(dy) != 1:
            raise ValueError(
                f"A move must be exactly one cell in one direction: got dx={dx}, dy={dy}"
            )

        self.moves_attempted += 1
        next_x = self.x + dx
        next_y = self.y + dy

        if world.is_blocked(next_x, next_y):
            self.blocked_moves += 1
            return False

        self.x = next_x
        self.y = next_y
        return True

    def reset(self) -> None:
        """Reset the robot state and counters back to the initial start position."""
        self.x = self.start_x
        self.y = self.start_y
        self.moves_attempted = 0
        self.blocked_moves = 0

    def manhattan_distance_to(self, target_x: int, target_y: int) -> int:
        """Calculate the geometric Manhattan distance to a target cell."""
        return abs(self.x - target_x) + abs(self.y - target_y)
