"""Dijkstra's shortest path planning algorithm on discrete 2D occupancy grids."""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from atlas_robotics.world import GridWorld
from atlas_robotics.mapping.occupancy_grid import OccupancyGrid


@dataclass
class PlanningResult:
    """Encapsulates the result and metrics of a path planning operation."""

    success: bool
    path: List[Tuple[int, int]] = field(default_factory=list)
    cost: float = float("inf")
    nodes_explored: int = 0
    planning_time_ms: float = 0.0
    status_message: str = ""


class DijkstraPlanner:
    """Shortest path graph search algorithm for discrete 4-connected grids.

    Guarantees optimal (shortest) path on non-negative edge weights.
    Supports both ground-truth GridWorld and robot-side OccupancyGrid.
    """

    def __init__(self, step_cost: float = 1.0) -> None:
        self.step_cost: float = step_cost
        # Orthogonal 4-connected movement offsets
        self.neighbor_offsets: List[Tuple[int, int]] = [
            (0, -1),  # Up
            (0, 1),   # Down
            (-1, 0),  # Left
            (1, 0),   # Right
        ]

    def plan(
        self,
        grid_source,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> PlanningResult:
        """Find the shortest path from start to goal in the given grid source.

        Args:
            grid_source: GridWorld or OccupancyGrid instance.
            start: (x, y) start cell coordinates.
            goal: (x, y) goal cell coordinates.

        Returns:
            PlanningResult containing path, cost, and diagnostic metrics.
        """
        start_time = time.perf_counter()

        # 1. Validate start and goal feasibility
        if grid_source.is_blocked(start[0], start[1]):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PlanningResult(
                success=False,
                planning_time_ms=elapsed_ms,
                status_message="Start position is blocked or out of bounds.",
            )

        if grid_source.is_blocked(goal[0], goal[1]):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PlanningResult(
                success=False,
                planning_time_ms=elapsed_ms,
                status_message="Goal position is blocked or out of bounds.",
            )

        # 2. Trivial Case: Start equals Goal
        if start == goal:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PlanningResult(
                success=True,
                path=[start],
                cost=0.0,
                nodes_explored=1,
                planning_time_ms=elapsed_ms,
                status_message="Start equals goal.",
            )

        # 3. Dijkstra Priority Queue and Bookkeeping
        # Priority queue stores tuples: (distance_so_far, (x, y))
        pq: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(pq, (0.0, start))

        distances: Dict[Tuple[int, int], float] = {start: 0.0}
        predecessors: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        visited = set()
        nodes_explored = 0

        goal_found = False

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)
            nodes_explored += 1

            # Early exit on goal extraction
            if current_node == goal:
                goal_found = True
                break

            curr_x, curr_y = current_node

            # Explore 4-connected orthogonal neighbors
            for dx, dy in self.neighbor_offsets:
                nx, ny = curr_x + dx, curr_y + dy
                neighbor = (nx, ny)

                # Skip blocked or out-of-bounds neighbors
                if grid_source.is_blocked(nx, ny):
                    continue

                if neighbor in visited:
                    continue

                new_cost = current_dist + self.step_cost

                if neighbor not in distances or new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (new_cost, neighbor))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not goal_found:
            return PlanningResult(
                success=False,
                nodes_explored=nodes_explored,
                planning_time_ms=elapsed_ms,
                status_message="No valid path exists (goal is unreachable).",
            )

        # 4. Reconstruct path by backtracking predecessors from goal to start
        path: List[Tuple[int, int]] = []
        curr: Optional[Tuple[int, int]] = goal
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()

        return PlanningResult(
            success=True,
            path=path,
            cost=distances[goal],
            nodes_explored=nodes_explored,
            planning_time_ms=elapsed_ms,
            status_message=f"Optimal path found ({len(path)} steps).",
        )
