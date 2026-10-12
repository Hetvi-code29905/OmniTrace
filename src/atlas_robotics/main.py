"""Atlas Robotics - 3 Fully Isolated Demonstration Environments.

- [Mode 1] GLOBAL (GOD MODE): Known static map + Odometry Drift. Test sudden dynamic hazards
          WITHOUT LiDAR to see how the robot reacts to sudden obstacles blindly vs globally!
- [Mode 2] SENSOR (LIDAR EXPLORATION): True Fog of War, 36-ray LiDAR, Occupancy Grid with ray-clearing,
          where the laser sees the moving hazard 6 meters away and detours safely before collision.
- [Mode 3] HYBRID RL AUTONOMY: Hierarchical architecture coupling Dijkstra global shortest path
          with an explainable 8-state local Q-Learning policy for real-time dynamic obstacle handling.

Key Features:
- 100% Independent World Grids: Painting obstacles in one mode does NOT affect other modes.
- Press [R] in any mode to instantly restore that specific mode back to factory defaults.
- Left-Click / Drag: Paint obstacles in real time.
- Right-Click / Drag: Erase obstacles / relocate Goal.
- Shift + Click: Relocate Robot.
"""

import math
import sys
import pygame
from enum import Enum
from typing import Dict, List, Optional, Tuple

from atlas_robotics.world import GridWorld
from atlas_robotics.robot import Robot
from atlas_robotics.planning.dijkstra import DijkstraPlanner, PlanningResult
from atlas_robotics.sensors.lidar import LidarSensor, LidarScan
from atlas_robotics.mapping.occupancy_grid import OccupancyGrid, CellState
from atlas_robotics.localization.odometry import OdometryEstimator, Pose2D
from atlas_robotics.control.waypoint_follower import (
    KinematicState,
    VelocityCommand,
    WaypointFollower,
    normalize_angle,
)
from atlas_robotics.learning.local_rl_policy import LocalAction, LocalRLPolicy, LocalState

# Window and layout constants
GRID_COLS: int = 20
GRID_ROWS: int = 15
CELL_SIZE: int = 40
GRID_WIDTH: int = GRID_COLS * CELL_SIZE   # 800 px
GRID_HEIGHT: int = GRID_ROWS * CELL_SIZE # 600 px
TOPBAR_HEIGHT: int = 44                  # 44 px (Mode selector tabs)
HUD_HEIGHT: int = 92                     # 92 px (Bottom telemetry bar)
WINDOW_WIDTH: int = GRID_WIDTH           # 800 px
WINDOW_HEIGHT: int = TOPBAR_HEIGHT + GRID_HEIGHT + HUD_HEIGHT # 736 px
TARGET_FPS: int = 30

# Initial robot configuration
ROBOT_START: Tuple[int, int] = (1, 1)
GOAL_POSITION: Tuple[int, int] = (18, 13)

# Color Palette
COLOR_BG = (10, 13, 20)
COLOR_GRID_BG = (18, 23, 36)
COLOR_GRID_LINE = (30, 36, 54)
COLOR_OBSTACLE = (52, 60, 82)
COLOR_OBSTACLE_BORDER = (70, 80, 110)
COLOR_DYNAMIC_OBS = (239, 68, 68)    # Red for dynamic hazard
COLOR_DYNAMIC_BORDER = (185, 28, 28)
COLOR_UNKNOWN_CELL = (8, 10, 15)     # Fog of War dark cell
COLOR_UNKNOWN_GRID = (16, 20, 28)
COLOR_GOAL = (34, 197, 94)           # Green goal
COLOR_GOAL_BORDER = (22, 101, 52)
COLOR_ROBOT = (56, 189, 248)         # Cyan (True robot ground truth)
COLOR_ROBOT_INNER = (2, 132, 199)
COLOR_ROBOT_BORDER = (240, 249, 255)
COLOR_ROBOT_HEADING = (255, 255, 255)# Direction pointer
COLOR_ODOM_GHOST = (217, 70, 239)    # Magenta (Estimated odometry pose)
COLOR_PATH_LINE = (14, 165, 233)
COLOR_PATH_NODE = (125, 211, 252)
COLOR_HUD_BG = (7, 9, 14)
COLOR_HUD_BORDER = (36, 44, 62)
COLOR_TOPBAR_BG = (14, 18, 28)
COLOR_TAB_ACTIVE = (30, 41, 64)
COLOR_TAB_INACTIVE = (18, 23, 36)
COLOR_TEXT_PRIMARY = (248, 250, 252)
COLOR_TEXT_MUTED = (148, 163, 184)
COLOR_ACCENT_YELLOW = (250, 204, 21)
COLOR_ACCENT_RED = (248, 113, 113)
COLOR_ACCENT_GREEN = (74, 222, 128)
COLOR_ACCENT_CYAN = (56, 189, 248)
COLOR_ACCENT_ORANGE = (251, 146, 60)
COLOR_ACCENT_MAGENTA = (232, 121, 249)
COLOR_ACCENT_PURPLE = (168, 85, 247)


class AppStage(Enum):
    STAGE_1_GLOBAL = 1
    STAGE_2_LIDAR = 2
    STAGE_3_HYBRID_RL = 3


class DynamicObstacle:
    """A moving hazard that patrols back and forth across an aisle."""

    def __init__(self, start_x: int, start_y: int, min_y: int, max_y: int) -> None:
        self.x: int = start_x
        self.y: int = start_y
        self.min_y: int = min_y
        self.max_y: int = max_y
        self.direction: int = 1

    def update(self, world: GridWorld) -> None:
        world.set_obstacle(self.x, self.y, blocked=False)
        next_y = self.y + self.direction
        if next_y > self.max_y or next_y < self.min_y:
            self.direction *= -1
            next_y = self.y + self.direction

        self.y = next_y
        world.set_obstacle(self.x, self.y, blocked=True)


class AtlasSimulatorApp:
    """Modular application presenting 3 completely independent robotics environments."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Atlas Robotics - 3 Isolated Robotics Demonstration Environments")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.lidar_surface = pygame.Surface((GRID_WIDTH, GRID_HEIGHT), pygame.SRCALPHA)

        # Typography
        self.font_main = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13, bold=True)
        self.font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 14, bold=True)
        self.font_tab = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12, bold=True)
        self.font_status = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12)
        self.font_controls = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 11)

        # Active Stage: 1 = Global, 2 = LiDAR, 3 = Fleet
        self.current_stage: AppStage = AppStage.STAGE_1_GLOBAL

        # Planner & Controller
        self.planner = DijkstraPlanner()
        self.controller = WaypointFollower(
            max_linear_vel=3.5,
            max_angular_vel=5.0,
            k_p_linear=2.5,
            k_p_angular=5.0,
            waypoint_tolerance=0.20,
            goal_tolerance=0.10,
            pivot_angle_threshold=math.radians(25.0),
        )
        self.lidar = LidarSensor(num_rays=36, max_range=6.5)

        # -------------------------------------------------------------
        # ISOLATED STAGE 1 ENVIRONMENT: Global God Mode
        # -------------------------------------------------------------
        self.world_s1 = GridWorld(width=GRID_COLS, height=GRID_ROWS)
        self.world_s1.load_default_obstacles(start=ROBOT_START, goal=GOAL_POSITION)
        self.robot_s1 = Robot(start_x=ROBOT_START[0], start_y=ROBOT_START[1])
        self.goal_s1 = list(GOAL_POSITION)
        self.kinematics_s1 = KinematicState(x=float(ROBOT_START[0]) + 0.5, y=float(ROBOT_START[1]) + 0.5, theta=0.0)
        self.odometry_s1 = OdometryEstimator(start_x=self.kinematics_s1.x, start_y=self.kinematics_s1.y)
        self.hazard_s1 = DynamicObstacle(start_x=7, start_y=4, min_y=3, max_y=11)
        self.hazard_s1_active: bool = False  # Toggleable with [D] in Stage 1
        self.path_s1: List[Tuple[int, int]] = []
        self.waypoint_idx_s1: int = 0
        self.cmd_s1 = VelocityCommand()

        # -------------------------------------------------------------
        # ISOLATED STAGE 2 ENVIRONMENT: LiDAR & Fog of War Exploration
        # -------------------------------------------------------------
        self.world_s2 = GridWorld(width=GRID_COLS, height=GRID_ROWS)
        self.world_s2.load_default_obstacles(start=ROBOT_START, goal=GOAL_POSITION)
        self.robot_s2 = Robot(start_x=ROBOT_START[0], start_y=ROBOT_START[1])
        self.goal_s2 = list(GOAL_POSITION)
        self.kinematics_s2 = KinematicState(x=float(ROBOT_START[0]) + 0.5, y=float(ROBOT_START[1]) + 0.5, theta=0.0)
        self.odometry_s2 = OdometryEstimator(start_x=self.kinematics_s2.x, start_y=self.kinematics_s2.y)
        self.occupancy_map_s2 = OccupancyGrid(width=GRID_COLS, height=GRID_ROWS)
        self.last_scan_s2: Optional[LidarScan] = None
        self.hazard_s2 = DynamicObstacle(start_x=7, start_y=4, min_y=3, max_y=11)
        self.hazard_s2_active: bool = True
        self.path_s2: List[Tuple[int, int]] = []
        self.waypoint_idx_s2: int = 0
        self.cmd_s2 = VelocityCommand()

        # -------------------------------------------------------------
        # ISOLATED STAGE 3 ENVIRONMENT: Hybrid Autonomy (Dijkstra + Local RL)
        # -------------------------------------------------------------
        self.world_s3 = GridWorld(width=GRID_COLS, height=GRID_ROWS)
        self.world_s3.load_default_obstacles(start=ROBOT_START, goal=GOAL_POSITION)
        self.robot_s3 = Robot(start_x=ROBOT_START[0], start_y=ROBOT_START[1])
        self.goal_s3 = list(GOAL_POSITION)
        self.kinematics_s3 = KinematicState(x=float(ROBOT_START[0]) + 0.5, y=float(ROBOT_START[1]) + 0.5, theta=0.0)
        self.hazard_s3 = DynamicObstacle(start_x=7, start_y=4, min_y=3, max_y=11)
        self.hazard_s3_active: bool = True
        self.cmd_s3 = VelocityCommand()
        self.path_s3: List[Tuple[int, int]] = []
        self.waypoint_idx_s3: int = 0
        self.last_scan_s3: Optional[LidarScan] = None
        self.rl_policy = LocalRLPolicy()
        self.last_rl_action: LocalAction = LocalAction.CRUISE
        self.last_rl_state: int = 0
        self.last_sector_clearances: Dict[str, float] = {"front": 99.0, "left": 99.0, "right": 99.0}
        self.yield_wait_counter_s3: int = 0
        self.crash_recovery_timer_s3: int = 0

        # Shared timing & interaction states
        self.last_hazard_tick: int = 0
        self.hazard_interval_ms: int = 400
        self.is_autonomous: bool = False
        self.mouse_dragging: bool = False
        self.drag_button: int = 0
        self.crash_pos: Optional[Tuple[float, float]] = None
        self.crash_timer: int = 0
        self.status_message = "Stage 1 Active. Click/Drag to paint obstacles, [Space] to drive!"
        self.status_color = COLOR_TEXT_MUTED

        # Initial plans
        self._recompute_stage1_plan()
        self._trigger_stage2_lidar()
        self._recompute_stage2_plan()
        self._recompute_stage3_plan()

    def set_stage(self, stage: AppStage) -> None:
        """Switch active stage without contaminating world grids."""
        self.current_stage = stage
        self.is_autonomous = False

        if stage == AppStage.STAGE_1_GLOBAL:
            self._recompute_stage1_plan()
            self.status_message = "Stage 1: Global God Mode. [D] Toggle Sudden Hazard without LiDAR!"
            self.status_color = COLOR_ACCENT_CYAN

        elif stage == AppStage.STAGE_2_LIDAR:
            self._trigger_stage2_lidar()
            self._recompute_stage2_plan()
            self.status_message = "Stage 2: LiDAR Exploration. Laser sees moving hazard from 6m and detours!"
            self.status_color = COLOR_ACCENT_ORANGE

        elif stage == AppStage.STAGE_3_HYBRID_RL:
            self._recompute_stage3_plan()
            self.status_message = "Stage 3: Hybrid Autonomy. Dijkstra Global Route + 8-State Local RL Reactive Policy."
            self.status_color = COLOR_ACCENT_PURPLE

    def reset_current_stage(self) -> None:
        """Reset only the currently active stage back to clean default state."""
        self.is_autonomous = False

        if self.current_stage == AppStage.STAGE_1_GLOBAL:
            self.world_s1.load_default_obstacles(start=ROBOT_START, goal=GOAL_POSITION)
            self.robot_s1.reset()
            self.goal_s1 = list(GOAL_POSITION)
            self.kinematics_s1 = KinematicState(x=float(ROBOT_START[0]) + 0.5, y=float(ROBOT_START[1]) + 0.5, theta=0.0)
            self.odometry_s1.reset()
            self.cmd_s1 = VelocityCommand()
            self._recompute_stage1_plan()
            self.status_message = "Stage 1 world and robot reset to factory default."
            self.status_color = COLOR_ACCENT_YELLOW

        elif self.current_stage == AppStage.STAGE_2_LIDAR:
            self.world_s2.load_default_obstacles(start=ROBOT_START, goal=GOAL_POSITION)
            self.robot_s2.reset()
            self.goal_s2 = list(GOAL_POSITION)
            self.kinematics_s2 = KinematicState(x=float(ROBOT_START[0]) + 0.5, y=float(ROBOT_START[1]) + 0.5, theta=0.0)
            self.odometry_s2.reset()
            self.occupancy_map_s2.reset()
            self.cmd_s2 = VelocityCommand()
            self._trigger_stage2_lidar()
            self._recompute_stage2_plan()
            self.status_message = "Stage 2 world, LiDAR map, and odometry reset."
            self.status_color = COLOR_ACCENT_YELLOW

        elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
            self.world_s3.load_default_obstacles(start=ROBOT_START, goal=GOAL_POSITION)
            self.robot_s3.reset()
            self.goal_s3 = list(GOAL_POSITION)
            self.kinematics_s3 = KinematicState(x=float(ROBOT_START[0]) + 0.5, y=float(ROBOT_START[1]) + 0.5, theta=0.0)
            self.cmd_s3 = VelocityCommand()
            self.last_scan_s3 = None
            self.yield_wait_counter_s3 = 0
            self.crash_recovery_timer_s3 = 0
            self._recompute_stage3_plan()
            self.status_message = "Stage 3 Hybrid RL environment reset."
            self.status_color = COLOR_ACCENT_YELLOW

    def _trigger_stage2_lidar(self) -> None:
        origin = (self.kinematics_s2.x, self.kinematics_s2.y)
        self.last_scan_s2 = self.lidar.scan(self.world_s2, origin=origin)
        self.occupancy_map_s2.update_from_scan(self.last_scan_s2)

    def _compute_stage3_clearances(self) -> Dict[str, float]:
        """Convert 36-ray LiDAR measurements into Front, Left, and Right sector clearances."""
        origin = (self.kinematics_s3.x, self.kinematics_s3.y)
        self.last_scan_s3 = self.lidar.scan(self.world_s3, origin=origin)

        # Align forward sector with upcoming waypoint to prevent corner-turning confusion
        if self.path_s3 and self.waypoint_idx_s3 < len(self.path_s3):
            target_pt = self.path_s3[self.waypoint_idx_s3]
            target_cont = (target_pt[0] + 0.5, target_pt[1] + 0.5)
            ref_heading = math.atan2(target_cont[1] - self.kinematics_s3.y, target_cont[0] - self.kinematics_s3.x)
        else:
            ref_heading = self.kinematics_s3.theta

        d_front = 99.0
        d_left = 99.0
        d_right = 99.0

        if self.last_scan_s3:
            for angle, r in zip(self.last_scan_s3.angles, self.last_scan_s3.ranges):
                rel = normalize_angle(angle - ref_heading)
                # Narrow forward driving lane: +-22 deg matches robot physical envelope
                if abs(rel) <= math.radians(22.0):
                    d_front = min(d_front, r)
                elif 0.0 < rel <= math.radians(95.0):
                    d_right = min(d_right, r)
                elif -math.radians(95.0) <= rel < 0.0:
                    d_left = min(d_left, r)

        if self.hazard_s3_active:
            hx = float(self.hazard_s3.x) + 0.5
            hy = float(self.hazard_s3.y) + 0.5
            dist_haz = math.hypot(self.kinematics_s3.x - hx, self.kinematics_s3.y - hy)
            if dist_haz < 6.0:
                angle_haz = math.atan2(hy - self.kinematics_s3.y, hx - self.kinematics_s3.x)
                rel_h = normalize_angle(angle_haz - ref_heading)
                if abs(rel_h) <= math.radians(22.0):
                    d_front = min(d_front, dist_haz)
                elif 0.0 < rel_h <= math.radians(95.0):
                    d_right = min(d_right, dist_haz)
                elif -math.radians(95.0) <= rel_h < 0.0:
                    d_left = min(d_left, dist_haz)

        self.last_sector_clearances = {"front": d_front, "left": d_left, "right": d_right}
        return self.last_sector_clearances

    def _recompute_stage1_plan(self) -> None:
        curr_grid_x = int(math.floor(self.kinematics_s1.x))
        curr_grid_y = int(math.floor(self.kinematics_s1.y))
        res = self.planner.plan(
            self.world_s1, start=(curr_grid_x, curr_grid_y), goal=(self.goal_s1[0], self.goal_s1[1])
        )
        if res.success:
            self.path_s1 = res.path
            self.waypoint_idx_s1 = 1 if len(self.path_s1) > 1 else 0
            self.status_message = f"Stage 1: Shortest Path Found ({len(self.path_s1)} waypoints)."
            self.status_color = COLOR_ACCENT_GREEN
        else:
            self.path_s1 = []
            self.waypoint_idx_s1 = 0
            self.is_autonomous = False
            self.status_message = res.status_message
            self.status_color = COLOR_ACCENT_RED

    def _recompute_stage2_plan(self) -> None:
        curr_grid_x = int(math.floor(self.kinematics_s2.x))
        curr_grid_y = int(math.floor(self.kinematics_s2.y))
        res = self.planner.plan(
            self.occupancy_map_s2, start=(curr_grid_x, curr_grid_y), goal=(self.goal_s2[0], self.goal_s2[1])
        )
        if res.success:
            self.path_s2 = res.path
            self.waypoint_idx_s2 = 1 if len(self.path_s2) > 1 else 0
            self.status_message = f"Stage 2: Discovered Path ({len(self.path_s2)} waypoints)."
            self.status_color = COLOR_ACCENT_GREEN
        else:
            self.path_s2 = []
            self.waypoint_idx_s2 = 0
            self.is_autonomous = False
            self.status_message = res.status_message
            self.status_color = COLOR_ACCENT_RED

    def _recompute_stage3_plan(self) -> None:
        curr_grid_x = int(math.floor(self.kinematics_s3.x))
        curr_grid_y = int(math.floor(self.kinematics_s3.y))
        res = self.planner.plan(
            self.world_s3, start=(curr_grid_x, curr_grid_y), goal=(self.goal_s3[0], self.goal_s3[1])
        )
        if res.success:
            self.path_s3 = res.path
            self.waypoint_idx_s3 = 1 if len(self.path_s3) > 1 else 0
            self.status_message = f"Stage 3: Dijkstra Global Route Found ({len(self.path_s3)} waypoints)."
            self.status_color = COLOR_ACCENT_PURPLE
        else:
            self.path_s3 = []
            self.waypoint_idx_s3 = 0
            self.is_autonomous = False
            self.status_message = res.status_message
            self.status_color = COLOR_ACCENT_RED

    def handle_input(self) -> bool:
        current_time = pygame.time.get_ticks()
        dt = 1.0 / TARGET_FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                # Stage selection
                elif event.key == pygame.K_1:
                    self.set_stage(AppStage.STAGE_1_GLOBAL)
                elif event.key == pygame.K_2:
                    self.set_stage(AppStage.STAGE_2_LIDAR)
                elif event.key == pygame.K_3:
                    self.set_stage(AppStage.STAGE_3_HYBRID_RL)

                # Reset Active Stage
                elif event.key == pygame.K_r:
                    self.reset_current_stage()

                # Autonomous Nav Toggle
                elif event.key == pygame.K_SPACE:
                    self.is_autonomous = not self.is_autonomous
                    if self.is_autonomous:
                        self.status_message = "Autonomous Pursuit ACTIVE..."
                        self.status_color = COLOR_ACCENT_GREEN
                    else:
                        self.cmd_s1 = VelocityCommand()
                        self.cmd_s2 = VelocityCommand()
                        self.cmd_s3 = VelocityCommand()
                        self.status_message = "Navigation paused."
                        self.status_color = COLOR_ACCENT_YELLOW

                # Hazard toggle in Stage 1, 2, or 3
                elif event.key == pygame.K_d:
                    if self.current_stage == AppStage.STAGE_1_GLOBAL:
                        self.hazard_s1_active = not self.hazard_s1_active
                        if not self.hazard_s1_active:
                            self.world_s1.set_obstacle(self.hazard_s1.x, self.hazard_s1.y, blocked=False)
                        state = "ACTIVE" if self.hazard_s1_active else "OFF"
                        self.status_message = f"Stage 1 Sudden Moving Hazard {state} (Watch what happens without LiDAR!)."
                        self.status_color = COLOR_ACCENT_RED
                    elif self.current_stage == AppStage.STAGE_2_LIDAR:
                        self.hazard_s2_active = not self.hazard_s2_active
                        if not self.hazard_s2_active:
                            self.world_s2.set_obstacle(self.hazard_s2.x, self.hazard_s2.y, blocked=False)
                        state = "ACTIVE" if self.hazard_s2_active else "OFF"
                        self.status_message = f"Stage 2 Hazard {state}."
                        self.status_color = COLOR_ACCENT_RED
                    elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
                        self.hazard_s3_active = not self.hazard_s3_active
                        if not self.hazard_s3_active:
                            self.world_s3.set_obstacle(self.hazard_s3.x, self.hazard_s3.y, blocked=False)
                        state = "ACTIVE" if self.hazard_s3_active else "OFF"
                        self.status_message = f"Stage 3 Moving Hazard {state}."
                        self.status_color = COLOR_ACCENT_RED

                # AMCL / Odometry sync
                elif event.key == pygame.K_c:
                    if self.current_stage == AppStage.STAGE_1_GLOBAL:
                        self.odometry_s1.correct_pose(self.kinematics_s1.x, self.kinematics_s1.y)
                        self.status_message = "Stage 1 Odometry Synced!"
                    elif self.current_stage == AppStage.STAGE_2_LIDAR:
                        self.odometry_s2.correct_pose(self.kinematics_s2.x, self.kinematics_s2.y)
                        self.status_message = "Stage 2 Odometry Synced!"
                    self.status_color = COLOR_ACCENT_GREEN

                else:
                    if not self.is_autonomous:
                        self._handle_manual_move(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if my < TOPBAR_HEIGHT:
                    tab_w = WINDOW_WIDTH // 3
                    if mx < tab_w:
                        self.set_stage(AppStage.STAGE_1_GLOBAL)
                    elif mx < 2 * tab_w:
                        self.set_stage(AppStage.STAGE_2_LIDAR)
                    else:
                        self.set_stage(AppStage.STAGE_3_HYBRID_RL)
                elif my < TOPBAR_HEIGHT + GRID_HEIGHT:
                    self.mouse_dragging = True
                    self.drag_button = event.button
                    self._apply_mouse_interaction(event.pos, event.button)

            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    mx, my = event.pos
                    if TOPBAR_HEIGHT <= my < TOPBAR_HEIGHT + GRID_HEIGHT:
                        self._apply_mouse_interaction(event.pos, self.drag_button)

        # Update Dynamic Hazards
        if current_time - self.last_hazard_tick >= self.hazard_interval_ms:
            # Stage 1 Hazard: Blind to moving objects (No LiDAR)
            if self.current_stage == AppStage.STAGE_1_GLOBAL and self.hazard_s1_active:
                self.hazard_s1.update(self.world_s1)
                h_dist = math.hypot(self.kinematics_s1.x - (self.hazard_s1.x + 0.5), self.kinematics_s1.y - (self.hazard_s1.y + 0.5))
                if h_dist < 0.85:
                    self.is_autonomous = False
                    self.cmd_s1 = VelocityCommand(v=0.0, omega=0.0)
                    self.crash_pos = (self.kinematics_s1.x, self.kinematics_s1.y)
                    self.crash_timer = 90
                    self.status_message = "💥 DIRECT CRASH with moving hazard! Robot had NO LiDAR to see it."
                    self.status_color = COLOR_ACCENT_RED

            # Stage 2 Hazard
            elif self.current_stage == AppStage.STAGE_2_LIDAR and self.hazard_s2_active:
                self.hazard_s2.update(self.world_s2)
                self._trigger_stage2_lidar()
                if self.path_s2 and any(self.occupancy_map_s2.is_blocked(px, py) for px, py in self.path_s2[self.waypoint_idx_s2:]):
                    self.status_message = "LiDAR laser spotted hazard ahead! Replanning detour..."
                    self._recompute_stage2_plan()

            # Stage 3 Hazard
            elif self.current_stage == AppStage.STAGE_3_HYBRID_RL and self.hazard_s3_active:
                self.hazard_s3.update(self.world_s3)
                h_dist = math.hypot(self.kinematics_s3.x - (self.hazard_s3.x + 0.5), self.kinematics_s3.y - (self.hazard_s3.y + 0.5))
                if h_dist < 0.85:
                    self.cmd_s3 = VelocityCommand(v=0.0, omega=0.0)
                    self.crash_pos = (self.kinematics_s3.x, self.kinematics_s3.y)
                    self.crash_timer = 90
                    self.crash_recovery_timer_s3 = 60
                    self.status_message = "💥 COLLISION ALERT! Hazard bumped AMR! Holding 2s..."
                    self.status_color = COLOR_ACCENT_RED

            self.last_hazard_tick = current_time

        # Update Autonomous Navigation Step
        if self.is_autonomous:
            if self.current_stage == AppStage.STAGE_1_GLOBAL:
                self._step_stage1_kinematic(dt)
            elif self.current_stage == AppStage.STAGE_2_LIDAR:
                self._step_stage2_kinematic(dt)
            elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
                self._step_stage3_hybrid(dt)

        return True

    def _apply_mouse_interaction(self, mouse_pos: Tuple[int, int], button: int) -> None:
        """Apply obstacle painting, erasing, or relocation into active stage world."""
        mx, my = mouse_pos
        if not (TOPBAR_HEIGHT <= my < TOPBAR_HEIGHT + GRID_HEIGHT):
            return

        grid_x = mx // CELL_SIZE
        grid_y = (my - TOPBAR_HEIGHT) // CELL_SIZE

        # Get active stage models
        if self.current_stage == AppStage.STAGE_1_GLOBAL:
            active_world = self.world_s1
            curr_cell = (int(math.floor(self.kinematics_s1.x)), int(math.floor(self.kinematics_s1.y)))
            goal_cell = (self.goal_s1[0], self.goal_s1[1])
        elif self.current_stage == AppStage.STAGE_2_LIDAR:
            active_world = self.world_s2
            curr_cell = (int(math.floor(self.kinematics_s2.x)), int(math.floor(self.kinematics_s2.y)))
            goal_cell = (self.goal_s2[0], self.goal_s2[1])
        else:
            active_world = self.world_s3
            curr_cell = (int(math.floor(self.kinematics_s3.x)), int(math.floor(self.kinematics_s3.y)))
            goal_cell = (self.goal_s3[0], self.goal_s3[1])

        if not active_world.in_bounds(grid_x, grid_y):
            return

        keys = pygame.key.get_pressed()
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        # Shift + Click: Relocate Robot
        if button == 1 and shift_held:
            if not active_world.is_blocked(grid_x, grid_y) and (grid_x, grid_y) != goal_cell:
                if self.current_stage == AppStage.STAGE_1_GLOBAL:
                    self.kinematics_s1.x = float(grid_x) + 0.5
                    self.kinematics_s1.y = float(grid_y) + 0.5
                    self.robot_s1.x = grid_x
                    self.robot_s1.y = grid_y
                    self.odometry_s1.correct_pose(self.kinematics_s1.x, self.kinematics_s1.y)
                    self._recompute_stage1_plan()
                elif self.current_stage == AppStage.STAGE_2_LIDAR:
                    self.kinematics_s2.x = float(grid_x) + 0.5
                    self.kinematics_s2.y = float(grid_y) + 0.5
                    self.robot_s2.x = grid_x
                    self.robot_s2.y = grid_y
                    self.odometry_s2.correct_pose(self.kinematics_s2.x, self.kinematics_s2.y)
                    self._trigger_stage2_lidar()
                    self._recompute_stage2_plan()
                elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
                    self.kinematics_s3.x = float(grid_x) + 0.5
                    self.kinematics_s3.y = float(grid_y) + 0.5
                    self.robot_s3.x = grid_x
                    self.robot_s3.y = grid_y
                    self._recompute_stage3_plan()
                self.status_message = f"Robot relocated to ({grid_x}, {grid_y})."
                self.status_color = COLOR_ACCENT_CYAN
            return

        # Left Click / Drag: Paint Obstacle (Place)
        if button == 1:
            if (grid_x, grid_y) in (curr_cell, goal_cell):
                return
            if not active_world.is_blocked(grid_x, grid_y):
                active_world.set_obstacle(grid_x, grid_y, blocked=True)
                if self.current_stage == AppStage.STAGE_1_GLOBAL:
                    self._recompute_stage1_plan()
                elif self.current_stage == AppStage.STAGE_2_LIDAR:
                    self._trigger_stage2_lidar()
                    self._recompute_stage2_plan()
                elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
                    self._recompute_stage3_plan()

        # Right Click / Drag: Erase Obstacle OR Move Goal
        elif button == 3:
            if active_world.is_blocked(grid_x, grid_y):
                # Erase obstacle
                active_world.set_obstacle(grid_x, grid_y, blocked=False)
                if self.current_stage == AppStage.STAGE_1_GLOBAL:
                    self._recompute_stage1_plan()
                elif self.current_stage == AppStage.STAGE_2_LIDAR:
                    self._trigger_stage2_lidar()
                    self._recompute_stage2_plan()
                elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
                    self._recompute_stage3_plan()
            elif not shift_held and (grid_x, grid_y) != curr_cell:
                # Move Goal
                if self.current_stage == AppStage.STAGE_1_GLOBAL:
                    self.goal_s1 = [grid_x, grid_y]
                    self._recompute_stage1_plan()
                elif self.current_stage == AppStage.STAGE_2_LIDAR:
                    self.goal_s2 = [grid_x, grid_y]
                    self._recompute_stage2_plan()
                elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
                    self.goal_s3 = [grid_x, grid_y]
                    self._recompute_stage3_plan()
                self.status_message = f"Goal relocated to ({grid_x}, {grid_y})."
                self.status_color = COLOR_GOAL

    def _handle_manual_move(self, key: int) -> None:
        """Manual keyboard controls."""
        move_map = {
            pygame.K_UP: (0, -1, -math.pi / 2),
            pygame.K_w: (0, -1, -math.pi / 2),
            pygame.K_DOWN: (0, 1, math.pi / 2),
            pygame.K_s: (0, 1, math.pi / 2),
            pygame.K_LEFT: (-1, 0, math.pi),
            pygame.K_a: (-1, 0, math.pi),
            pygame.K_RIGHT: (1, 0, 0.0),
            pygame.K_d: (1, 0, 0.0),
        }
        if key in move_map:
            dx, dy, target_theta = move_map[key]
            if self.current_stage == AppStage.STAGE_1_GLOBAL:
                success = self.robot_s1.move(dx, dy, self.world_s1)
                if success:
                    self.kinematics_s1.x = float(self.robot_s1.x) + 0.5
                    self.kinematics_s1.y = float(self.robot_s1.y) + 0.5
                    self.kinematics_s1.theta = target_theta
                    self.odometry_s1.update_from_discrete_move(dx, dy)
                    self._recompute_stage1_plan()
                else:
                    self.status_message = "Collision rejected!"
                    self.status_color = COLOR_ACCENT_RED
            elif self.current_stage == AppStage.STAGE_2_LIDAR:
                success = self.robot_s2.move(dx, dy, self.world_s2)
                if success:
                    self.kinematics_s2.x = float(self.robot_s2.x) + 0.5
                    self.kinematics_s2.y = float(self.robot_s2.y) + 0.5
                    self.kinematics_s2.theta = target_theta
                    self.odometry_s2.update_from_discrete_move(dx, dy)
                    self._trigger_stage2_lidar()
                    self._recompute_stage2_plan()
                else:
                    self.status_message = "Collision rejected!"
                    self.status_color = COLOR_ACCENT_RED
            elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
                success = self.robot_s3.move(dx, dy, self.world_s3)
                if success:
                    self.kinematics_s3.x = float(self.robot_s3.x) + 0.5
                    self.kinematics_s3.y = float(self.robot_s3.y) + 0.5
                    self.kinematics_s3.theta = target_theta
                    self._recompute_stage3_plan()
                else:
                    self.status_message = "Collision rejected!"
                    self.status_color = COLOR_ACCENT_RED

    def _step_stage1_kinematic(self, dt: float) -> None:
        if not self.path_s1 or self.waypoint_idx_s1 >= len(self.path_s1):
            self.is_autonomous = False
            self.cmd_s1 = VelocityCommand()
            return

        target_grid_pt = self.path_s1[self.waypoint_idx_s1]
        target_cont = (target_grid_pt[0] + 0.5, target_grid_pt[1] + 0.5)
        is_final = (self.waypoint_idx_s1 == len(self.path_s1) - 1)

        cmd, reached = self.controller.compute_command(self.kinematics_s1, target_cont, is_final_goal=is_final)
        self.cmd_s1 = cmd

        prev_x, prev_y = self.kinematics_s1.x, self.kinematics_s1.y
        self.kinematics_s1 = WaypointFollower.integrate_kinematics(self.kinematics_s1, cmd, dt)

        dx = self.kinematics_s1.x - prev_x
        dy = self.kinematics_s1.y - prev_y
        self.odometry_s1.estimated_pose.x += dx
        self.odometry_s1.estimated_pose.y += dy
        self.odometry_s1.estimated_pose.theta = self.kinematics_s1.theta

        self.robot_s1.x = int(math.floor(self.kinematics_s1.x))
        self.robot_s1.y = int(math.floor(self.kinematics_s1.y))

        # Continuous collision check with dynamic hazard (if active)
        if self.hazard_s1_active:
            haz_center_x = float(self.hazard_s1.x) + 0.5
            haz_center_y = float(self.hazard_s1.y) + 0.5
            if math.hypot(self.kinematics_s1.x - haz_center_x, self.kinematics_s1.y - haz_center_y) < 0.85:
                self.is_autonomous = False
                self.cmd_s1 = VelocityCommand(v=0.0, omega=0.0)
                self.crash_pos = (self.kinematics_s1.x, self.kinematics_s1.y)
                self.crash_timer = 120  # Keep flashing for 2 full seconds (120 frames at 60 FPS)
                self.status_message = "💥 DIRECT COLLISION: Smashed into moving hazard! (No LiDAR to detect it)"
                self.status_color = COLOR_ACCENT_RED
                return

        # Continuous collision check with static obstacle or boundary wall
        if self.world_s1.is_blocked(self.robot_s1.x, self.robot_s1.y):
            self.is_autonomous = False
            self.cmd_s1 = VelocityCommand(v=0.0, omega=0.0)
            self.crash_pos = (self.kinematics_s1.x, self.kinematics_s1.y)
            self.crash_timer = 120
            self.status_message = "💥 EMERGENCY STOP: Wall/Obstacle Collision! (Blind without LiDAR perception)"
            self.status_color = COLOR_ACCENT_RED
            return

        if reached:
            if is_final:
                self.is_autonomous = False
                self.cmd_s1 = VelocityCommand()
                self.status_message = "Goal reached successfully!"
                self.status_color = COLOR_GOAL
            else:
                self.waypoint_idx_s1 += 1

    def _step_stage2_kinematic(self, dt: float) -> None:
        if not self.path_s2 or self.waypoint_idx_s2 >= len(self.path_s2):
            self.is_autonomous = False
            self.cmd_s2 = VelocityCommand()
            return

        target_grid_pt = self.path_s2[self.waypoint_idx_s2]
        target_cont = (target_grid_pt[0] + 0.5, target_grid_pt[1] + 0.5)
        is_final = (self.waypoint_idx_s2 == len(self.path_s2) - 1)

        cmd, reached = self.controller.compute_command(self.kinematics_s2, target_cont, is_final_goal=is_final)
        self.cmd_s2 = cmd

        prev_x, prev_y = self.kinematics_s2.x, self.kinematics_s2.y
        self.kinematics_s2 = WaypointFollower.integrate_kinematics(self.kinematics_s2, cmd, dt)

        dx = self.kinematics_s2.x - prev_x
        dy = self.kinematics_s2.y - prev_y
        self.odometry_s2.estimated_pose.x += dx
        self.odometry_s2.estimated_pose.y += dy
        self.odometry_s2.estimated_pose.theta = self.kinematics_s2.theta

        self.robot_s2.x = int(math.floor(self.kinematics_s2.x))
        self.robot_s2.y = int(math.floor(self.kinematics_s2.y))

        self._trigger_stage2_lidar()

        # Dynamic replan if laser spots obstacle on path
        if any(self.occupancy_map_s2.is_blocked(px, py) for px, py in self.path_s2[self.waypoint_idx_s2:]):
            self.status_message = "LiDAR laser spotted obstacle ahead! Replanning detour..."
            self.status_color = COLOR_ACCENT_ORANGE
            self._recompute_stage2_plan()
            return

        if reached:
            if is_final:
                self.is_autonomous = False
                self.cmd_s2 = VelocityCommand()
                self.status_message = "Goal reached safely with LiDAR perception!"
                self.status_color = COLOR_GOAL
            else:
                self.waypoint_idx_s2 += 1

    def _step_stage3_hybrid(self, dt: float) -> None:
        """Stage 3: Hierarchical Hybrid Autonomy (Dijkstra Global Waypoints + 8-State Local RL Policy)."""
        if not self.path_s3 or self.waypoint_idx_s3 >= len(self.path_s3):
            self.is_autonomous = False
            self.cmd_s3 = VelocityCommand()
            return

        # Crash recovery countdown: hold stopped for 2 seconds (60 frames at 30 FPS)
        if self.crash_recovery_timer_s3 > 0:
            self.crash_recovery_timer_s3 -= 1
            self.cmd_s3 = VelocityCommand(v=0.0, omega=0.0)
            if self.crash_recovery_timer_s3 == 0:
                # 2 seconds elapsed: inspect surroundings via RL policy
                clearances = self._compute_stage3_clearances()
                state = self.rl_policy.discretize_state(clearances, front_thresh=0.68, side_thresh=0.52)
                if state != LocalState.ALL_BLOCKED:
                    self.status_message = "Corridor clear! Resuming autonomous route to destination..."
                    self.status_color = COLOR_ACCENT_GREEN
                    self._recompute_stage3_plan()
                else:
                    self.status_message = "Corridor still blocked. Holding until hazard clears..."
                    self.status_color = COLOR_ACCENT_YELLOW
                    self.crash_recovery_timer_s3 = 20  # re-inspect in 20 frames
            return

        # Check if an obstacle now blocks the upcoming path waypoints
        if any(self.world_s3.is_blocked(px, py) for px, py in self.path_s3[self.waypoint_idx_s3:]):
            self.status_message = "Path obstructed ahead! Replanning global route..."
            self.status_color = COLOR_ACCENT_ORANGE
            self._recompute_stage3_plan()
            return

        # 1. Perception: Read 36-ray LiDAR and compute 3 sector clearances (F, L, R)
        clearances = self._compute_stage3_clearances()

        # 2. State Discretization:
        # front_thresh = 0.68m: obstacle directly in front within stopping room
        # side_thresh = 0.52m: corridor pinched narrower than standard 1m warehouse aisle
        state = self.rl_policy.discretize_state(
            clearances,
            front_thresh=0.68,
            side_thresh=0.52,
        )
        self.last_rl_state = state

        # 3. Policy Execution: Query Q-table for greedy action
        action = self.rl_policy.select_action(state, greedy=True)
        self.last_rl_action = action

        # 4. Global Dijkstra Waypoint Pursuit
        target_grid_pt = self.path_s3[self.waypoint_idx_s3]
        target_cont = (target_grid_pt[0] + 0.5, target_grid_pt[1] + 0.5)
        is_final = (self.waypoint_idx_s3 == len(self.path_s3) - 1)

        base_cmd, reached = self.controller.compute_command(
            self.kinematics_s3, target_cont, is_final_goal=is_final
        )

        # 5. Local RL Tactical Action Arbitration
        if action == LocalAction.CRUISE:
            self.yield_wait_counter_s3 = 0
            cmd = base_cmd
        elif action == LocalAction.YIELD_WAIT:
            self.yield_wait_counter_s3 += 1
            cmd = VelocityCommand(v=0.0, omega=0.0)
            reached = False
            # Freezing Robot Problem Watchdog:
            # If robot is stopped waiting for > 40 frames (~1.3s) and corridor remains blocked,
            # trigger Dijkstra replan to compute an unfreezing detour!
            if self.yield_wait_counter_s3 > 40:
                self.status_message = "Corridor blocked! Replanning detour to unfreeze..."
                self.status_color = COLOR_ACCENT_ORANGE
                self._recompute_stage3_plan()
                self.yield_wait_counter_s3 = 0
                return
        elif action == LocalAction.BYPASS_LEFT:
            self.yield_wait_counter_s3 = 0
            cmd = VelocityCommand(v=min(base_cmd.v, 1.0), omega=base_cmd.omega + 0.6)
        elif action == LocalAction.BYPASS_RIGHT:
            self.yield_wait_counter_s3 = 0
            cmd = VelocityCommand(v=min(base_cmd.v, 1.0), omega=base_cmd.omega - 0.6)
        else:
            self.yield_wait_counter_s3 = 0
            cmd = base_cmd

        self.cmd_s3 = cmd
        self.kinematics_s3 = WaypointFollower.integrate_kinematics(self.kinematics_s3, cmd, dt)
        self.robot_s3.x = int(math.floor(self.kinematics_s3.x))
        self.robot_s3.y = int(math.floor(self.kinematics_s3.y))

        # Check collision with dynamic hazard
        if self.hazard_s3_active:
            haz_center_x = float(self.hazard_s3.x) + 0.5
            haz_center_y = float(self.hazard_s3.y) + 0.5
            if math.hypot(self.kinematics_s3.x - haz_center_x, self.kinematics_s3.y - haz_center_y) < 0.85:
                self.cmd_s3 = VelocityCommand(v=0.0, omega=0.0)
                self.crash_pos = (self.kinematics_s3.x, self.kinematics_s3.y)
                self.crash_timer = 90
                self.crash_recovery_timer_s3 = 60  # Hold stopped for 2 seconds
                self.status_message = "💥 COLLISION ALERT! Hazard hit robot. Stopping 2s to yield..."
                self.status_color = COLOR_ACCENT_RED
                return

        # Check collision with static obstacle / boundary wall
        if self.world_s3.is_blocked(self.robot_s3.x, self.robot_s3.y):
            self.cmd_s3 = VelocityCommand(v=0.0, omega=0.0)
            self.crash_pos = (self.kinematics_s3.x, self.kinematics_s3.y)
            self.crash_timer = 90
            self.crash_recovery_timer_s3 = 60  # Hold stopped for 2 seconds
            self.status_message = "💥 WALL COLLISION ALERT! Holding 2s before rerouting..."
            self.status_color = COLOR_ACCENT_RED
            return

        if reached:
            if is_final:
                self.is_autonomous = False
                self.cmd_s3 = VelocityCommand()
                self.status_message = "Goal reached safely with Hybrid RL Autonomy!"
                self.status_color = COLOR_GOAL
            else:
                self.waypoint_idx_s3 += 1

    def render(self) -> None:
        self.screen.fill(COLOR_BG)
        self.lidar_surface.fill((0, 0, 0, 0))

        # 1. Render Top Navigation Bar
        self._render_topbar()

        # 2. Render Active World Grid
        if self.current_stage == AppStage.STAGE_1_GLOBAL:
            active_world = self.world_s1
        elif self.current_stage == AppStage.STAGE_2_LIDAR:
            active_world = self.world_s2
        else:
            active_world = self.world_s3

        for row in range(active_world.height):
            for col in range(active_world.width):
                cell_rect = pygame.Rect(
                    col * CELL_SIZE, TOPBAR_HEIGHT + row * CELL_SIZE, CELL_SIZE, CELL_SIZE
                )

                if self.current_stage == AppStage.STAGE_2_LIDAR:
                    state = self.occupancy_map_s2.get_state(col, row)
                    if state == int(CellState.UNKNOWN):
                        pygame.draw.rect(self.screen, COLOR_UNKNOWN_CELL, cell_rect)
                        pygame.draw.rect(self.screen, COLOR_UNKNOWN_GRID, cell_rect, 1)
                    elif state == int(CellState.OCCUPIED):
                        pygame.draw.rect(self.screen, COLOR_OBSTACLE, cell_rect)
                        pygame.draw.rect(self.screen, COLOR_OBSTACLE_BORDER, cell_rect, 1)
                    else:
                        pygame.draw.rect(self.screen, COLOR_GRID_BG, cell_rect)
                        pygame.draw.rect(self.screen, COLOR_GRID_LINE, cell_rect, 1)
                else:
                    if active_world.is_blocked(col, row):
                        # Moving hazard
                        if (self.current_stage == AppStage.STAGE_1_GLOBAL and self.hazard_s1_active and (col, row) == (self.hazard_s1.x, self.hazard_s1.y)) or \
                           (self.current_stage == AppStage.STAGE_2_LIDAR and self.hazard_s2_active and (col, row) == (self.hazard_s2.x, self.hazard_s2.y)) or \
                           (self.current_stage == AppStage.STAGE_3_HYBRID_RL and self.hazard_s3_active and (col, row) == (self.hazard_s3.x, self.hazard_s3.y)):
                            pygame.draw.rect(self.screen, COLOR_DYNAMIC_OBS, cell_rect)
                            pygame.draw.rect(self.screen, COLOR_DYNAMIC_BORDER, cell_rect, 1)
                        else:
                            pygame.draw.rect(self.screen, COLOR_OBSTACLE, cell_rect)
                            pygame.draw.rect(self.screen, COLOR_OBSTACLE_BORDER, cell_rect, 1)
                    else:
                        pygame.draw.rect(self.screen, COLOR_GRID_BG, cell_rect)
                        pygame.draw.rect(self.screen, COLOR_GRID_LINE, cell_rect, 1)

        # 3. Render Active Stage Specific Actors
        if self.current_stage == AppStage.STAGE_1_GLOBAL:
            self._render_stage1_scene()
        elif self.current_stage == AppStage.STAGE_2_LIDAR:
            self._render_stage2_scene()
        elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
            self._render_stage3_scene()

        # 4. Render Telemetry HUD
        self._render_hud()
        pygame.display.flip()

    def _render_topbar(self) -> None:
        topbar_rect = pygame.Rect(0, 0, WINDOW_WIDTH, TOPBAR_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_TOPBAR_BG, topbar_rect)
        pygame.draw.line(self.screen, COLOR_HUD_BORDER, (0, TOPBAR_HEIGHT), (WINDOW_WIDTH, TOPBAR_HEIGHT), 2)

        tab_width = WINDOW_WIDTH // 3
        stages = [
            (AppStage.STAGE_1_GLOBAL, "[1] 1: GLOBAL GOD", COLOR_ACCENT_CYAN),
            (AppStage.STAGE_2_LIDAR, "[2] 2: LIDAR FOG", COLOR_ACCENT_ORANGE),
            (AppStage.STAGE_3_HYBRID_RL, "[3] 3: HYBRID RL", COLOR_ACCENT_PURPLE),
        ]

        for i, (stage, label, accent_col) in enumerate(stages):
            is_active = (self.current_stage == stage)
            rect = pygame.Rect(i * tab_width, 0, tab_width, TOPBAR_HEIGHT)
            bg_col = COLOR_TAB_ACTIVE if is_active else COLOR_TAB_INACTIVE
            pygame.draw.rect(self.screen, bg_col, rect)
            pygame.draw.rect(self.screen, COLOR_HUD_BORDER, rect, 1)

            if is_active:
                pygame.draw.line(self.screen, accent_col, (i * tab_width, TOPBAR_HEIGHT - 3), ((i + 1) * tab_width, TOPBAR_HEIGHT - 3), 3)

            text_col = accent_col if is_active else COLOR_TEXT_MUTED
            surf = self.font_tab.render(label, True, text_col)
            tx = rect.centerx - surf.get_width() // 2
            ty = rect.centery - surf.get_height() // 2
            self.screen.blit(surf, (tx, ty))

    def _render_stage1_scene(self) -> None:
        """Render Stage 1: Path, Goal, Odometry Ghost, and Robot."""
        # 1. Path
        if len(self.path_s1) > 1:
            pts = [(int(px * CELL_SIZE + CELL_SIZE / 2), int(TOPBAR_HEIGHT + py * CELL_SIZE + CELL_SIZE / 2)) for px, py in self.path_s1]
            pygame.draw.lines(self.screen, COLOR_ACCENT_CYAN, False, pts, 3)
            for pt in pts:
                pygame.draw.circle(self.screen, (125, 211, 252), pt, 3)

        # 2. Goal
        g_rect = pygame.Rect(self.goal_s1[0] * CELL_SIZE, TOPBAR_HEIGHT + self.goal_s1[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, COLOR_GOAL, g_rect)
        pygame.draw.rect(self.screen, COLOR_GOAL_BORDER, g_rect, 2)
        inner_g = g_rect.inflate(-16, -16)
        pygame.draw.rect(self.screen, (255, 255, 255), inner_g)

        # 3. Odometry Ghost (Magenta circle)
        est_cx = int(self.odometry_s1.estimated_pose.x * CELL_SIZE)
        est_cy = int(TOPBAR_HEIGHT + self.odometry_s1.estimated_pose.y * CELL_SIZE)
        pygame.draw.circle(self.screen, COLOR_ODOM_GHOST, (est_cx, est_cy), int(CELL_SIZE * 0.36), 2)

        # 4. Robot Body & Heading
        rcx = int(self.kinematics_s1.x * CELL_SIZE)
        rcy = int(TOPBAR_HEIGHT + self.kinematics_s1.y * CELL_SIZE)
        radius = int(CELL_SIZE * 0.38)
        pygame.draw.circle(self.screen, COLOR_ACCENT_CYAN, (rcx, rcy), radius)
        pygame.draw.circle(self.screen, (2, 132, 199), (rcx, rcy), radius - 4)
        pygame.draw.circle(self.screen, COLOR_ROBOT_BORDER, (rcx, rcy), radius, 2)

        hx = rcx + int(math.cos(self.kinematics_s1.theta) * (radius + 6))
        hy = rcy + int(math.sin(self.kinematics_s1.theta) * (radius + 6))
        pygame.draw.line(self.screen, COLOR_ROBOT_HEADING, (rcx, rcy), (hx, hy), 3)

        # 5. Visual Crash Shockwave & Impact Indicator
        if self.crash_timer > 0 and self.crash_pos:
            cpx = int(self.crash_pos[0] * CELL_SIZE)
            cpy = int(TOPBAR_HEIGHT + self.crash_pos[1] * CELL_SIZE)
            pulse_radius = int(CELL_SIZE * (0.6 + ((120 - self.crash_timer) % 30) / 12.0))
            # Draw expanding pulsing red shockwave rings
            pygame.draw.circle(self.screen, (239, 68, 68), (cpx, cpy), pulse_radius, 3)
            pygame.draw.circle(self.screen, (252, 165, 165), (cpx, cpy), max(6, pulse_radius - 8), 2)
            # Draw flashing impact X
            pygame.draw.line(self.screen, (255, 255, 255), (cpx - 14, cpy - 14), (cpx + 14, cpy + 14), 4)
            pygame.draw.line(self.screen, (255, 255, 255), (cpx - 14, cpy + 14), (cpx + 14, cpy - 14), 4)
            pygame.draw.line(self.screen, (239, 68, 68), (cpx - 14, cpy - 14), (cpx + 14, cpy + 14), 2)
            pygame.draw.line(self.screen, (239, 68, 68), (cpx - 14, cpy + 14), (cpx + 14, cpy - 14), 2)
            self.crash_timer -= 1

    def _render_stage2_scene(self) -> None:
        """Render Stage 2: Path, LiDAR Laser Beams with Red Contact Dots, Goal, Ghost, and Robot."""
        # 1. Path
        if len(self.path_s2) > 1:
            pts = [(int(px * CELL_SIZE + CELL_SIZE / 2), int(TOPBAR_HEIGHT + py * CELL_SIZE + CELL_SIZE / 2)) for px, py in self.path_s2]
            pygame.draw.lines(self.screen, COLOR_ACCENT_ORANGE, False, pts, 3)
            for pt in pts:
                pygame.draw.circle(self.screen, (253, 186, 116), pt, 3)

        # 2. LiDAR Laser Beams & Red Endpoints
        if self.last_scan_s2:
            ox = int(self.last_scan_s2.origin[0] * CELL_SIZE)
            oy = int(self.last_scan_s2.origin[1] * CELL_SIZE)
            for hx, hy in self.last_scan_s2.hit_points:
                px = int(hx * CELL_SIZE)
                py = int(hy * CELL_SIZE)
                pygame.draw.line(self.lidar_surface, (34, 197, 94, 70), (ox, oy), (px, py), 1)
                pygame.draw.circle(self.lidar_surface, (239, 68, 68, 220), (px, py), 2)
            self.screen.blit(self.lidar_surface, (0, TOPBAR_HEIGHT))

        # 3. Goal
        g_rect = pygame.Rect(self.goal_s2[0] * CELL_SIZE, TOPBAR_HEIGHT + self.goal_s2[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, COLOR_GOAL, g_rect)
        pygame.draw.rect(self.screen, COLOR_GOAL_BORDER, g_rect, 2)
        inner_g = g_rect.inflate(-16, -16)
        pygame.draw.rect(self.screen, (255, 255, 255), inner_g)

        # 4. Odometry Ghost
        est_cx = int(self.odometry_s2.estimated_pose.x * CELL_SIZE)
        est_cy = int(TOPBAR_HEIGHT + self.odometry_s2.estimated_pose.y * CELL_SIZE)
        pygame.draw.circle(self.screen, COLOR_ODOM_GHOST, (est_cx, est_cy), int(CELL_SIZE * 0.36), 2)

        # 5. Robot Body & Heading
        rcx = int(self.kinematics_s2.x * CELL_SIZE)
        rcy = int(TOPBAR_HEIGHT + self.kinematics_s2.y * CELL_SIZE)
        radius = int(CELL_SIZE * 0.38)
        pygame.draw.circle(self.screen, COLOR_ACCENT_CYAN, (rcx, rcy), radius)
        pygame.draw.circle(self.screen, (2, 132, 199), (rcx, rcy), radius - 4)
        pygame.draw.circle(self.screen, COLOR_ROBOT_BORDER, (rcx, rcy), radius, 2)

        hx = rcx + int(math.cos(self.kinematics_s2.theta) * (radius + 6))
        hy = rcy + int(math.sin(self.kinematics_s2.theta) * (radius + 6))
        pygame.draw.line(self.screen, COLOR_ROBOT_HEADING, (rcx, rcy), (hx, hy), 3)

    def _render_stage3_scene(self) -> None:
        """Render Stage 3: Global Dijkstra Path, LiDAR rays, Goal, and Robot."""
        # 1. Planned Global Dijkstra Path (Purple)
        if len(self.path_s3) > 1:
            pts = [
                (int(px * CELL_SIZE + CELL_SIZE / 2), int(TOPBAR_HEIGHT + py * CELL_SIZE + CELL_SIZE / 2))
                for px, py in self.path_s3
            ]
            pygame.draw.lines(self.screen, COLOR_ACCENT_PURPLE, False, pts, 3)
            for pt in pts:
                pygame.draw.circle(self.screen, (216, 180, 254), pt, 3)

        # 2. 36-ray LiDAR Laser Beams & Contact Endpoints
        if self.last_scan_s3:
            ox = int(self.last_scan_s3.origin[0] * CELL_SIZE)
            oy = int(self.last_scan_s3.origin[1] * CELL_SIZE)
            for hx, hy in self.last_scan_s3.hit_points:
                px = int(hx * CELL_SIZE)
                py = int(hy * CELL_SIZE)
                pygame.draw.line(self.lidar_surface, (168, 85, 247, 70), (ox, oy), (px, py), 1)
                pygame.draw.circle(self.lidar_surface, (239, 68, 68, 220), (px, py), 2)
            self.screen.blit(self.lidar_surface, (0, TOPBAR_HEIGHT))

        # 3. Goal
        g_rect = pygame.Rect(
            self.goal_s3[0] * CELL_SIZE,
            TOPBAR_HEIGHT + self.goal_s3[1] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(self.screen, COLOR_GOAL, g_rect)
        pygame.draw.rect(self.screen, COLOR_GOAL_BORDER, g_rect, 2)
        inner_g = g_rect.inflate(-16, -16)
        pygame.draw.rect(self.screen, (255, 255, 255), inner_g)

        # 4. Robot Body & Heading Pointer
        rcx = int(self.kinematics_s3.x * CELL_SIZE)
        rcy = int(TOPBAR_HEIGHT + self.kinematics_s3.y * CELL_SIZE)
        radius = int(CELL_SIZE * 0.38)
        pygame.draw.circle(self.screen, COLOR_ACCENT_PURPLE, (rcx, rcy), radius)
        pygame.draw.circle(self.screen, (88, 28, 135), (rcx, rcy), radius - 4)
        pygame.draw.circle(self.screen, COLOR_ROBOT_BORDER, (rcx, rcy), radius, 2)

        hx = rcx + int(math.cos(self.kinematics_s3.theta) * (radius + 6))
        hy = rcy + int(math.sin(self.kinematics_s3.theta) * (radius + 6))
        pygame.draw.line(self.screen, COLOR_ROBOT_HEADING, (rcx, rcy), (hx, hy), 3)

        # 5. Visual Crash Shockwave & Impact Indicator
        if self.crash_timer > 0 and self.crash_pos:
            cpx = int(self.crash_pos[0] * CELL_SIZE)
            cpy = int(TOPBAR_HEIGHT + self.crash_pos[1] * CELL_SIZE)
            pulse_radius = int(CELL_SIZE * (0.6 + ((120 - self.crash_timer) % 30) / 12.0))
            pygame.draw.circle(self.screen, (239, 68, 68), (cpx, cpy), pulse_radius, 3)
            pygame.draw.line(self.screen, (255, 255, 255), (cpx - 14, cpy - 14), (cpx + 14, cpy + 14), 4)
            pygame.draw.line(self.screen, (255, 255, 255), (cpx - 14, cpy + 14), (cpx + 14, cpy - 14), 4)
            self.crash_timer -= 1

    def _render_hud(self) -> None:
        hud_y = TOPBAR_HEIGHT + GRID_HEIGHT
        hud_rect = pygame.Rect(0, hud_y, WINDOW_WIDTH, HUD_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HUD_BG, hud_rect)
        pygame.draw.line(self.screen, COLOR_HUD_BORDER, (0, hud_y), (WINDOW_WIDTH, hud_y), 2)

        if self.current_stage == AppStage.STAGE_1_GLOBAL:
            path_cost = f"{len(self.path_s1) - 1}" if self.path_s1 else "N/A"
            loc_err = self.odometry_s1.calculate_localization_error(self.kinematics_s1.x, self.kinematics_s1.y)
            hazard_str = "ACTIVE" if self.hazard_s1_active else "OFF"

            title_surf = self.font_title.render("STAGE 1: GLOBAL GOD MODE", True, COLOR_ACCENT_CYAN)
            pos_surf = self.font_main.render(f"Pose: ({self.kinematics_s1.x:.1f}, {self.kinematics_s1.y:.1f})", True, COLOR_TEXT_PRIMARY)
            drift_surf = self.font_main.render(f"Drift: {loc_err:.2f}c", True, COLOR_ACCENT_MAGENTA)
            hazard_surf = self.font_main.render(f"Sudden Hazard [D]: {hazard_str}", True, COLOR_ACCENT_RED if self.hazard_s1_active else COLOR_TEXT_MUTED)
            cost_surf = self.font_main.render(f"Cost: {path_cost}", True, COLOR_ACCENT_GREEN)

            self.screen.blit(title_surf, (16, hud_y + 8))
            self.screen.blit(pos_surf, (230, hud_y + 8))
            self.screen.blit(drift_surf, (350, hud_y + 8))
            self.screen.blit(hazard_surf, (470, hud_y + 8))
            self.screen.blit(cost_surf, (690, hud_y + 8))

            status_surf = self.font_status.render(f"Status: {self.status_message}", True, self.status_color)
            self.screen.blit(status_surf, (16, hud_y + 34))

            controls_surf = self.font_controls.render(
                "[Space] Auto Nav | [D] Sudden Hazard | [O] Drift | [C] Sync | [Left-Drag] Paint | [Right-Drag] Erase | [R] Reset | [1-3] Tabs",
                True,
                COLOR_TEXT_MUTED,
            )
            self.screen.blit(controls_surf, (16, hud_y + 60))

        elif self.current_stage == AppStage.STAGE_2_LIDAR:
            explored_pct = int(self.occupancy_map_s2.get_explored_ratio() * 100)
            loc_err = self.odometry_s2.calculate_localization_error(self.kinematics_s2.x, self.kinematics_s2.y)
            hazard_str = "ACTIVE" if self.hazard_s2_active else "OFF"

            title_surf = self.font_title.render("STAGE 2: LIDAR EXPLORATION", True, COLOR_ACCENT_ORANGE)
            explored_surf = self.font_main.render(f"Explored: {explored_pct}%", True, COLOR_ACCENT_YELLOW)
            drift_surf = self.font_main.render(f"Odom Drift: {loc_err:.2f}c", True, COLOR_ACCENT_MAGENTA)
            hazard_surf = self.font_main.render(f"Hazard [D]: {hazard_str}", True, COLOR_ACCENT_RED if self.hazard_s2_active else COLOR_TEXT_MUTED)

            self.screen.blit(title_surf, (16, hud_y + 8))
            self.screen.blit(explored_surf, (240, hud_y + 8))
            self.screen.blit(drift_surf, (380, hud_y + 8))
            self.screen.blit(hazard_surf, (550, hud_y + 8))

            status_surf = self.font_status.render(f"Status: {self.status_message}", True, self.status_color)
            self.screen.blit(status_surf, (16, hud_y + 34))

            controls_surf = self.font_controls.render(
                "[Space] Auto Nav | [D] Hazard | [C] Sync | [O] Drift | [Left-Drag] Paint | [Right-Drag] Erase | [R] Reset | [1-3] Tabs",
                True,
                COLOR_TEXT_MUTED,
            )
            self.screen.blit(controls_surf, (16, hud_y + 60))

        elif self.current_stage == AppStage.STAGE_3_HYBRID_RL:
            state_enum = LocalState(self.last_rl_state)
            action_name = self.last_rl_action.name
            q_vals = self.rl_policy.get_q_values(self.last_rl_state)
            haz_str = "ACTIVE" if self.hazard_s3_active else "OFF"

            title_surf = self.font_title.render("STAGE 3: HYBRID RL AUTONOMY", True, COLOR_ACCENT_PURPLE)
            state_surf = self.font_main.render(f"State: {state_enum.name} ({self.last_rl_state})", True, COLOR_ACCENT_YELLOW)
            action_surf = self.font_main.render(f"Action: {action_name}", True, COLOR_ACCENT_CYAN)
            hazard_surf = self.font_main.render(f"Hazard [D]: {haz_str}", True, COLOR_ACCENT_RED if self.hazard_s3_active else COLOR_TEXT_MUTED)

            self.screen.blit(title_surf, (16, hud_y + 8))
            self.screen.blit(state_surf, (270, hud_y + 8))
            self.screen.blit(action_surf, (480, hud_y + 8))
            self.screen.blit(hazard_surf, (640, hud_y + 8))

            c = self.last_sector_clearances
            clear_str = f"LiDAR: Front={c['front']:.1f}m | Left={c['left']:.1f}m | Right={c['right']:.1f}m"
            q_str = f"Q(s,·): [CRUISE:{q_vals.get('CRUISE', 0.0):.1f}, WAIT:{q_vals.get('YIELD_WAIT', 0.0):.1f}, L:{q_vals.get('BYPASS_LEFT', 0.0):.1f}, R:{q_vals.get('BYPASS_RIGHT', 0.0):.1f}]"
            detail_surf = self.font_status.render(f"{clear_str}   |   {q_str}", True, COLOR_TEXT_PRIMARY)
            self.screen.blit(detail_surf, (16, hud_y + 34))

            controls_surf = self.font_controls.render(
                "[Space] Auto Nav | [D] Hazard | [Left-Drag] Paint | [Right-Drag] Erase/Goal | [Shift+Click] Relocate | [R] Reset | [1-3] Tabs",
                True,
                COLOR_TEXT_MUTED,
            )
            self.screen.blit(controls_surf, (16, hud_y + 60))

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_input()
            self.render()
            self.clock.tick(TARGET_FPS)
        pygame.quit()


def main() -> None:
    app = AtlasSimulatorApp()
    app.run()


if __name__ == "__main__":
    main()
