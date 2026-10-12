"""Kinematic motion control and waypoint tracking algorithms."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


def normalize_angle(angle: float) -> float:
    """Normalize an angle in radians to the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class KinematicState:
    """Continuous 2D kinematic pose of the mobile robot."""

    x: float
    y: float
    theta: float = 0.0  # Heading angle in radians (0 = Facing Right +X, pi/2 = Facing Down +Y)

    @property
    def heading_degrees(self) -> float:
        """Heading in human-readable degrees [0, 360)."""
        deg = math.degrees(self.theta) % 360.0
        return deg


@dataclass
class VelocityCommand:
    """Body-frame velocity control signals for a differential drive / unicycle robot."""

    v: float = 0.0      # Linear velocity (grid cells / second)
    omega: float = 0.0  # Angular velocity (radians / second)


class WaypointFollower:
    """Closed-loop kinematic controller for smooth path tracking.

    Computes bounded linear and angular velocity commands to steer the robot
    towards sequential waypoints and smoothly converge at the final goal.
    """

    def __init__(
        self,
        max_linear_vel: float = 3.0,
        max_angular_vel: float = 4.5,
        k_p_linear: float = 2.5,
        k_p_angular: float = 5.0,
        waypoint_tolerance: float = 0.25,
        goal_tolerance: float = 0.10,
        pivot_angle_threshold: float = math.radians(45.0),
    ) -> None:
        self.max_linear_vel: float = max_linear_vel
        self.max_angular_vel: float = max_angular_vel
        self.k_p_linear: float = k_p_linear
        self.k_p_angular: float = k_p_angular
        self.waypoint_tolerance: float = waypoint_tolerance
        self.goal_tolerance: float = goal_tolerance
        self.pivot_angle_threshold: float = pivot_angle_threshold

    def compute_command(
        self,
        current_state: KinematicState,
        target_waypoint: Tuple[float, float],
        is_final_goal: bool = False,
    ) -> Tuple[VelocityCommand, bool]:
        """Compute velocity command [v, omega] to track the current target waypoint.

        Args:
            current_state: Current continuous robot pose (x, y, theta).
            target_waypoint: Target (x, y) coordinates in continuous grid units.
            is_final_goal: True if this is the destination goal waypoint.

        Returns:
            Tuple of (VelocityCommand, waypoint_reached_flag).
        """
        tx, ty = target_waypoint
        dx = tx - current_state.x
        dy = ty - current_state.y
        distance = math.hypot(dx, dy)

        tolerance = self.goal_tolerance if is_final_goal else self.waypoint_tolerance

        # Check if waypoint is reached
        if distance <= tolerance:
            if is_final_goal:
                return VelocityCommand(v=0.0, omega=0.0), True
            return VelocityCommand(v=0.0, omega=0.0), True

        # Desired heading towards waypoint
        desired_theta = math.atan2(dy, dx)
        heading_error = normalize_angle(desired_theta - current_state.theta)

        # Proportional angular steering
        raw_omega = self.k_p_angular * heading_error
        omega = max(-self.max_angular_vel, min(self.max_angular_vel, raw_omega))

        # If heading error is large, pivot in place to avoid wild arcs
        if abs(heading_error) > self.pivot_angle_threshold:
            v = 0.0
        else:
            # Scale linear speed with distance and heading alignment
            alignment_factor = max(0.0, math.cos(heading_error))
            raw_v = self.k_p_linear * distance * alignment_factor
            v = max(0.0, min(self.max_linear_vel, raw_v))

        return VelocityCommand(v=v, omega=omega), False

    @staticmethod
    def integrate_kinematics(
        state: KinematicState, cmd: VelocityCommand, dt: float
    ) -> KinematicState:
        """Advance the kinematic state forward by time step dt using unicycle model."""
        new_theta = normalize_angle(state.theta + cmd.omega * dt)
        new_x = state.x + cmd.v * math.cos(state.theta) * dt
        new_y = state.y + cmd.v * math.sin(state.theta) * dt

        return KinematicState(x=new_x, y=new_y, theta=new_theta)
