"""Motion control and kinematic tracking module for Atlas Robotics."""

from atlas_robotics.control.waypoint_follower import (
    KinematicState,
    VelocityCommand,
    WaypointFollower,
    normalize_angle,
)

__all__ = [
    "KinematicState",
    "VelocityCommand",
    "WaypointFollower",
    "normalize_angle",
]
