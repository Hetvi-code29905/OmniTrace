import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Pose2D:
    """Represents a 2D continuous pose (x, y, heading in radians)."""

    x: float
    y: float
    theta: float = 0.0

    @property
    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


class OdometryEstimator:
    """Simulates wheel encoder odometry subject to cumulative drift and noise.

    Maintains estimated robot pose separate from simulator ground truth.
    """

    def __init__(
        self,
        start_x: float = 1.0,
        start_y: float = 1.0,
        start_theta: float = 0.0,
        translational_noise_std: float = 0.04,
        rotational_noise_std: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        self.start_pose = Pose2D(start_x, start_y, start_theta)
        self.estimated_pose = Pose2D(start_x, start_y, start_theta)
        self.translational_noise_std: float = translational_noise_std
        self.rotational_noise_std: float = rotational_noise_std
        self.noise_enabled: bool = True
        self.random_gen = random.Random(seed)

    def update_from_discrete_move(self, dx: int, dy: int) -> Pose2D:
        """Update dead-reckoning position estimate after a discrete movement.

        Injects Gaussian drift to simulate wheel slip and encoder quantization.
        """
        noise_x = 0.0
        noise_y = 0.0

        if self.noise_enabled and (dx != 0 or dy != 0):
            noise_x = self.random_gen.gauss(0.0, self.translational_noise_std)
            noise_y = self.random_gen.gauss(0.0, self.translational_noise_std)

        self.estimated_pose.x += dx + noise_x
        self.estimated_pose.y += dy + noise_y

        # Update heading towards movement direction
        if dx != 0 or dy != 0:
            target_theta = math.atan2(dy, dx)
            rot_noise = (
                self.random_gen.gauss(0.0, self.rotational_noise_std)
                if self.noise_enabled
                else 0.0
            )
            self.estimated_pose.theta = target_theta + rot_noise

        return self.estimated_pose

    def calculate_localization_error(self, true_x: float, true_y: float) -> float:
        """Compute Euclidean distance between estimated pose and simulator ground truth."""
        return math.hypot(self.estimated_pose.x - true_x, self.estimated_pose.y - true_y)

    def correct_pose(self, true_x: float, true_y: float) -> None:
        """Simulate landmark/GPS/AMCL pose correction resetting drift."""
        self.estimated_pose.x = true_x
        self.estimated_pose.y = true_y

    def reset(self) -> None:
        """Reset estimated pose back to starting coordinates."""
        self.estimated_pose = Pose2D(
            self.start_pose.x, self.start_pose.y, self.start_pose.theta
        )
