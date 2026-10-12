"""Local Reactive Reinforcement Learning Policy for AMR Obstacle Avoidance.

Hierarchical Hybrid Navigation:
- Layer 1 (Global): Dijkstra computes the optimal topological route.
- Layer 2 (Local): This 8-state Tabular Q-Learning policy handles real-time dynamic hazard avoidance.

MDP Formulation:
- State Space (8 States): 3 binary clearance sectors from LiDAR (Front, Left, Right).
- Action Space (4 Actions): CRUISE, YIELD_WAIT, BYPASS_LEFT, BYPASS_RIGHT.
- Algorithm: Off-policy Q-Learning with Bellman Temporal Difference updates:
    Q(s, a) <- Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
"""

from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np


class LocalAction(IntEnum):
    """4 tactical behaviors for reactive obstacle handling."""
    CRUISE = 0       # Follow Dijkstra global guidepath
    YIELD_WAIT = 1   # Stop in place and yield for moving hazard
    BYPASS_LEFT = 2  # Evade / steer around obstacle to the left
    BYPASS_RIGHT = 3 # Evade / steer around obstacle to the right


class LocalState(IntEnum):
    """8 discrete perception states from 3 binary LiDAR clearance zones:
    Bit 2 (4): Front blocked?
    Bit 1 (2): Left blocked?
    Bit 0 (1): Right blocked?
    """
    ALL_CLEAR = 0              # (F=0, L=0, R=0) -> Road ahead and sides wide open
    RIGHT_BLOCKED = 1          # (F=0, L=0, R=1) -> Front clear, right side blocked
    LEFT_BLOCKED = 2           # (F=0, L=1, R=0) -> Front clear, left side blocked
    SIDES_BLOCKED = 3          # (F=0, L=1, R=1) -> Narrow corridor, front is clear
    FRONT_BLOCKED = 4          # (F=1, L=0, R=0) -> Obstacle directly ahead, both sides open
    FRONT_RIGHT_BLOCKED = 5    # (F=1, L=0, R=1) -> Front & right blocked, left is open
    FRONT_LEFT_BLOCKED = 6     # (F=1, L=1, R=0) -> Front & left blocked, right is open
    ALL_BLOCKED = 7            # (F=1, L=1, R=1) -> Obstacle ahead in narrow aisle (trapped)


class LocalRLPolicy:
    """Explainable 8-state Tabular Q-Learning Agent for local reactive navigation."""

    ACTION_NAMES: List[str] = ["CRUISE", "YIELD_WAIT", "BYPASS_LEFT", "BYPASS_RIGHT"]
    STATE_DESCRIPTIONS: List[str] = [
        "All Clear",
        "Right Blocked",
        "Left Blocked",
        "Sides Blocked",
        "Front Blocked",
        "Front+Right Blocked",
        "Front+Left Blocked",
        "All Blocked",
    ]

    def __init__(
        self,
        learning_rate: float = 0.25,
        discount_factor: float = 0.90,
        epsilon: float = 0.10,
    ) -> None:
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-Table shape: (8 states, 4 actions) -> exactly 32 values!
        self.q_table: np.ndarray = np.zeros((8, 4), dtype=np.float32)
        self.episodes_trained: int = 0

        # Pre-train the 8 states so the policy is ready for immediate deployment
        self.train(episodes=300)

    @staticmethod
    def discretize_state(
        clearances: Dict[str, float],
        threshold: float = 1.35,
        front_thresh: Optional[float] = None,
        side_thresh: Optional[float] = None,
    ) -> int:
        """Convert continuous LiDAR sector clearances (meters) into an integer state 0..7.

        Args:
            clearances: Dict with keys 'front', 'left', 'right'.
            threshold: Distance in meters below which a sector is considered BLOCKED.
            front_thresh: Optional specific threshold for front sector.
            side_thresh: Optional specific threshold for side sectors.
        """
        f_lim = front_thresh if front_thresh is not None else threshold
        s_lim = side_thresh if side_thresh is not None else threshold

        f_blocked = clearances.get("front", 99.0) < f_lim
        l_blocked = clearances.get("left", 99.0) < s_lim
        r_blocked = clearances.get("right", 99.0) < s_lim

        state_idx = (4 if f_blocked else 0) + (2 if l_blocked else 0) + (1 if r_blocked else 0)
        return int(state_idx)

    def select_action(self, state: int, greedy: bool = True) -> LocalAction:
        """Select action using epsilon-greedy during training, argmax during execution."""
        if not greedy and np.random.rand() < self.epsilon:
            return LocalAction(np.random.randint(4))
        return LocalAction(int(np.argmax(self.q_table[state])))

    def bellman_update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool = False,
    ) -> float:
        """Standard Sutton & Barto Q-Learning Bellman Update:
        Q(s, a) <- Q(s, a) + alpha * [R + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    @staticmethod
    def compute_reward(state: int, action: int) -> Tuple[float, int, bool]:
        """Reward function enforcing collision avoidance and smooth forward progress.

        Returns:
            Tuple of (reward, next_state, done).
        """
        front_blocked = (state & 4) != 0
        left_blocked = (state & 2) != 0
        right_blocked = (state & 1) != 0

        # Scenario 1: Front is CLEAR (States 0, 1, 2, 3)
        if not front_blocked:
            if action == LocalAction.CRUISE:
                return +3.0, LocalState.ALL_CLEAR, True  # Correct: Cruise ahead!
            elif action == LocalAction.YIELD_WAIT:
                return -2.0, state, True                # Wrong: unnecessary hesitation
            else:
                return -0.5, state, True                # Suboptimal: unnecessary detour

        # Scenario 2: Front is BLOCKED, both sides blocked (State 7 - Narrow Aisle Trap)
        if left_blocked and right_blocked:
            if action == LocalAction.YIELD_WAIT:
                # Correct: Wait patiently for dynamic hazard to pass!
                return +4.0, LocalState.ALL_CLEAR, True
            else:
                # Wrong: moving forward or sideways causes collision!
                return -15.0, state, True

        # Scenario 3: Front is BLOCKED, left side is OPEN (States 4, 5)
        if not left_blocked and right_blocked:
            if action == LocalAction.BYPASS_LEFT:
                return +3.5, LocalState.ALL_CLEAR, True  # Correct: safely bypass left!
            elif action == LocalAction.YIELD_WAIT:
                return +1.5, state, True                # Acceptable: yielding is safe
            elif action == LocalAction.BYPASS_RIGHT:
                return -12.0, state, True               # Collision: right side is blocked!
            else:
                return -15.0, state, True               # Collision: crashing straight!

        # Scenario 4: Front is BLOCKED, right side is OPEN (State 6)
        if left_blocked and not right_blocked:
            if action == LocalAction.BYPASS_RIGHT:
                return +3.5, LocalState.ALL_CLEAR, True  # Correct: safely bypass right!
            elif action == LocalAction.YIELD_WAIT:
                return +1.5, state, True                # Acceptable: yielding is safe
            elif action == LocalAction.BYPASS_LEFT:
                return -12.0, state, True               # Collision: left side is blocked!
            else:
                return -15.0, state, True               # Collision: crashing straight!

        # Scenario 5: Front is BLOCKED, both sides are OPEN (State 4)
        if not left_blocked and not right_blocked:
            if action in (LocalAction.BYPASS_LEFT, LocalAction.BYPASS_RIGHT):
                return +3.5, LocalState.ALL_CLEAR, True  # Correct: evade around either side!
            elif action == LocalAction.YIELD_WAIT:
                return +1.5, state, True                # Yielding is also safe
            else:
                return -15.0, state, True               # Straight collision!

        return -1.0, state, True

    def train(self, episodes: int = 300) -> None:
        """Train the Q-table over simulated transitions until convergence."""
        for _ in range(episodes):
            for s in range(8):
                a = self.select_action(s, greedy=False)
                reward, next_s, done = self.compute_reward(s, int(a))
                self.bellman_update(s, int(a), reward, next_s, done)
            self.episodes_trained += 1

    def get_q_values(self, state: int) -> Dict[str, float]:
        """Human-readable Q-values for the current state."""
        return {
            self.ACTION_NAMES[a]: round(float(self.q_table[state, a]), 2)
            for a in range(4)
        }
