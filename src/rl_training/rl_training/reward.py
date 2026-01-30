"""
Reward Function for SARSA RL Training.
"""

import numpy as np
from typing import Optional
from .state_action import RobotState, RobotAction


class RewardFunction:
    """
    Compute rewards for robot trajectory following.

    Reward components:
    - Distance to target (negative)
    - Bonus for reaching waypoint
    - Penalty for joint limit violations
    - Penalty for excessive velocity
    - Reward for smooth motion
    """

    def __init__(
        self,
        distance_weight: float = -1.0,
        waypoint_bonus: float = 10.0,
        joint_limit_penalty: float = -5.0,
        velocity_penalty_weight: float = -0.1,
        smoothness_weight: float = 0.5,
        waypoint_threshold: float = 0.05,
        joint_limit_buffer: float = 0.1
    ):
        self.distance_weight = distance_weight
        self.waypoint_bonus = waypoint_bonus
        self.joint_limit_penalty = joint_limit_penalty
        self.velocity_penalty_weight = velocity_penalty_weight
        self.smoothness_weight = smoothness_weight
        self.waypoint_threshold = waypoint_threshold
        self.joint_limit_buffer = joint_limit_buffer

        # Joint limits (from URDF)
        self.joint_limits = [
            (-np.pi, np.pi),
            (-np.pi/2, np.pi/2),
            (-np.pi*0.75, np.pi*0.75),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]

        # Previous action for smoothness calculation
        self.prev_action: Optional[RobotAction] = None

    def compute_reward(
        self,
        state: RobotState,
        action: RobotAction,
        next_state: RobotState
    ) -> float:
        """Compute total reward for a state-action-next_state transition."""
        reward = 0.0

        # 1. Distance reward (negative distance to target)
        distance_reward = self.distance_weight * next_state.distance_to_target
        reward += distance_reward

        # 2. Waypoint reached bonus
        if next_state.distance_to_target < self.waypoint_threshold:
            reward += self.waypoint_bonus

        # 3. Joint limit penalty
        joint_penalty = self._compute_joint_limit_penalty(next_state.joint_positions)
        reward += joint_penalty

        # 4. Velocity penalty
        velocity_penalty = self._compute_velocity_penalty(next_state.joint_velocities)
        reward += velocity_penalty

        # 5. Smoothness reward
        smoothness_reward = self._compute_smoothness_reward(action)
        reward += smoothness_reward

        # 6. Progress reward (getting closer to target)
        progress = state.distance_to_target - next_state.distance_to_target
        reward += progress * 5.0  # Bonus for making progress

        # Store action for next smoothness calculation
        self.prev_action = action

        return reward

    def _compute_joint_limit_penalty(self, joint_positions: np.ndarray) -> float:
        """Penalize positions close to joint limits."""
        penalty = 0.0

        for i, (pos, (low, high)) in enumerate(zip(joint_positions, self.joint_limits)):
            # Distance from lower limit
            dist_to_low = pos - low
            # Distance from upper limit
            dist_to_high = high - pos

            # Penalize if within buffer zone
            if dist_to_low < self.joint_limit_buffer:
                penalty += self.joint_limit_penalty * (1 - dist_to_low / self.joint_limit_buffer)
            if dist_to_high < self.joint_limit_buffer:
                penalty += self.joint_limit_penalty * (1 - dist_to_high / self.joint_limit_buffer)

        return penalty

    def _compute_velocity_penalty(self, joint_velocities: np.ndarray) -> float:
        """Penalize excessive joint velocities."""
        velocity_magnitude = np.linalg.norm(joint_velocities)
        return self.velocity_penalty_weight * velocity_magnitude

    def _compute_smoothness_reward(self, action: RobotAction) -> float:
        """Reward smooth motion (small changes in action)."""
        if self.prev_action is None:
            return 0.0

        action_change = np.linalg.norm(
            action.joint_velocity_adjustments - self.prev_action.joint_velocity_adjustments
        )

        # Reward for small action changes (smooth motion)
        smoothness = np.exp(-action_change)
        return self.smoothness_weight * smoothness

    def reset(self):
        """Reset reward function state (e.g., at episode start)."""
        self.prev_action = None


class ShapedRewardFunction(RewardFunction):
    """
    Shaped reward function with potential-based shaping.
    Provides denser rewards for faster learning.
    """

    def __init__(self, gamma: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.prev_potential = None

    def compute_reward(
        self,
        state: RobotState,
        action: RobotAction,
        next_state: RobotState
    ) -> float:
        """Compute shaped reward."""
        # Base reward
        base_reward = super().compute_reward(state, action, next_state)

        # Potential-based shaping
        current_potential = self._compute_potential(next_state)
        shaping_reward = 0.0

        if self.prev_potential is not None:
            shaping_reward = self.gamma * current_potential - self.prev_potential

        self.prev_potential = current_potential

        return base_reward + shaping_reward

    def _compute_potential(self, state: RobotState) -> float:
        """
        Compute potential function.
        Higher potential = closer to goal.
        """
        # Negative distance as potential (closer = higher)
        distance_potential = -state.distance_to_target * 10.0

        # Bonus potential for being near waypoint
        if state.distance_to_target < self.waypoint_threshold * 2:
            distance_potential += 5.0

        return distance_potential

    def reset(self):
        """Reset shaped reward function state."""
        super().reset()
        self.prev_potential = None
