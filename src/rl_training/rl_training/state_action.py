"""
State and Action Space Definitions for SARSA RL Training.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class RobotState:
    """Continuous robot state representation."""
    joint_positions: np.ndarray  # 6D: current joint angles
    joint_velocities: np.ndarray  # 6D: current joint velocities
    target_position: np.ndarray  # 3D: target waypoint in Cartesian space
    distance_to_target: float  # Scalar: Euclidean distance to target
    ee_position: np.ndarray  # 3D: current end-effector position

    def to_array(self) -> np.ndarray:
        """Convert state to flat array."""
        return np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            self.target_position,
            [self.distance_to_target]
        ])


class StateSpace:
    """Discretized state space for tabular SARSA."""

    def __init__(
        self,
        position_bins: int = 10,
        velocity_bins: int = 5,
        target_distance_bins: int = 10,
        num_joints: int = 6
    ):
        self.num_joints = num_joints
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.target_distance_bins = target_distance_bins

        # Joint limits (from URDF)
        self.joint_limits = [
            (-np.pi, np.pi),      # joint1
            (-np.pi/2, np.pi/2),  # joint2
            (-np.pi*0.75, np.pi*0.75),  # joint3
            (-np.pi, np.pi),      # joint4
            (-np.pi, np.pi),      # joint5
            (-np.pi, np.pi),      # joint6
        ]

        # Velocity limits
        self.velocity_limits = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.5, 1.5),
            (-2.0, 2.0),
            (-2.5, 2.5),
        ]

        # Distance range (meters)
        self.distance_range = (0.0, 1.5)

        # Calculate total state space size
        self.state_dims = (
            [position_bins] * num_joints +  # Joint positions
            [velocity_bins] * num_joints +  # Joint velocities
            [target_distance_bins]  # Distance to target
        )
        self.total_states = np.prod(self.state_dims)

    def discretize(self, state: RobotState) -> Tuple[int, ...]:
        """Convert continuous state to discrete state tuple."""
        discrete_state = []

        # Discretize joint positions
        for i, pos in enumerate(state.joint_positions):
            low, high = self.joint_limits[i]
            bin_idx = self._to_bin(pos, low, high, self.position_bins)
            discrete_state.append(bin_idx)

        # Discretize joint velocities
        for i, vel in enumerate(state.joint_velocities):
            low, high = self.velocity_limits[i]
            bin_idx = self._to_bin(vel, low, high, self.velocity_bins)
            discrete_state.append(bin_idx)

        # Discretize distance to target
        dist_bin = self._to_bin(
            state.distance_to_target,
            self.distance_range[0],
            self.distance_range[1],
            self.target_distance_bins
        )
        discrete_state.append(dist_bin)

        return tuple(discrete_state)

    def _to_bin(self, value: float, low: float, high: float, num_bins: int) -> int:
        """Convert continuous value to bin index."""
        # Clip to range
        value = np.clip(value, low, high)
        # Normalize to [0, 1]
        normalized = (value - low) / (high - low + 1e-10)
        # Convert to bin index
        bin_idx = int(normalized * (num_bins - 1))
        return np.clip(bin_idx, 0, num_bins - 1)

    def state_to_index(self, discrete_state: Tuple[int, ...]) -> int:
        """Convert discrete state tuple to single index."""
        index = 0
        multiplier = 1
        for i, (state_val, dim_size) in enumerate(zip(reversed(discrete_state), reversed(self.state_dims))):
            index += state_val * multiplier
            multiplier *= dim_size
        return index


@dataclass
class RobotAction:
    """Robot action representation."""
    joint_velocity_adjustments: np.ndarray  # 6D: velocity changes per joint


class ActionSpace:
    """Discretized action space for tabular SARSA."""

    def __init__(
        self,
        num_velocity_levels: int = 5,
        num_joints: int = 6
    ):
        self.num_joints = num_joints
        self.num_velocity_levels = num_velocity_levels

        # Velocity adjustment range (rad/s)
        self.velocity_range = 0.2  # Max adjustment per step

        # Generate discrete velocity adjustments
        self.velocity_adjustments = np.linspace(
            -self.velocity_range,
            self.velocity_range,
            num_velocity_levels
        )

        # Total number of actions (each joint can have num_velocity_levels choices)
        # For simplicity, we'll use a combined action space
        # Action index encodes velocity adjustment for all joints
        self.num_actions = num_velocity_levels ** num_joints

        # Pre-compute action mappings for efficiency
        self._action_cache = {}

    def get_action(self, action_idx: int) -> RobotAction:
        """Convert action index to RobotAction."""
        if action_idx in self._action_cache:
            return self._action_cache[action_idx]

        adjustments = np.zeros(self.num_joints)
        remaining = action_idx

        for i in range(self.num_joints):
            level_idx = remaining % self.num_velocity_levels
            adjustments[i] = self.velocity_adjustments[level_idx]
            remaining //= self.num_velocity_levels

        action = RobotAction(joint_velocity_adjustments=adjustments)
        self._action_cache[action_idx] = action
        return action

    def sample_random(self) -> int:
        """Sample random action index."""
        return np.random.randint(0, self.num_actions)

    def get_zero_action(self) -> int:
        """Get action index for zero velocity adjustment."""
        middle_level = self.num_velocity_levels // 2
        action_idx = 0
        for i in range(self.num_joints):
            action_idx += middle_level * (self.num_velocity_levels ** i)
        return action_idx


class SimplifiedActionSpace:
    """
    Simplified action space with fewer dimensions.
    Instead of controlling all joints independently, use macro-actions.
    """

    def __init__(self, num_joints: int = 6):
        self.num_joints = num_joints

        # Define macro-actions
        self.actions = [
            np.zeros(num_joints),  # Stop
        ]

        # Add single-joint movements
        for i in range(num_joints):
            # Positive movement
            action = np.zeros(num_joints)
            action[i] = 0.1
            self.actions.append(action)

            # Negative movement
            action = np.zeros(num_joints)
            action[i] = -0.1
            self.actions.append(action)

        # Add combined movements for common patterns
        # Move toward target (approximate)
        self.actions.append(np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0]))
        self.actions.append(np.array([-0.05, -0.05, -0.05, 0.0, 0.0, 0.0]))

        self.num_actions = len(self.actions)

    def get_action(self, action_idx: int) -> RobotAction:
        """Get action by index."""
        return RobotAction(
            joint_velocity_adjustments=self.actions[action_idx].copy()
        )

    def sample_random(self) -> int:
        """Sample random action."""
        return np.random.randint(0, self.num_actions)
