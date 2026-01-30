"""
ROS2/Gazebo Environment Wrapper for RL Training.
"""

import numpy as np
from typing import Optional, Tuple, List
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.srv import SetEntityState, GetEntityState
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration

from .state_action import RobotState, RobotAction, StateSpace, ActionSpace


class RobotEnvironment:
    """
    Environment wrapper for robot RL training.

    Interfaces with ROS2/Gazebo to:
    - Get robot state (joint positions, velocities, end-effector position)
    - Execute actions (send velocity commands)
    - Reset the robot to initial state
    """

    def __init__(
        self,
        node: Node,
        waypoints: List[np.ndarray],
        waypoint_threshold: float = 0.05,
        max_steps: int = 200
    ):
        self.node = node
        self.waypoints = waypoints
        self.waypoint_threshold = waypoint_threshold
        self.max_steps = max_steps

        # Joint configuration
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.num_joints = len(self.joint_names)

        # Joint limits
        self.joint_limits = [
            (-np.pi, np.pi),
            (-np.pi/2, np.pi/2),
            (-np.pi*0.75, np.pi*0.75),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]

        # Link lengths for FK (from URDF)
        self.link_lengths = {
            'base_height': 0.1,
            'joint_length': 0.06,
            'link1': 0.3,
            'link2': 0.25,
            'link3': 0.2,
            'link4': 0.1,
            'link5': 0.08,
            'link6': 0.05,
        }

        # Current state
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)

        # Episode state
        self.current_waypoint_idx = 0
        self.step_count = 0
        self.episode_reward = 0.0

        # Publishers and subscribers
        self.trajectory_pub = node.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.joint_state_sub = node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        # Service clients for Gazebo
        self.reset_world_client = node.create_client(Empty, '/reset_world')
        self.pause_physics_client = node.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = node.create_client(Empty, '/unpause_physics')

        self.node.get_logger().info('Robot Environment initialized')

    def _joint_state_callback(self, msg: JointState):
        """Update current joint state from subscription."""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joint_positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.current_joint_velocities[i] = msg.velocity[idx]

    def get_state(self) -> RobotState:
        """Get current robot state."""
        # Compute end-effector position using FK
        ee_position = self._forward_kinematics(self.current_joint_positions)

        # Get current target waypoint
        target_waypoint = self.waypoints[self.current_waypoint_idx]

        # Compute distance to target
        distance = np.linalg.norm(ee_position - target_waypoint)

        return RobotState(
            joint_positions=self.current_joint_positions.copy(),
            joint_velocities=self.current_joint_velocities.copy(),
            target_position=target_waypoint.copy(),
            distance_to_target=distance,
            ee_position=ee_position.copy()
        )

    def step(self, action: RobotAction) -> Tuple[RobotState, float, bool, dict]:
        """
        Execute action and return next state, reward, done, info.

        Args:
            action: RobotAction with velocity adjustments

        Returns:
            next_state: New robot state after action
            reward: Reward for this transition (computed externally)
            done: Whether episode is finished
            info: Additional information
        """
        # Apply velocity adjustments to get target positions
        target_positions = self.current_joint_positions + action.joint_velocity_adjustments

        # Clip to joint limits
        for i, (low, high) in enumerate(self.joint_limits):
            target_positions[i] = np.clip(target_positions[i], low, high)

        # Send trajectory command
        self._send_trajectory_command(target_positions, duration=0.1)

        # Wait for execution (in simulation time)
        self._wait_for_execution(0.1)

        # Get new state
        next_state = self.get_state()

        # Check if waypoint reached
        waypoint_reached = next_state.distance_to_target < self.waypoint_threshold

        # Update waypoint index if reached
        if waypoint_reached:
            self.current_waypoint_idx = min(
                self.current_waypoint_idx + 1,
                len(self.waypoints) - 1
            )

        # Increment step count
        self.step_count += 1

        # Check if episode is done
        done = (
            self.step_count >= self.max_steps or
            self.current_waypoint_idx >= len(self.waypoints) - 1
        )

        info = {
            'waypoint_reached': waypoint_reached,
            'current_waypoint_idx': self.current_waypoint_idx,
            'step_count': self.step_count,
        }

        return next_state, 0.0, done, info  # Reward computed externally

    def reset(self) -> RobotState:
        """Reset environment for new episode."""
        # Reset episode state
        self.current_waypoint_idx = 0
        self.step_count = 0
        self.episode_reward = 0.0

        # Reset robot to home position
        home_positions = np.zeros(self.num_joints)
        self._send_trajectory_command(home_positions, duration=2.0)
        self._wait_for_execution(2.0)

        # Optionally reset Gazebo world
        # self._reset_gazebo()

        return self.get_state()

    def _send_trajectory_command(self, positions: np.ndarray, duration: float = 1.0):
        """Send joint trajectory command."""
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.node.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.velocities = [0.0] * self.num_joints
        point.time_from_start = Duration(
            sec=int(duration),
            nanosec=int((duration % 1) * 1e9)
        )
        trajectory.points.append(point)

        self.trajectory_pub.publish(trajectory)

    def _wait_for_execution(self, duration: float):
        """Wait for trajectory execution (blocking)."""
        import time
        time.sleep(duration)

    def _reset_gazebo(self):
        """Reset Gazebo world."""
        if self.reset_world_client.wait_for_service(timeout_sec=1.0):
            request = Empty.Request()
            future = self.reset_world_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)

    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position from joint positions.
        Simplified FK for the 6-DOF arm.
        """
        q = joint_positions

        # Extract link lengths
        l0 = self.link_lengths['base_height'] + self.link_lengths['joint_length']
        l1 = self.link_lengths['link1']
        l2 = self.link_lengths['link2']
        l3 = self.link_lengths['link3']
        l4 = self.link_lengths['link4']
        l5 = self.link_lengths['link5']
        l6 = self.link_lengths['link6']

        # Simplified FK (treating as a planar arm in the q[0] rotation plane)
        # This is an approximation; full FK would use DH parameters

        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        c23 = np.cos(q[1] + q[2])
        s23 = np.sin(q[1] + q[2])
        c234 = np.cos(q[1] + q[2] + q[3])
        s234 = np.sin(q[1] + q[2] + q[3])

        # End effector position
        r = (l1 * c2 + l2 * c23 + l3 * c234 + (l4 + l5 + l6) * c234)
        z = l0 + l1 * s2 + l2 * s23 + l3 * s234 + (l4 + l5 + l6) * s234

        x = r * c1
        y = r * s1

        return np.array([x, y, z])

    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Set new waypoints for the episode."""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0

    def get_current_target(self) -> np.ndarray:
        """Get current target waypoint."""
        return self.waypoints[self.current_waypoint_idx]


class SimulatedEnvironment(RobotEnvironment):
    """
    Simulated environment for faster training without ROS/Gazebo.
    Uses kinematic simulation only.
    """

    def __init__(
        self,
        waypoints: List[np.ndarray],
        waypoint_threshold: float = 0.05,
        max_steps: int = 200,
        dt: float = 0.1
    ):
        # Initialize without ROS node
        self.waypoints = waypoints
        self.waypoint_threshold = waypoint_threshold
        self.max_steps = max_steps
        self.dt = dt

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.num_joints = len(self.joint_names)

        self.joint_limits = [
            (-np.pi, np.pi),
            (-np.pi/2, np.pi/2),
            (-np.pi*0.75, np.pi*0.75),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]

        self.link_lengths = {
            'base_height': 0.1,
            'joint_length': 0.06,
            'link1': 0.3,
            'link2': 0.25,
            'link3': 0.2,
            'link4': 0.1,
            'link5': 0.08,
            'link6': 0.05,
        }

        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)
        self.current_waypoint_idx = 0
        self.step_count = 0

    def step(self, action: RobotAction) -> Tuple[RobotState, float, bool, dict]:
        """Execute action in simulation."""
        # Apply velocity adjustments
        self.current_joint_velocities = action.joint_velocity_adjustments.copy()
        self.current_joint_positions += self.current_joint_velocities * self.dt

        # Clip to limits
        for i, (low, high) in enumerate(self.joint_limits):
            self.current_joint_positions[i] = np.clip(
                self.current_joint_positions[i], low, high
            )

        # Get new state
        next_state = self.get_state()

        # Check waypoint
        waypoint_reached = next_state.distance_to_target < self.waypoint_threshold
        if waypoint_reached:
            self.current_waypoint_idx = min(
                self.current_waypoint_idx + 1,
                len(self.waypoints) - 1
            )

        self.step_count += 1
        done = (
            self.step_count >= self.max_steps or
            self.current_waypoint_idx >= len(self.waypoints) - 1
        )

        info = {
            'waypoint_reached': waypoint_reached,
            'current_waypoint_idx': self.current_waypoint_idx,
            'step_count': self.step_count,
        }

        return next_state, 0.0, done, info

    def reset(self) -> RobotState:
        """Reset simulation."""
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)
        self.current_waypoint_idx = 0
        self.step_count = 0
        return self.get_state()

    def get_state(self) -> RobotState:
        """Get current state."""
        ee_position = self._forward_kinematics(self.current_joint_positions)
        target = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(ee_position - target)

        return RobotState(
            joint_positions=self.current_joint_positions.copy(),
            joint_velocities=self.current_joint_velocities.copy(),
            target_position=target.copy(),
            distance_to_target=distance,
            ee_position=ee_position.copy()
        )

    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute FK."""
        q = joint_positions

        l0 = self.link_lengths['base_height'] + self.link_lengths['joint_length']
        l1 = self.link_lengths['link1']
        l2 = self.link_lengths['link2']
        l3 = self.link_lengths['link3']
        l4 = self.link_lengths['link4']
        l5 = self.link_lengths['link5']
        l6 = self.link_lengths['link6']

        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        c23 = np.cos(q[1] + q[2])
        s23 = np.sin(q[1] + q[2])
        c234 = np.cos(q[1] + q[2] + q[3])
        s234 = np.sin(q[1] + q[2] + q[3])

        r = (l1 * c2 + l2 * c23 + l3 * c234 + (l4 + l5 + l6) * c234)
        z = l0 + l1 * s2 + l2 * s23 + l3 * s234 + (l4 + l5 + l6) * s234

        x = r * c1
        y = r * s1

        return np.array([x, y, z])
