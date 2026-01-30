#!/usr/bin/env python3
"""
ROS2 Training Node for SARSA RL Robot Training.
"""

import os
import csv
import numpy as np
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32, Int32

from rl_training.sarsa_agent import SARSAAgent, SARSALambdaAgent, ExpectedSARSAAgent
from rl_training.environment import RobotEnvironment, SimulatedEnvironment
from rl_training.state_action import StateSpace, SimplifiedActionSpace
from rl_training.reward import RewardFunction, ShapedRewardFunction


class RLTrainingNode(Node):
    """ROS2 node for SARSA RL training."""

    def __init__(self):
        super().__init__('rl_training_node')

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self.alpha = self.get_parameter('alpha').get_parameter_value().double_value
        self.gamma = self.get_parameter('gamma').get_parameter_value().double_value
        self.epsilon = self.get_parameter('epsilon').get_parameter_value().double_value
        self.epsilon_decay = self.get_parameter('epsilon_decay').get_parameter_value().double_value
        self.epsilon_min = self.get_parameter('epsilon_min').get_parameter_value().double_value

        self.max_episodes = self.get_parameter('max_episodes').get_parameter_value().integer_value
        self.max_steps = self.get_parameter('max_steps_per_episode').get_parameter_value().integer_value
        self.save_interval = self.get_parameter('save_interval').get_parameter_value().integer_value

        self.position_bins = self.get_parameter('position_bins').get_parameter_value().integer_value
        self.velocity_bins = self.get_parameter('velocity_bins').get_parameter_value().integer_value
        self.distance_bins = self.get_parameter('target_distance_bins').get_parameter_value().integer_value

        self.waypoint_threshold = self.get_parameter('waypoint_threshold').get_parameter_value().double_value

        self.q_table_path = self.get_parameter('q_table_path').get_parameter_value().string_value
        self.log_path = self.get_parameter('training_log_path').get_parameter_value().string_value

        # Initialize state and action spaces
        self.state_space = StateSpace(
            position_bins=self.position_bins,
            velocity_bins=self.velocity_bins,
            target_distance_bins=self.distance_bins
        )
        self.action_space = SimplifiedActionSpace()

        # Initialize SARSA agent
        self.agent = SARSAAgent(
            state_space=self.state_space,
            action_space=self.action_space,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min
        )

        # Try to load existing Q-table
        if os.path.exists(self.q_table_path):
            self.agent.load(self.q_table_path)
            self.get_logger().info(f'Loaded Q-table from {self.q_table_path}')

        # Initialize reward function
        self.reward_function = ShapedRewardFunction(
            gamma=self.gamma,
            distance_weight=self.get_parameter('distance_reward_weight').get_parameter_value().double_value,
            waypoint_bonus=self.get_parameter('waypoint_bonus').get_parameter_value().double_value,
            joint_limit_penalty=self.get_parameter('joint_limit_penalty').get_parameter_value().double_value,
            velocity_penalty_weight=self.get_parameter('velocity_penalty_weight').get_parameter_value().double_value,
            smoothness_weight=self.get_parameter('smoothness_reward_weight').get_parameter_value().double_value,
            waypoint_threshold=self.waypoint_threshold
        )

        # Generate sample waypoints
        self.waypoints = self._generate_sample_waypoints()

        # Initialize environment (simulated for now)
        self.use_simulation = True  # Set to False to use ROS/Gazebo
        if self.use_simulation:
            self.env = SimulatedEnvironment(
                waypoints=self.waypoints,
                waypoint_threshold=self.waypoint_threshold,
                max_steps=self.max_steps
            )
        else:
            self.env = RobotEnvironment(
                node=self,
                waypoints=self.waypoints,
                waypoint_threshold=self.waypoint_threshold,
                max_steps=self.max_steps
            )

        # Publishers for monitoring
        self.reward_pub = self.create_publisher(Float32, 'training/episode_reward', 10)
        self.episode_pub = self.create_publisher(Int32, 'training/episode', 10)

        # Training state
        self.current_episode = 0
        self.training_log = []

        # Initialize log file
        self._init_log_file()

        # Start training timer
        self.training_timer = self.create_timer(0.01, self.training_step)
        self.training_active = False
        self.episode_state = None
        self.episode_action = None

        self.get_logger().info('RL Training Node initialized')
        self.get_logger().info(f'State space size: {self.state_space.total_states}')
        self.get_logger().info(f'Action space size: {self.action_space.num_actions}')

    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        self.declare_parameter('alpha', 0.1)
        self.declare_parameter('gamma', 0.99)
        self.declare_parameter('epsilon', 0.3)
        self.declare_parameter('epsilon_decay', 0.995)
        self.declare_parameter('epsilon_min', 0.01)

        self.declare_parameter('max_episodes', 1000)
        self.declare_parameter('max_steps_per_episode', 200)
        self.declare_parameter('save_interval', 100)

        self.declare_parameter('position_bins', 10)
        self.declare_parameter('velocity_bins', 5)
        self.declare_parameter('target_distance_bins', 10)
        self.declare_parameter('num_velocity_levels', 5)

        self.declare_parameter('distance_reward_weight', -1.0)
        self.declare_parameter('waypoint_bonus', 10.0)
        self.declare_parameter('joint_limit_penalty', -5.0)
        self.declare_parameter('velocity_penalty_weight', -0.1)
        self.declare_parameter('smoothness_reward_weight', 0.5)

        self.declare_parameter('waypoint_threshold', 0.05)
        self.declare_parameter('joint_limit_buffer', 0.1)

        self.declare_parameter('q_table_path', 'q_table.pkl')
        self.declare_parameter('training_log_path', 'training_log.csv')

    def _generate_sample_waypoints(self) -> List[np.ndarray]:
        """Generate sample waypoints for training."""
        waypoints = []

        # Generate circular path in XZ plane
        center = np.array([0.3, 0.0, 0.5])
        radius = 0.1
        num_points = 10

        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            waypoint = center + np.array([
                radius * np.cos(angle),
                0.0,
                radius * np.sin(angle)
            ])
            waypoints.append(waypoint)

        return waypoints

    def _init_log_file(self):
        """Initialize training log file."""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'steps', 'epsilon',
                'waypoints_reached', 'avg_reward_100'
            ])

    def start_training(self):
        """Start training loop."""
        self.training_active = True
        self.current_episode = self.agent.episode_count
        self._start_episode()
        self.get_logger().info('Training started')

    def stop_training(self):
        """Stop training."""
        self.training_active = False
        self.agent.save(self.q_table_path)
        self.get_logger().info('Training stopped, Q-table saved')

    def _start_episode(self):
        """Start a new training episode."""
        # Reset environment
        self.episode_state = self.env.reset()
        self.reward_function.reset()

        # Select first action
        self.episode_action = self.agent.select_action(self.episode_state)

        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.waypoints_reached = 0

    def training_step(self):
        """Execute one training step (called by timer)."""
        if not self.training_active:
            return

        if self.current_episode >= self.max_episodes:
            self.stop_training()
            return

        # Get action object
        action = self.action_space.get_action(self.episode_action)

        # Execute action
        next_state, _, done, info = self.env.step(action)

        # Compute reward
        reward = self.reward_function.compute_reward(
            self.episode_state, action, next_state
        )

        # Select next action
        next_action = self.agent.select_action(next_state)

        # SARSA update
        self.agent.update(
            self.episode_state,
            self.episode_action,
            reward,
            next_state,
            next_action,
            done
        )

        # Update episode tracking
        self.episode_reward += reward
        self.episode_steps += 1
        if info.get('waypoint_reached', False):
            self.waypoints_reached += 1

        # Transition to next state
        self.episode_state = next_state
        self.episode_action = next_action

        # Check if episode is done
        if done:
            self._end_episode()

    def _end_episode(self):
        """Handle end of episode."""
        # Decay epsilon
        self.agent.decay_epsilon()

        # Record statistics
        self.agent.episode_rewards.append(self.episode_reward)
        avg_reward = np.mean(self.agent.episode_rewards[-100:])

        # Log episode
        self._log_episode(avg_reward)

        # Publish metrics
        reward_msg = Float32()
        reward_msg.data = float(self.episode_reward)
        self.reward_pub.publish(reward_msg)

        episode_msg = Int32()
        episode_msg.data = self.current_episode
        self.episode_pub.publish(episode_msg)

        # Save periodically
        if self.current_episode % self.save_interval == 0:
            self.agent.save(self.q_table_path)
            self.get_logger().info(f'Episode {self.current_episode}: reward={self.episode_reward:.2f}, '
                                   f'avg={avg_reward:.2f}, epsilon={self.agent.epsilon:.3f}')

        # Increment episode
        self.current_episode += 1

        # Start next episode
        if self.current_episode < self.max_episodes:
            self._start_episode()

    def _log_episode(self, avg_reward: float):
        """Log episode to file."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_episode,
                self.episode_reward,
                self.episode_steps,
                self.agent.epsilon,
                self.waypoints_reached,
                avg_reward
            ])

    def run_evaluation(self, num_episodes: int = 10):
        """Run evaluation episodes (no learning)."""
        self.get_logger().info(f'Running {num_episodes} evaluation episodes')

        total_rewards = []

        for ep in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action_idx = self.agent.select_action(state, training=False)
                action = self.action_space.get_action(action_idx)
                next_state, _, done, _ = self.env.step(action)
                reward = self.reward_function.compute_reward(state, action, next_state)
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        self.get_logger().info(f'Evaluation: avg_reward={avg_reward:.2f} +/- {std_reward:.2f}')

        return avg_reward, std_reward


def main(args=None):
    rclpy.init(args=args)
    node = RLTrainingNode()

    # Start training
    node.start_training()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_training()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
