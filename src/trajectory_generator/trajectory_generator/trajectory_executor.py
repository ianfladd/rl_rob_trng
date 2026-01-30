#!/usr/bin/env python3
"""
Trajectory Executor Node
Executes joint trajectories via action client to joint_trajectory_controller.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

from builtin_interfaces.msg import Duration
import time


class TrajectoryExecutorNode(Node):
    """Execute joint trajectories on the robot."""

    def __init__(self):
        super().__init__('trajectory_executor')

        # Parameters
        self.declare_parameter('action_server', '/joint_trajectory_controller/follow_joint_trajectory')
        self.declare_parameter('timeout', 30.0)

        self.action_server_name = self.get_parameter('action_server').get_parameter_value().string_value
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        # Callback group
        self.callback_group = ReentrantCallbackGroup()

        # Joint names
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        # Current state
        self.current_joint_state = None
        self.is_executing = False

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            'planned_trajectory',
            self.trajectory_callback,
            10
        )

        # Action client
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.action_server_name,
            callback_group=self.callback_group
        )

        self.get_logger().info('Trajectory Executor Node started')
        self.get_logger().info(f'Action server: {self.action_server_name}')

    def joint_state_callback(self, msg: JointState):
        """Store current joint state."""
        self.current_joint_state = msg

    def trajectory_callback(self, msg: JointTrajectory):
        """Handle incoming trajectory to execute."""
        if self.is_executing:
            self.get_logger().warn('Already executing a trajectory, ignoring new one')
            return

        self.get_logger().info(f'Received trajectory with {len(msg.points)} points')
        self.execute_trajectory(msg)

    def execute_trajectory(self, trajectory: JointTrajectory) -> bool:
        """Execute trajectory via action client."""
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        # Create goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # Set tolerances
        goal_msg.goal_time_tolerance = Duration(sec=1, nanosec=0)

        self.is_executing = True
        self.get_logger().info('Sending trajectory goal...')

        # Send goal
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

        return True

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected')
            self.is_executing = False
            return

        self.get_logger().info('Trajectory goal accepted')

        # Get result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback during execution."""
        feedback = feedback_msg.feedback
        # Log progress
        if hasattr(feedback, 'desired') and feedback.desired.positions:
            progress = len(feedback.actual.positions)
            self.get_logger().debug(f'Execution progress: joint positions received')

    def result_callback(self, future):
        """Handle trajectory execution result."""
        result = future.result().result
        status = future.result().status

        self.is_executing = False

        if status == 4:  # SUCCEEDED
            self.get_logger().info('Trajectory execution succeeded')
        elif status == 5:  # CANCELED
            self.get_logger().warn('Trajectory execution was canceled')
        elif status == 6:  # ABORTED
            self.get_logger().error(f'Trajectory execution aborted: {result.error_string}')
        else:
            self.get_logger().warn(f'Trajectory execution finished with status: {status}')

    def cancel_execution(self):
        """Cancel current trajectory execution."""
        if not self.is_executing:
            self.get_logger().info('No trajectory being executed')
            return

        self.get_logger().info('Canceling trajectory execution...')
        # Note: Would need to store goal handle to cancel

    def move_to_position(self, positions: list, duration: float = 2.0) -> bool:
        """Move to a specific joint position."""
        if len(positions) != len(self.joint_names):
            self.get_logger().error(f'Expected {len(self.joint_names)} positions, got {len(positions)}')
            return False

        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = self.joint_names

        from trajectory_msgs.msg import JointTrajectoryPoint
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)
        point.time_from_start = Duration(
            sec=int(duration),
            nanosec=int((duration % 1) * 1e9)
        )
        trajectory.points.append(point)

        return self.execute_trajectory(trajectory)

    def move_to_home(self) -> bool:
        """Move robot to home position."""
        home_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.get_logger().info('Moving to home position')
        return self.move_to_position(home_positions, duration=3.0)

    def move_to_ready(self) -> bool:
        """Move robot to ready position."""
        ready_positions = [0.0, -0.785, 1.57, -0.785, 0.0, 0.0]
        self.get_logger().info('Moving to ready position')
        return self.move_to_position(ready_positions, duration=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
