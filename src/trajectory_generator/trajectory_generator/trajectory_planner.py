#!/usr/bin/env python3
"""
Trajectory Planner Node
Converts end-effector paths to joint trajectories using MoveIt2.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from moveit_msgs.msg import (
    RobotState,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from moveit_msgs.action import MoveGroup
from shape_msgs.msg import SolidPrimitive

from trajectory_generator.msg import SamplePath
from trajectory_generator.srv import GenerateTrajectory

from builtin_interfaces.msg import Duration
import math


class TrajectoryPlannerNode(Node):
    """Plan joint trajectories from end-effector paths using MoveIt2."""

    def __init__(self):
        super().__init__('trajectory_planner')

        # Parameters
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('ee_link', 'ee_link')
        self.declare_parameter('max_velocity_scaling', 0.5)
        self.declare_parameter('max_acceleration_scaling', 0.5)

        self.planning_group = self.get_parameter('planning_group').get_parameter_value().string_value
        self.ee_link = self.get_parameter('ee_link').get_parameter_value().string_value
        self.max_vel_scale = self.get_parameter('max_velocity_scaling').get_parameter_value().double_value
        self.max_acc_scale = self.get_parameter('max_acceleration_scaling').get_parameter_value().double_value

        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()

        # Joint names for the arm
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        # Current joint state
        self.current_joint_state = None

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.path_sub = self.create_subscription(
            SamplePath,
            'sample_path',
            self.path_callback,
            10
        )

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            'planned_trajectory',
            10
        )

        # Service clients for MoveIt2
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path',
            callback_group=self.callback_group
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik',
            callback_group=self.callback_group
        )

        # Service server
        self.generate_traj_srv = self.create_service(
            GenerateTrajectory,
            'generate_trajectory',
            self.generate_trajectory_callback
        )

        self.get_logger().info('Trajectory Planner Node started')
        self.get_logger().info(f'Planning group: {self.planning_group}, EE link: {self.ee_link}')

    def joint_state_callback(self, msg: JointState):
        """Store current joint state."""
        self.current_joint_state = msg

    def path_callback(self, msg: SamplePath):
        """Handle incoming sample path."""
        self.get_logger().info(f'Received path: {msg.path_id} with {len(msg.waypoints)} waypoints')

        if len(msg.waypoints) == 0:
            self.get_logger().warn('Empty path received')
            return

        # Generate trajectory from path
        trajectory = self.generate_trajectory_from_waypoints(
            msg.waypoints,
            self.max_vel_scale,
            self.max_acc_scale
        )

        if trajectory:
            self.trajectory_pub.publish(trajectory)
            self.get_logger().info(f'Published trajectory with {len(trajectory.points)} points')

    def generate_trajectory_callback(self, request, response):
        """Service callback to generate trajectory."""
        self.get_logger().info(f'Generating trajectory for {len(request.waypoints)} waypoints')

        velocity_scaling = request.velocity_scaling if request.velocity_scaling > 0 else self.max_vel_scale
        acceleration_scaling = request.acceleration_scaling if request.acceleration_scaling > 0 else self.max_acc_scale

        trajectory = self.generate_trajectory_from_waypoints(
            request.waypoints,
            velocity_scaling,
            acceleration_scaling,
            request.avoid_collisions
        )

        if trajectory:
            response.success = True
            response.message = f'Generated trajectory with {len(trajectory.points)} points'
            response.trajectory = trajectory
        else:
            response.success = False
            response.message = 'Failed to generate trajectory'
            response.trajectory = JointTrajectory()

        return response

    def generate_trajectory_from_waypoints(
        self,
        waypoints: list,
        velocity_scaling: float = 0.5,
        acceleration_scaling: float = 0.5,
        avoid_collisions: bool = True
    ) -> JointTrajectory:
        """Generate joint trajectory from Cartesian waypoints."""

        if not self.current_joint_state:
            self.get_logger().warn('No current joint state available')
            return self.generate_simple_trajectory(waypoints)

        # Try using MoveIt Cartesian path service
        if self.cartesian_path_client.wait_for_service(timeout_sec=1.0):
            return self.compute_cartesian_path(
                waypoints, velocity_scaling, acceleration_scaling, avoid_collisions
            )
        else:
            self.get_logger().warn('Cartesian path service not available, using simple IK')
            return self.generate_simple_trajectory(waypoints)

    def compute_cartesian_path(
        self,
        waypoints: list,
        velocity_scaling: float,
        acceleration_scaling: float,
        avoid_collisions: bool
    ) -> JointTrajectory:
        """Use MoveIt2 to compute Cartesian path."""
        request = GetCartesianPath.Request()
        request.header.frame_id = 'world'
        request.header.stamp = self.get_clock().now().to_msg()
        request.group_name = self.planning_group
        request.link_name = self.ee_link

        # Set start state
        request.start_state = RobotState()
        request.start_state.joint_state = self.current_joint_state

        # Set waypoints
        for wp in waypoints:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'world'
            pose_stamped.pose = wp
            request.waypoints.append(pose_stamped.pose)

        request.max_step = 0.01  # Resolution
        request.jump_threshold = 0.0  # Disable jump threshold
        request.avoid_collisions = avoid_collisions
        request.max_velocity_scaling_factor = velocity_scaling
        request.max_acceleration_scaling_factor = acceleration_scaling

        # Call service
        future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is not None:
            response = future.result()
            if response.fraction > 0.9:  # At least 90% of path computed
                self.get_logger().info(f'Cartesian path computed: {response.fraction * 100:.1f}%')
                return response.solution.joint_trajectory
            else:
                self.get_logger().warn(f'Only {response.fraction * 100:.1f}% of path computed')
                return response.solution.joint_trajectory if response.fraction > 0.5 else None
        else:
            self.get_logger().error('Failed to compute Cartesian path')
            return None

    def generate_simple_trajectory(self, waypoints: list) -> JointTrajectory:
        """Generate simple trajectory using analytical IK (fallback)."""
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.header.frame_id = 'world'
        trajectory.joint_names = self.joint_names

        # Generate trajectory points
        time_from_start = 0.0
        time_step = 1.0  # seconds between waypoints

        for i, waypoint in enumerate(waypoints):
            point = JointTrajectoryPoint()

            # Simple IK approximation (for testing without MoveIt)
            joint_positions = self.simple_ik(waypoint)
            point.positions = joint_positions
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6

            time_from_start += time_step
            point.time_from_start = Duration(
                sec=int(time_from_start),
                nanosec=int((time_from_start % 1) * 1e9)
            )

            trajectory.points.append(point)

        return trajectory

    def simple_ik(self, pose: Pose) -> list:
        """
        Simple analytical IK for 6-DOF arm.
        This is a simplified approximation for testing.
        """
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        # Link lengths (from URDF)
        l1 = 0.16  # base height + joint1 housing
        l2 = 0.3   # link1_length
        l3 = 0.25  # link2_length
        l4 = 0.2   # link3_length
        l5 = 0.1   # link4_length
        l6 = 0.08  # link5_length

        # Joint 1: Base rotation
        joint1 = math.atan2(y, x)

        # Distance in XY plane
        r = math.sqrt(x*x + y*y)

        # Height above shoulder
        z_adj = z - l1

        # Distance to wrist
        d = math.sqrt(r*r + z_adj*z_adj)

        # Two-link IK for joints 2 and 3
        l_upper = l2
        l_lower = l3 + l4

        # Clamp d to valid range
        d = min(d, l_upper + l_lower - 0.01)
        d = max(d, abs(l_upper - l_lower) + 0.01)

        # Elbow angle (joint 3)
        cos_joint3 = (d*d - l_upper*l_upper - l_lower*l_lower) / (2 * l_upper * l_lower)
        cos_joint3 = max(-1.0, min(1.0, cos_joint3))
        joint3 = math.acos(cos_joint3)

        # Shoulder angle (joint 2)
        alpha = math.atan2(z_adj, r)
        beta = math.acos((l_upper*l_upper + d*d - l_lower*l_lower) / (2 * l_upper * d))
        joint2 = alpha + beta - math.pi/2

        # Wrist joints (simplified - keep end effector pointing down)
        joint4 = -joint2 - joint3 + math.pi/2
        joint5 = 0.0
        joint6 = 0.0

        # Clamp to joint limits
        joint1 = max(-3.14, min(3.14, joint1))
        joint2 = max(-1.57, min(1.57, joint2))
        joint3 = max(-2.35, min(2.35, joint3))
        joint4 = max(-3.14, min(3.14, joint4))
        joint5 = max(-3.14, min(3.14, joint5))
        joint6 = max(-3.14, min(3.14, joint6))

        return [joint1, joint2, joint3, joint4, joint5, joint6]


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
