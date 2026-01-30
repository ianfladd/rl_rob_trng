#!/usr/bin/env python3
"""
Path Loader Node
Loads sample end-effector paths from CSV/JSON files and publishes them.
"""

import json
import csv
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from trajectory_generator.msg import SamplePath


class PathLoaderNode(Node):
    """Load and publish sample end-effector paths."""

    def __init__(self):
        super().__init__('path_loader')

        # Parameters
        self.declare_parameter('path_file', '')
        self.declare_parameter('publish_rate', 1.0)

        self.path_file = self.get_parameter('path_file').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        # Publisher
        self.path_publisher = self.create_publisher(SamplePath, 'sample_path', 10)

        # Timer for publishing
        if publish_rate > 0:
            self.timer = self.create_timer(1.0 / publish_rate, self.publish_path)

        # Load path on startup if file specified
        self.current_path = None
        if self.path_file:
            self.current_path = self.load_path(self.path_file)
            if self.current_path:
                self.get_logger().info(f'Loaded path with {len(self.current_path.waypoints)} waypoints')

        self.get_logger().info('Path Loader Node started')

    def load_path(self, filepath: str) -> SamplePath:
        """Load path from CSV or JSON file."""
        if not os.path.exists(filepath):
            self.get_logger().error(f'Path file not found: {filepath}')
            return None

        _, ext = os.path.splitext(filepath)

        if ext.lower() == '.json':
            return self.load_json_path(filepath)
        elif ext.lower() == '.csv':
            return self.load_csv_path(filepath)
        else:
            self.get_logger().error(f'Unsupported file format: {ext}')
            return None

    def load_json_path(self, filepath: str) -> SamplePath:
        """Load path from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            path_msg = SamplePath()
            path_msg.header = Header()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = data.get('frame_id', 'world')
            path_msg.path_id = data.get('path_id', os.path.basename(filepath))
            path_msg.description = data.get('description', '')

            for wp in data.get('waypoints', []):
                pose = Pose()
                pose.position = Point(
                    x=float(wp.get('x', 0.0)),
                    y=float(wp.get('y', 0.0)),
                    z=float(wp.get('z', 0.0))
                )
                pose.orientation = Quaternion(
                    x=float(wp.get('qx', 0.0)),
                    y=float(wp.get('qy', 0.0)),
                    z=float(wp.get('qz', 0.0)),
                    w=float(wp.get('qw', 1.0))
                )
                path_msg.waypoints.append(pose)

                if 'time' in wp:
                    path_msg.time_stamps.append(float(wp['time']))

            return path_msg

        except Exception as e:
            self.get_logger().error(f'Error loading JSON path: {e}')
            return None

    def load_csv_path(self, filepath: str) -> SamplePath:
        """Load path from CSV file.

        Expected format: x,y,z,qx,qy,qz,qw[,time]
        """
        try:
            path_msg = SamplePath()
            path_msg.header = Header()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'world'
            path_msg.path_id = os.path.basename(filepath)
            path_msg.description = f'Loaded from {filepath}'

            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header if present

                for row in reader:
                    if len(row) < 3:
                        continue

                    pose = Pose()
                    pose.position = Point(
                        x=float(row[0]),
                        y=float(row[1]),
                        z=float(row[2])
                    )

                    if len(row) >= 7:
                        pose.orientation = Quaternion(
                            x=float(row[3]),
                            y=float(row[4]),
                            z=float(row[5]),
                            w=float(row[6])
                        )
                    else:
                        # Default orientation (pointing down Z)
                        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                    path_msg.waypoints.append(pose)

                    if len(row) >= 8:
                        path_msg.time_stamps.append(float(row[7]))

            return path_msg

        except Exception as e:
            self.get_logger().error(f'Error loading CSV path: {e}')
            return None

    def publish_path(self):
        """Publish the current path."""
        if self.current_path:
            self.current_path.header.stamp = self.get_clock().now().to_msg()
            self.path_publisher.publish(self.current_path)

    def create_sample_path(self) -> SamplePath:
        """Create a sample circular path for testing."""
        import math

        path_msg = SamplePath()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'world'
        path_msg.path_id = 'sample_circle'
        path_msg.description = 'Sample circular path for testing'

        # Create circular path
        center_x, center_y, center_z = 0.3, 0.0, 0.5
        radius = 0.1
        num_points = 20

        for i in range(num_points):
            angle = 2.0 * math.pi * i / num_points

            pose = Pose()
            pose.position = Point(
                x=center_x + radius * math.cos(angle),
                y=center_y + radius * math.sin(angle),
                z=center_z
            )
            pose.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)
            path_msg.waypoints.append(pose)
            path_msg.time_stamps.append(float(i) * 0.5)

        return path_msg


def main(args=None):
    rclpy.init(args=args)
    node = PathLoaderNode()

    # If no file specified, create and publish sample path
    if not node.current_path:
        node.current_path = node.create_sample_path()
        node.get_logger().info('Created sample circular path')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
