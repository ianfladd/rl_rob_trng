#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
import yaml


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_launch_description():
    # Get package directories
    pkg_description = get_package_share_directory('robot_description')
    pkg_moveit_config = get_package_share_directory('robot_moveit_config')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Robot description
    xacro_file = os.path.join(pkg_description, 'urdf', 'robot_arm.urdf.xacro')
    robot_description = Command(['xacro ', xacro_file])

    # SRDF
    srdf_file = os.path.join(pkg_moveit_config, 'config', 'robot_arm.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()

    # Load YAML configurations
    kinematics_yaml = load_yaml('robot_moveit_config', 'config/kinematics.yaml')
    joint_limits_yaml = load_yaml('robot_moveit_config', 'config/joint_limits.yaml')
    ompl_planning_yaml = load_yaml('robot_moveit_config', 'config/ompl_planning.yaml')
    moveit_controllers_yaml = load_yaml('robot_moveit_config', 'config/moveit_controllers.yaml')

    # MoveIt configuration
    moveit_config = {
        'robot_description': robot_description,
        'robot_description_semantic': robot_description_semantic,
        'robot_description_kinematics': kinematics_yaml,
        'robot_description_planning': joint_limits_yaml,
        'planning_pipelines': ['ompl'],
        'ompl': ompl_planning_yaml,
    }
    moveit_config.update(moveit_controllers_yaml)

    # RViz config for MoveIt
    rviz_config_file = os.path.join(pkg_description, 'rviz', 'robot_arm.rviz')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': use_sim_time
            }]
        ),

        # MoveIt move_group node
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            name='move_group',
            output='screen',
            parameters=[
                moveit_config,
                {'use_sim_time': use_sim_time},
                {'publish_robot_description_semantic': True},
            ],
        ),

        # RViz with MoveIt
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
            parameters=[
                moveit_config,
                {'use_sim_time': use_sim_time}
            ],
        ),
    ])
