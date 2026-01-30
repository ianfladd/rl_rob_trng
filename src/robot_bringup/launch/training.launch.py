#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_description = get_package_share_directory('robot_description')
    pkg_bringup = get_package_share_directory('robot_bringup')
    pkg_rl_training = get_package_share_directory('rl_training')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='')
    headless = LaunchConfiguration('headless', default='false')

    # Robot description
    xacro_file = os.path.join(pkg_description, 'urdf', 'robot_arm.urdf.xacro')
    robot_description = Command(['xacro ', xacro_file])

    # Training params
    training_params_file = os.path.join(pkg_rl_training, 'config', 'training_params.yaml')

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock'
        ),
        DeclareLaunchArgument(
            'world',
            default_value='',
            description='World file to load'
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Gazebo in headless mode (no GUI)'
        ),

        # Gazebo server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items()
        ),

        # Gazebo client (conditional on headless)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
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

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_robot',
            output='screen',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'robot_arm',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.0'
            ]
        ),

        # Controller Manager - spawn joint state broadcaster (delayed)
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                         'joint_state_broadcaster'],
                    output='screen'
                )
            ]
        ),

        # Controller Manager - spawn joint trajectory controller (delayed)
        TimerAction(
            period=4.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                         'joint_trajectory_controller'],
                    output='screen'
                )
            ]
        ),

        # RL Training Node (delayed to ensure everything is ready)
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='rl_training',
                    executable='train_node',
                    name='rl_training_node',
                    output='screen',
                    parameters=[
                        training_params_file,
                        {'use_sim_time': use_sim_time}
                    ]
                )
            ]
        ),
    ])
