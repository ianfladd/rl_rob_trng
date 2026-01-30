# ROS2 RL Robot Training Workspace

A ROS2 Humble workspace for training a 6-DOF robotic arm using SARSA reinforcement learning with sample end-effector paths.

## Overview

This workspace provides:
- A 6-DOF robotic arm URDF with ROS2 control integration
- MoveIt2 configuration for motion planning and IK
- Trajectory generation from sample end-effector paths
- SARSA reinforcement learning for trajectory optimization
- Complete simulation setup with Gazebo and RViz

## Workspace Structure

```
rl_rob_trng/
├── src/
│   ├── robot_description/      # URDF and visualization
│   ├── robot_moveit_config/    # MoveIt2 configuration
│   ├── trajectory_generator/   # Path → trajectory conversion
│   ├── rl_training/            # SARSA RL module
│   └── robot_bringup/          # Launch files
├── README.md
└── colcon.meta
```

## Dependencies

Install ROS2 Humble and the following packages:

```bash
sudo apt install ros-humble-ros2-control \
                 ros-humble-ros2-controllers \
                 ros-humble-gazebo-ros2-control \
                 ros-humble-moveit \
                 ros-humble-joint-state-publisher-gui \
                 ros-humble-xacro
```

## Build

```bash
cd rl_rob_trng
colcon build
source install/setup.bash
```

## Usage

### 1. Visualize Robot in RViz

```bash
ros2 launch robot_description display.launch.py
```

Use the joint_state_publisher_gui to move joints interactively.

### 2. Run Gazebo Simulation

```bash
ros2 launch robot_bringup simulation.launch.py
```

This launches:
- Gazebo with the robot
- RViz for visualization
- Joint state broadcaster
- Joint trajectory controller

### 3. Run MoveIt2 Planning

```bash
ros2 launch robot_moveit_config move_group.launch.py
```

### 4. Generate Trajectories

```bash
# Load sample paths
ros2 run trajectory_generator path_loader

# Plan trajectories
ros2 run trajectory_generator trajectory_planner

# Execute trajectories
ros2 run trajectory_generator trajectory_executor
```

### 5. Run RL Training

```bash
ros2 launch robot_bringup training.launch.py
```

Or run in simulated mode (faster, no Gazebo):

```bash
ros2 run rl_training train_node
```

## Packages

### robot_description

Contains the 6-DOF robotic arm URDF:
- 6 revolute joints: base, shoulder, elbow, wrist1, wrist2, wrist3
- Simple cylinder/box geometry
- ros2_control hardware interface

### robot_moveit_config

MoveIt2 configuration:
- SRDF with planning group "arm"
- KDL kinematics solver
- OMPL motion planning
- Joint limits and controller configuration

### trajectory_generator

Trajectory generation services:
- `path_loader.py` - Load waypoints from CSV/JSON
- `trajectory_planner.py` - Generate joint trajectories via MoveIt2
- `trajectory_executor.py` - Execute trajectories via action client

Custom interfaces:
- `SamplePath.msg` - End-effector path message
- `GenerateTrajectory.srv` - Trajectory generation service

### rl_training

SARSA reinforcement learning:
- `sarsa_agent.py` - SARSA, SARSA(λ), Expected SARSA agents
- `environment.py` - ROS2/Gazebo environment wrapper
- `state_action.py` - Discretized state/action spaces
- `reward.py` - Reward function with shaping
- `train_node.py` - ROS2 training node

**State Space:**
- Joint positions (6D)
- Joint velocities (6D)
- Distance to target (1D)

**Action Space:**
- Discretized joint velocity adjustments

**Reward Function:**
- Negative distance to target
- Waypoint reached bonus
- Joint limit penalty
- Velocity penalty
- Smoothness reward

### robot_bringup

Launch files:
- `gazebo.launch.py` - Gazebo only
- `rviz.launch.py` - RViz only
- `simulation.launch.py` - Full simulation
- `training.launch.py` - RL training

## Configuration

### Training Parameters

Edit `src/rl_training/config/training_params.yaml`:

```yaml
rl_training_node:
  ros__parameters:
    alpha: 0.1          # Learning rate
    gamma: 0.99         # Discount factor
    epsilon: 0.3        # Exploration rate
    max_episodes: 1000
    max_steps_per_episode: 200
```

### Joint Limits

Edit `src/robot_moveit_config/config/joint_limits.yaml`:

```yaml
joint_limits:
  joint1:
    max_velocity: 1.0
    max_acceleration: 2.0
```

## Monitoring Training

Training progress is published to:
- `/training/episode_reward` - Episode reward (Float32)
- `/training/episode` - Episode number (Int32)

Training logs are saved to `training_log.csv`.

Q-table checkpoints are saved to `q_table.pkl`.

## License

MIT License
