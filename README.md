# Alien-Cave-Hunters
**Alien-Cave-Hunters** is a ROS2 Framework for a Mars rover that explores and understands a Martian cave environment.

# Getting Started

## Clone Repo
```bash
git clone https://github.com/JayDeeBot/Alien-Cave-Hunters.git
```

## Link to your ROS Workspace
```bash
ln -s ~/YOUR_GIT_WS/Alien-Cave-Hunters/cave_explorer/ ~/YOUR_ROS_WS/src/cave_explorer
```

## Build
```bash
cd ~/YOUR_ROS_WS/ # Maybe cd ~/ros_ws/
colcon build --symlink-install --packages-select cave_explorer
source install/setup.bash
```

## Launch
```bash
ros2 launch cave_explorer cave_explorer_startup.launch.py
```
