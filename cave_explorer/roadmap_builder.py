#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cave_explorer import PathPlanner
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

def main():
    # Initialize ROS2
    rclpy.init()
    
    # Create node
    node = Node('roadmap_builder')
    
    # Create PathPlanner instance
    planner = PathPlanner(node)

    # Set up a timer to repeatedly call the roadmap update
    timer_period = 0.5  # seconds
    def timer_callback():
        planner.roadmap_update()
    
    node.create_timer(timer_period, timer_callback)
    
    node.get_logger().info("Roadmap Builder Node started.")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Roadmap Builder Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
