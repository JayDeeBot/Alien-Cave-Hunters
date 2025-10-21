#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math, time, heapq
from cave_explorer.roadmap_builder import RoadmapBuilder

# -----------------------------
# Helper Class: RoadmapPath
# -----------------------------
class RoadmapPath:
    def __init__(self, roadmap: RoadmapBuilder):
        # Map & robot state
        self.latest_map_ = None
        self.current_pose = None  # Pose2D

        # PRM path-following state
        self.roadmap = roadmap
        self.current_path_nodes = []
        self.current_path_index = 0
        self.arrival_tol = 0.3

        # Frontier & artifact state
        self.current_goal = None
        self.visited_artifacts = []
        self.active_artifact_goal = None
        self.standoff_distance = 0.3

    # -----------------------------
    # Robot & Map Setters
    # -----------------------------
    def update_pose(self, pose: Pose2D):
        self.current_pose = pose

    def update_map(self, map_msg: OccupancyGrid):
        self.latest_map_ = map_msg

    # -----------------------------
    # Movement helpers
    # -----------------------------
    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def get_yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def move_toward(self, target_pose: Pose2D) -> Twist:
        """Return Twist command to move toward target."""
        cmd = Twist()
        if self.current_pose is None:
            return cmd

        dx = target_pose.x - self.current_pose.x
        dy = target_pose.y - self.current_pose.y
        distance = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        yaw = getattr(self.current_pose, 'theta', 0.0)
        angle_error = self.normalize_angle(angle_to_target - yaw)

        # Simple proportional controller
        linear_k = 0.3
        angular_k = 1.2

        if abs(angle_error) > 0.3:
            cmd.angular.z = angular_k * angle_error
        else:
            cmd.linear.x = min(0.2, linear_k * distance)
            cmd.angular.z = angular_k * angle_error

        return cmd

    # -----------------------------
    # Frontier / Artifact logic
    # -----------------------------
    def find_frontiers(self):
        if self.latest_map_ is None:
            return []

        width = self.latest_map_.info.width
        height = self.latest_map_.info.height
        data = np.array(self.latest_map_.data).reshape((height, width))

        frontiers = []
        for y in range(1, height-1):
            for x in range(1, width-1):
                if data[y,x] == -1 and 0 in [data[y+1,x], data[y-1,x], data[y,x+1], data[y,x-1]]:
                    map_x = self.latest_map_.info.origin.position.x + x*self.latest_map_.info.resolution
                    map_y = self.latest_map_.info.origin.position.y + y*self.latest_map_.info.resolution
                    frontiers.append((map_x, map_y))
        return frontiers

    @staticmethod
    def choose_frontier(frontiers, robot_pose: Pose2D):
        if not frontiers or robot_pose is None:
            return None
        # simple: choose closest frontier
        best = min(frontiers, key=lambda f: math.hypot(f[0]-robot_pose.x, f[1]-robot_pose.y))
        return Pose2D(x=best[0], y=best[1], theta=0.0)
    

    ### -----------------------------
    # PRM Path Planning
    # -----------------------------

    def get_nearest_node(self, x, y):
        if self.roadmap is None or not self.roadmap.nodes_:
            return None
        min_dist = float('inf')
        closest = None
        for node in self.roadmap.nodes_:
            dist = math.hypot(node.x - x, node.y - y)
            if dist < min_dist:
                min_dist = dist
                closest = node
        return closest
    
    def plan_path_to_goal(self, goal_x, goal_y):
        if self.roadmap is None or not self.roadmap.nodes_:
            return False
        
        # Simple: sort all nodes by distance to goal, follow nearest first
        nodes_sorted = sorted(self.roadmap.nodes_, key=lambda n: math.hypot(n.x - goal_x, n.y - goal_y))
        
        # Build path as sequence of nodes toward goal
        self.current_path_nodes = nodes_sorted[:min(10, len(nodes_sorted))]  # pick first 10 nodes toward goal
        self.current_path_index = 0
        return True

    
    def follow_prm_path(self):
        if not self.current_path_nodes or self.current_path_index >= len(self.current_path_nodes):
            return None  # nothing to follow

        target_node = self.current_path_nodes[self.current_path_index]
        cmd = self.move_toward(Pose2D(x=target_node.x, y=target_node.y, theta=0.0))

        # Advance if close enough
        distance = math.hypot(target_node.x - self.current_pose.x, target_node.y - self.current_pose.y)
        if distance < self.arrival_tol:
            self.current_path_index += 1

        return cmd




# -----------------------------
# ROS 2 Node: RoadmapPathNode
# -----------------------------
class RoadmapPathNode(Node):
    def __init__(self):
        super().__init__('roadmap_path_node')
        self.get_logger().info("Starting RoadmapPathNode")

        # Planner
        self.builder = RoadmapBuilder()  # live PRM
        self.planner = RoadmapPath(self.builder)

        # self.planner = RoadmapPath(roadmap=None)

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_prm', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/planner_markers', 10)
        self.create_subscription(Odometry, '/odometry', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Timer for control loop
        self.create_timer(0.2, self.timer_callback)  # 5 Hz

        # Subscribe to roadmap_builder markers
        self.create_subscription(
            MarkerArray,
            '/roadmap_markers',
            self.roadmap_callback,
            10
        )

    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        pose2d = Pose2D()
        pose2d.x = msg.pose.pose.position.x
        pose2d.y = msg.pose.pose.position.y
        pose2d.theta = RoadmapPath.get_yaw_from_quaternion(msg.pose.pose.orientation)
        self.planner.update_pose(pose2d)
        self.get_logger().info(f"[DEBUG] Got odometry: x={pose2d.x}, y={pose2d.y}, theta={pose2d.theta}")

    def map_callback(self, msg: OccupancyGrid):
        self.planner.update_map(msg)
        self.get_logger().info(f"[DEBUG] Got map: width={msg.info.width}, height={msg.info.height}")

    def roadmap_callback(self, msg: MarkerArray):
        nodes = []

        # Extract SPHERE markers as nodes
        for m in msg.markers:
            if m.type == Marker.SPHERE:
                node = type('GraphNode', (), {})()  # lightweight GraphNode
                node.x = m.pose.position.x
                node.y = m.pose.position.y
                node.idx = m.id
                node.neighbours = []  # optional, can be filled later
                nodes.append(node)

        if nodes:
            roadmap = type('Roadmap', (), {})()  # dummy object to hold nodes_
            roadmap.nodes_ = nodes
            roadmap.edges_ = []  # optional: can parse LINE_LIST markers if needed
            self.planner.roadmap = roadmap
            self.get_logger().info(f"[DEBUG] Updated roadmap with {len(nodes)} nodes")

    # -----------------------------
    # Timer / Control
    # -----------------------------
    # def timer_callback(self):
    #     robot_pose = self.planner.current_pose
    #     if robot_pose is None:
    #         return

    #     frontiers = self.planner.find_frontiers()
    #     self.get_logger().info(f"[DEBUG] Found {len(frontiers)} frontiers")

    #     goal = self.planner.choose_frontier(frontiers, robot_pose)
    #     if goal:
    #         cmd = self.planner.move_toward(goal)
    #         self.get_logger().info(f"[DEBUG] Publishing velocity: {cmd.linear.x}, {cmd.angular.z}")
    #         self.cmd_pub.publish(cmd)
    #         self.get_logger().info(f"Moving toward frontier at ({goal.x:.2f},{goal.y:.2f})")
    
    # -----------------------------
    # Timer / Control
    # -----------------------------
    def timer_callback(self):
        if self.planner.current_pose is None:
            return

        # If no path exists or path completed
        if not self.planner.current_path_nodes or self.planner.current_path_index >= len(self.planner.current_path_nodes):
            # 1. Find frontiers
            frontiers = self.planner.find_frontiers()
            self.get_logger().info(f"[DEBUG] Found {len(frontiers)} frontiers")

            # 2. Choose closest frontier
            goal = self.planner.choose_frontier(frontiers, self.planner.current_pose)
            if goal:
                # 3. Find closest PRM node to frontier
                nearest_node = self.planner.get_nearest_node(goal.x, goal.y)
                if nearest_node:
                    # 4. Plan path along PRM nodes toward this node
                    self.planner.plan_path_to_goal(nearest_node.x, nearest_node.y)
                    self.get_logger().info(f"[DEBUG] Planning path to PRM node at ({nearest_node.x:.2f},{nearest_node.y:.2f})")
                else:
                    self.get_logger().warn("[WARN] No PRM nodes available to plan path")
            else:
                self.get_logger().warn("[WARN] No frontier to explore")

        # 5. Follow PRM path
        cmd = self.planner.follow_prm_path()
        if cmd:
            self.cmd_pub.publish(cmd)


# -----------------------------
# Main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RoadmapPathNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
