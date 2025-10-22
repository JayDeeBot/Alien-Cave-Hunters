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
from enum import Enum

class PlannerType(Enum):
    PRM_EXPLORATION = 1
    FRONTIER_EXPLORATION = 2
    ARTIFACT_EXPLORATION = 3


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
        resolution = self.latest_map_.info.resolution
        origin = self.latest_map_.info.origin.position

        frontiers = []
        OCC_THRESHOLD = 100  # occupied cell threshold

        for y in range(1, height-1):
            for x in range(1, width-1):
                val = data[y, x]
                if val != 0:
                    # only consider explicitly free cells as boundary candidates
                    continue

                # neighborhood
                nb = data[y-1:y+2, x-1:x+2]

                # must have at least one unknown neighbour
                if not np.any(nb == -1):
                    continue

                # **avoid cells that are adjacent to occupied pixels** (these often are walls)
                if np.any(nb >= OCC_THRESHOLD):
                    continue

                wx = origin.x + x * resolution
                wy = origin.y + y * resolution
                frontiers.append((wx, wy))
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
    def dijkstra_path(self, start_node, goal_node):
        """Return a list of nodes representing a shortest path from start to goal."""
        if start_node is None or goal_node is None:
            return []

        dist = {n.idx: float('inf') for n in self.roadmap.nodes_}
        prev = {n.idx: None for n in self.roadmap.nodes_}
        dist[start_node.idx] = 0

        # Priority queue: (distance, node)
        queue = [(0, start_node)]

        while queue:
            queue.sort(key=lambda x: x[0])  # simple min-heap replacement
            current_dist, current = queue.pop(0)

            if current.idx == goal_node.idx:
                break

            for neighbor in getattr(current, 'neighbours', []):
                # compute edge distance
                edge_dist = math.hypot(neighbor.x - current.x, neighbor.y - current.y)
                new_dist = current_dist + edge_dist
                if new_dist < dist[neighbor.idx]:
                    dist[neighbor.idx] = new_dist
                    prev[neighbor.idx] = current
                    queue.append((new_dist, neighbor))

        # Reconstruct path
        path = []
        node = goal_node
        while node is not None:
            path.append(node)
            node = prev[node.idx]
        path.reverse()
        return path


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
        
        start_node = self.get_nearest_node(self.current_pose.x, self.current_pose.y)
        goal_node = self.get_nearest_node(goal_x, goal_y)
        
        if start_node is None or goal_node is None:
            return False
        
        path_nodes = self.dijkstra_path(start_node, goal_node)
        if not path_nodes:
            return False
        
        # # Simple: sort all nodes by distance to goal, follow nearest first
        # nodes_sorted = sorted(self.roadmap.nodes_, key=lambda n: math.hypot(n.x - goal_x, n.y - goal_y))
        
        # Build path as sequence of nodes toward goal
        self.current_path_nodes = path_nodes
        self.current_path_index = 0
        
        return True
    
    def project_onto_edge(self, prev_node, next_node):
        """Project current robot position onto PRM edge."""
        if prev_node is None or next_node is None or self.current_pose is None:
            return None

        # Vector from prev_node to next_node
        dx = next_node.x - prev_node.x
        dy = next_node.y - prev_node.y
        if dx == 0 and dy == 0:
            return Pose2D(x=prev_node.x, y=prev_node.y, theta=0.0)

        # Vector from prev_node to robot
        rx = self.current_pose.x - prev_node.x
        ry = self.current_pose.y - prev_node.y

        # Project robot vector onto edge vector
        t = (rx*dx + ry*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))  # clamp to [0,1]

        # Compute closest point on edge
        x_proj = prev_node.x + t*dx
        y_proj = prev_node.y + t*dy
        return Pose2D(x=x_proj, y=y_proj, theta=0.0)
    # -----------------------------
    # Artifact detection (dummy)
    # -----------------------------
    def dummy_artifact_check(self):
        # For testing, return None (no artifact) or a mock object
        class DummyArtifact:
            def __init__(self):
                self.id = 0
                self.x = 1.0
                self.y = 1.0
                self.time_examined = 0.0

        # Example: return None if no artifact detected
        return None

        # Or return DummyArtifact() to simulate finding one


    
    def follow_prm_path(self):
        """Return Twist command to follow current PRM path."""

        cmd = Twist()
        path_completed = False

        if not self.current_path_nodes or self.current_path_index >= len(self.current_path_nodes):
            path_completed = True
            return cmd, path_completed

        if self.current_pose is None:
            return cmd, False

        #target node is next to path
        target_node = self.current_path_nodes[self.current_path_index]
        prev_node = self.current_path_nodes[self.current_path_index - 1] if self.current_path_index > 0 else target_node

        # Project robot onto edge between prev_node and target_node
        projected_pose = self.project_onto_edge(prev_node, target_node)

        # if projected_pose is None:
        #     projected_pose = Pose2D(x=target_node.x, y=target_node.y, theta=0.0)

        if projected_pose is None or math.isnan(projected_pose.x) or math.isnan(projected_pose.y):
            return Twist(), False
    
        cmd = self.move_toward(projected_pose)

        # Stop if dangerously close to a wall 
        if self.latest_map_ is not None:
            map_data = np.array(self.latest_map_.data)
            occ_threshold = 70
            wx, wy = projected_pose.x, projected_pose.y
            ox, oy = self.latest_map_.info.origin.position.x, self.latest_map_.info.origin.position.y
            res = self.latest_map_.info.resolution
            mx = int((wx - ox) / res)
            my = int((wy - oy) / res)
            width = self.latest_map_.info.width
            height = self.latest_map_.info.height

            if 0 <= mx < width and 0 <= my < height:
                if map_data[my * width + mx] > occ_threshold:
                    # Too close to obstacle — stop instead of crashing
                    return Twist(), False

        # Check if reached next node
        edge_length = math.hypot(projected_pose.x - self.current_pose.x,
                                projected_pose.y - self.current_pose.y)

        if edge_length < self.arrival_tol:
            self.current_path_index += 1
            if self.current_path_index >= len(self.current_path_nodes):
                path_completed = True  # signal to Node that path is done

        return cmd, path_completed

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
        # self.get_logger().info(f"[DEBUG] Got odometry: x={pose2d.x}, y={pose2d.y}, theta={pose2d.theta}")

    def map_callback(self, msg: OccupancyGrid):
        self.planner.update_map(msg)
        # self.get_logger().info(f"[DEBUG] Got map: width={msg.info.width}, height={msg.info.height}")

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

        for m in msg.markers:
            if m.type == Marker.LINE_LIST:
                # Each LINE_LIST marker has points in pairs
                for i in range(0, len(m.points), 2):
                    p1, p2 = m.points[i], m.points[i+1]
                    n1 = next((n for n in nodes if abs(n.x - p1.x) < 1e-3 and abs(n.y - p1.y) < 1e-3), None)
                    n2 = next((n for n in nodes if abs(n.x - p2.x) < 1e-3 and abs(n.y - p2.y) < 1e-3), None)
                    if n1 and n2:
                        if n2 not in n1.neighbours:
                            n1.neighbours.append(n2)
                        if n1 not in n2.neighbours:
                            n2.neighbours.append(n1)

        if nodes:
            roadmap = type('Roadmap', (), {})()  # lightweight container
            roadmap.nodes_ = nodes
            roadmap.edges_ = []  # optional
            self.planner.roadmap = roadmap 

    # -----------------------------
    # Timer / Control
    # -----------------------------
    def timer_callback(self):
        if self.planner.current_pose is None:
            return
 

        # -----------------------------
        # Always use PRM exploration
        # -----------------------------
        self.planner_type_ = PlannerType.PRM_EXPLORATION

        # -----------------------------
        # 1. Follow PRM path
        # -----------------------------
        path_finished = (not self.planner.current_path_nodes or
                        self.planner.current_path_index >= len(self.planner.current_path_nodes))
        
        self.cmd_pub.publish(cmd)

        # -----------------------------
        # 2. Plan new PRM path if finished
        # -----------------------------
        if path_finished:
            frontiers = self.planner.find_frontiers()
            goal = self.planner.choose_frontier(frontiers, self.planner.current_pose)

            if goal:
                # Ensure robot is on roadmap
                robot_node = self.planner.get_nearest_node(self.planner.current_pose.x, self.planner.current_pose.y)
                if robot_node is None:
                    robot_node = type('GraphNode', (), {})()
                    robot_node.x = self.planner.current_pose.x
                    robot_node.y = self.planner.current_pose.y
                    robot_node.idx = -1
                    robot_node.neighbours = []
                    self.planner.roadmap.nodes_.append(robot_node)

                nearest_node = self.planner.get_nearest_node(goal.x, goal.y)
                if nearest_node is not None:
                    success = self.planner.plan_path_to_goal(nearest_node.x, nearest_node.y)
                    if not success:
                        self.get_logger().warn("[WARN] Failed to plan PRM path")
                        self.cmd_pub.publish(Twist())
                        return
                else:
                    self.get_logger().warn("[PRM] No reachable PRM node near goal — waiting for roadmap expansion.")
                    self.cmd_pub.publish(Twist())
                    return
            else:
                self.get_logger().warn("[WARN] No frontier to explore")
                self.cmd_pub.publish(Twist())
                return

        # -----------------------------
        # 3. Re-project and follow path
        # -----------------------------
        cmd, completed = self.planner.follow_prm_path()
        if cmd:
            if self.planner.current_path_nodes:
                i = self.planner.current_path_index
                if i < len(self.planner.current_path_nodes):
                    node_a = self.planner.current_path_nodes[max(0, i - 1)]
                    node_b = self.planner.current_path_nodes[i]
                    projected = self.planner.project_onto_edge(node_a, node_b)
                    if projected is not None:
                        cmd = self.planner.move_toward(projected)
            self.cmd_pub.publish(cmd)
        else:
            self.cmd_pub.publish(Twist())

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
