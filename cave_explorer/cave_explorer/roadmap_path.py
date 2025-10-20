#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy # 
from geometry_msgs.msg import PoseStamped, Pose2D #global publishing of position/orientation, simpler pose (x,y,theta)
from nav_msgs.msg import Odometry, OccupancyGrid #odometry data, occupancy grid map
from visualization_msgs.msg import MarkerArray #for visualizing the PRM graph
import math, heapq, random 


class RoadmapPathNode(Node):
    """
    Node for navigating through a PRM graph built by roadmap_builder.py
    and using it to perform basic frontier-style exploration.
    """

    def __init__(self):
        super().__init__('roadmap_path')

        # Parameters
        self.declare_parameter('use_sim_time', False)
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.get_logger().info(f"RoadmapPathNode started (sim time: {self.use_sim_time})")

        # Internal state
        self.nodes_ = []                # PRM graph nodes stored and reconstructed from /roadmap_markers, MarkerArray
        self.latest_map_ = None         # Latest occupancy grid map
        self.current_pose_2d = None     # Current robot pose in (x,y,theta)
        self.visited_frontiers = set()  # Keep track of already-visited frontiers

        # ROS interfaces
        qos = QoSProfile(
            reliability=QoSDurabilityPolicy.VOLATILE,   # meaning messages are not saved for later subscribers
            history=QoSHistoryPolicy.KEEP_LAST,         # only keep last 'depth' messages
            depth=10                                    # number of messages to store
        )

        # Subscriptions, triggering callback functions when a new message arrives
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.create_subscription(MarkerArray, '/roadmap_markers', self.roadmap_callback, qos)

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10) # Publish goal poses for navigation

        # Main timer
        self.timer = self.create_timer(3.0, self.exploration_tick)  # 3 seconds timer for main explore loop


    # -----------------------------------------------------
    #  Callbacks
    # -----------------------------------------------------

    def map_callback(self, msg):
        self.latest_map_ = msg  # Store the latest occupancy grid map

    def odom_callback(self, msg):   # extract robot pose from quaternion, convert to 2D, store as current_pose_2d
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        theta = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        self.current_pose_2d = Pose2D(x=x, y=y, theta=theta)

    def roadmap_callback(self, msg: MarkerArray):
        """Reconstruct the PRM node list from visualization markers."""
        nodes = []
        for m in msg.markers:
            for p in m.points:
                node = type('GraphNode', (), {})()
                node.x = p.x
                node.y = p.y
                node.neighbours = []  # If builder publishes connectivity, parse it here
                nodes.append(node)
        self.nodes_ = nodes

    # -----------------------------------------------------
    #  Utility functions
    # -----------------------------------------------------

    def get_pose_2d(self):
        return self.current_pose_2d # Return current robot pose 
    
    def send_goal(self, x, y):      #create and publish a nav goal message
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0   # neutral orientation
        self.goal_pub.publish(goal)
        self.get_logger().info(f"Sent goal: ({x:.2f}, {y:.2f})")

    def distance(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y) # Euclidean distance between two nodes

    # -----------------------------------------------------
    #  Core logic
    # -----------------------------------------------------

    def find_random_frontier(self):
        """Mock frontier finder – just pick a random unexplored spot."""
        if not self.latest_map_:
            return None
        w = self.latest_map_.info.width
        h = self.latest_map_.info.height
        res = self.latest_map_.info.resolution
        ox = self.latest_map_.info.origin.position.x
        oy = self.latest_map_.info.origin.position.y

        for _ in range(100):
            gx = ox + random.uniform(0, w * res)
            gy = oy + random.uniform(0, h * res)
            if (gx, gy) not in self.visited_frontiers:
                return (gx, gy)
        return None

    def nearest_node(self, x, y):
        """Find the closest PRM node to a coordinate."""
        if not self.nodes_:
            return None
        return min(self.nodes_, key=lambda n: math.hypot(n.x - x, n.y - y))

    def a_star(self, start_node, goal_node):
        """A* pathfinding on PRM graph (neighbours must exist)."""
        open_set = [(0, start_node)]
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self.distance(start_node, goal_node)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_node:
                return self.reconstruct_path(came_from, current)
            for neighbor in getattr(current, "neighbours", []):
                tentative = g_score[current] + self.distance(current, neighbor)
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + self.distance(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def exploration_tick(self):
        """Main periodic exploration loop."""
        if not self.nodes_:
            self.get_logger().warn("No PRM nodes yet.")
            return
        if self.current_pose_2d is None:
            self.get_logger().warn("No odometry data yet.")
            return
        if self.latest_map_ is None:
            self.get_logger().warn("No map data yet.")
            return

        # Choose a random frontier
        frontier = self.find_random_frontier()
        if frontier is None:
            self.get_logger().info("No new frontiers found.")
            return
        self.visited_frontiers.add(frontier)

        # Find PRM nodes near start and goal
        start_node = self.nearest_node(self.current_pose_2d.x, self.current_pose_2d.y)
        goal_node = self.nearest_node(frontier[0], frontier[1])

        if not start_node or not goal_node:
            self.get_logger().warn("Could not find nearest PRM nodes.")
            return

        path_nodes = self.a_star(start_node, goal_node)
        if not path_nodes:
            self.get_logger().warn("A* failed to find a path.")
            return

        # Send first few waypoints
        self.get_logger().info(f"Following PRM path of {len(path_nodes)} nodes.")
        for wp in path_nodes[:3]:  # send first 3 for now
            self.send_goal(wp.x, wp.y)


# -----------------------------------------------------
#  Main
# -----------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RoadmapPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
