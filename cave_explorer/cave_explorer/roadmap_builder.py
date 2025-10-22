#!/usr/bin/env python3
from platform import node
import rclpy    
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Pose2D
from nav_msgs.msg import Odometry, OccupancyGrid
import numpy as np
import math, random

# Simple graph node container
class GraphNode:
    def __init__(self, x, y, idx):
        self.x = float(x)
        self.y = float(y)
        self.idx = idx
        self.neighbours = []    # store connected nodes for path planning later

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)   # Euclidean distance



class RoadmapBuilder(Node):
    # -------------------------
    #  Utility functions
    # -------------------------
    def is_too_close(self, x, y, threshold=None):
        threshold = threshold or self.min_node_spacing
        return any((n.x - x)**2 + (n.y - y)**2 < (threshold**2) for n in self.nodes_) # check existing nodes, return True if too close
    
    def world_to_map(self, x, y):   # convert world (x,y) to map (mx,my)
        mx = int((x - self.map_origin_x) / self.map_resolution)
        my = int((y - self.map_origin_y) / self.map_resolution)
        return mx, my

    def map_to_world(self, mx, my): # convert map (mx,my) to world (x,y)
        x = mx * self.map_resolution + self.map_origin_x
        y = my * self.map_resolution + self.map_origin_y
        return x, y
    
    # -------------------------
    #  PRM Node Management
    # -------------------------
    def node_at_robot_pos(self):
        """Add a node at the robot's current position if not already present."""
        if self.map_data is None:
            return None
        
        rx = float(self.current_pose_2d.x)
        ry = float(self.current_pose_2d.y)
        
        # Check if a node is already close to robot
        if self.is_too_close(rx, ry):
            return None
        

        # Mark map cell as known, only if map_data is available
        mx, my = self.world_to_map(rx, ry)

        if not (0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]):
            return None

        cell_value = self.map_data[my, mx]
        if cell_value != 0:  # only mark free cells
            return None

        # if not self.is_free_space(rx, ry, self.latest_map_msg, clearance=0.2):
        #     return None
       
        node = GraphNode(rx, ry, len(self.nodes_))
        self.nodes_.append(node)
        self.known_map[my, mx] = True

        return node
    # -------------------------
    # Collision checking helper
    # -------------------------
    def edge_valid(self, x1, y1, x2, y2, step=0.05, clearance=None):
        """Return True if straight line from (x1,y1) to (x2,y2) does not cross occupied map cells."""
        if self.map_data is None:
            return True

        clearance = clearance or self.min_node_spacing # minimum clearance from obstacles

        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(dist / step))

        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            #convert to map coords
            mx, my = self.world_to_map(x, y)

            buffer_cells = max(1, int((clearance * 1.0) / self.map_resolution)) # buffer around edge

            x0 = max(0, mx - buffer_cells)
            x1 = min(self.map_data.shape[1], mx + buffer_cells + 1)
            y0 = max(0, my - buffer_cells)
            y1 = min(self.map_data.shape[0], my + buffer_cells + 1)

            local = self.map_data[y0:y1, x0:x1]
            if np.any(local > 50):  # occupied threshold
                return False
            # if not (0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]):
            #     return False
            # if self.map_data[my, mx] > 50:  # occupied threshold
            #     return False

        return True
    
    # -------------------------
    # Node validity helper
    # -------------------------
    def node_valid(self, x, y):
        """Return True if world (x,y) maps to a free map cell."""
        if self.map_data is None:
            return True
        mx, my = self.world_to_map(x, y)
        if not (0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]):
            return False
        if self.map_data[my, mx] > 50:  # occupied
            return False
        return True

    def local_node_density(self, x, y, radius=None):
        radius = radius or self.min_node_spacing * 2
        count = 0
        for node in self.nodes_:
            if math.hypot(node.x - x, node.y - y) < radius:
                count += 1
        return count
    
    def prune_clumped_nodes(self, radius=4.0, max_density=10, min_spacing=None):
        """
        Remove nodes that are excessively clumped together to reduce computational load.

        Parameters:
        radius (float)      : used for optional spatial checks (kept for backwards compat)
        max_density (int)   : maximum allowed number of nodes inside min_spacing radius (inclusive)
        min_spacing (float) : radius (m) used to compute local density; if None, uses
                                a multiple of self.min_node_spacing.
        """
        if min_spacing is None:
            # Use a slightly larger radius for pruning than regular min spacing
            min_spacing = max(self.min_node_spacing * 2.0, 0.5) # at least 0.5 m

        if not self.nodes_:
            return

        removed_indices = set()

        # First pass: mark nodes outside the valid map or with too-high local density
        for i, node in enumerate(self.nodes_):
            # remove nodes outside the map
            if not self.node_valid(node.x, node.y):
                removed_indices.add(i)
                continue

            # compute local density (number of nodes within min_spacing)
            density = self.local_node_density(node.x, node.y, radius=min_spacing)
            if density > max_density:
                removed_indices.add(i)

        old_count = len(self.nodes_)
        # keep only nodes not marked for removal
        self.nodes_ = [n for idx, n in enumerate(self.nodes_) if idx not in removed_indices]
        new_count = len(self.nodes_)

        # Filter edges to only those connecting remaining nodes
        self.edges_ = [(n1, n2) for (n1, n2) in self.edges_ if n1 in self.nodes_ and n2 in self.nodes_]

        # Re-index nodes and clean up neighbour lists
        valid_nodes_set = set(self.nodes_)  # objects are hashable by id; this is fine
        for idx, node in enumerate(self.nodes_):
            node.idx = idx
            node.neighbours = [n for n in node.neighbours if n in valid_nodes_set]

        # Log result
        try:
            self.get_logger().info(
                f"[PRUNE] Removed {old_count - new_count} clumped nodes (kept {new_count}), "
                f"min_spacing={min_spacing:.2f} m, max_density={max_density}"
            )
        except Exception:
            # fallback if this object isn't a Node (defensive)
            print(f"[PRUNE] Removed {old_count - new_count} clumped nodes (kept {new_count})")


    def __init__(self):
        super().__init__('roadmap_builder')

        # State
        self.map_received = False
        self.current_pose_2d = Pose2D()
        self.map_data = None
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_resolution = 0.05

        # Graph
        self.nodes_ = []            # list of GraphNode
        self.edges_ = []            # list of (GraphNode, GraphNode) tuples
        self.known_map = None       # boolean mask same shape as explored map_data
        self.marker_id_counter = 0  # persistent marker id counter for RViz

        # Sampling / tuning parameters (tweak these for performance)
        self.initial_prm_nodes = 200        # initial PRM size
        self.incremental_samples_per_tick = 30
        self.incremental_sample_radius = 4.0   # meters around robot to sample
        self.min_node_spacing = 0.5           # meters minimum spacing between nodes
        self.connection_radius = 5           # meters to attempt connecting nodes
        self.publish_throttle_sec = 2.0        # don't publish more often than this

        self.node_buffer_distance = 1.0  # meters

        # QoS + publishers/subscribers
        qos = QoSProfile(depth=10,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                         history=QoSHistoryPolicy.KEEP_LAST)
        self.marker_pub = self.create_publisher(MarkerArray, '/roadmap_markers', qos)

        # subscribe odom for robot pose (use odom because robot_pose was not publishing)
        self.pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Timer: main loop (sampling + incremental add + publish incremental)
        # Use timer_callback wrapper so we can throttle publishes
        self._last_publish_time = self.get_clock().now().nanoseconds * 1e-9
        self.timer = self.create_timer(1.0, self.timer_callback)    #1 second timer for incremental growth

        self.get_logger().info("Roadmap Builder Node started.")

    # -------------------------
    # Callbacks
    # -------------------------

    def pose_callback(self, msg: Odometry):  # up to date robot pose from odom 
        self.current_pose_2d.x = msg.pose.pose.position.x
        self.current_pose_2d.y = msg.pose.pose.position.y
        self.current_pose_2d.theta = 0.0
        # debug log
        # self.get_logger().debug(f"pose: {self.current_pose_2d.x:.2f}, {self.current_pose_2d.y:.2f}")

    def map_callback(self, msg: OccupancyGrid): # process incoming occupancy grid map
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        width, height = msg.info.width, msg.info.height

        # reshape into (height, width)
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        self.map_data = data
        self.map_received = True

      

        # initialize or resize known_map
        if self.known_map is None or self.known_map.shape != (height, width):
            self.known_map = np.zeros((height, width), dtype=bool)

            # mark unknown/occupied 
            occupied = (self.map_data == -1) | (self.map_data > 50)

            # buffer
            buffer_cells = int(self.node_buffer_distance / self.map_resolution)  # 0.7 m safety distance
            from scipy.ndimage import binary_dilation
            self.known_map = binary_dilation(occupied, structure=np.ones((2*buffer_cells+1, 2*buffer_cells+1)))

            # initial PRM covering currently known free cells
            initial = self.create_prm(num_nodes=self.initial_prm_nodes)
            self.connect_nearby_nodes(radius=self.connection_radius)

            # publish initial markers fully (but limit to prevent RViz overload)
            self.publish_markers(full=True)
            self.get_logger().info(f"Initial PRM: added {len(initial)} nodes, total nodes {len(self.nodes_)}")

    # -------------------------
    # PRM creation and updates
    # -------------------------
    def create_prm(self, num_nodes=200):
        """Create PRM nodes by sampling unexplored/free map pixels, with density based on local openness and proximity to robot."""

        if self.map_data is None:
            return []

        # Find all free, unexplored pixels
        height, width = self.map_data.shape
        free_pixels = np.argwhere((self.map_data >= 0) & (self.map_data <= 50) & (~self.known_map))
        
        if free_pixels.size == 0:
            return []

        # Get robot position in map coordinates
        rx = float(self.current_pose_2d.x)
        ry = float(self.current_pose_2d.y)
        mx_r, my_r = self.world_to_map(rx, ry)

        # Score each pixel by local openness and proximity to robot
        openness_radius = 6  # pixels
        openness_scores = []
        proximity_scores = []

        for (y, x) in free_pixels:
            y0 = max(0, y - openness_radius)    # top
            y1 = min(height, y + openness_radius + 1)   # bottom
            x0 = max(0, x - openness_radius)    # left
            x1 = min(width, x + openness_radius + 1)   # right

            local = self.map_data[y0:y1, x0:x1]
            free_count = np.sum((local >= 0) & (local <= 50))
            openness_scores.append(free_count)

            # Proximity: closer to robot = higher score
            dist = math.hypot(x - mx_r, y - my_r)
            proximity_scores.append(dist)

        scores = np.array(openness_scores)
        prox = np.array(proximity_scores)

        # Invert and normalize scores for sampling probability
            # Lower openness = tighter space, so sample more nodes there
            # Closer to robot = higher probability, but not exclusively
            # blend: 60% proximity, 40% tightness

        norm_openness = (scores.max() - scores + 1)
        norm_prox = (prox.max() - prox + 1)
        blend = 0.6 * norm_prox + 0.4 * norm_openness
        prob = blend / blend.sum()
        chosen_indices = np.random.choice(len(free_pixels), size=min(num_nodes, len(free_pixels)), replace=False, p=prob)
        chosen = free_pixels[chosen_indices]

        new_nodes = []

        for (y, x) in chosen:
            world_x = x * self.map_resolution + self.map_origin_x   
            world_y = y * self.map_resolution + self.map_origin_y

            # spacing check
            if self.is_too_close(world_x, world_y) or not self.node_valid(world_x, world_y):
                self.known_map[y, x] = True
                continue

            node = GraphNode(world_x, world_y, len(self.nodes_))
            self.nodes_.append(node)
            new_nodes.append(node)
            self.known_map[y, x] = True

        return new_nodes

    def roadmap_incremental(self):
        """Add new PRM nodes near the robot and in observed but unreached areas."""
        
        if not self.map_received or self.map_data is None:
            return []

        rx = float(self.current_pose_2d.x)
        ry = float(self.current_pose_2d.y)

        height, width = self.map_data.shape
        new_nodes = []

        # Candidates near robot
        mx_c, my_c = self.world_to_map(rx, ry)
        rad_pix = int(self.incremental_sample_radius / self.map_resolution) # radius in pixels

        x0 = max(0, mx_c - rad_pix) # left
        x1 = min(width, mx_c + rad_pix + 1) # right
        y0 = max(0, my_c - rad_pix) # top
        y1 = min(height, my_c + rad_pix + 1) # bottom

        candidates_robot = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                if self.map_data[y, x] >= 0 and self.map_data[y, x] <= 50 and not self.known_map[y, x]: 
                    candidates_robot.append((y, x))

        # Candidates in observed but unreached areas (visible free space not near robot)
        candidates_explore = np.argwhere((self.map_data >= 0) & (self.map_data <= 50) & (~self.known_map))  # visible free space

        # Remove those already near robot
        candidates_explore = [tuple(c) for c in candidates_explore if abs(c[0] - my_c) > rad_pix//2 or abs(c[1] - mx_c) > rad_pix//2]   # avoid overlap

        # Sample from both sets
        total_samples = self.incremental_samples_per_tick
        n_robot = max(1, total_samples // 2)
        n_explore = total_samples - n_robot 

        chosen_robot = random.sample(candidates_robot, min(n_robot, len(candidates_robot))) if candidates_robot else []     # sample from robot candidates
        chosen_explore = random.sample(candidates_explore, min(n_explore, len(candidates_explore))) if candidates_explore else []       # sample from explore candidates
        chosen = chosen_robot + chosen_explore  # combine

        for (y, x) in chosen:
            world_x = x * self.map_resolution + self.map_origin_x
            world_y = y * self.map_resolution + self.map_origin_y

            if self.is_too_close(world_x, world_y) or not self.node_valid(world_x, world_y):
                self.known_map[y, x] = True
                continue

            node = GraphNode(world_x, world_y, len(self.nodes_))
            self.nodes_.append(node)
            new_nodes.append(node)
            self.known_map[y, x] = True

        return new_nodes

    def connect_nearby_nodes(self, radius=None, nodes_to_check=None, max_neighbors=6):
        """Connect each node to up to max_neighbors nearest nodes within radius."""

        if radius is None:
            radius = self.connection_radius

        new_edges = []

        if nodes_to_check is None:
            nodes_to_check = self.nodes_

        for node in nodes_to_check:

            # Find all other nodes within radius
            candidates = [(other, node.distance_to(other)) for other in self.nodes_ if other is not node and node.distance_to(other) < radius]
            
            # Sort by distance and take up to max_neighbors
            candidates.sort(key=lambda x: x[1])
            for other, _ in candidates[:max_neighbors]:
                already_connected = any((e1 is node and e2 is other) or (e1 is other and e2 is node) for (e1,e2) in self.edges_)

                if already_connected or not self.edge_valid(node.x, node.y, other.x, other.y):
                    continue

                node.neighbours.append(other)
                other.neighbours.append(node)
                self.edges_.append((node, other))
                new_edges.append((node, other))

        return new_edges


    # -------------------------
    # Publish markers (safe, incremental)
    # -------------------------
    def publish_markers(self, new_nodes=None, new_edges=None, full=False):
        """Publish markers.
           - If full=True: publish a snapshot of all nodes+edges (limited to avoid RViz crash).
           - Otherwise publish only new_nodes/new_edges (incremental).
        """

        if not self.nodes_:
            self.get_logger().info("No nodes to visualise yet.")
            return

        ma = MarkerArray()
        marker_count = 0

        if full:
            # full publish (used initially). Limit total markers to avoid crashing RViz.
            max_markers = 1200  # max total markers (nodes + edges)

            # create node markers (but sample if too many)
            nodes_list = self.nodes_    # make a copy

            if len(nodes_list) > 500:
                nodes_list = random.sample(nodes_list, 500) # sample 500 nodes

            for node in nodes_list:
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "roadmap_nodes"
                m.id = self.marker_id_counter; self.marker_id_counter += 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = node.x
                m.pose.position.y = node.y
                m.pose.position.z = 0.08
                m.scale.x = m.scale.y = m.scale.z = 0.12
                m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
                ma.markers.append(m)
                marker_count += 1

                if marker_count >= max_markers:
                    break

            # edges (sample a subset if many)
            for (n1, n2) in self.edges_[:max_markers - marker_count]:
                e = Marker()
                e.header.frame_id = "map"
                e.header.stamp = self.get_clock().now().to_msg()
                e.ns = "roadmap_edges"
                e.id = self.marker_id_counter; self.marker_id_counter += 1
                e.type = Marker.LINE_LIST
                e.action = Marker.ADD
                e.points = [Point(x=n1.x, y=n1.y, z=0.05), Point(x=n2.x, y=n2.y, z=0.05)]
                e.scale.x = 0.03
                e.color.r = 1.0; e.color.g = 0.0; e.color.b = 0.0; e.color.a = 0.6
                ma.markers.append(e)
        else:
            # Incremental publication: prefer new_nodes/new_edges if provided, otherwise nothing
            if new_nodes:   # publish new nodes
                for node in new_nodes:
                    m = Marker()
                    m.header.frame_id = "map"
                    m.header.stamp = self.get_clock().now().to_msg()
                    m.ns = "roadmap_nodes"
                    m.id = self.marker_id_counter; self.marker_id_counter += 1
                    m.type = Marker.SPHERE
                    m.action = Marker.ADD
                    m.pose.position.x = node.x
                    m.pose.position.y = node.y
                    m.pose.position.z = 0.08
                    m.scale.x = m.scale.y = m.scale.z = 0.12
                    m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
                    ma.markers.append(m)

            if new_edges:   # publish new edges
                for (n1, n2) in new_edges:
                    e = Marker()
                    e.header.frame_id = "map"    
                    e.header.stamp = self.get_clock().now().to_msg()    # current time
                    e.ns = "roadmap_edges"
                    e.id = self.marker_id_counter; self.marker_id_counter += 1
                    e.type = Marker.LINE_LIST
                    e.action = Marker.ADD
                    e.points = [Point(x=n1.x, y=n1.y, z=0.05), Point(x=n2.x, y=n2.y, z=0.05)]
                    e.scale.x = 0.03
                    e.color.r = 1.0; e.color.g = 0.0; e.color.b = 0.0; e.color.a = 0.6
                    ma.markers.append(e)

        # publish if anything to publish
        if ma.markers:
            self.marker_pub.publish(ma)
            self.get_logger().info(f"Published {len(ma.markers)} markers to /roadmap_markers")

    # -------------------------
    # Timer wrapper
    # -------------------------
    def timer_callback(self):
        """Main timer callback for incremental PRM growth and marker publishing."""
        # Incremental PRM growth near the robot
        # Always add node at robot position
        now_s = self.get_clock().now().seconds_nanoseconds()[0]
        # robot_node = self.add_node_at_robot_position()
        rx, ry = self.current_pose_2d.x, self.current_pose_2d.y
        new_nodes = []


        ######## check if lingering #######
        add_nodes = True
        if hasattr(self, 'last_prm_robot_pos') and self.last_prm_robot_pos:
                dist_moved = math.hypot(rx - self.last_prm_robot_pos[0], ry - self.last_prm_robot_pos[1])
                if dist_moved < getattr(self, 'robot_stay_threshold', 1.0):  # same spot
                    time_stationary = now_s - getattr(self, 'time_at_last_pos', now_s)
                    if time_stationary < getattr(self, 'prm_pause_duration', 10):
                        add_nodes = False  # skip adding nodes

        if add_nodes:
            self.last_prm_robot_pos = (rx, ry)
            self.time_at_last_pos = now_s

            robot_node = self.node_at_robot_pos()
            if robot_node:
                new_nodes.append(robot_node)

        # Incremental PRM growth near the robot
            inc_nodes = self.roadmap_incremental()

            if inc_nodes:
                new_nodes.extend(inc_nodes)
        else:
            self.prune_clumped_nodes(min_spacing=self.min_node_spacing*1.5, max_density=20)

        ### connect nodes up
        new_edges = []

        if new_nodes:
            new_edges = self.connect_nearby_nodes(radius=self.connection_radius, nodes_to_check=self.nodes_)

        # Failsafe: prune clumped nodes periodically (every 10 seconds)
        if int(now_s) % 10 == 0:
            self.prune_clumped_nodes()

        # throttle publishing: only publish at most every publish_throttle_sec seconds
        if (now_s - self._last_publish_time) >= self.publish_throttle_sec:
            self.publish_markers(new_nodes=new_nodes, new_edges=new_edges, full=False)
            self._last_publish_time = now_s

def main(args=None):
    rclpy.init(args=args)
    node = RoadmapBuilder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()