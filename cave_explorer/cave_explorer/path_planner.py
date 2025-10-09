import math
import numpy as np

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class PathPlanner():
    def __init__(self, node):
        self.node = node
        self.latest_map_ = None
        self.marker_pub_ = node.marker_pub_

        self.current_goal = None
        #Artifact handling
        self.visited_artifacts = []      #list of (x, y) in map frame
        self.active_artifact_goal = None #current artifact standoff goal
        self.standoff_distance = 0.3     #m, safe viewing distance
        self.artifact_timeout = 10

########################################
##### ----- Helper Functions ----- #####
########################################

    #imported needed functions from cave_explorer
    def get_pose_2d(self):
        return self.node.get_pose_2d()

    def planner_go_to_pose2d(self, pose2d):
        self.node.planner_go_to_pose2d(pose2d)

    def get_logger(self):
        return self.node.get_logger()

#############################################
##### ----- Frontier Path Planner ----- #####
#############################################

    #Function to find fronties from map occupancy grid
    def find_frontiers(self, map_msg: OccupancyGrid):
        """
        Identify frontier cells: unknown cells (value=-1) adjacent to free space (value=0)
        """
        width = map_msg.info.width
        height = map_msg.info.height
        data = np.array(map_msg.data).reshape((height, width))
        
        frontiers = []

        #iterate through grid data
        for y in range(1, height-1):
            for x in range(1, width-1):
                if data[y, x] == -1:  #check if the cell is unknown

                    #Check 4-neighbourhood for free space
                    if 0 in [data[y+1, x], data[y-1, x], data[y, x+1], data[y, x-1]]:
                        #Convert grid index to map coordinates
                        map_x = map_msg.info.origin.position.x + x * map_msg.info.resolution
                        map_y = map_msg.info.origin.position.y + y * map_msg.info.resolution

                        #add point to frontiers array
                        frontiers.append((map_x, map_y))

        #return array will points of fronties(unkown cells)
        return frontiers
    
    def cluster_frontiers(self, frontiers, threshold=0.5, min_size=6):
        """
        Cluster frontier points based on proximity using BFS.
        Returns a list of clusters, each cluster is a list of (x, y) points.
        Ignores clusters with fewer than `min_size` points.
        """
        clusters = []
        visited = set()

        for i, f in enumerate(frontiers):
            if i in visited:
                continue
            cluster = [f]
            queue = [f]
            visited.add(i)

            while queue:
                fx, fy = queue.pop(0)
                for j, other in enumerate(frontiers):
                    if j in visited:
                        continue
                    ox, oy = other
                    if math.hypot(fx - ox, fy - oy) <= threshold:
                        queue.append(other)
                        cluster.append(other)
                        visited.add(j)

            # Only add clusters with enough points
            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters


    
    #Function to pick which frontier to go to
    def choose_frontier(self, frontiers, robot_pose,w_cluster=0.2, w_cost=0.7, diff_goal_pen=0.5):
        """
        Choose the best frontier cluster based on weighted score:
        score = w_cluster * cluster_size
            - w_cost * travel_cost
            + w_stick * w_cost

        w_stick encourages staying near the current goal.
        """

        if not frontiers:
            return None

        clusters = self.cluster_frontiers(frontiers)

        best_score = float("-inf")
        best_centroid = None

        for cluster in clusters:
            # Compute centroid
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            centroid = (cx, cy)

            # Frontier properties
            cluster_size = len(cluster)

            # Distance from robot to centroid
            dx = centroid[0] - robot_pose.x
            dy = centroid[1] - robot_pose.y
            path_cost = math.hypot(dx, dy)

            # Goal persistence bonus
            diff_goal_multiplier = 1
            if isinstance(self.current_goal, (tuple, list)):
                if not math.isclose(centroid[0], self.current_goal[0], abs_tol=0.1) or \
                not math.isclose(centroid[1], self.current_goal[1], abs_tol=0.1):
                    diff_goal_multiplier = diff_goal_pen

            # Compute total score
            score = (w_cluster * cluster_size - w_cost * path_cost) * diff_goal_multiplier

            if score > best_score:
                best_score = score
                best_centroid = centroid

        # Update and return
        if best_centroid is None:
            return None

        # Store for next iteration
        self.current_goal = best_centroid

        return Pose2D(x=best_centroid[0], y=best_centroid[1], theta=0.0)
        
    #Function called every frame/step to coninitusly update the robot with the most recent data
    def frontier_exploration_step(self):
        """Perform one iteration of frontier-based exploration with dynamic replanning."""
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return

        if self.latest_map_ is None:
            self.get_logger().warn("No map available yet for frontier exploration.")
            return

        frontiers = self.find_frontiers(self.latest_map_)
        if not frontiers:
            self.get_logger().warn("No frontiers found! Exploration complete.")
            return

        goal_pose = self.choose_frontier(frontiers, robot_pose)

        if goal_pose is not None:
            self.planner_go_to_pose2d(goal_pose)
            self.get_logger().info(f'Exploring largest frontier at [{goal_pose.x:.2f}, {goal_pose.y:.2f}]')
            self.publish_frontier_markers(frontiers, goal_pose)



    #Function to publish markers
    def publish_frontier_markers(self, frontiers, goal):
        marker_array = MarkerArray()

        # Frontier points (check if frontiers exist)
        frontier_marker = Marker()
        frontier_marker.header.frame_id = 'map'
        frontier_marker.type = Marker.POINTS
        frontier_marker.action = Marker.ADD
        frontier_marker.scale.x = 0.3
        frontier_marker.scale.y = 0.3
        frontier_marker.color.b = 1.0
        frontier_marker.color.a = 1.0
        frontier_marker.points = [Point(x=f[0], y=f[1], z=0.0) for f in frontiers] if frontiers else []
        frontier_marker.id = 0
        marker_array.markers.append(frontier_marker)

        # Goal point
        goal_marker = Marker()
        goal_marker.header.frame_id = 'map'
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.scale.x = 0.5
        goal_marker.scale.y = 0.5
        goal_marker.scale.z = 0.5
        goal_marker.color.g = 1.0
        goal_marker.color.a = 1.0
        if goal is not None:
            goal_marker.pose.position.x = goal.x
            goal_marker.pose.position.y = goal.y
            goal_marker.pose.position.z = 0.0
        goal_marker.id = 1
        marker_array.markers.append(goal_marker)

        self.marker_pub_.publish(marker_array)

#############################################
##### ----- Artifact Path Planner ----- #####
#############################################

    def check_artifact_register(self, ax, ay, tol=0.5):
        """
        Return True if an artifact close to (ax, ay) has been visited before
        """
        for (vx, vy) in self.visited_artifacts:
            if math.hypot(vx - ax, vy - ay) < tol:
                return True
        return False



    def register_artifact(self, ax, ay):
        """
        Add artifact to visited list and publish a marker in RViz
        """
        self.visited_artifacts.append((ax, ay))

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = ax
        marker.pose.position.y = ay
        marker.pose.position.z = 0.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.a = 1.0
        marker.id = len(self.visited_artifacts)

        self.marker_pub_.publish(marker)

    
    def artifact_exploration_step(self, ax, ay,):
        """
        Drive to the standoff point near the detected artifact
        returns true when done
        """
        print("MOVEING TO ARTFIACTS")
        #Check if artfect is visited
        if self.check_artifact_register(ax, ay):
            self.get_logger().info("Artifact already inspected — ignoring.")
            return True


        # --- if we haven’t yet computed a goal, do it ---
        if self.active_artifact_goal is None:
            if self.node.artifact_pose_map is None:
                self.get_logger().warn("Artifact flag set but no pose available.")
                return True

            # Compute standoff pose
            robot_pose = self.get_pose_2d()
            ax = self.node.artifact_pose_map.position.x
            ay = self.node.artifact_pose_map.position.y

            dx = ax - robot_pose.x
            dy = ay - robot_pose.y
            dist = math.hypot(dx, dy)

            if dist < 1e-3:
                self.get_logger().warn("Artifact and robot pose are the same!")
                return True

            # Standoff distance and orientation
            standoff = 1.0   # metre
            sx = ax - (dx/dist) * standoff
            sy = ay - (dy/dist) * standoff
            theta = math.atan2(dy, dx)

            self.active_artifact_goal = Pose2D(x=sx, y=sy, theta=theta)
            self.get_logger().info(
                f"Artifact goal set at [{sx:.2f}, {sy:.2f}] facing [{theta:.2f}]"
            )

        # --- move to goal ---
        self.planner_go_to_pose2d(self.active_artifact_goal)

        # --- check arrival ---
        robot_pose = self.get_pose_2d()
        d = math.hypot(robot_pose.x - self.active_artifact_goal.x,
                       robot_pose.y - self.active_artifact_goal.y)

        if d < 0.3:
            self.get_logger().info("Reached artifact — registering it.")
            self.register_artifact(self.node.artifact_pose_map.position.x,self.node.artifact_pose_map.position.y)

        return False


        
