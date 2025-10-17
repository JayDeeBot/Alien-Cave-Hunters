import math
import random
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
        self.active_artifact_goal = None #current artifact standoff goal
        self.standoff_distance = 0.3     #m, safe viewing distance

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
    def choose_frontier(self, frontiers, robot_pose, w_cluster=1.0, w_cost=1.0, w_turn=1.0, w_persist=0.5):
        """
        Choose the best frontier cluster based on weighted score:
        score = + w_cluster * normalized_cluster_size
                - w_cost * normalized_path_cost
                - w_turn * normalized_turn_penalty
                + w_persist * goal_persistence_bonus
        """

        if not frontiers:
            return None

        clusters = self.cluster_frontiers(frontiers)
        if not clusters:
            return None

        # Precompute normalization factors
        max_cluster_size = max(len(c) for c in clusters)
        max_path_cost = 0.0
        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dx = cx - robot_pose.x
            dy = cy - robot_pose.y
            cost = math.hypot(dx, dy)
            max_path_cost = max(max_path_cost, cost)

        best_score = float("-inf")
        best_centroid = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            centroid = (cx, cy)

            cluster_size = len(cluster)
            dx = cx - robot_pose.x
            dy = cy - robot_pose.y
            path_cost = math.hypot(dx, dy)
            desired_heading = math.atan2(dy, dx)
            heading_diff = abs((desired_heading - robot_pose.theta + math.pi) % (2 * math.pi) - math.pi)

            # Normalize terms
            norm_cluster = cluster_size / max_cluster_size
            norm_cost = path_cost / max_path_cost if max_path_cost > 0 else 0.0
            norm_turn = heading_diff / math.pi

            # Goal persistence bonus
            goal_persistence_bonus = 0.0
            if isinstance(self.current_goal, (tuple, list)):
                if math.isclose(cx, self.current_goal[0], abs_tol=0.1) and \
                math.isclose(cy, self.current_goal[1], abs_tol=0.1):
                    goal_persistence_bonus = 1.0

            score = (
                + w_cluster * norm_cluster
                - w_cost * norm_cost
                - w_turn * norm_turn
                + w_persist * goal_persistence_bonus
            )

            if score > best_score:
                best_score = score
                best_centroid = centroid

        if best_centroid is None:
            return None

        self.current_goal = best_centroid
        return Pose2D(x=best_centroid[0], y=best_centroid[1], theta=0.0)


        
    #Function called every frame/step to coninitusly update the robot with the most recent data
    def frontier_exploration_step(self):
        """Perform one iteration of frontier-based exploration with dynamic replanning."""
        robot_pose = self.node.get_pose_2d()
        if robot_pose is None:
            return

        if self.latest_map_ is None:
            self.node.get_logger().warn("No map available yet for frontier exploration.")
            return

        frontiers = self.find_frontiers(self.latest_map_)
        if not frontiers:
            # self.node.get_logger().warn("No frontiers found! Exploration complete.")
            return

        goal_pose = self.choose_frontier(frontiers, robot_pose)

        if goal_pose is not None:
            self.node.planner_go_to_pose2d(goal_pose)
            # self.node.get_logger().info(f'Exploring largest frontier at [{goal_pose.x:.2f}, {goal_pose.y:.2f}]')
            self.publish_frontier_markers(frontiers, goal_pose)



    #Function to publish markers
    def publish_frontier_markers(self, frontiers, goal):
        marker_array = MarkerArray()

        # Frontier points
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

    def get_artifact(self):
        """
        Navigate to the nearest unvisited artifact.
        If all artifacts are visited, fallback to frontier exploration.
        """
        if not self.artifacts:
            self.node.get_logger().info("No artifects")
            return None

        # Current robot position
        pose = self.node.get_pose_2d()

        # Sort artifacts by distance to robot
        sorted_artifacts = sorted(self.artifacts,key=lambda a: math.hypot(a.x - pose.x, a.y - pose.y))

        registered_artfiacts = self.path_planner.get_artifact_register()
        registered_ids = [r[2] for r in registered_artfiacts]  # all registered artifact IDs

        for a in sorted_artifacts:
            # Check if artifact is in the register
            for reg in registered_artfiacts:
                reg_x, reg_y, reg_id, timer = reg
                
                if a.id == reg_id:
                    # Found a match — check the timer
                    if timer <= self.artifact_timeout:
                        self.node.get_logger().info(f"Artifact {a.id} timer expired ({timer:.2f}s >= {self.artifact_timeout}s)")
                        return a.id
                    break  # no need to keep checking other registers
            
            # If artifact not registered at all — prioritise it
            if a.id not in registered_ids:
                self.node.get_logger().info(f"New artifact found (ID: {a.id}) — prioritising visit")
                return a.id

        # If none found
        return None
    
    def artifact_exploration_step(self, artifact):
        """
        Drive to the standoff point near the detected artifact
        returns true when done
        """
        # self.node.get_logger().info(f" artifac ID {artifact_id}")
        # artifact = self.registered_artifacts[artifact_id]
        #if we haven’t yet computed a goal, do it
        self.node.get_logger().info(f"Mvoing to artfact, it has id: {artifact.id}, it has pose x: {artifact.x}, y: {artifact.y}")
        if self.active_artifact_goal is None:
            # Compute standoff pose
            robot_pose = self.node.get_pose_2d()
            dx = artifact.x - robot_pose.x
            dy = artifact.y - robot_pose.y
            dist = math.hypot(dx, dy)

            #Standoff distance and orientation
            standoff = 1.0   # metre
            sx = artifact.x - (dx/dist) * standoff
            sy = artifact.y - (dy/dist) * standoff
            theta = math.atan2(dy, dx)

            self.active_artifact_goal = Pose2D(x=sx, y=sy, theta=theta)
            self.node.get_logger().info(f"Artifact goal set at [{sx:.2f}, {sy:.2f}] facing [{theta:.2f}]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        
        # --- move to goal ---
        self.node.get_logger().info(f"artact goal, it has pose x: {self.active_artifact_goal.x}, y: {self.active_artifact_goal.y}") 
        self.node.planner_go_to_pose2d(self.active_artifact_goal)

        # --- check arrival ---
        robot_pose = self.node.get_pose_2d()
        d = math.hypot(robot_pose.x - self.active_artifact_goal.x,
                       robot_pose.y - self.active_artifact_goal.y)
        
        artifact.time_examined = artifact.time_examined + 0.5
        self.node.get_logger().info(f" artifact timer: {artifact.time_examined}")
        if  artifact.time_examined >= self.node.artifact_timeout:
            self.active_artifact_goal = None
            artifact.visited = True

##################################################
##### ----- Original Planner Functions ----- #####
##################################################
        
    def planner_move_forwards(self, distance):
            """Simply move forward by the specified distance"""

            pose_2d = self.node.get_pose_2d()
            pose_2d.x += distance * math.cos(pose_2d.theta)
            pose_2d.y += distance * math.sin(pose_2d.theta)
            self.node.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.node.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.node.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.node.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location
        goal_valid = False
        while not goal_valid:
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
            else:
                self.node.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.node.planner_go_to_pose2d(goal_pose2d)