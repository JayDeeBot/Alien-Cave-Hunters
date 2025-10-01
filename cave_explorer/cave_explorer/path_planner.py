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
    
    def cluster_frontiers(self, frontiers, threshold=0.5):
        """
        Cluster frontier points based on proximity using BFS.
        Returns a list of clusters, each cluster is a list of (x, y) points.
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
            clusters.append(cluster)

        return clusters

    
    #Function to pick which frontier to go to
    def choose_frontier(self, frontiers, robot_pose):
        """
        Choose the frontier cluster with the largest number of points.
        Returns the centroid of that cluster as a Pose2D.
        """
        if not frontiers:
            return None

        # Cluster the frontiers
        clusters = self.cluster_frontiers(frontiers)

        # Pick the largest cluster
        largest_cluster = max(clusters, key=len)

        # Return the centroid of the cluster
        x_mean = sum(p[0] for p in largest_cluster) / len(largest_cluster)
        y_mean = sum(p[1] for p in largest_cluster) / len(largest_cluster)

        return Pose2D(x=x_mean, y=y_mean, theta=0.0)
    
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

    def check_artifact_register(self):
        ''' Checks to see if artifact has already been visted and if so how much time has been spent visiting it'''
        pass

    
    def artifact_exploration_step(self):
        pass

    def register_artifact():
        '''After artifact inspection add it to register and plot it on rviz'''
        pass
        
