import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class PathPlanner():
    def __init__(self):
        pass

    def find_frontiers(self, map_msg: OccupancyGrid):
        """
        Identify frontier cells: unknown cells (value=-1) adjacent to free space (value=0)
        """
        width = map_msg.info.width
        height = map_msg.info.height
        data = np.array(map_msg.data).reshape((height, width))
        
        frontiers = []

        for y in range(1, height-1):
            for x in range(1, width-1):
                if data[y, x] == -1:  # Unknown
                    # Check 4-neighbourhood for free space
                    if 0 in [data[y+1, x], data[y-1, x], data[y, x+1], data[y, x-1]]:
                        # Convert grid index to map coordinates
                        map_x = map_msg.info.origin.position.x + x * map_msg.info.resolution
                        map_y = map_msg.info.origin.position.y + y * map_msg.info.resolution
                        frontiers.append((map_x, map_y))
    
        return frontiers
    

    def choose_frontier(self, frontiers, robot_pose):
        if not frontiers:
            return None

        best_frontier = min(frontiers, key=lambda f: math.hypot(f[0]-robot_pose.x, f[1]-robot_pose.y))
        return Pose2D(x=best_frontier[0], y=best_frontier[1], theta=0.0)
    


    def planner_frontier_exploration(self):
        """Perform frontier-based exploration"""
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return

        frontiers = self.find_frontiers(self.latest_map_)
        goal_pose = self.choose_frontier(frontiers, robot_pose)

        if goal_pose is not None:
            self.planner_go_to_pose2d(goal_pose)
            self.get_logger().info(f'Exploring frontier at [{goal_pose.x:.2f}, {goal_pose.y:.2f}]')
            
            # Optional: visualize frontiers in RViz
            self.publish_frontier_markers(frontiers, goal_pose)
        else:
            self.get_logger().warn("No frontiers found! Exploration complete.")

    def publish_frontier_markers(self, frontiers, goal):
        marker_array = MarkerArray()

        # Frontier points
        frontier_marker = Marker()
        frontier_marker.header.frame_id = 'map'
        frontier_marker.type = Marker.POINTS
        frontier_marker.action = Marker.ADD
        frontier_marker.scale.x = 0.3
        frontier_marker.scale.y = 0.3
        frontier_marker.color.r = 1.0
        frontier_marker.color.a = 1.0
        frontier_marker.points = [Point(x=f[0], y=f[1], z=0.0) for f in frontiers]
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
        goal_marker.pose.position.x = goal.x
        goal_marker.pose.position.y = goal.y
        goal_marker.pose.position.z = 0.0
        marker_array.markers.append(goal_marker)

        self.marker_pub_.publish(marker_array)