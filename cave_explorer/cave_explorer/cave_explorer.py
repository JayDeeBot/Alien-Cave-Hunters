#!/usr/bin/env python3

# ### Perception 2 Imports ###
import numpy as np
from sensor_msgs.msg import CameraInfo

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception as e:
    _HAS_YOLO = False
### --------------------- ###

### Perception 3 Imports ###
from geometry_msgs.msg import PointStamped
import time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_geometry_msgs  # <-- add this line (side-effect registers PointStamped)

from dataclasses import dataclass, field
### --------------------- ###


import math
import random
from enum import Enum

import cv2  # OpenCV2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from cave_explorer.path_planner import PathPlanner

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    FRONTIER_EXPLORATION = 6
    ARTIFACT_EXPLORATION = 7
    # Add more!

# --- Perception 3: Artifact record (no Kalman, simple merge) ---
@dataclass
class Artifact:
    """
    Minimal world-coordinate artifact record.

    We maintain:
      - (x, y): fused position in map frame
      - votes: {class_name: count} for majority voting
      - n: total detections merged at this location
      - last_update: timestamp for housekeeping
    Fusion uses a lightweight EMA (alpha) to keep points stable.
    """
    id: int
    x: float
    y: float
    votes: dict = field(default_factory=dict)  # {cls_name: count}
    n: int = 0                                  # total detections fused here
    last_update: float = 0.0

    @property
    def cls(self) -> str:
        """Majority class at this location."""
        if not self.votes:
            return "unknown"
        return max(self.votes.items(), key=lambda kv: kv[1])[0]

    def add(self, x: float, y: float, cls_name: str, alpha: float = 0.5):
        """
        Fuse position with a simple EMA (keeps it stable and cheap).
        alpha is the weight of the new measurement.
        """
        self.x = (1.0 - alpha) * self.x + alpha * x
        self.y = (1.0 - alpha) * self.y + alpha * y
        self.votes[cls_name] = self.votes.get(cls_name, 0) + 1
        self.n += 1
        self.last_update = time.time()
# --- end Perception 3 ---

class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False

        # Marker for artifact locations
        # See https://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_artifacts_ = Marker()
        self.marker_artifacts_.header.frame_id = "map"
        self.marker_artifacts_.ns = "artifacts"
        self.marker_artifacts_.id = 0
        self.marker_artifacts_.type = Marker.SPHERE_LIST
        self.marker_artifacts_.action = Marker.ADD
        self.marker_artifacts_.pose.position.x = 0.0
        self.marker_artifacts_.pose.position.y = 0.0
        self.marker_artifacts_.pose.position.z = 0.0
        self.marker_artifacts_.pose.orientation.x = 0.0
        self.marker_artifacts_.pose.orientation.y = 0.0
        self.marker_artifacts_.pose.orientation.z = 0.0
        self.marker_artifacts_.pose.orientation.w = 1.0
        self.marker_artifacts_.scale.x = 1.5
        self.marker_artifacts_.scale.y = 1.5
        self.marker_artifacts_.scale.z = 1.5
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Remember the artifact locations
        # Array of type geometry_msgs.Point
        self.artifact_locations_ = []

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        #Create path_planner
        self.path_planner = PathPlanner(self)
        self.use_classic = False

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)

        # Publisher for the goal pose visualisation
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)

        # QoS profile for image and camera info subscriptions
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)

        # ### Original Computer Vision Model ###
        # self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)
        # self.computer_vision_model_ = cv2.CascadeClassifier(self.get_parameter('computer_vision_model_filename').value)
        # ### --------------------- ###

        # --- Perception 2: YOLO setup ---
        # Declare YOLO params (set from launch)
        self.declare_parameter('yolo_model_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('yolo_conf', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('yolo_iou', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('yolo_imgsz', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('yolo_classes', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('use_depth_for_localisation', False)

        # Read params
        self.yolo_conf  = float(self.get_parameter('yolo_conf').value)
        self.yolo_iou   = float(self.get_parameter('yolo_iou').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.class_names = list(self.get_parameter('yolo_classes').value)
        self.use_depth_for_localisation = bool(self.get_parameter('use_depth_for_localisation').value)

        # Subscribe to camera (adjust topic elsewhere if needed)
        self.image_msgs_seen = 0
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, sensor_qos)

        # Load YOLO model (CPU first; switch to CUDA later if desired)
        self.yolo_model = None
        if _HAS_YOLO:
            try:
                model_path = self.get_parameter('yolo_model_path').value
                self.yolo_model = YOLO(model_path)
                self.get_logger().warn(f"[YOLO] Loaded model: {model_path}")
                try:
                    self.get_logger().warn(f"[YOLO] model.names: {self.yolo_model.names}")
                except Exception:
                    pass
                if self.class_names:
                    self.get_logger().warn(f"[YOLO] class_names (param): {self.class_names}")
            except Exception as e:
                self.get_logger().error(f"[YOLO] Failed to load model: {e}")
        else:
            self.get_logger().error("[YOLO] Ultralytics not installed. Try: python3 -m pip install --user ultralytics opencv-python")
        # --- end Perception 2 ---

        # --- Perception 3: intrinsics & depth ---
        self.declare_parameter('use_manual_intrinsics', True)         # default True since your topic is missing
        self.declare_parameter('camera_hfov_deg', 90.0)               # horizontal FOV in degrees (adjust if known)
        self.use_manual_intrinsics = bool(self.get_parameter('use_manual_intrinsics').value)
        self.camera_hfov_deg = float(self.get_parameter('camera_hfov_deg').value)
        self.camera_info = None
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None
        self.last_depth_header = None
        self.depth_w = self.depth_h = None
        self.last_image_header = None
        self.camera_frame_id = None  # set from CameraInfo or image header

        # Subscribe to camera info and depth image
        self.camera_info_sub_ = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_cb, sensor_qos
        )
        self.depth_sub_ = self.create_subscription(
            Image, 'camera/depth/image', self.depth_callback, sensor_qos
        )

        # --- Perception 3: simple artifacts store (fresh every run) ---
        self.artifacts: list[Artifact] = []
        self.next_artifact_id = 1
        self.merge_dist_m = 1.5  # meters; detections within this distance are merged
        # --- end Perception 3 ---

        # Cache detections for planning/localisation
        self.latest_detections = []  # list of dicts per frame

        # Timer for main loop
        if self.use_classic:
            self.main_loop_timer_ = self.create_timer(0.2, self.main_loop_classic)
        else:
            self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

    ### Perception 3 ###
    def _upsert_artifact(self, x: float, y: float, cls_name: str):
        """
        If a stored artifact is within self.merge_dist_m of (x,y), fuse into it and
        add a vote for cls_name. Otherwise create a new artifact.
        """
        best = None
        best_d = 1e9
        for a in self.artifacts:
            d = math.hypot(a.x - x, a.y - y)
            if d < best_d:
                best = a
                best_d = d

        if best is not None and best_d <= self.merge_dist_m:
            best.add(x, y, cls_name, alpha=0.5)
            return best
        else:
            a = Artifact(self.next_artifact_id, x, y)
            a.add(x, y, cls_name, alpha=1.0)  # initialize exactly at measurement
            self.next_artifact_id += 1
            self.artifacts.append(a)
            return a
    
    def camera_info_cb(self, msg: CameraInfo):
        self.camera_info = msg
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]
        self.camera_frame_id = msg.header.frame_id or "camera_link"
        if self.use_depth_for_localisation and self.last_depth_header is not None:
            if (msg.header.frame_id or '') != (self.last_depth_header.frame_id or ''):
                self.get_logger().warn(
                    f"[Intrinsics] camera_info frame ({msg.header.frame_id}) != depth frame "
                    f"({self.last_depth_header.frame_id}). Use a camera_info for the depth stream or an aligned depth topic."
                )

    def depth_callback(self, msg: Image):
        depth = self.cv_bridge_.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if depth is None:
            self.latest_depth = None
            self.last_depth_header = None
            return
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001  # mm -> m
        elif depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        self.latest_depth = depth
        self.depth_h, self.depth_w = depth.shape[:2]
        self.last_depth_header = msg.header
        # Prefer the depth frame as the camera frame when using depth
        if self.use_depth_for_localisation:
            self.camera_frame_id = (msg.header.frame_id or self.camera_frame_id)

    def _project_pixel_to_ray(self, u: float, v: float):
        """Return a (unit-ish) ray direction in camera frame for pixel (u,v)."""
        if not all([self.fx, self.fy, self.cx, self.cy]):
            return None
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        norm = math.sqrt(x*x + y*y + z*z)
        return (x / norm, y / norm, z / norm)

    def _depth_at(self, depth_img: np.ndarray, box_xyxy, fallback_center):
        """
        Robust depth: median inside a small central window of the detection.
        box_xyxy: [x1,y1,x2,y2]; fallback_center: (u,v).
        Returns depth in meters or None.
        """
        if depth_img is None:
            return None
        h, w = depth_img.shape[:2]
        x1,y1,x2,y2 = box_xyxy
        u0 = int((x1 + x2) * 0.5); v0 = int((y1 + y2) * 0.5)
        # Small window around center (clip to image)
        k = 5
        u1 = max(0, u0 - k); u2 = min(w-1, u0 + k)
        v1 = max(0, v0 - k); v2 = min(h-1, v0 + k)
        patch = depth_img[v1:v2+1, u1:u2+1]
        if patch.size == 0:
            u0, v0 = map(int, fallback_center)
            if 0 <= v0 < h and 0 <= u0 < w:
                d = float(depth_img[v0, u0])
                return d if d > 0 else None
            return None
        vals = patch.reshape(-1)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _transform_cam_point_to_map(self, px, py, pz, stamp_msg, cam_frame: str):
        """TF transform a camera-frame 3D point to map frame."""
        ps = PointStamped()
        ps.header.frame_id = cam_frame
        ps.header.stamp = stamp_msg  # try with the image time first
        ps.point.x = float(px); ps.point.y = float(py); ps.point.z = float(pz)

        try:
            return self.tf_buffer.transform(
                ps, 'map', timeout=Duration(seconds=0.5)
            ).point
        except Exception as e1:
            # Fallback: try again with "latest" time (now) in case of cache miss
            try:
                ps.header.stamp = self.get_clock().now().to_msg()
                return self.tf_buffer.transform(
                    ps, 'map', timeout=Duration(seconds=0.5)
                ).point
            except Exception as e2:
                self.get_logger().warn(f"TF transform failed (stamped & latest): {e1} | {e2}")
                return None
    ### --------------------- ###
    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

        # Lookup the latest transform
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        self.get_logger().warn(f'Pose: {pose}')

        return pose
    
    def localise_artifact(self):
        """
        Compute artifact locations in the map frame using:
        - pixel center of each detection (u,v)
        - depth median in a small window around the box center
        - camera intrinsics (fx, fy, cx, cy)
        - TF transform from camera frame to 'map'

        Each measurement is merged into an artifact bucket if within self.merge_dist_m.
        Class is decided by majority vote over all detections at that location.
        """
        # Preconditions (keep this lightweight)
        intrinsics_ready = all([self.fx, self.fy, self.cx, self.cy])
        if not intrinsics_ready:
            self.get_logger().warn("[Artifacts] Missing intrinsics (fx,fy,cx,cy).")
            return
        if self.use_depth_for_localisation and self.latest_depth is None:
            self.get_logger().warn("[Artifacts] No depth yet.")
            return
        if not self.latest_detections:
            return

        # Choose stamp/frame: depth if we rely on depth, otherwise image header
        if self.use_depth_for_localisation and self.last_depth_header is not None:
            stamp_for_tf = self.last_depth_header.stamp
            cam_frame = (self.last_depth_header.frame_id or
                        self.camera_frame_id or
                        (self.last_image_header.frame_id if self.last_image_header else 'camera_link'))
        else:
            stamp_for_tf = (self.last_image_header.stamp if self.last_image_header else self.get_clock().now().to_msg())
            cam_frame = (self.camera_frame_id or
                        (self.last_image_header.frame_id if self.last_image_header else 'camera_link'))

        updates = 0
        for det in self.latest_detections:
            x1, y1, x2, y2 = det["xyxy"]
            u = 0.5 * (x1 + x2); v = 0.5 * (y1 + y2)

            # ray direction in camera frame
            ray = self._project_pixel_to_ray(u, v)
            if ray is None:
                continue

            # depth at box center (median in small window). If depth use is disabled, assume nominal 2.0 m
            depth = self._depth_at(self.latest_depth, [x1, y1, x2, y2], (u, v)) if self.use_depth_for_localisation else 2.0
            if depth is None or not np.isfinite(depth) or depth <= 0.0:
                continue

            # 3D point in camera frame: P_cam = depth * unit-ray
            px_cam = ray[0] * depth
            py_cam = ray[1] * depth
            pz_cam = ray[2] * depth

            # Transform to map frame
            p_map = self._transform_cam_point_to_map(px_cam, py_cam, pz_cam, stamp_for_tf, cam_frame)
            if p_map is None:
                continue

            # Class name
            cls_id = det.get("cls", -1)
            cls_name = (self.class_names[cls_id]
                        if (isinstance(cls_id, int) and 0 <= cls_id < len(self.class_names))
                        else f"class_{cls_id}")

            # Upsert artifact (distance-based merge + majority vote)
            self._upsert_artifact(p_map.x, p_map.y, cls_name)
            updates += 1

        if updates > 0:
            self.publish_artifact_markers()

    def publish_artifact_markers(self):
        """
        Publish all saved artifact estimates as:
        - a SPHERE_LIST in 'map' frame
        - one TEXT_VIEW_FACING label per artifact showing majority class and count

        Uses self.marker_artifacts_ (already initialised in __init__) and the existing
        marker publisher 'marker_array_artifacts'.
        """
        from std_msgs.msg import ColorRGBA

        points, colors, texts = [], [], []

        for a in self.artifacts:
            p = Point(x=float(a.x), y=float(a.y), z=1.0)
            points.append(p)
            texts.append((p, f"{a.cls}#{a.id} (n={a.n})", a.id))

            # deterministic color per class
            h = (hash(a.cls) % 255) / 255.0
            r = 0.1 + 0.2 * h
            g = 0.7 + 0.2 * (1.0 - h)
            b = 0.2 + 0.2 * (0.5 - abs(0.5 - h))
            colors.append(ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0))

        # MarkerArray with DELETEALL + spheres + text labels
        marr = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marr.markers.append(delete_all)

        # Sphere list
        self.marker_artifacts_.header.stamp = self.get_clock().now().to_msg()
        self.marker_artifacts_.points = points
        self.marker_artifacts_.colors = colors if len(colors) == len(points) else []
        marr.markers.append(self.marker_artifacts_)

        # Text labels (stable per-id)
        text_id_base = 10000
        for p, txt, aid in texts:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.marker_artifacts_.header.stamp
            m.ns = "artifact_labels"
            m.id = text_id_base + int(aid)
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = p.x; m.pose.position.y = p.y; m.pose.position.z = p.z + 0.8
            m.scale.z = 0.8
            m.color.a = 1.0; m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0
            m.text = txt
            marr.markers.append(m)

        self.marker_pub_.publish(marr)

##########################################
##### ----- Callback Functions ----- #####
##########################################

    def map_callback(self, map_msg: OccupancyGrid):
        """New map received, so update x and y limits"""

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x, 
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        self.latest_map_ = map_msg
        self.path_planner.latest_map_ = map_msg  # forward map to PathPlanner

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')

    # ### Original Computer Vision Model ###
    # def image_callback(self, image_msg):
    #     """
    #     Recieve an RGB image.
    #     Use this method to detect artifacts of interest.
        
    #     A simple method has been provided to begin with for detecting stop signs (which is not what we're actually looking for) 
    #     adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
    #     """
    
    #     # Copy the image message to a cv image
    #     # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
    #     image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

    #     # Create a grayscale version (some simple models use this)
    #     # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Retrieve the pre-trained model
    #     stop_sign_model = self.computer_vision_model_

    #     # Detect artifacts in the image
    #     # The minSize is used to avoid very small detections that are probably noise
    #     detections = stop_sign_model.detectMultiScale(image, minSize=(20,20))

    #     # You can set "artifact_found_" to true to signal to "main_loop" that you have found a artifact
    #     # You may want to communicate more information
    #     # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
    #     # with a mutex
    #     # "artifact_found_" doesn't need a mutex because it's an atomic
    #     num_detections = len(detections)

    #     if num_detections > 0:
    #         self.artifact_found_ = True
    #     else:
    #         self.artifact_found_ = False

    #     # Draw a bounding box rectangle on the image for each detection
    #     for(x, y, width, height) in detections:
    #         cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 0), 5)

    #     # Publish the image with the detection bounding boxes
    #     image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8")
    #     self.image_detections_pub_.publish(image_detection_message)

    #     if self.artifact_found_:
    #         self.get_logger().info('Artifact found!')
    #         self.localise_artifact()

    # ### --------------------- ###

    ### Perception 2 ###
    def image_callback(self, image_msg):
        # Count frames
        self.image_msgs_seen += 1
        if self.image_msgs_seen % 10 == 0:
            self.get_logger().info(f"[image] frames seen: {self.image_msgs_seen}")

        # Decode with correct encoding (avoid RGB/BGR mix-ups)
        enc = (image_msg.encoding or '').lower()
        try:
            if enc in ('rgb8', 'rgba8'):
                rgb = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                # default to BGR
                bgr = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion error: {e}")
            return
        
        # Sanity: depth & color should match resolution when using per-pixel depth
        if self.use_depth_for_localisation and self.latest_depth is not None:
            h_rgb, w_rgb = rgb.shape[:2]
            if (self.depth_h, self.depth_w) != (h_rgb, w_rgb):
                if self.image_msgs_seen % 15 == 0:
                    self.get_logger().warn(
                        f"[Depth] Size mismatch: rgb=({w_rgb}x{h_rgb}) depth=({self.depth_w}x{self.depth_h}); skipping localisation this frame."
                    )
                self.artifact_found_ = False
                return

        # --- Manual intrinsics fallback if no CameraInfo ---
        if self.camera_info is None and self.use_manual_intrinsics:
            h, w = rgb.shape[:2]
            hfov = math.radians(self.camera_hfov_deg)
            fx = w / (2.0 * math.tan(hfov / 2.0))
            fy = fx
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5
            if not all([self.fx, self.fy, self.cx, self.cy]):
                self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
                self.camera_frame_id = image_msg.header.frame_id or "camera_link"
                self.get_logger().warn(
                    f"[Intrinsics] Using manual intrinsics fx=fy={fx:.1f}, cx={cx:.1f}, cy={cy:.1f} "
                    f"(w={w}, h={h}, hfov={self.camera_hfov_deg:.1f} deg)"
                )

        annotated = bgr.copy()
        detections, num_boxes = [], 0

        # --- YOLO inference on CPU (robust default) ---
        hud_text = "YOLO: off"
        try:
            if self.yolo_model is not None:
                results = self.yolo_model.predict(
                    source=rgb,
                    conf=self.yolo_conf,
                    iou=self.yolo_iou,
                    imgsz=self.yolo_imgsz,
                    verbose=False,
                    device='cpu'   # switch to 'cuda:0' later once verified
                )
                if results and len(results) > 0 and getattr(results[0], 'boxes', None) is not None:
                    res = results[0]
                    num_boxes = len(res.boxes)
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                        name = (self.class_names[cls_id] if 0 <= cls_id < len(self.class_names)
                                else (self.yolo_model.names.get(cls_id, f"class_{cls_id}")
                                    if hasattr(self.yolo_model, 'names') else f"class_{cls_id}"))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, f"{name} {conf:.2f}", (x1, max(0, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        detections.append({"xyxy":[x1,y1,x2,y2], "cls":cls_id, "conf":conf,
                                        "stamp": image_msg.header.stamp})
                    hud_text = f"YOLO: {num_boxes} boxes"
                else:
                    hud_text = "YOLO: 0 boxes"
            else:
                hud_text = "YOLO: not loaded"
        except Exception as e:
            self.get_logger().error(f"YOLO predict error: {e}")
            hud_text = "YOLO: error"

        # Planner flags
        self.latest_detections = detections

        # HUD so you can see status in RViz
        cv2.putText(annotated, hud_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"conf={self.yolo_conf:.2f}, img={self.yolo_imgsz}", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Publish overlay (always publish so RViz never shows "No image")
        out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        msg_out = self.cv_bridge_.cv2_to_imgmsg(out_rgb, encoding="rgb8")
        msg_out.header = image_msg.header
        self.image_detections_pub_.publish(msg_out)

        self.last_image_header = image_msg.header
        # Kick off localisation & fusion
        ready_to_localise = (self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None) and \
                    (self.latest_depth is not None or not self.use_depth_for_localisation)

        self.artifact_found_ = (len(detections) > 0) and ready_to_localise

        # if ready_to_localise:
        #     # Minimal Perception 3 pipeline:
        #     # YOLO boxes -> pixel(center) -> depth median -> 3D cam -> TF to map -> merge -> RViz
        #     self.localise_artifact()
        # else:
        #     # Throttle this warning
        #     if self.image_msgs_seen % 15 == 0:
        #         if self.fx is None:
        #             self.get_logger().warn("Waiting for intrinsics (CameraInfo or manual).")
        #         if self.use_depth_for_localisation and self.latest_depth is None:
        #             self.get_logger().warn("Waiting for depth frames…")
        ### --------------------- ###

    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        # Goal accepted: get result when it's completed
        self.get_logger().warn(f'Goal accepted')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""

        result = future.result().result
        self.get_logger().info(f'Goal reached!')
        self.ready_for_next_goal_ = True

#########################################
##### ----- Planner Functions ----- #####
#########################################

    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        # Decide whether to show feedback or not
        if self.get_parameter('print_feedback').value:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        # Send goal to action server
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)

    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

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
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

#################################
##### ----- Main Loop ----- #####
#################################

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        """
        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return
        
        #Set planner type to frontier
        if self.artifact_found_:
            self.planner_type_ = PlannerType.FRONTIER_EXPLORATION
            #self.planner_type_ = PlannerType.ARTIFACT_EXPLORATION
        else:
            self.planner_type_ = PlannerType.FRONTIER_EXPLORATION

        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')

        #Run though control logic
        if self.planner_type_ == PlannerType.FRONTIER_EXPLORATION:
            if hasattr(self.path_planner, 'latest_map_') and self.path_planner.latest_map_ is not None:
                #run frontier explortion
                self.path_planner.frontier_exploration_step()
            else:
                self.get_logger().warn('No map received yet. Cannot perform frontier exploration.')

        # elif self.planner_type_ == PlannerType.ARTIFACT_EXPLORATION:
        #     done = self.path_planner.artifact_exploration_step()
        #     if done:
        #         self.planner_type_ = PlannerType.FRONTIER_EXPLORATION
        else:
            self.get_logger().error('No valid planner selected')

    def main_loop_classic(self):
        """
        Original main loop
        """
        
        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        #######################################################
        # Update flags related to the progress of the current planner

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.get_logger().info('Successfully reached first artifact!')
            self.reached_first_artifact_ = True
        if self.planner_type_ == PlannerType.RETURN_HOME:
            self.get_logger().info('Successfully returned home!')
            self.returned_home_ = True

        #######################################################
        # Select the next planner to execute
        # Update this logic as you see fit!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.RANDOM_GOAL

        #######################################################
        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()
        #######################################################

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)