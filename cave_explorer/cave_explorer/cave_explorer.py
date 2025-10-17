#!/usr/bin/env python3
import cv2
import math
import time
import rclpy
import numpy as np
import tf2_geometry_msgs

from ultralytics import YOLO

from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from enum import Enum
from cv_bridge import CvBridge
from dataclasses import dataclass, field

from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point
from cave_explorer.path_planner import PathPlanner

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

###########################################
##### ----- Classless Functions ----- #####
###########################################

dummy_artifacts = [
    [23,5],
    [52,6],
    [-4,30],
    [54,33]
]
    

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

#########################################
##### ----- Classes n Structs ----- #####
#########################################

class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    FRONTIER_EXPLORATION = 6
    ARTIFACT_EXPLORATION = 7

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
    time_examined: float = 0.0
    visited: bool = False

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

##################################
##### ----- Cave Class ----- #####
##################################

class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False

        #### ---- Artifact Vars ---- ####
        # Marker for artifact locations
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

        #Remember the artifact locations
        self.artifact_locations_ = []
        self.artifact_timeout = 8
        self.artifact_found_ = False
        self.artifacts: list[Artifact] = []
        self.next_artifact_id = 0
        self.merge_dist_m = 2 # meters; detections within this distance are merged
        self.target_artifact = None

        self.dummy_artfacts = []
        for art_pose in dummy_artifacts:
            art = Artifact(id=self.next_artifact_id,x=art_pose[0],y=art_pose[1])
            self.next_artifact_id += 1

            self.dummy_artfacts.append(art)

        # Cache detections for planning/localisation
        self.latest_detections = []  # list of dicts per frame

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #### ---- Navigation Vars ---- ####
        self.path_planner = PathPlanner(self)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)

        #### ---- YOLO Vars ---- ####
        self.declare_parameter('yolo_model_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('yolo_conf', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('yolo_iou', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('yolo_imgsz', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('yolo_classes', rclpy.Parameter.Type.STRING_ARRAY)

        #Read params
        self.yolo_conf  = float(self.get_parameter('yolo_conf').value)
        self.yolo_iou   = float(self.get_parameter('yolo_iou').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.class_names = list(self.get_parameter('yolo_classes').value)

        self.image_msgs_seen = 0
        
        # Load YOLO model
        self.yolo_model = None
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


        #### ---- Camera Params ---- ####
        self.latest_depth = None
        self.last_depth_header = None
        self.depth_w = self.depth_h = None

        camera_width = 720
        camera_height = 480
        hfov_rad = 2.0944
        self.camera_hfov_deg = 120

        self.fx = camera_width / (2.0 * math.tan(hfov_rad / 2.0))
        vfov_rad = 2.0 * math.atan((camera_height / camera_width) * math.tan(hfov_rad / 2.0))
        self.fy = camera_height / (2.0 * math.tan(vfov_rad / 2.0))
        self.cx = (camera_width - 1) * 0.5
        self.cy = (camera_height - 1) * 0.5

        self.camera_frame_id = "camera_link"


        # QoS profile for image and camera info subscriptions
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        #### ---- Pubs/Subs ---- ####
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)

        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, sensor_qos)
        self.depth_sub_ = self.create_subscription(Image, 'camera/depth/image', self.depth_callback, sensor_qos)

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.5, self.main_loop)

#############################################
##### ----- Dummy Functions ----- #####
#############################################
    def dummy_artifact_check(self, range=12):
        rob_pose = self.get_pose_2d()
        for art in self.dummy_artfacts:
            dist = math.sqrt((art.x - rob_pose.x)**2 + (art.y - rob_pose.y)**2)
            # self.get_logger().warn(f"calculated distance {dist}")
            if dist <= range and art.time_examined < self.artifact_timeout:
                return art
            
        return None
#############################################
##### ----- Calculation Functions ----- #####
#############################################

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
        k = 10
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
                    ps, 'map', timeout=Duration(seconds=0.8)
                ).point
            except Exception as e2:
                self.get_logger().warn(f"TF transform failed (stamped & latest): {e1} | {e2}")
                return None
    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

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

        # self.get_logger().warn(f'Pose: {pose}')

        return pose
    
##########################################
##### ----- Artifact Functions ----- #####
##########################################  

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
        # Preconditions
        if not all([self.fx, self.fy, self.cx, self.cy]):
            self.get_logger().warn("[Artifacts] Missing camera intrinsics.")
            return
        
        if self.latest_depth is None:
            self.get_logger().warn("[Artifacts] No depth available.")
            return
        
        if not self.latest_detections:
            return

        cam_frame = self.camera_frame_id
        # stamp_for_tf = rclpy.time.Time().to_msg()
        stamp_for_tf = self.last_depth_header.stamp

        updates = 0

        for i, det in enumerate(self.latest_detections):
            # Class name
            cls_id = det.get("cls", -1)
            cls_name = (self.class_names[cls_id]
                        if (isinstance(cls_id, int) and 0 <= cls_id < len(self.class_names))
                        else f"class_{cls_id}")
            # self.get_logger().warn(f"[Debug] Detection {i}: cls_name={cls_name}")

            # Ignore ICE_CASTLE
            if cls_name.lower() == "ice_castle":
                # self.get_logger().warn(f"[Debug] Detection {i}: ignored ICE_CASTLE")
                continue

            x1, y1, x2, y2 = det["xyxy"]
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)
            # self.get_logger().warn(f"[Debug] Detection {i}: bbox={det['xyxy']}, center_pixel=({u:.1f},{v:.1f})")

            # ray direction in camera frame
            ray = self._project_pixel_to_ray(u, v)
            if ray is None:
                # self.get_logger().warn(f"[Debug] Detection {i}: ray projection failed")
                continue
            # self.get_logger().warn(f"[Debug] Detection {i}: ray={ray}")

            # depth at box center
            depth = self._depth_at(self.latest_depth, [x1, y1, x2, y2], (u, v))
            if depth is None or not np.isfinite(depth) or depth <= 0.0:
                # self.get_logger().warn(f"[Debug] Detection {i}: invalid depth={depth}")
                continue
            # self.get_logger().warn(f"[Debug] Detection {i}: depth={depth:.3f}")

            # 3D point in camera frame
            px_cam = ray[0] * depth
            py_cam = ray[1] * depth
            pz_cam = ray[2] * depth
            # self.get_logger().warn(f"[Debug] Detection {i}: P_cam=({px_cam:.3f},{py_cam:.3f},{pz_cam:.3f})")

            # Transform to map frame
            p_map = self._transform_cam_point_to_map(px_cam, py_cam, pz_cam, stamp_for_tf, cam_frame)
            if p_map is None:
                # self.get_logger().warn(f"[Debug] Detection {i}: TF transform failed")
                continue
            # self.get_logger().warn(f"[Debug] Detection {i}: P_map=({p_map.x:.3f},{p_map.y:.3f},{p_map.z:.3f})")

            # Upsert artifact
            artifact = self._upsert_artifact(p_map.x, p_map.y, cls_name)
            # self.get_logger().warn(f"[Debug] Detection {i}: artifact_id={artifact.id}, x={artifact.x:.3f}, y={artifact.y:.3f}, votes={artifact.votes}")
            updates += 1


        if updates > 0:
            self.publish_artifact_markers()

    def _upsert_artifact(self, x: float, y: float, cls_name: str):
        """
        If a stored artifact is within self.merge_dist_m of (x,y), fuse into it and
        add a vote for cls_name. Otherwise create a new artifact.
        """
        best_artifact = None
        best_dist = 1e9
        for artifact in self.artifacts:
            dist = math.hypot(artifact.x - x, artifact.y - y)
            if dist < best_dist:
                best_artifact = artifact
                best_dist = dist

        if best_artifact is not None and best_dist <= self.merge_dist_m:
            best_artifact.add(x, y, cls_name, alpha=0.5)
            return best_artifact
        else:
            artifact = Artifact(self.next_artifact_id, x, y)
            artifact.add(x, y, cls_name, alpha=1.0)  # initialize exactly at measurement
            self.next_artifact_id += 1
            self.artifacts.append(artifact)
            return artifact


    def publish_artifact_markers(self):
        """ Publish the artifact location markers"""
        points, colors, texts = [], [], []

        for a in self.artifacts:
            p = Point(x=float(a.x), y=float(a.y), z=1.0)
            points.append(p)
            texts.append((p, f"{a.cls}#{a.id} (n={a.n})", a.id))

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
        self.path_planner.latest_map_ = map_msg #forward map to PathPlanner

    def image_callback(self, image_msg):
        self.image_msgs_seen += 1

        # Decode with correct encoding
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
        if self.latest_depth is not None:
            h_rgb, w_rgb = rgb.shape[:2]
            if (self.depth_h, self.depth_w) != (h_rgb, w_rgb):
                if self.image_msgs_seen % 15 == 0:
                    self.get_logger().warn(f"[Depth] Size mismatch: rgb=({w_rgb}x{h_rgb}) depth=({self.depth_w}x{self.depth_h}); Skipping")
                self.artifact_found_ = False
                return

        annotated = bgr.copy()
        detections, num_boxes = [], 0

        # YOLO inference on CPU
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
        cv2.putText(annotated, f"conf={self.yolo_conf:.2f}, img={self.yolo_imgsz}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Publish overlay
        out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        msg_out = self.cv_bridge_.cv2_to_imgmsg(out_rgb, encoding="rgb8")
        msg_out.header = image_msg.header
        self.image_detections_pub_.publish(msg_out)

        self.artifact_found_ = (len(detections) > 0)
        if self.artifact_found_ :
            self.localise_artifact()


    def depth_callback(self, msg: Image):
        depth = self.cv_bridge_.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if depth is None:
            self.latest_depth = None
            self.last_depth_header = None
            return
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        elif depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        self.latest_depth = depth
        self.depth_h, self.depth_w = depth.shape[:2]
        self.last_depth_header = msg.header
        # Prefer the depth frame as the camera frame when using depth
        self.camera_frame_id = (msg.header.frame_id or self.camera_frame_id)


    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        # Goal accepted: get result when it's completed
        # self.get_logger().warn(f'Goal accepted')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback
        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""

        result = future.result().result
        # self.get_logger().info(f'Goal reached!')
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
        # self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)
            
#################################
##### ----- Main Loop ----- #####
#################################

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        """
        self.get_logger().info("------------------------------------------------")

        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return
        
        #######################################################
        #Set planner type
        artfiact_result = self.dummy_artifact_check()
        if artfiact_result != None:
            self.get_logger().info("Found artiact")
            if artfiact_result.time_examined < self.artifact_timeout:
                self.planner_type_ = PlannerType.ARTIFACT_EXPLORATION
            else:
                self.planner_type_ = PlannerType.FRONTIER_EXPLORATION 
        else:
            self.planner_type_ = PlannerType.FRONTIER_EXPLORATION

        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')

        #######################################################
        #Execute Planner
        if self.planner_type_ == PlannerType.FRONTIER_EXPLORATION:
            if hasattr(self.path_planner, 'latest_map_') and self.path_planner.latest_map_ is not None:
                self.path_planner.frontier_exploration_step()
            else:
                self.get_logger().warn('No map received yet. Cannot perform frontier exploration.')
        elif self.planner_type_ == PlannerType.ARTIFACT_EXPLORATION:
                self.get_logger().info(f"Mvoing to artfact, it has id: {artfiact_result.id}, it has pose x: {artfiact_result.x}, y: {artfiact_result.y}")
                self.path_planner.artifact_exploration_step(artfiact_result)
        else:
            self.get_logger().error('No valid planner selected')

def main():
    rclpy.init()
    cave_explorer = CaveExplorer()
    while rclpy.ok():
        rclpy.spin(cave_explorer)