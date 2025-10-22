# ### Perception 2 Imports ###
import numpy as np

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception as e:
    _HAS_YOLO = False

import cv2
### --------------------- ###

### Perception 3 Imports ###
import json
import math
import time
import rclpy
import numpy as np
import tf2_geometry_msgs

from pathlib import Path
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
    cls: str   
    x: float
    y: float
    conf_avg: float = 0.0
    weight_sum: float = 0.0
    #votes: dict = field(default_factory=dict)  # {cls_name: count}
    n: int = 0                                  # total detections fused here
    last_update: float = 0.0
    time_examined: float = 0.0
    visited: bool = False

    def add(self, x_new: float, y_new: float, conf_new: float):
        """
        Update (x,y) and conf_avg using cumulative confidence-weighted averages.
        """
        w_prev = self.weight_sum
        w_new  = max(0.0, float(conf_new))
        if w_new == 0.0:
            # No contribution if conf=0
            return

        self.x = (self.x * w_prev + x_new * w_new) / (w_prev + w_new)
        self.y = (self.y * w_prev + y_new * w_new) / (w_prev + w_new)
        self.conf_avg = (self.conf_avg * w_prev + conf_new * w_new) / (w_prev + w_new)
        self.weight_sum = w_prev + w_new

        self.n += 1
        self.last_update = time.time()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "class": self.cls,
            "x": self.x,
            "y": self.y,
            "smoothed_confidence": self.conf_avg,
            "n": self.n,
            "last_update": self.last_update,
        }

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

        # Remember the artifact locations
        self.artifact_locations_ = []
        self.artifact_timeout = 8
        self.artifact_found_ = False
        self.artifacts: list[Artifact] = []
        self.next_artifact_id = 0
        self.merge_dist_m = 5.0  # meters; detections within this distance are merged
        self.target_artifact = None

        self.dummy_artfacts = []
        for art_pose in dummy_artifacts:
            art = Artifact(id=self.next_artifact_id,x=art_pose[0],y=art_pose[1], cls='mushroom')
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


        # --- Perception 2: YOLO setup ---
        self.declare_parameter('yolo_model_path', rclpy.Parameter.Type.STRING)
        self.declare_parameter('yolo_conf', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('yolo_iou', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('yolo_imgsz', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('yolo_classes', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('use_depth_for_localisation', False)
        self.declare_parameter('yolo_allowed_class_names', rclpy.Parameter.Type.STRING_ARRAY)

        # Read params
        self.yolo_conf  = float(self.get_parameter('yolo_conf').value)
        self.yolo_iou   = float(self.get_parameter('yolo_iou').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.class_names = list(self.get_parameter('yolo_classes').value)
        self.use_depth_for_localisation = bool(self.get_parameter('use_depth_for_localisation').value)

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
      
        # Build a name->id map from the model
        name_to_id = {}
        if hasattr(self.yolo_model, 'names'):
            if isinstance(self.yolo_model.names, dict):       # common in YOLOv8
                # model.names: {id: name}
                name_to_id = {v: k for k, v in self.yolo_model.names.items()}
            else:                                             # list-style
                name_to_id = {n: i for i, n in enumerate(self.yolo_model.names)}
        else:
            self.get_logger().warn("[YOLO] Model has no 'names' attribute; class filtering by name will be disabled.")

        # Read the allow-list from ROS params
        allowed_names = list(self.get_parameter('yolo_allowed_class_names').value or [])

        # Map names -> ids; keep only those that exist
        allowed_ids = []
        missing = []
        for n in allowed_names:
            if n in name_to_id:
                allowed_ids.append(int(name_to_id[n]))
            else:
                missing.append(n)

        self.allowed_class_ids = allowed_ids     # store for use in predict()

        # Helpful diagnostics
        if allowed_names:
            if allowed_ids:
                self.get_logger().warn(f"[YOLO] Restricting to classes: {allowed_names} -> ids {allowed_ids}")
            if missing:
                self.get_logger().warn(f"[YOLO] Ignored unknown class names (not in model): {missing}")
        else:
            self.get_logger().info("[YOLO] No yolo_allowed_class_names provided; detecting all classes.")


        # Use SDF-provided intrinsics instead of /camera_info
        self.declare_parameter('use_sdf_intrinsics', True)       # True since /camera_info is missing
        self.declare_parameter('sdf_hfov_rad', 2.0944)           # ≈120 deg
        self.declare_parameter('sdf_camera_width', 720)          # nominal SDF width
        self.declare_parameter('sdf_camera_height', 480)         # nominal SDF height

        self.use_sdf_intrinsics = bool(self.get_parameter('use_sdf_intrinsics').value)
        self.sdf_hfov_rad = float(self.get_parameter('sdf_hfov_rad').value)
        self.sdf_cam_w = int(self.get_parameter('sdf_camera_width').value)
        self.sdf_cam_h = int(self.get_parameter('sdf_camera_height').value)

        # Intrinsics will be set on first image with the actual resolution,
        # using the SDF hfov and implied vfov from the current aspect ratio.
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None
        self.last_depth_header = None
        self.depth_w = self.depth_h = None
        self.last_image_header = None
        self.camera_frame_id = None  # Set from image header

        # Subscribe to depth image
        self.depth_sub_ = self.create_subscription(
            Image, 'camera/depth/image', self.depth_callback, sensor_qos
        )

        # --- Perception 3: simple artifacts store ---
        self.artifacts: list[Artifact] = []
        self.next_artifact_id = 1
        self.merge_dist_m = 15.0  # meters; detections within this distance are merged

        # Portable path: <this_file_dir>/artifact_detections/detections.json
        self.artifact_json_path = (Path(__file__).resolve().parent
                                / "artifact_detections" / "detections.json")
        self.artifact_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Clear JSON on each run
        with open(self.artifact_json_path, "w") as f:
            json.dump({"artifacts": []}, f, indent=2)
        # --- end Perception 3 ---


        #### ---- Pubs/Subs ---- ####
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)

        # QoS profile for image and camera info subscriptions
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, sensor_qos)
        self.depth_sub_ = self.create_subscription(Image, 'camera/depth/image', self.depth_callback, sensor_qos)

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

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

    ### Perception 3 ###
    def _set_intrinsics_from_sdf(self, img_w: int, img_h: int):
        """
        Compute camera intrinsics from SDF data using the *current* image resolution.
        Uses horizontal FOV from SDF and derives vertical FOV from aspect ratio.
        """
        hfov = float(self.sdf_hfov_rad)
        # Derive vfov from aspect ratio (matches your colleague's SDF approach)
        vfov = 2.0 * math.atan((img_h / img_w) * math.tan(hfov / 2.0))

        fx = img_w / (2.0 * math.tan(hfov / 2.0))
        fy = img_h / (2.0 * math.tan(vfov / 2.0))
        cx = (img_w - 1) * 0.5
        cy = (img_h - 1) * 0.5

        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.get_logger().warn(
            f"[Intrinsics:SDF] w={img_w} h={img_h} hfov={hfov:.4f} rad "
            f"-> fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        )

    def _upsert_artifact(self, x: float, y: float, cls_name: str, conf: float):
        """
        Merge with the nearest EXISTING artifact of the SAME CLASS within self.merge_dist_m,
        using a confidence-weighted running average. Otherwise, create a new artifact.

        Returns the updated/created Artifact.
        """
        # Find nearest artifact of the same class
        best = None
        best_d = 1e9
        for a in self.artifacts:
            if a.cls != cls_name:
                continue
            d = math.hypot(a.x - x, a.y - y)
            if d < best_d:
                best = a
                best_d = d

        if best is not None and best_d <= self.merge_dist_m:
            best.add(x, y, conf)
            return best
        else:
            # Create a new class-locked artifact
            a = Artifact(
                id=self.next_artifact_id,
                cls=cls_name,
                x=float(x),
                y=float(y),
                conf_avg=float(conf),
                weight_sum=max(0.0, float(conf)),
                n=1,
                last_update=time.time(),
            )
            self.next_artifact_id += 1
            self.artifacts.append(a)
            return a
        
    def _persist_artifacts_json(self):
        """Write current artifact set to JSON (portable path, pretty)."""
        data = {"artifacts": [a.as_dict() for a in self.artifacts]}
        try:
            with open(self.artifact_json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"[Artifacts] Failed to persist JSON: {e}")


    def estimate_artifact_direction(self, u: float, v: float, cam_frame: str):
        """
        From a pixel (u,v), compute:
        - ray_base: unit 3D look vector in base_link
        - dir_map_xy: unit 2D ground-plane direction in map frame
        Uses intrinsics + the static TF (base_link <- cam_frame) + robot yaw from get_pose_2d().
        """
        # ---- 1) Pixel -> camera ray (optical frame: +Z forward, +X right, +Y down) ----
        if not all([self.fx, self.fy, self.cx, self.cy]):
            self.get_logger().warn("[Artifacts] Missing intrinsics; cannot estimate direction.")
            return None, None

        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        z_cam = 1.0
        nrm = math.sqrt(x_cam*x_cam + y_cam*y_cam + z_cam*z_cam)
        ray_cam = np.array([x_cam/nrm, y_cam/nrm, z_cam/nrm], dtype=np.float32)

        # ---- 2) Rotate camera ray into base_link using static extrinsics ----
        # Default: identity if TF not available
        R_base_cam = np.eye(3, dtype=np.float32)
        try:
            # latest is fine (static)
            T = self.tf_buffer.lookup_transform('base_link', cam_frame, rclpy.time.Time())
            # Quaternion to rotation matrix
            qw = T.transform.rotation.w
            qx = T.transform.rotation.x
            qy = T.transform.rotation.y
            qz = T.transform.rotation.z
            # 3x3 rotation
            R_base_cam = np.array([
                [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),         2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),         1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),         1 - 2*(qx*qx + qy*qy)]
            ], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"[Artifacts] Using identity R_base_cam; TF lookup failed: {e}")

        ray_base = R_base_cam @ ray_cam  # 3D unit vector in base_link

        # ---- 3) Ground-plane projection in base_link, then rotate by robot yaw into map ----
        # Project to XY (base_link frame has Z up)
        vx, vy, vz = float(ray_base[0]), float(ray_base[1]), float(ray_base[2])
        horiz_norm = math.hypot(vx, vy)
        if horiz_norm < 1e-6:
            self.get_logger().warn("[Artifacts] Ray nearly vertical; cannot form ground-plane direction.")
            return None, None
        dir_base_xy = (vx / horiz_norm, vy / horiz_norm)  # unit on ground

        # Rotate into map by robot yaw
        pose = self.get_pose_2d()
        if pose is None:
            return None, None
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        dir_map_xy = (c*dir_base_xy[0] - s*dir_base_xy[1],
                    s*dir_base_xy[0] + c*dir_base_xy[1])

        return (vx, vy, vz), dir_map_xy
    
    def estimate_artifact_depth(self, depth_img: np.ndarray, box_xyxy) -> float | None:
        """
        Estimate range (meters) by taking the MEAN of all valid depth pixels inside the YOLO box.
        Returns None if no valid pixels.
        """
        if depth_img is None:
            return None
        h, w = depth_img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        patch = depth_img[y1:y2+1, x1:x2+1].reshape(-1)
        vals = patch[np.isfinite(patch)]
        vals = vals[vals > 0.0]
        if vals.size == 0:
            return None
        return float(np.mean(vals))


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
        New localisation pipeline:
        - Direction = f(pixel, intrinsics, base<-cam extrinsics, robot yaw)
        - Distance  = mean depth inside full box
        - Position  = (robot_xy + Rz(theta)*t_base_cam_xy) + d_horiz * dir_map_xy
        Notes:
        - d_horiz projects the 3D range onto the ground plane using the ray's Z in base_link.
        - Requires: intrinsics, camera_frame_id, latest_depth (if depth localisation is enabled).
        """
        if not self.latest_detections:
            return
        if not all([self.fx, self.fy, self.cx, self.cy]):
            self.get_logger().warn("[Artifacts] Missing intrinsics.")
            return
        if self.use_depth_for_localisation and self.latest_depth is None:
            self.get_logger().warn("[Artifacts] No depth image yet.")
            return

        # Determine the camera frame to use (must match your URDF/tf)
        cam_frame = (self.camera_frame_id or
                    (self.last_depth_header.frame_id if self.last_depth_header else None) or
                    (self.last_image_header.frame_id if self.last_image_header else None) or
                    'camera_link')

        # Robot pose (in map)
        pose = self.get_pose_2d()
        if pose is None:
            return
        c, s = math.cos(pose.theta), math.sin(pose.theta)

        # Camera translation wrt base_link (use TF; static)
        t_base_cam = np.zeros(3, dtype=np.float32)
        try:
            T = self.tf_buffer.lookup_transform('base_link', cam_frame, rclpy.time.Time())
            t_base_cam[:] = np.array([
                T.transform.translation.x,
                T.transform.translation.y,
                T.transform.translation.z
            ], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"[Artifacts] No base_link->camera static TF; assuming camera at base_link. {e}")

        # Camera position in map (XY only)
        cam_map_x = pose.x + (c * float(t_base_cam[0]) - s * float(t_base_cam[1]))
        cam_map_y = pose.y + (s * float(t_base_cam[0]) + c * float(t_base_cam[1]))

        updates = 0

        for det in self.latest_detections:
            x1, y1, x2, y2 = det["xyxy"]
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)

            # 1) Direction
            ray_base, dir_map_xy = self.estimate_artifact_direction(u, v, cam_frame)
            if ray_base is None or dir_map_xy is None:
                continue
            vz = ray_base[2]  # base_link Z component

            # 2) Distance (mean depth over the full box)
            if self.use_depth_for_localisation:
                d = self.estimate_artifact_depth(self.latest_depth, [x1, y1, x2, y2])
            else:
                d = 2.0  # fallback heuristic if no depth by design
            if d is None or not np.isfinite(d) or d <= 0.0:
                continue

            # Convert 3D range along ray to horizontal ground distance
            # If ray is perfectly horizontal (vz≈0), this is ~d
            horiz_scale = math.sqrt(max(1e-12, 1.0 - float(vz)*float(vz)))  # clamp for safety
            d_horiz = d * horiz_scale

            # 3) Position in map
            px_map = cam_map_x + d_horiz * dir_map_xy[0]
            py_map = cam_map_y + d_horiz * dir_map_xy[1]

            # Class name
            cls_id = det.get("cls", -1)
            cls_name = (self.class_names[cls_id]
                        if (isinstance(cls_id, int) and 0 <= cls_id < len(self.class_names))
                        else f"class_{cls_id}")
            conf = float(det.get("conf", 0.5))  # default if model didn't supply
            self._upsert_artifact(float(px_map), float(py_map), cls_name, conf)
            
            updates += 1

        if updates > 0:
            self._persist_artifacts_json()
            self.publish_artifact_markers()

    def publish_artifact_markers(self):
        """
        Publish all saved artifact estimates as:
        - a SPHERE_LIST in 'map' frame
        - one TEXT_VIEW_FACING label per artifact showing majority class and count
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

        marr = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marr.markers.append(delete_all)

        self.marker_artifacts_.header.stamp = self.get_clock().now().to_msg()
        self.marker_artifacts_.points = points
        self.marker_artifacts_.colors = colors if len(colors) == len(points) else []
        marr.markers.append(self.marker_artifacts_)

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
        if self.use_depth_for_localisation:
            self.camera_frame_id = (msg.header.frame_id or self.camera_frame_id)

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
        self.path_planner.latest_map_ = map_msg  #forward map to PathPlanner

    def image_callback(self, image_msg):
        # Count frames
        self.image_msgs_seen += 1
        if self.image_msgs_seen % 10 == 0:
            self.get_logger().info(f"[image] frames seen: {self.image_msgs_seen}")

        # Decode with correct encoding
        enc = (image_msg.encoding or '').lower()
        try:
            if enc in ('rgb8', 'rgba8'):
                rgb = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                bgr = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion error: {e}")
            return
        
        # Ensure intrinsics are set from SDF as soon as we know the stream size
        if self.use_sdf_intrinsics and (self.fx is None or self.fy is None):
            h, w = rgb.shape[:2]
            self._set_intrinsics_from_sdf(w, h)
            self.camera_frame_id = image_msg.header.frame_id or "camera_link"

        # Depth/RGB size check when using per-pixel depth
        if self.use_depth_for_localisation and self.latest_depth is not None:
            h_rgb, w_rgb = rgb.shape[:2]
            if (self.depth_h, self.depth_w) != (h_rgb, w_rgb):
                if self.image_msgs_seen % 15 == 0:
                    self.get_logger().warn(
                        f"[Depth] Size mismatch: rgb=({w_rgb}x{h_rgb}) depth=({self.depth_w}x{self.depth_h}); skipping localisation this frame."
                    )
                self.artifact_found_ = False
                return

        annotated = bgr.copy()
        detections, num_boxes = [], 0

        # --- YOLO inference ---
        hud_text = "YOLO: off"
        try:
            if self.yolo_model is not None:
                results = self.yolo_model.predict(
                    source=rgb,
                    conf=self.yolo_conf,
                    iou=self.yolo_iou,
                    imgsz=self.yolo_imgsz,
                    verbose=False,
                    device='cpu',   # switch to 'cpu' if required or keep 'cuda:0'
                    classes=self.allowed_class_ids if self.allowed_class_ids else None
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

        # HUD
        cv2.putText(annotated, hud_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"conf={self.yolo_conf:.2f}, img={self.yolo_imgsz}", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Publish overlay
        out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        msg_out = self.cv_bridge_.cv2_to_imgmsg(out_rgb, encoding="rgb8")
        msg_out.header = image_msg.header
        self.image_detections_pub_.publish(msg_out)

        self.last_image_header = image_msg.header

        ready_to_localise = (self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None) and \
                    (self.latest_depth is not None or not self.use_depth_for_localisation)

        self.artifact_found_ = (len(detections) > 0) and ready_to_localise

        # Uncomment to run localisation on every frame once ready:
        if ready_to_localise:
            self.localise_artifact()
        else:
            if self.image_msgs_seen % 5 == 0:
                if self.fx is None:
                    self.get_logger().warn("Waiting for SDF intrinsics.")
                if self.use_depth_for_localisation and self.latest_depth is None:
                    self.get_logger().warn("Waiting for depth frames…")

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
            ########### toggle prm #######
            if artfiact_result is not None and artfiact_result.time_examined < self.artifact_timeout:
                self.planner_type_ = PlannerType.ARTIFACT_EXPLORATION
            elif hasattr(self.path_planner, 'latest_map_') and self.path_planner.latest_map_ is not None:
                self.planner_type_ = PlannerType.PRM_EXPLORATION
            else:
                self.planner_type_ = PlannerType.FRONTIER_EXPLORATION
            ############################## to replace below

            # if artfiact_result != None:
            #     self.get_logger().info("Found artiact")
            #     if artfiact_result.time_examined < self.artifact_timeout:
            #         self.planner_type_ = PlannerType.ARTIFACT_EXPLORATION
            #     else:
            #         self.planner_type_ = PlannerType.FRONTIER_EXPLORATION 
            # else:
            #     self.planner_type_ = PlannerType.FRONTIER_EXPLORATION

            # self.get_logger().info(f'Calling planner: {self.planner_type_.name}')

            #######################################################
            #Execute Planner
            

            if self.planner_type_ == PlannerType.FRONTIER_EXPLORATION:
                if hasattr(self.path_planner, 'latest_map_') and self.path_planner.latest_map_ is not None:
                    self.path_planner.frontier_exploration_step()
                else:
                    self.get_logger().warn('No map received yet. Cannot perform frontier exploration.')
            
            ##########toggle prm ###############
            elif self.planner_type_ == PlannerType.PRM_EXPLORATION:
                if hasattr(self.path_planner, 'latest_map_') and self.path_planner.latest_map_ is not None:
                    self.path_planner.prm_exploration_step()
                else:
                    self.get_logger().warn("[PRM] No map available — skipping PRM step.")

            ############################# to replace below
            # elif self.planner_type_ == PlannerType.ARTIFACT_EXPLORATION:
            #         self.get_logger().info(f"Mvoing to artfact, it has id: {artfiact_result.id}, it has pose x: {artfiact_result.x}, y: {artfiact_result.y}")
            #         self.path_planner.artifact_exploration_step(artfiact_result)
            # else:
            #     self.get_logger().error('No valid planner selected')

    
def main():
    rclpy.init()
    cave_explorer = CaveExplorer()
    while rclpy.ok():
        rclpy.spin(cave_explorer)

    ###### toggle prm ##############
    try:
        while rclpy.ok():
            rclpy.spin_once(cave_explorer)
            time.sleep(0.05)  # 20 Hz loop
    finally:
        cave_explorer.destroy_node()
        rclpy.shutdown()
    ################
