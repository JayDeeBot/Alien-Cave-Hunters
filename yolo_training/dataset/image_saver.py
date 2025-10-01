#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, os, time


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()  # OpenCV bridge

        # save local
        self.save_dir = os.path.join(
            os.path.dirname(__file__),
            'images/train'   # change to 'images/val' for validation set
        )
        os.makedirs(self.save_dir, exist_ok=True)

        # avoid overwriting existing images
        self.counter = self.get_next_counter()

        # subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        # every 2 sec
        self.timer = self.create_timer(2.0, self.save_image)  
        self.latest_image = None
        self.get_logger().info("ImageSaver ready, saving every 2s.")

    def get_next_counter(self):
        import re
        pattern = re.compile(r'frame_(\d{5})\.jpg') # match filenames like frame_00001.jpg
        nums = [int(m.group(1)) for f in os.listdir(self.save_dir) # list all files in save_dir
                if (m := pattern.match(f))]
        return max(nums) + 1 if nums else 0

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge failed: {e}")

    def save_image(self):
        if self.latest_image is None:
            self.get_logger().warn("No image yet.")
            return
        filename = os.path.join(self.save_dir, f"frame_{self.counter:05d}.jpg")
        cv2.imwrite(filename, self.latest_image)
        self.get_logger().info(f"✅ saved {filename}")
        self.counter += 1


def main():
    rclpy.init()
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
