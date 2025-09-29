#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, os, time


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.counter = 0

        # Folder to save
        self.save_dir = os.path.join(
            os.path.dirname(__file__),
            'images/train'
        )
        os.makedirs(self.save_dir, exist_ok=True)

        # Subscriber
        self.sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        # Timer every 2 sec
        self.timer = self.create_timer(2.0, self.save_image)
        self.latest_image = None
        self.get_logger().info("ImageSaver ready. Will save every 2s.")

    def image_callback(self, msg):
        self.get_logger().info("📸 Received image")
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
        self.get_logger().info(f"✅ Saved {filename}")
        self.counter += 1


def main():
    rclpy.init()
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
