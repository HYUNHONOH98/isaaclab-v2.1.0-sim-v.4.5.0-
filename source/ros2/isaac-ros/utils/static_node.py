#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import math
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from rclpy.parameter import Parameter

class StaticTFPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_publisher',
        parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ]
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster_1 = StaticTransformBroadcaster(self)
        self.static_tf_broadcaster_2 = StaticTransformBroadcaster(self)
        
        self.static_tf_timer_1 = self.create_timer(1.0, self.publish_static_tf_link_lidar)
        self.static_tf_timer_2 = self.create_timer(1.0, self.publish_static_tf_lidar_basescan)
    
    def publish_static_tf_link_lidar(self):
        # Cancel the timer so this callback runs only one time.
        

        try:
            # Try to look up an existing transform from "world" to "camera_init".
            # Here, rclpy.time.Time() (i.e. time=0) means "the latest available".
            self.tf_buffer.lookup_transform(
                "mid360_link_frame", "lidar_sensor", rclpy.time.Time()
            )
            self.get_logger().info(
                "Static transform from 'mid360_link_frame' to 'lidar_sensor' already exists. Not publishing a new one."
            )
            self.static_tf_timer_1.cancel()
        except:
            # Header
            static_transform = TransformStamped()
            
            static_transform.header.stamp = self.get_clock().now().to_msg()
            static_transform.header.frame_id = 'mid360_link_frame'
            static_transform.child_frame_id = 'lidar_sensor'

            # Translation (meters)
            static_transform.transform.translation.x = 0.0
            static_transform.transform.translation.y = 0.0
            static_transform.transform.translation.z = 0.0

            # Rotation (quaternion) — identity here
            static_transform.transform.rotation.x = 0.0
            static_transform.transform.rotation.y = 0.0
            static_transform.transform.rotation.z = 0.0
            static_transform.transform.rotation.w = 1.0

            # Broadcast once
            self.static_tf_broadcaster_1.sendTransform(static_transform)
            self.get_logger().info('Published static transform from mid360_link_frame → lidar_sensor')

    def publish_static_tf_lidar_basescan(self):
        # Cancel the timer so this callback runs only one time.
        

        try:
            # Try to look up an existing transform from "world" to "camera_init".
            # Here, rclpy.time.Time() (i.e. time=0) means "the latest available".
            self.tf_buffer.lookup_transform(
                "odom", "camera_init", rclpy.time.Time()
            )
            self.get_logger().info(
                "Static transform from 'odom' to 'camera_init' already exists. Not publishing a new one."
            )
            self.static_tf_timer_2.cancel()
        except:
            # Header
            static_transform = TransformStamped()

            try:
                tf = self.tf_buffer.lookup_transform(
                    "odom",
                    'mid360_link',
                    rclpy.time.Time(),  # latest available
                )
            except:
                return
            
            static_transform.header.stamp = self.get_clock().now().to_msg()
            static_transform.header.frame_id = 'odom'
            static_transform.child_frame_id = 'camera_init'

            # Translation (meters)
            static_transform.transform.translation = tf.transform.translation

            # Rotation (quaternion) — identity here
            static_transform.transform.rotation = tf.transform.rotation

            # Broadcast once
            self.static_tf_broadcaster_2.sendTransform(static_transform)
            self.get_logger().info('Published static transform from lidar_sensor → base_scan')
    

def main():
    rclpy.init()
    node = StaticTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
