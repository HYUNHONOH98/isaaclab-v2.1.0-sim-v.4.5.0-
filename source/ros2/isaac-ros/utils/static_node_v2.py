#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.parameter import Parameter

class StaticAsDynamicTFPublisher(Node):
    def __init__(self):
        super().__init__('static_as_dynamic_tf',
        parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ])

        # Optional: Use simulation time
        # self.declare_parameter('use_sim_time', True)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publish at 100 Hz
        self.timer = self.create_timer(0.01, self.publish_tf_pair)

    def publish_tf_pair(self):
        now = self.get_clock().now().to_msg()

        tf1 = TransformStamped()
        tf1.header.stamp = now
        tf1.header.frame_id = 'mid360_link_frame'
        tf1.child_frame_id = 'lidar_sensor'
        # tf1.transform.translation.z = 0.1
        tf1.transform.rotation.w = 1.0

        tf2 = TransformStamped()
        tf2.header.stamp = now
        tf2.header.frame_id = 'lidar_sensor'
        tf2.child_frame_id = 'base_scan'
        # tf2.transform.translation.x = 0.2
        tf2.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform([tf1, tf2])  # Can send both at once

def main():
    rclpy.init()
    node = StaticAsDynamicTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
