#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from builtin_interfaces.msg import Time
import numpy as np
from tf2_ros import TransformBroadcaster, TransformStamped
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.parameter import Parameter

class TimePublisher(Node):
    def __init__(self):
        super().__init__('time_publisher')
        self.simtime_pub = self.create_publisher(TimeMsg, '/clock', 10)
        self.tf_broadcaster = TransformBroadcaster(self)


    def publish_simtime(self, sim_time):
        # advance sim_time
        # split into sec / nanosec
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        msg = TimeMsg(sec=sec, nanosec=nanosec)
        self.simtime_pub.publish(msg)

def main():
    rclpy.init()
    node = TimePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()