import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math
from geometry_msgs.msg import Quaternion
import random
from builtin_interfaces.msg import Time
from rclpy.parameter import Parameter
import numpy as np
from scipy.spatial.transform import Rotation as R
from icecream import ic


from .functions import to_ros_time_from_seconds

class ImuPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher',
        # parameter_overrides=[
        #     Parameter('use_sim_time', Parameter.Type.BOOL, True)
        # ]
        )
        self.publisher_ = self.create_publisher(Imu, 'imu/data', 10)

    def publish_imu(self, data, simtime):
        msg = Imu()
        quat, ang_vel, lin_acc  = data["orientation"], data["ang_vel"], data["lin_acc"]

        # Header
        # breakpoint()
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.stamp = to_ros_time_from_seconds(simtime)
        msg.header.frame_id = 'imu_sensor'

        # Orientation (simulate rotation)
        msg.orientation.w = 1.# float(quat[0])
        msg.orientation.x = 0.#float(quat[1])
        msg.orientation.y = 0.#float(quat[2])
        msg.orientation.z = 0.#float(quat[3])


        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])

        # Linear acceleration
        msg.linear_acceleration.x = float(lin_acc[0])
        msg.linear_acceleration.y = float(lin_acc[1])
        msg.linear_acceleration.z = float(lin_acc[2])

        self.publisher_.publish(msg)
