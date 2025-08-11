from math import sin, cos, pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
# from tf2_ros import TransformBroadcaster, TransformStamped
import yaml
from sensor_msgs.msg import JointState
from rclpy.parameter import Parameter

class StatePublisher(Node):

    def __init__(self):
        super().__init__('state_publisher',
        parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ]
        )
        qos_profile = QoSProfile(depth=10)

        with open('/workspace/isaaclab/source/ros2/isaac-ros/g1_data/g1.yaml', 'r') as fp:
            self.joint_names = yaml.safe_load(fp)['raw_joint_order']

        self.low_state = JointState()
        self.low_state_subscriber = self.create_subscription(JointState,
                    '/g1/joint_states', self.on_low_state, 10)
        self.joint_pub = self.create_publisher(JointState,
                                               'joint_states', qos_profile)
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))
        self.joint_state = JointState()

    def on_low_state(self, msg: JointState):
        self.low_state = msg
        joint_state = self.joint_state
        now = self.get_clock().now()
        joint_state.header.stamp = now.to_msg()
        joint_state.name = self.joint_names
        joint_state.position = [0.0 for _ in self.joint_names]
        joint_state.velocity = [0.0 for _ in self.joint_names]
        
        n:int = min(len(self.joint_names), len(self.low_state.position))
        for i in range(n):
            joint_state.position[i] = self.low_state.position[i]
            joint_state.velocity[i] = self.low_state.velocity[i]
        self.joint_pub.publish(joint_state)
    


def main():
    rclpy.init()
    node = StatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()