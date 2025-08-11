import numpy as np

def index_map(k_to, k_from):
    """
    Returns an index mapping from k_from to k_to.

    Given k_to=a, k_from=b,
    returns an index map "a_from_b" such that
    array_a[a_from_b] = array_b

    Missing values are set to -1.
    """
    index_dict = {k: i for i, k in enumerate(k_to)}  # O(len(k_from))
    return [index_dict.get(k, -1) for k in k_from]  # O(len(k_to))

def get_gravity_orientation(quaternion):
    qx = quaternion[0]
    qy = quaternion[1]
    qz = quaternion[2]
    qw = quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation



def create_nodes():
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("PrimService", "isaacsim.ros2.bridge.ROS2ServicePrim"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),  # ★ 추가

    ]

    def robot_nodes(name, type_str):
        return [(f"{name}_g1", type_str)]

    nodes += robot_nodes("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState")
    # nodes += robot_nodes("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState")
    # nodes += robot_nodes("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController")
    # nodes += robot_nodes("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry")
    # nodes += robot_nodes("PublishRawTransformTree", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")
    # nodes += robot_nodes("PublishRawTransformTree_Odom", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")
    # nodes += robot_nodes("PublishTransformTree", "isaacsim.ros2.bridge.ROS2PublishTransformTree")

    return nodes

def connect_nodes():
    def connect_tick(target, no_robot=False):
        if no_robot:
            return [("OnPlaybackTick.outputs:tick", f"{target}.inputs:execIn")]
        return [("OnPlaybackTick.outputs:tick", f"{target}_g1.inputs:execIn") ]
    
    def connect_time(target):
        return [("ReadSimTime.outputs:simulationTime", f"{target}_g1.inputs:timeStamp")]

    def connect(source, dest, attr_src, attr_dest):
        return [(f"{source}_g1.outputs:{attr_src}", f"{dest}_g1.inputs:{attr_dest}")]

    connections = []
    # Tick connections
    connections += connect_tick("PublishJointState")
    connections += [("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn")] 
    connections += [("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp")]
    # connections += connect_tick("SubscribeJointState")
    # connections += connect_tick("ArticulationController")
    # connections += connect_tick("PublishRawTransformTree")
    # connections += connect_tick("PublishTransformTree")
    # connections += connect_tick("PrimService", no_robot=True)

    # Time connections
    connections += connect_time("PublishJointState")
    # connections += connect_time("PublishTransformTree")
    # connections += connect_time("PublishRawTransformTree")
    # connections += connect_time("PublishRawTransformTree_Odom")
    # connections += connect_time("PublishOdometry")

    # Controller connections
    # connections += connect("SubscribeJointState", "ArticulationController", "jointNames", "jointNames")
    # connections += connect("SubscribeJointState", "ArticulationController", "positionCommand", "positionCommand")
    # connections += connect("SubscribeJointState", "ArticulationController", "velocityCommand", "velocityCommand")
    # connections += connect("SubscribeJointState", "ArticulationController", "effortCommand", "effortCommand")

        
    return connections

def set_values():
    def set_value(name, attr, target):
        return [(f"{name}_g1.inputs:{attr}", target)]

    setvals = []
    setvals += [("PublishClock.inputs:topicName", "/clock")]

    # Controller
    # setvals += set_value("ArticulationController", "robotPath", lambda i: f"/World/G1_{i}/pelvis")

    # Joint state
    setvals += set_value("PublishJointState", "targetPrim", "/World/g1/pelvis")

    # Namespace
    for name in ["PublishJointState", "SubscribeJointState", "PublishOdometry",
                 "PublishRawTransformTree", "PublishRawTransformTree_Odom", "PublishTransformTree"]:
        setvals += set_value(name, "nodeNamespace", "g1")

    # Odometry
    # setvals += set_value("PublishRawTransformTree", "parentFrameId", "odom")
    # setvals += set_value("PublishRawTransformTree", "childFrameId", "pelvis")
    
    return setvals

from builtin_interfaces.msg import Time

def to_ros_time_from_seconds(simtime_sec: float) -> Time:
    sec = int(simtime_sec)
    nanosec = int(round((simtime_sec - sec) * 1e9))
    if nanosec == 1_000_000_000:  # 반올림 넘침 방지
        sec += 1
        nanosec = 0
    return Time(sec=sec, nanosec=nanosec)