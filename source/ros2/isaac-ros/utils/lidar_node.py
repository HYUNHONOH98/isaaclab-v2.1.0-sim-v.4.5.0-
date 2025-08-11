#!/usr/bin/env python3
"""
Publish a {x,y,z,intensity} PointCloud2 built from the dict returned by annotator.get_data().
Call   node.publish_from_dict(your_dict)   whenever new data arrive.
"""

from __future__ import annotations
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2   # part of ros-<distro>-sensor-msgs-py
from icecream import ic
from .functions import to_ros_time_from_seconds



# --------------------------------------------------------------------------- #
# Helper: list of PointField describing <float32 x,y,z,intensity>
FIELDS_XYZIRT = [
    PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='ring',      offset=16, datatype=PointField.UINT16,  count=1),
    # 2-byte padding → offset 18–19 (아무 필드도 지정 안 함)
    PointField(name='t',      offset=20, datatype=PointField.FLOAT32, count=1),
]
# --------------------------------------------------------------------------- #


POINTCLOUD_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('intensity', '<f4'), ('ring', '<u2'), ('pad', 'V2'),
    ('t', '<f4')
])

class PointCloudPublisher(Node):
    def __init__(self, topic='points'):
        super().__init__('pcl_fast_pub')
        self.pub = self.create_publisher(PointCloud2, topic, 5)

    def publish_from_dict(self, data, frame_id='lidar', scan_rate=10.0, simtime=0.0):
        msg = PointCloud2()
        msg.fields = FIELDS_XYZIRT

        xyz = data['data']
        intensity = data['intensity']
        idx = data['index']
        N = xyz.shape[0]

        info = data['info']
        numEchos = info['numEchos']
        numChannels = info['numChannels']
        ticksPerScan = info['ticksPerScan']

        div = numChannels * numEchos
        tick = idx // div
        channel = (idx % div) // numEchos  # 중간 곱셈 제거

        delta_t_tick = 1.0 / (scan_rate * ticksPerScan)
        delta_t_channel = delta_t_tick / numChannels

        # 0 초기화 불필요 → np.empty 사용
        arr = np.empty(N, dtype=POINTCLOUD_DTYPE)

        arr['x'] = xyz[:, 0]
        arr['y'] = xyz[:, 1]
        arr['z'] = xyz[:, 2]
        arr['intensity'] = intensity
        arr['ring'] = channel.astype(np.uint16, copy=False)
        arr['t'] = tick * delta_t_tick + channel * delta_t_channel

        msg.header.stamp = to_ros_time_from_seconds(simtime)
        msg.header.frame_id = frame_id
        msg.data = memoryview(arr).cast("B")

        self.pub.publish(msg)