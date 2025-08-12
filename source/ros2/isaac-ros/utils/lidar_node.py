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
FIELDS_XYZI_TRRAR = [
    PointField(name='x',            offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',            offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',            offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity',    offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name='t',            offset=16, datatype=PointField.UINT32,  count=1),
    PointField(name='reflectivity', offset=20, datatype=PointField.UINT16,  count=1),
    PointField(name='ring',         offset=22, datatype=PointField.UINT8,   count=1),
    # 23: 1-byte pad (no field)
    PointField(name='ambient',      offset=24, datatype=PointField.UINT16,  count=1),
    # 26: 2-byte pad (no field)
    PointField(name='range',        offset=28, datatype=PointField.UINT32,  count=1),
]

POINTCLOUD_DTYPE = np.dtype([
    ('x','<f4'),('y','<f4'),('z','<f4'),
    ('intensity','<f4'),
    ('t','<u4'),
    ('reflectivity','<u2'),
    ('ring','u1'),
    ('pad1','V1'),          # offset 23
    ('ambient','<u2'),
    ('pad2','V2'),          # offset 26–27
    ('range','<u4'),
])
class PointCloudPublisher(Node):
    def __init__(self, topic='points'):
        super().__init__('pcl_fast_pub')
        self.pub = self.create_publisher(PointCloud2, topic, 5)

    def publish_from_dict(self, data, frame_id='lidar', scan_rate=10.0, simtime=0.0):
        msg = PointCloud2()
        msg.fields = FIELDS_XYZI_TRRAR

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

        arr['x'] = xyz[:, 0].astype(np.float32, copy=False)
        arr['y'] = xyz[:, 1].astype(np.float32, copy=False)
        arr['z'] = xyz[:, 2].astype(np.float32, copy=False)
        arr['intensity'] = intensity.astype(np.float32, copy=False)
        arr['ring'] = channel.astype(np.uint8, copy=False)
        arr['t'] = ((tick * delta_t_tick + channel * delta_t_channel) * 1e9).astype(np.uint32, copy=False)

        msg.header.stamp = to_ros_time_from_seconds(simtime)
        msg.header.frame_id = frame_id
        msg.is_bigendian = False
        msg.height = 1
        msg.width = N
        msg.point_step = POINTCLOUD_DTYPE.itemsize  # 32
        msg.row_step = msg.point_step * N
        msg.is_dense = True
        msg.data = memoryview(arr).cast("B")

        self.pub.publish(msg)