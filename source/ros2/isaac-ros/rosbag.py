from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import rclpy.serialization as rz
import csv, pathlib

bag = '/workspace/isaac-ros/g1_data/lidar_data/lidar_data_0.db3'
storage_opt = StorageOptions(uri=bag, storage_id='sqlite3')   # or 'mcap'
converter_opt = ConverterOptions('', '')  # 자동 타입 변환
reader = SequentialReader()
reader.open(storage_opt, converter_opt)

with open('imu.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == '/livox/imu':
            msg = rz.deserialize_message(data, 'sensor_msgs/msg/Imu')
            writer.writerow([
                t, 
                msg.header.frame_id,
            msg.orientation.x, 
            msg.orientation.y, 
            msg.orientation.z, 
            msg.orientation.w, 
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            ])
        # if topic == '/livox/lidar':
        #     msg = rz.deserialize_message(data, 'sensor_msgs/msg/PointCloud2')
        #     writer.writerow([t, msg.orientation.x, msg.linear_acceleration.x])
