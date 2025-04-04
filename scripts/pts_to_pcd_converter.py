#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pcl
import numpy as np
import struct


class PTS2PCDConverter(Node):
    def __init__(self, pts_file, pcd_save_path):
        super().__init__('pts_to_pcd_converter')
        self.pts_file = pts_file
        self.pcd_save_path = pcd_save_path
        self.convert_pts_to_pcd()

    def load_pts_file(self):
        points = []
        field_indices = None  # To store the indices of the fields dynamically
        
        with open(self.pts_file, 'r') as file:
            for line_num, line in enumerate(file):
                data = line.strip().split()

                # Skip header (first line if it's a single integer)
                if line_num == 0 and len(data) == 1 and data[0].isdigit():
                    continue

                # Dynamically determine the field indices based on the data length
                if field_indices is None:
                    field_count = len(data)
                    field_indices = {
                        "x": 0, "y": 1, "z": 2,
                        "r": 3 if field_count > 3 else None,
                        "g": 4 if field_count > 4 else None,
                        "b": 5 if field_count > 5 else None,
                    }

                # Parse the data fields dynamically based on available fields
                x = float(data[field_indices["x"]]) if field_indices["x"] is not None else 0.0
                y = float(data[field_indices["y"]]) if field_indices["y"] is not None else 0.0
                z = float(data[field_indices["z"]]) if field_indices["z"] is not None else 0.0
                
                r = g = b = 0.0
                if field_indices["r"] is not None and field_indices["g"] is not None and field_indices["b"] is not None:
                    r = float(data[field_indices["r"]]) / 255.0
                    g = float(data[field_indices["g"]]) / 255.0
                    b = float(data[field_indices["b"]]) / 255.0

                # Pack RGB as an unsigned integer
                rgb = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)

                # Add the point to the list
                points.append([x, y, z, rgb])

        return np.array(points)

    def convert_pts_to_pcd(self):
        self.get_logger().info('Loading .pts file...')
        points = self.load_pts_file()

        # Convert numpy array to pcl PointCloud with RGB
        pcl_cloud = pcl.PointCloud_PointXYZRGB()  # Using PointCloud_PointXYZRGB to store XYZ and RGB
        pcl_cloud.from_array(points.astype(np.float32))

        # Save to PCD file
        pcl.save(pcl_cloud, self.pcd_save_path)

        self.get_logger().info(f"Saved PointCloud to {self.pcd_save_path}")


def main(args=None):
    rclpy.init(args=args)
    pts_file = '/home/simson/simson_ws/CMU_Capstone_Project/faro_mapping/data/V3/FARO_data_3.pts'  # Update this
    pcd_save_path = '/home/simson/simson_ws/CMU_Capstone_Project/faro_mapping/data/V3/FARO_data_3.pcd'  # Update this
    node = PTS2PCDConverter(pts_file, pcd_save_path)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
