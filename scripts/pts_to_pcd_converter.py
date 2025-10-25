#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import open3d as o3d
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
        field_indices = None
        
        with open(self.pts_file, 'r') as file:
            for line_num, line in enumerate(file):
                data = line.strip().split()

                if line_num == 0 and len(data) == 1 and data[0].isdigit():
                    continue

                if field_indices is None:
                    field_count = len(data)
                    field_indices = {
                        "x": 0, "y": 1, "z": 2,
                        "r": 3 if field_count > 3 else None,
                        "g": 4 if field_count > 4 else None,
                        "b": 5 if field_count > 5 else None,
                    }

                x = float(data[field_indices["x"]]) if field_indices["x"] is not None else 0.0
                y = float(data[field_indices["y"]]) if field_indices["y"] is not None else 0.0
                z = float(data[field_indices["z"]]) if field_indices["z"] is not None else 0.0
                
                r = g = b = 0.0
                if field_indices["r"] is not None and field_indices["g"] is not None and field_indices["b"] is not None:
                    r = float(data[field_indices["r"]]) / 255.0
                    g = float(data[field_indices["g"]]) / 255.0
                    b = float(data[field_indices["b"]]) / 255.0

                rgb = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)

                points.append([x, y, z, rgb])

        return np.array(points)

    def convert_pts_to_pcd(self):
        self.get_logger().info('Loading .pts file...')
        points_data = self.load_pts_file()

        xyz_points = points_data[:, 0:3].astype(np.float64)         
        rgb_int = points_data[:, 3].astype(np.uint32)

        r = ((rgb_int >> 16) & 0xFF).astype(np.float64)
        g = ((rgb_int >> 8) & 0xFF).astype(np.float64)
        b = (rgb_int & 0xFF).astype(np.float64)

        rgb_normalized = np.stack([r, g, b], axis=1) / 255.0

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(xyz_points)
        o3d_cloud.colors = o3d.utility.Vector3dVector(rgb_normalized)

        o3d.io.write_point_cloud(self.pcd_save_path, o3d_cloud, write_ascii=False)

        self.get_logger().info(f"Saved PointCloud to {self.pcd_save_path}")


def main(args=None):
    rclpy.init(args=args)
    pts_file = '/home/simson/CMU/MRSD_Capstone_Project/faro_mapping/data/moon_yard_scan.pts'  # Update this
    pcd_save_path = '/home/simson/CMU/MRSD_Capstone_Project/faro_mapping/data/moon_yard_scan.pcd'  # Update this
    node = PTS2PCDConverter(pts_file, pcd_save_path)
    rclpy.shutdown()


if __name__ == '__main__':
    main()