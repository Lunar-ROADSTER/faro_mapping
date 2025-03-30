#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import rclpy
import cv2
import yaml
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

# Parameters
pcd_file = "/home/simson/simson_ws/CMU_Capstone_Project/Lunar_ROADSTER_ws/src/mapping/data/V2/FARO_data_2.pcd"
resolution = 0.01  # Grid resolution in meters
obstacle_threshold_below = -0.03 #-0.03
obstacle_threshold_above = 0.026 # 0.03
map_size = (750, 700)

# Define offsets
x_offset = 3.3#7
y_offset = 0.1#4

# Define rotation angle in degrees and convert to radians
theta_degrees = -134  # Example: 30-degree rotation
theta = np.radians(theta_degrees)

# Rotation matrix for 2D rotation around Z-axis
R = np.array([[np.cos(theta), -np.sin(theta), 0],
              [np.sin(theta),  np.cos(theta), 0],
              [0,              0,             1]])

# Load point cloud
def load_pcd(file):
    pcd = o3d.io.read_point_cloud(file)
    points = np.asarray(pcd.points)
    points = points @ R.T 
    points[:, 0] -= x_offset  # Adjust X
    points[:, 1] -= y_offset  # Adjust Y
    return points

# Fit a plane using RANSAC
def get_ground_plane(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=10,
                                             num_iterations=1000)
    return plane_model, inliers

# Create costmap from point cloud
def generate_costmap(points, plane_model, resolution, map_size, threshold_below, threshold_above):
    a, b, c, d = plane_model  # Plane equation: ax + by + cz + d = 0
    costmap = np.zeros(map_size, dtype=np.int8)
    
    for point in points:
        x, y, z = point
        height = (a * x + b * y + c * z + d) / c  # Distance from plane
        
        # Check for obstacles below or above the plane
        if height < threshold_below or threshold_above < height < 0.1:
            px = int((x / resolution) + map_size[0] / 2)
            py = int((y / resolution) + map_size[1] / 2)
            if 0 <= px < map_size[0] and 0 <= py < map_size[1]:
                costmap[px, py] = 100  # Mark as occupied
    
    return costmap

class CostmapPublisher(Node):
    def __init__(self, costmap, resolution):
        super().__init__('costmap_publisher')
        qos_profile = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL  # Ensure latched behavior
        )

        self.publisher_ = self.create_publisher(OccupancyGrid, '/map', qos_profile)
        self.timer = self.create_timer(1.0, self.publish_costmap)
        self.costmap = costmap
        self.resolution = resolution

    def publish_costmap(self):
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.info.resolution = self.resolution
        grid.info.width = self.costmap.shape[1]
        grid.info.height = self.costmap.shape[0]
        grid.info.origin = Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        grid.data = self.costmap.flatten().tolist()
        self.publisher_.publish(grid)

if __name__ == "__main__":
    rclpy.init()
    points = load_pcd(pcd_file)
    plane_model, _ = get_ground_plane(points)
    costmap = generate_costmap(points, plane_model, resolution, map_size, obstacle_threshold_below, obstacle_threshold_above)
    
    node = CostmapPublisher(costmap, resolution)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
