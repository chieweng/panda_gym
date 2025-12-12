import numpy as np
import cv2
import open3d as o3d

def load_intrinsics(path):
    K = np.load(path)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    return fx, fy, cx, cy

def depth_to_pointcloud(depth, K):
    fx, fy, cx, cy = K
    h, w = depth.shape

    # (u, v) pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Convert depth to meters
    d = depth.astype(np.float32) / 1000.0

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    xyz = np.stack((x, y, z), axis=-1)
    xyz = xyz.reshape(-1, 3)

    # remove invalid points
    xyz = xyz[z.reshape(-1) > 0]

    return xyz

def visualize_pointcloud(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])


depth = cv2.imread("../graspnet/train_2/scene_0056/realsense/depth/0000.png", -1)  # read as uint16
K = load_intrinsics("../graspnet/train_2/scene_0056/realsense/camK.npy")

pc = depth_to_pointcloud(depth, K)
print(pc.shape)
visualize_pointcloud(pc)
