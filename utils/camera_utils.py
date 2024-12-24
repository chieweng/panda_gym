from typing import List, Tuple, Optional
import os
import numpy as np
import open3d as o3d
import copy

def compute_intrinsics_matrix(
    image_width: int, 
    fov: float
    ) -> np.ndarray:
    """
    Computes the intrinsic matrix for a pinhole camera model.

    Args:
        image_width (int): Width of the image (in pixels).
        fov (float): Field of view (in degrees) of the camera.

    Returns:
        np.ndarray: 3x3 intrinsic matrix.
    """
    fx = fy = image_width / (2 * np.tan(fov * np.pi / 180 / 2))  # Focal lengths
    cx = cy = image_width / 2  # Principal point (center of the image)

    # Intrinsic matrix
    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])

    return intrinsics

def compute_extrinsics_matrix(
    Rx: np.ndarray, 
    Ry: np.ndarray, 
    Rz: np.ndarray, 
    camera_position: np.ndarray
) -> np.ndarray:
    """
    Computes the extrinsic matrix, which maps points from the world frame to the camera frame.

    Args:
        Rx (np.ndarray): Rotation vector along the X-axis (3x1).
        Ry (np.ndarray): Rotation vector along the Y-axis (3x1).
        Rz (np.ndarray): Rotation vector along the Z-axis (3x1).
        camera_position (np.ndarray): Camera position in world coordinates (3x1).

    Returns:
        np.ndarray: 4x4 extrinsic matrix.
    """
    extrinsics = np.eye(4)  # Create a 4x4 identity matrix

    # Assign rotation vectors
    extrinsics[:3, 0] = Rx
    extrinsics[:3, 1] = Ry
    extrinsics[:3, 2] = Rz

    # Assign translation (camera center in world coordinates)
    extrinsics[:3, 3] = camera_position

    return extrinsics

def depth_to_camera_frame(
    depth_image: np.ndarray, 
    intrinsics: np.ndarray
    ) -> np.ndarray:
    """
    Converts a depth image to 3D points in the camera frame.

    Args:
        depth_image (np.ndarray): 2D array representing the depth image.
        intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.

    Returns:
        np.ndarray: 3D point cloud in the camera frame (shape: HxWx3).
    """
    height, width = depth_image.shape[:2]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    # Create arrays of pixel coordinates
    i, j = np.indices((height, width))

    # Compute 3D coordinates in the camera frame
    x_cam = (j - cx) * depth_image / fx
    y_cam = (i - cy) * depth_image / fy
    z_cam = depth_image  # z_cam is the depth value

    # Stack into 3D point vectors
    points_camera_frame = np.stack((x_cam, y_cam, z_cam), axis=-1)

    return points_camera_frame

def camera_to_world(
    points_camera_frame: np.ndarray, 
    extrinsics: np.ndarray
    ) -> np.ndarray:
    """
    Transforms 3D points from the camera frame to the world frame using the extrinsic matrix.

    Args:
        points_camera_frame (np.ndarray): 3D points in the camera frame (HxWx3).
        extrinsics (np.ndarray): 4x4 extrinsic matrix of the camera.

    Returns:
        np.ndarray: 3D points in the world frame (HxWx3).
    """
    # Add homogeneous coordinate (1) to each point for matrix multiplication
    points_homogeneous = np.concatenate(
        [points_camera_frame, np.ones((*points_camera_frame.shape[:2], 1))], axis=-1
    )

    # Reshape to (N, 4) for matrix multiplication
    points_homogeneous = points_homogeneous.reshape(-1, 4)

    # Apply the extrinsic matrix to each point
    points_world_frame = (extrinsics @ points_homogeneous.T).T

    # Remove homogeneous coordinate
    points_world_frame = points_world_frame[:, :3]

    return points_world_frame.reshape(*points_camera_frame.shape)


def depth_image_to_point_cloud(
    depth_image: np.ndarray, 
    intrinsics: np.ndarray, 
    extrinsics: np.ndarray
    ) -> np.ndarray:
    """
    Converts a depth image into a 3D point cloud in the world frame.

    Args:
        depth_image (np.ndarray): 2D array representing the depth image.
        intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.
        extrinsics (np.ndarray): 4x4 extrinsic matrix of the camera.

    Returns:
        np.ndarray: Point cloud array (2D) in the world frame (N, 3) - np.array([[x1, y1, z1], [x2, y2, z2], ..., [xN, yN, zN]])
    """
    # Convert depth image pixels to camera frame points
    points_camera_frame = depth_to_camera_frame(depth_image, intrinsics)

    # Convert camera frame points to world frame
    points_world_frame = camera_to_world(points_camera_frame, extrinsics) # (H,W,3)

    # Flatten the point cloud to shape (N, 3)
    point_cloud = points_world_frame.reshape(-1, 3) # (N, 3)

    return point_cloud


def plot_point_cloud(file_name: str) -> None:
    """
    Loads and visualizes a point cloud from a .npy file.

    Args:
        file_name (str): Name of the point cloud file (e.g., 'point_cloud_00x.npy').

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the loaded data does not have the correct shape.
    """
    # Construct the full path to the .npy file
    file_path = os.path.join("scans", file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    # Load the point cloud data from the .npy file
    points = np.load(file_path)

    # Validate the shape (should be Nx3 for 3D points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud data must be of shape (N, 3).")

    # Create an Open3D point cloud object and assign points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    


