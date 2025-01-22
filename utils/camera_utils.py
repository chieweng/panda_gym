from typing import List, Tuple, Optional
import os
import numpy as np
import open3d as o3d
from env.franka_env import FrankaPandaCam

def render_from_robot_cam(
    robot: FrankaPandaCam,
    img_width: int = 512,
    img_height: int = 512,
    fov: float = 58, # FOV in degrees
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render images from D405 Camera fixed to the panda robot arm
    """
    import pybullet
    cam_pos = robot.sim.get_link_position("panda_cam", robot.cam_render_link) # Eye position
    cam_rot = robot.sim.get_link_orientation("panda_cam", robot.cam_render_link) # In quaternion
    
    # Compute rotation matrix to transform in world coordinates
    rot_matrix = np.array(robot.sim.physics_client.getMatrixFromQuaternion(cam_rot)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    forward_vec = rot_matrix.dot(np.array((0, 0, -1))) # (0, 0, -1) is the default forward direction in camera frame, look-at direction
    up_vec = rot_matrix.dot(np.array((0, 1, 0))) # (0, 1, 0) is the default up direction in camera frame, up direction
    
    target_position = cam_pos + 0.1 * forward_vec # Target 0.1 units in front of camera
    
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
    
    # Compute view and projection matrices
    view_matrix = robot.sim.physics_client.computeViewMatrix(cam_pos, target_position, up_vec) # Transform the world coord into camera coord space
    aspect_ratio = img_width / img_height
    nearVal, farVal = 0.01, 1.5 # Near, far clipping plane
    proj_matrix = robot.sim.physics_client.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal) # Transforms 3D points from camera coord space into 2D screen space
    
    # Get RBG and depth images from camera
    images = robot.sim.physics_client.getCameraImage(
        img_width, 
        img_height, 
        view_matrix, 
        proj_matrix, 
        renderer = pybullet.ER_BULLET_HARDWARE_OPENGL
        )
    
    rgb_img = np.array(images[2]).reshape((img_height, img_width, 4))[:, :, :3]
    
    depth_img = np.array(images[3]).reshape((img_height, img_width))        

    # Normalize deapth values
    """
    The depth data from the simulation is captured in a non-linear way (as a depth buffer). 
    This needs to be converted into actual depth values using the camera's near and far clipping planes.
    """
    depth_img = (-farVal * nearVal / (farVal - (farVal - nearVal) * depth_img)) # Convert raw depth buffer values (non-linear, scaled between 0-1) into linear depth values
    
    intrinsics = compute_intrinsics_matrix(img_width, img_height, fov)

    return rgb_img, depth_img, intrinsics

def compute_intrinsics_matrix(
    image_width: int,
    image_height: int, 
    fov: float
    ) -> np.ndarray:
    """
    Computes the intrinsic matrix for a pinhole camera model.

    Args:
        image_width (int): Width of the image (in pixels).
        image_height (int): Height of the image (in pixels).
        fov (float): Field of view (in degrees) of the camera.

    Returns:
        np.ndarray: 3x3 intrinsic matrix.
    """
    fx = image_width / (2 * np.tan(fov * np.pi / 360))  # Focal lengths
    fy = image_height / (2 * np.tan(fov * np.pi / 360))
    cx = image_width / 2  # Principal point x-coordinate
    cy = image_height / 2  # Principal point y-coordinate

    # Intrinsic matrix
    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])

    return intrinsics

def compute_extrinsics_matrix(
    R: np.ndarray,
    t: np.ndarray
    ) -> np.ndarray:
    """
    Computes the extrinsic matrix, which maps points from the camera frame to the world frame.

    Args:
        R (np.ndarray): Camera orientation in world coordinates (3x3).
        t (np.ndarray): Camera position in world coordinates (3x1).

    Returns:
        np.ndarray: 4x4 extrinsic matrix.
    """
    extrinsics = np.eye(4)  # Create a 4x4 identity matrix
    
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t
    
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
    x, y = np.indices((height, width))

    # Compute 3D coordinates in the camera frame
    x_cam = (y - cx) * depth_image / fx
    y_cam = (x - cy) * depth_image / fy
    z_cam = depth_image  # z_cam is the depth value

    # Stack into 3D point vectors
    points_camera_frame = np.stack((x_cam, y_cam, z_cam), axis=-1)

    return points_camera_frame

def camera_to_world(
    points_camera: np.ndarray, 
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
        [points_camera, np.ones((*points_camera.shape[:2], 1))], axis=-1
    )

    # Reshape to (N, 4) for matrix multiplication
    points_homogeneous = points_homogeneous.reshape(-1, 4)

    # Apply the extrinsic matrix to each point
    points_world = (extrinsics @ points_homogeneous.T).T
    
    # Remove homogeneous coordinate
    points_world = points_world[:, :3]
    
    return points_world.reshape(*points_camera.shape)


def depth_image_to_point_cloud(
    depth_image: np.ndarray, 
    intrinsics: np.ndarray, 
    extrinsics: np.ndarray,
    object_center,
    object_rotation
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
    points_world_reshaped = points_world_frame.reshape(-1, 3) # (N, 3)
    
    # translated_points = points_world_reshaped - object_center
    # rotated_points = translated_points @ object_rotation
    # pcd_array = rotated_points + object_center

    return points_world_reshaped


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
    


