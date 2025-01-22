from typing import List, Tuple, Optional, Union
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from pyb_class import PyBullet as p
from env.franka_env import FrankaPandaCam, Scene
from utils.camera_utils import *
from utils.pcd_utils import *

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class Task:
    """ Base class for robot tasks. """
    def __init__(self, robot):
        self.robot = robot

class MultiScanTask(Task):
    """
    A class to perform multiple scans using the Franka Panda robot with a Realsense D405 camera.

    Args:
        robot (FrankaPandaCam): The robot object.
        scan_positions (List[np.ndarray]): List of positions where the robot's camera will capture scans.
        output_dir (str): Directory where the scans (RGB and depth images) will be saved.
        hold_duration (float, optional): Time to hold the position for each scan. Default is 1.0 second.
    """

    def __init__(
        self, 
        robot: FrankaPandaCam, 
        scan_positions: List[np.ndarray], 
        output_dir: str = "scans", 
        hold_duration: float = 1.0
    ) -> None:
        self.robot = robot
        self.scan_positions = scan_positions
        self.output_dir = output_dir
        self.hold_duration = hold_duration

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def perform_scan(
        self,
        point_to_cog: bool,
        scene: Scene
        ) -> None:
        """
        Perform the multi-view scan by moving the robot to each scan position and capturing images.

        Args:
            point_to_cog (bool): If True, orient the end-effector towards the object's base position.
            scene (Scene): The simulation scene containing the object.
        """
        
        object_base_pos, object_center_height = scene.object_pos, scene.object_size[2]/2 # Get default object base position and object CoG height from scene
        object_center = object_base_pos + [0, 0, object_center_height]

        for i, scan_pos in enumerate(self.scan_positions):
            logging.info("-----------------------------------------------------------------------")
            logging.info(f"Moving to scan position {i+1} / {len(self.scan_positions)}: {scan_pos}")

            # Calculate the quaternion to point towards the CoG if point_to_cog == True
            if point_to_cog:
                cam_ori_quat = self.calculate_orientation(scan_pos, object_center, output_type="quaternion") # Lookat object 
                
            else:
                cam_ori_quat = [0, 0, 0, 1]  # Identity quaternion (no rotation)

            # Move the robot's camera to the target pose
            self.robot.move_robot(
                control_mode = "end_effector",
                target_pose = np.concatenate([scan_pos, cam_ori_quat]),
                mode = "static",
                hold_duration = self.hold_duration
            )
            
            self.robot.visualize_camera_pose()

            # Capture RGB and depth images from the camera
            rgb_img, depth_img, intrinsics = render_from_robot_cam(robot=self.robot)

            self._save_images(i+1, rgb_img)
            
            # Compute intrinsic and extrinsic camera parameters
            cam_pos = scene.robot.get_link_position(scene.robot.cam_render_link)
            cam_ori_R = self.calculate_orientation(cam_pos, object_center, output_type="rotation_matrix")
            extrinsics = compute_extrinsics_matrix(cam_ori_R, cam_pos)
            
            # print("cam_pos", cam_pos)
            # print("cam_ori", cam_ori_R)
            # print("extrinsics", extrinsics)
            
            # Convert depth image to point cloud in world frame
            point_cloud = depth_image_to_point_cloud(depth_img, intrinsics, extrinsics, object_center, cam_ori_R)
            
            np.save(os.path.join(self.output_dir, f"point_cloud_{i+1:03d}.npy"), point_cloud)
            logging.info(f"Point Cloud saved for scan {i+1}.")
            
            # plot_point_cloud(file_name=f"point_cloud_{i:03d}.npy")
            
            scene.robot.set_joint_neutral() 

    def calculate_orientation(
        self,
        position: np.ndarray,
        target: np.ndarray,
        output_type: str = 'quaternion'
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """
        Calculate the quaternion that orients the camera towards the object's CoG, given the camera position.
            
        Args:
            position (np.ndarray): The current position of the camera.
            target (np.ndarray): The target position (object CoG).
            output_type (str): The desired output type ('quaternion' or 'rotation_matrix').
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray]]: Quaternion or rotation matrix representing the orientation.
        """
        # Calculate the direction vector from the current position to the target
        direction_vector = target.astype(np.float32) - position.astype(np.float32)
        direction_vector /= np.linalg.norm(direction_vector)  # Normalize

        world_x = np.array([1, 0, 0], dtype=np.float32)

        # Calculate the x-axis (perpendicular to direction_vector and arbitrary_up)
        x_axis = np.cross(direction_vector, world_x).astype(np.float32)
        x_axis /= np.linalg.norm(x_axis)  # Normalize

        # Recalculate the y-axis to ensure orthogonality
        y_axis = np.cross(x_axis, direction_vector).astype(np.float32)
        y_axis /= np.linalg.norm(y_axis)  # Normalize

        # Build the rotation matrix with new x, y, and direction_vector as the z-axis
        rotation_matrix = np.column_stack((x_axis, y_axis, -direction_vector))

        if output_type == 'quaternion':
            # Convert the rotation matrix to a quaternion
            rotation = R.from_matrix(rotation_matrix)
            return rotation.as_quat()  # Quaternion [x, y, z, w]
        elif output_type == 'rotation_matrix':
            return rotation_matrix
        else:
            raise ValueError("Invalid output type specified. Use 'quaternion' or 'rotation_matrix'.")

    def _save_images(
        self, 
        scan_index: int, 
        rgb_img: np.ndarray, 
        depth_img: Optional[np.ndarray] = None
        ) -> None:
        """
        Save the RGB and depth images to the output directory.

        Args:
            scan_index (int): Index of the current scan position.
            rgb_img (np.ndarray): Captured RGB image.
            depth_img (Optional[np.ndarray]): Captured depth image (default: None).
        """
        # Create file paths with formatted scan index
        rgb_path = os.path.join(self.output_dir, f"rgb_scan_{scan_index:03d}.png")
        depth_path = os.path.join(self.output_dir, f"depth_scan_{scan_index:03d}.npy")

        # Save the RGB image as PNG
        plt.imsave(rgb_path, rgb_img)
        logging.info(f"RGB image saved at: {rgb_path}")

        # Save the depth image as a NumPy array if provided
        if depth_img is not None:
            np.save(depth_path, depth_img)
            logging.info(f"Depth image saved at: {depth_path}")

    def reset_robot(self) -> None:
        """Reset the robot to its neutral position after the scanning process is complete."""
        logging.info("Resetting the robot to its neutral position...")
        self.robot.reset()

    def close(self) -> None:
        """Close the robot simulation after completing the scanning task."""
        logging.info("Closing the robot simulation...")
        self.robot.close()

