from typing import Optional, Tuple, Dict

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pybullet
from pyb_class import PyBullet as p

class FrankaPandaCam:
    """Panda robot in PyBullet with Realsense D405 camera.
    
    Link Connections:
        base to link 0: pos: (0, 0, 0.333) ori: (0, 0, 0)
        link 0 to link 1: pos: (0, 0, 0) ori: (-1.57079632679, 0, 0)
        link 1 to link 2: pos: (0, -0.316, 0) ori: (1.57079632679, 0, 0)
        link 2 to link 3: pos: (0.0825, 0, 0) ori: (1.57079632679, 0, 0)
        link 3 to link 4: pos: (-0.0825, 0.384, 0) ori: (-1.57079632679, 0, 0)
        link 4 to link 5: pos: (0, 0, 0) ori: (1.57079632679, 0, 0)
        link 5 to link 6: pos: (0.088, 0, 0) ori: (1.57079632679, 0, 0)
        link 6 to link 7 (end-effector): pos: (0, 0, 0.107) ori: (0, 0, 0)
        link 7 (end_effector) to link 8: pos: (0, 0, 0) ori: (0, 0, -0.785398163397)
        link 8 to link 9: pos: (0, 0, 0.0584) ori: (0, 0, 0)
        link 9 to link 10: pos: (0, 0, 0.0584) ori: (0, 0, 0)
        link 10 to link 11: pos: (0, 0, 0.105) ori: (0, 0, 0)
        link 11 to link 12: pos: (0.035, 0, -0.05) ori: (3.1416, 0, 1.5708)
        link 12 to link 13 (camera): pos: (0, 0, -0.023) ori: (0, 0, 0)
        
    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
    """
    def __init__(
        self,
        sim: p,
        body_name = "panda_cam",
        file_name = os.path.join(os.path.dirname(__file__), "../assets/franka_panda_robot/panda_modified.urdf"),
        base_position: Optional[np.ndarray] = None,
        ) -> None:
        """Initialize environment."""
        self.sim = sim
        self.body_name = body_name
        self.file_name = file_name
        
        base_position = base_position if base_position is not None else np.zeros(3)
        self.ee_link = 7
        self.cam_link = 12
        self.cam_render_link = 13
        self.joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        self.fingers_indices = np.array([9, 10]) # Joint indices
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.neutral_pose = np.array([0.5, 0, 0.5, 0, 0, 0.7071, 0])

        self.joint_forces = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])
        
        self.dh_params = np.array([ # (a, alpha, d, theta), theta set to 0 for neutral position, update with self.get_joint_angles() when using
            [0, 0, 0.333, 0],        # Joint 1
            [0, -np.pi/2, 0, 0],     # Joint 2
            [0, np.pi/2, -0.316, 0],   # Joint 3
            [0.0825, np.pi/2, 0, 0],  # Joint 4
            [-0.0825, -np.pi/2, 0.384, 0], # Joint 5
            [0, np.pi/2, 0, 0],       # Joint 6
            [0.088, np.pi/2, 0, 0],   # Joint 7
        ])

    def _load_robot(self, base_position: np.ndarray) -> None:
        """Load Franka Panda robot.
        
        Args:
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name = self.body_name,
            fileName = self.file_name, 
            basePosition = base_position, 
            useFixedBase = True
        )
        
    def set_robot_transparency(self, alpha: float) -> None:
        """
        Sets the transparency for each link in the robot.

        Args:
            alpha (float): Transparency level between 0 (fully transparent) and 1 (fully opaque).
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")

        robot_id = self.sim._bodies_idx[self.body_name]
        num_joints = self.sim.physics_client.getNumJoints(robot_id)

        # Set the transparency for the base and each link of the robot
        for link_index in range(-1, num_joints):  # Start from -1 to include the base link
            pybullet.changeVisualShape(robot_id, link_index, rgbaColor=[1, 1, 1, alpha])
            
    def step(self):
        """Step simulation."""
        self.sim.step()

    def render(self):
        """Render scene."""
        return self.sim.render()
    
    def reset(self) -> None:
        """Reset robot to default position."""
        self.set_joint_neutral()

    def close(self):
        """Close the simulation."""
        self.sim.close()
    
    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)
    
    def get_joint_angles(self, joints: Optional[np.ndarray[int]] = None) -> np.ndarray[float]:
        """Returns the angles of multiple joints of the body.

        Args:
            joints (np.ndarray[int]): Array of joint indices

        Returns:
            np.ndarray[float]: An array of joint angles.
        """
        joints = self.joint_indices if joints is None else joints # If no joints specified: return all joint angles.
        return self.sim.get_joint_angles(self.body_name, joints)
    
    def get_link_indices(self, print_links: bool = False) -> np.ndarray:
        """Prints the index and name of each link in the robot and returns an array of link indices.
        
        Args:
            print_links (bool): If True, prints the link indices and names. If False, does not print.
            
        Returns:
            np.ndarray: An array of link indices.
        """
        robot_id = self.sim._bodies_idx[self.body_name]
        num_joints = self.sim.physics_client.getNumJoints(robot_id)

        link_indices = np.arange(num_joints)  # Create an array of link indices

        if print_links:
            print(f"Link Index - Link Name Mapping for Robot: {self.body_name}")
            for i in range(num_joints):
                info = self.sim.physics_client.getJointInfo(robot_id, i)
                link_name = info[12].decode('utf-8')
                print(f"Index: {i}, Link Name: {link_name}")

        return link_indices
        
    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the center of mass position of a link as (x, y, z) wrt robot base frame

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)
    
    def get_link_orientation(self, link: int) -> np.ndarray:
        """Get the orientation of the link of the body, in quaternion.

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Orientation as(rx, ry, rz, w)
        """
        return self.sim.get_link_orientation(self.body_name, link)
    
    def get_link_pose(self, link: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the position and orientation of a link.

        Args:
            link (int): The link index.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Position (x, y, z) and orientation (quaternion: rx, ry, rz, w).
        """
        position = self.get_link_position(link = link)
        orientation = self.get_link_orientation(link = link)
        return position, orientation
    
    # def get_camera_transform(self) -> np.ndarray:
    #     """Get the transformation matrix from the end effector to the camera."""

    #     # Get the positions and orientations
    #     cam_position = self.get_link_position(self.cam_render_link)
    #     cam_orientation = self.get_link_orientation(self.cam_render_link)
    #     ee_position = self.get_link_position(self.ee_link)
    #     ee_orientation = self.get_link_orientation(self.ee_link)

    #     # Convert orientations (quaternions) to rotation matrices
    #     cam_rotation_matrix = R.from_quat(cam_orientation).as_matrix()
    #     ee_rotation_matrix = R.from_quat(ee_orientation).as_matrix()

    #     # Relative rotation matrix from end-effector to camera
    #     relative_rotation_matrix = cam_rotation_matrix @ ee_rotation_matrix.T

    #     # Calculate the position offset
    #     # Transform the end-effector position into the camera's frame
    #     position_offset = cam_position - (ee_rotation_matrix @ ee_position)

    #     # Construct the transformation matrix
    #     transform_matrix = np.eye(4)
    #     transform_matrix[:3, :3] = relative_rotation_matrix  # Set rotation
    #     transform_matrix[:3, 3] = position_offset  # Set position offset

    #     return transform_matrix

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)
        
    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the angles of the joints of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints = self.joint_indices, angles = angles)
        
    def control_joints(self, angles: np.ndarray, forces: np.ndarray):
        """ Control the joints with motor dynamics and forces, suitable for real-time control in simulations.
        
        Args:
            angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.sim.control_joints(self.body_name, joints = self.joint_indices, target_angles = angles, forces = forces)
    
    # TODO: fix FK calculation, robot will not move to the exact scan location for some inputs, from IK obtain joint angles -> use joint angles to calculate link 7 position, account for transformation from link7 to TCP (camera link)
    def move_robot(
        self, 
        control_mode: str = "end_effector",
        target_joint_angles: Optional[np.ndarray] = None,
        target_pose: Optional[np.ndarray] = None,
        mode: str = "static",
        forces: Optional[np.ndarray] = None,
        hold_duration: Optional[float] = 1.0
        ) -> None:
        """Move robot to specified joint positions.
        
        Args:
            control_mode (str): "end_effector" or "joints" to specify control type.
            target_angles (np.ndarray, optional): List of target angles, as a list of floats.
            target_pose (np.ndarray): Desired position and orientation of the end-effector, as (x, y, z, rx, ry, rz, w).
            orientation (np.ndarray): 
            mode (str): "static" - Default mode, set_joint_angles(), or "dynamic" - control_joints()
            forces (np.ndarray, optional): Forces to apply, as a list of floats.
            hold_duration (float, optional): Time in seconds to hold the position after reaching the target, default duration is 1.0s

        """
        # Determine target angles based on control mode
        if control_mode == "joints":
            if target_joint_angles is None:
                raise ValueError("When control_mode is set to 'joints', target_angles must be provided.")
        elif control_mode == "end_effector":
            if target_pose is None:
                raise ValueError("When control_mode is set to 'end_effector', target_position must be provided.")
            
            target_joint_angles = self.inverse_kinematics( 
                link = self.cam_link, 
                position = target_pose[0:3],
                orientation = target_pose[3:7]
                )
        else:
            raise ValueError(f'Invalid control_mode "{control_mode}": Must be either "joints" or "end_effector".')

        # Execute the movement based on the mode
        if mode == "static":
            self.set_joint_angles(angles = target_joint_angles)

        elif mode == "dynamic":
            self.control_joints(
                angles = target_joint_angles,
                forces = self.joint_forces if forces is None else forces,
            )
        else:
            raise ValueError(f'Invalid mode "{mode}": Mode must be either "static" or "dynamic".')
                
        if hold_duration is not None:
            self.hold_position(hold_duration)

    def forward_kinematics(
        self,
        joint_angles: np.ndarray,
        dh_params: Optional[np.ndarray] = None
        ) -> np.ndarray:
        """
        Calculate the forward kinematics for Franka Panda Robot using joint angles (radians) and DH parameters.

        Args:
            joint_angles (np.ndarray): Array of joint angles (radians) for the robot.
            dh_params (np.ndarray): Array of DH parameters for the robot.

        Returns:
            np.ndarray: Transformation matrix of the end effector (link 7) in the base frame.
        """
        
        def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
            """Transformation matrix using DH parameters."""
            return np.array([
                [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
        
        if dh_params is None:
            dh_params = self.dh_params.copy()  # Make a copy to avoid modifying the original
            dh_params[:, 3] += np.array(joint_angles)  # Update theta with actual joint angles
    
        T = np.eye(4) # Identity matrix
        for (a, alpha, d, theta) in dh_params:
            T_i = dh_transform(a, alpha, d, theta)
            T = np.dot(T, T_i)
        
        return T

    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        
        # Error handling if target pose is unreachable
        if inverse_kinematics is None or len(inverse_kinematics) == 0:
            raise ValueError("Inverse Kinematics failed to find a solution.")

        return inverse_kinematics
    
    # def render_from_robot_cam(
    #     self,
    #     img_width: int = 512,
    #     img_height: int = 512,
    #     fov: float = 58, # FOV in degrees
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Render images from D405 Camera fixed to the panda robot arm
    #     """
    #     cam_pos = self.sim.get_link_position("panda_cam", self.cam_render_link) # Eye position
    #     cam_rot = self.sim.get_link_orientation("panda_cam", self.cam_render_link) # In quaternion
        
    #     # Compute rotation matrix to transform in world coordinates
    #     rot_matrix = np.array(self.sim.physics_client.getMatrixFromQuaternion(cam_rot)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    #     forward_vec = rot_matrix.dot(np.array((0, 0, -1))) # (0, 0, -1) is the default forward direction in camera frame, look-at direction
    #     up_vec = rot_matrix.dot(np.array((0, 1, 0))) # (0, 1, 0) is the default up direction in camera frame, up direction
        
    #     target_position = cam_pos + 0.1 * forward_vec # Target 0.1 units in front of camera
        
    #     pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
        
    #     # Compute view and projection matrices
    #     view_matrix = self.sim.physics_client.computeViewMatrix(cam_pos, target_position, up_vec) # Transform the world coord into camera coord space
    #     aspect_ratio = img_width / img_height
    #     nearVal, farVal = 0.01, 1.5 # Near, far clipping plane
    #     proj_matrix = self.sim.physics_client.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal) # Transforms 3D points from camera coord space into 2D screen space
        
    #     # Get RBG and depth images from camera
    #     images = self.sim.physics_client.getCameraImage(
    #         img_width, 
    #         img_height, 
    #         view_matrix, 
    #         proj_matrix, 
    #         renderer = pybullet.ER_BULLET_HARDWARE_OPENGL
    #         )
        
    #     rgb_img = np.array(images[2]).reshape((img_height, img_width, 4))[:, :, :3]
        
    #     depth_img = np.array(images[3]).reshape((img_height, img_width))        

    #     # Normalize deapth values
    #     """
    #     The depth data from the simulation is captured in a non-linear way (as a depth buffer). 
    #     This needs to be converted into actual depth values using the camera's near and far clipping planes.
    #     """
    #     depth_img = (-farVal * nearVal / (farVal - (farVal - nearVal) * depth_img)) # Convert raw depth buffer values (non-linear, scaled between 0-1) into linear depth values
        
    #     intrinsics = compute_intrinsics_matrix(img_width, img_height, fov)

    #     return rgb_img, depth_img, intrinsics
        
    def hold_position(self, hold_duration: float, step_interval: float = 0.01) -> None:
        """Hold the robot in its current position for a specified duration.

        Args:
            hold_duration (float): Time in seconds to hold the position.
            step_interval (float): Time in seconds between each step. Defaults to 0.01.
        """
        start_time = time.time()  # Record the start time
        
        # Get the current joint angles
        current_angles = self.get_joint_angles()
        
        # Hold the position for the specified duration
        while time.time() - start_time < hold_duration:
            self.control_joints(current_angles, self.joint_forces)  # Maintain position
            self.step()  # Advance the simulation
            time.sleep(step_interval)  # Wait for a short time
            
    def visualize_camera_pose(self):
        """Visualize the camera's position and orientation using PyBullet's debug lines."""
        cam_pos, cam_ori = self.get_link_pose(self.cam_render_link)

        # Draw a small sphere at the camera's position
        pybullet.addUserDebugLine(
            lineFromXYZ=cam_pos, 
            lineToXYZ=cam_pos + np.array([0, 0, 0.1]),  # Offset along camera's z-axis
            lineColorRGB=[0, 1, 0], 
            lineWidth=2
        )

        # Draw lines for the camera's orientation axes
        rot_matrix = R.from_quat(cam_ori).as_matrix()
        
        # Camera forward direction (Z-axis)
        cam_forward = rot_matrix[:, 2]  # Z-axis direction
        pybullet.addUserDebugLine(
            lineFromXYZ=cam_pos,
            lineToXYZ=cam_pos + cam_forward * 0.2,
            lineColorRGB=[0, 0, 1],  # Blue for Z-axis
            lineWidth=2
        )

        # Camera right direction (X-axis)
        cam_right = rot_matrix[:, 0]  # X-axis direction
        pybullet.addUserDebugLine(
            lineFromXYZ=cam_pos,
            lineToXYZ=cam_pos + cam_right * 0.2,
            lineColorRGB=[1, 0, 0],  # Red for X-axis
            lineWidth=2
        )

        # Camera up direction (Y-axis)
        cam_up = rot_matrix[:, 1]  # Y-axis direction
        pybullet.addUserDebugLine(
            lineFromXYZ=cam_pos,
            lineToXYZ=cam_pos + cam_up * 0.2,
            lineColorRGB=[0, 1, 0],  # Green for Y-axis
            lineWidth=2
        )

        # Add text for camera position
        pybullet.addUserDebugText(
            text=f"Cam Pos: {cam_pos.round(2)}",
            textPosition=cam_pos,
            textColorRGB=[1, 1, 0],
            lifeTime=0  # Keep the label until reset
        )
        
    def draw_robot_links(self, link_indices: Optional[np.ndarray] = None, color = (1, 0, 0)) -> None:
        """
        Draws a visible circle at each link position and a line showing displacement between each link of the robot.
        Link 0 corresponds to Joint 1,....Link 13 corresponds to Joint 14

        Args:
            link_indices (list, optional): Specific link indices to draw. If None, draws all links.    
            color (tuple): RGB color of the circle and line. Default is red (1, 0, 0).
        """
        
        # Fetch the robot's current joint positions
        if link_indices is None:
            link_indices = self.get_link_indices()
            
        link_positions = [self.get_link_position(link) for link in link_indices]

        # Draw circles at each link's position
        for i, position in enumerate(link_positions):
            position_text = f"Link {link_indices[i]}: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            pybullet.addUserDebugText(
                text=position_text,
                textPosition=position,
                textColorRGB=color,
                textSize=0.8,
                lifeTime=0
            )
            pybullet.addUserDebugLine(
                lineFromXYZ=position,
                lineToXYZ=link_positions[i+1] if i+1 < len(link_positions) else position,
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=0
            )

class Scene:
    """
    Simulation scene in PyBullet, including a ground plane,
    a table, and a Franka Panda robot. The scene is initialized with a
    specific simulation instance and allows for loading and managing
    various objects within the environment.

    Args:
        sim (PyBullet): The simulation instance that manages the physics 
            and rendering of the scene.
        distance_threshold (float, optional): A threshold value to 
            determine distances in the scene, defaulting to 0.2.
        obj_xy_range (float, optional): The range for object placement 
            in the X and Y dimensions, defaulting to 0.3.

    Attributes:
        sim (PyBullet): The simulation instance.
        distance_threshold (float): The distance threshold for object 
            interactions.
        object_size (float): The size of the objects created in the scene.
        obj_range_low (np.ndarray): The lower bounds for object placement 
            coordinates.
        obj_range_high (np.ndarray): The upper bounds for object placement 
            coordinates.
        robot (FrankaPandaCam): An instance of the Franka Panda robot 
            loaded into the scene.
    """
    def __init__(
        self,
        sim: p,
        robot_base_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
        object_pos: np.ndarray = np.array([0.6, 0.0, 0.0])
        
    ) -> None:
        self.sim = sim  
        self.distance_threshold:float = 0.2, # Collision detection

        self.robot_base_pos = robot_base_pos   

        self.object_size: np.ndarray = None
        self.object_pos: np.ndarray = object_pos
        self.obb_min: np.ndarray = None
        self.obb_max: np.ndarray = None
        
        # with self.sim.no_rendering():
        self._create_scene()

    def load_robot(self, base_position: np.ndarray) -> None:
        """Load the Franka Panda robot into the scene."""
        self.robot = FrankaPandaCam(sim=self.sim) 
        self.robot._load_robot(base_position)

    def _create_scene(self) -> None:
        """Create the scene with a plane and objects."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=3, width=3, height=0.4, x_offset=0.3, lateral_friction=1, spinning_friction=1)
        # self.sim.create_box(body_name="robot_stand", half_extents=np.array([0.2, 0.2, self.robot_base_pos[2]/2]), mass=1.0, position=np.array([0.0, 0.0, 0.0]), lateral_friction=1, spinning_friction=1)
        
        # Load the Franka Panda robot in the scene
        self.load_robot(base_position = self.robot_base_pos)
        
    def import_object(
            self, 
            mesh_name: str,
            mesh_file: Optional[str] = "model.urdf", 
            base_position: Optional[Tuple[float, float, float]] = None, 
            base_orientation: Optional[Tuple[float, float, float, float]] = None, 
            use_fixed_base: bool = False,
            scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
            ) -> int:
        """
        Import a object mesh into the scene.

        Args:
            mesh_name (str): The name of the mesh.
            mesh_file (str, Optional): The name of the load file (URDF, OBJ, STL). For ycb_objects, default to "model.urdf"
            base_position (Tuple[float, float, float], Optional): The position where the object will be placed. Defaults to (0, 0, 0).
            base_orientation (Tuple[float, float, float, float], Optional): The orientation of the object as a quaternion (x, y, z, w). Defaults to (0, 0, 0, 1).
            use_fixed_base (bool): If True, the object will have a fixed base and won't move.
            scale (Tuple[float, float, float]): Scaling factors for the object along x, y, and z axes.

        Returns:
            int: The unique ID of the imported object in the simulation.
        """
        # Construct the correct path to the URDF or other mesh files
        mesh_path = os.path.join(
            os.path.dirname(__file__), "../assets/ycb_objects", mesh_name, mesh_file
        )

        # Default position and orientation
        base_position = base_position if base_position is not None else self.object_pos
        base_orientation = base_orientation if base_orientation is not None else (0, 0, 0, 1)
        
        # Determine the file type and load accordingly
        file_ext = os.path.splitext(mesh_file)[-1].lower()

        if file_ext == ".urdf":
            # Ensure the correct path is being used
            mesh_path = os.path.join(os.path.dirname(__file__), "../assets/ycb_objects", mesh_name, "model.urdf")
            object_id = pybullet.loadURDF(
                mesh_path, 
                basePosition=base_position, 
                baseOrientation=base_orientation, 
                useFixedBase=use_fixed_base,
                globalScaling=scale[0]  # PyBullet's URDF loader only supports uniform scaling
                )
            
        elif file_ext in [".obj", ".stl"]:
            # Ensure the correct path is being used
            mesh_path = os.path.join(os.path.dirname(__file__), "../assets/ycb_objects", mesh_name, mesh_file)
            # Create the collision shape and return its index
            collision_id = pybullet.createCollisionShape(pybullet.GEOM_MESH, fileName=mesh_path, mesh_scale = scale)
            # Create the visual shape and return its index
            visual_id = pybullet.createVisualShape(pybullet.GEOM_MESH, fileName=mesh_path, mesh_scale = scale)
            # Create the multi-body using positional arguments for shape indices
            object_id = pybullet.createMultiBody(1.0, collision_id, visual_id, basePosition=base_position, baseOrientation=base_orientation)
        
        else:
            raise ValueError(f"Unsupported mesh file type '{file_ext}'.")

        print(f"Object '{mesh_file}' imported with ID: {object_id}")
        
        # Calculate and draw OBB
        # Obtain mesh data
        mesh_data = pybullet.getMeshData(object_id)
        vertices = np.array([v for v in mesh_data[1]])  # Vertices in local frame
        
        # Get object's world transformation
        position, orientation = pybullet.getBasePositionAndOrientation(object_id)
        rotation_matrix = np.array(pybullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Transform vertices to world frame
        world_vertices = [np.dot(rotation_matrix, v) + position for v in vertices]
        
        # Compute OBB min/max
        world_vertices = np.array(world_vertices)
        obb_min = world_vertices.min(axis=0)
        obb_max = world_vertices.max(axis=0)
        
        # Calculate the 8 corners of the OBB
        corners = [
            [obb_min[0], obb_min[1], obb_min[2]],
            [obb_min[0], obb_min[1], obb_max[2]],
            [obb_min[0], obb_max[1], obb_min[2]],
            [obb_min[0], obb_max[1], obb_max[2]],
            [obb_max[0], obb_min[1], obb_min[2]],
            [obb_max[0], obb_min[1], obb_max[2]],
            [obb_max[0], obb_max[1], obb_min[2]],
            [obb_max[0], obb_max[1], obb_max[2]],
        ]
        
        # Draw lines between the corners to form the box
        edges = [
            (0, 1), (0, 2), (0, 4), 
            (1, 3), (1, 5), 
            (2, 3), (2, 6),
            (3, 7), 
            (4, 5), (4, 6), 
            (5, 7), 
            (6, 7)
        ]
        
        for edge in edges:
            pybullet.addUserDebugLine(corners[edge[0]], corners[edge[1]], [0, 1, 0], 1.0)  # Green lines
    
        dimensions = obb_max - obb_min
        self.object_size = dimensions
        self.obb_min, self.obb_max = obb_min, obb_max
                
        return object_id

            