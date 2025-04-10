from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import  open3d as o3d
import copy

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class PointCloudMerger:
    """
    Point cloud alignment process:
    1. Global Registration (RANSAC-based): Obtain rough initial alignment of the point clouds, performed on heavily downsampled pcd
    2. Local Registration (ICP): Using the rough alignment from the global registration, apply ICP to finetune the alignment on original pcd
    """
    
    def __init__(
        self,
        scene_instance,
        voxel_size: float = 0.01,
        max_correspondence_dist: float = 0.01,
        rolling_ball_radius: float = 0.1,
        smoothing_iter: int = 10,
        generate_mesh: bool = False
    ) -> None:
        """
        Initialize parameters for point cloud merging and processing.f

        Args:
            ----- Global Registration/ICP -----
            voxel_size (float): Size of voxel grid used for downsampling. 
                Recommended range:
                - High-density point clouds: ~1% to 5% of the average point spacing.
                - Medium-density point clouds: ~0.02 to 0.05 times the point spacing.
                - Low-density point clouds: ~0.1 to 0.2 times the point spacing.
            max_correspondence_dist (float): Maximum distance for ICP point correspondence.
            
            ----- Mesh generation -----
            rolling_ball_radius (float): Radius for rolling ball surface reconstruction.
            smoothing_iter (int): Number of smoothing iterations.
            generate_mesh (bool): Generate the mesh using rolling ball + surface smoothing, False by default.
        """
        self.scene_instance = scene_instance
        self.voxel_size = voxel_size
        self.max_correspondence_dist = max_correspondence_dist
        self.rolling_ball_radius = rolling_ball_radius
        self.smoothing_iter = smoothing_iter
        self.generate_mesh = generate_mesh
        self.npy_file_path = os.path.join(os.path.dirname(__file__), '..', 'scans')
        self.source_pcd_list: List[o3d.geometry.PointCloud] = []
    
    def downsample_and_compute_fpfh(self, pcd):
        """
        Downsamples a given point cloud to a target size range and computes its Fast Point Feature Histogram (FPFH).

        Steps:
        1. Downsample the input point cloud using voxel downsampling, adjusting voxel size iteratively 
        to ensure the point count remains within the specified range.
        2. Estimate normals for the downsampled point cloud.
        3. Compute the FPFH feature, which encodes local geometric properties of each point 
        based on the distribution of angles between its normal and the normals of its neighbors.

        Args:
        - pcd (o3d.geometry.PointCloud): The input point cloud to be processed.

        Returns:
        - pcd_down (o3d.geometry.PointCloud): The downsampled point cloud.
        - fpfh (o3d.pipelines.registration.Feature): The computed FPFH feature for the downsampled point cloud.
        """
        def downsample_to_target(pcd, target_min=2500, target_max=3500, max_iter=50):
            """
            Downsample the point cloud to a size within the target range [target_min, target_max].
            """
            voxel_size = self.voxel_size
            downsampled_pcd = pcd.voxel_down_sample(self.voxel_size)
            
            # Iteratively adjust the voxel size based on the downsampled size
            for i in range(max_iter):
                downsampled_size = len(downsampled_pcd.points)
                
                # Check if we're within the target range
                if target_min <= downsampled_size <= target_max:
                    return downsampled_pcd  # Return the downsampled point cloud
                
                # Adjust the voxel size based on whether the size is too large or too small
                if downsampled_size < target_min:
                    voxel_size *= 0.5
                elif downsampled_size > target_max:
                    voxel_size *= 1.8 
                
                downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            
            return downsampled_pcd
        
        pcd_down = downsample_to_target(pcd)

        # Estimate normals for downsampled pcd
        radius_normal = self.voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30)
            )
        
        # Compute FPFH, which captures local geom properties of each point based on distribution of angles between the point's normal and neighbors' normals.
        radius_feature = self.voxel_size * 2
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, 
            o3d.geometry.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 175)
            )
        return pcd_down, fpfh

    def pcd_preprocessing(self): 
        """
        Loads and preprocesses all .npy point cloud files in the specified directory ("../scans/")

        Steps:
        1. Loads and converts .npy files into Open3D point clouds.
        2. Removes background points (wall/ floor/ clipping plane) by cropping the point cloud using an enlarged bounding box.
        3. Segments and removes the largest planar surface (e.g., a table) using RANSAC. This step is to ensure that all noise are filtered and removed.
        4. Downsamples the filtered point cloud and compute FPFH features.
        5. Stores the processed point clouds and FPFH features in respective lists.

        Modifies:
        - self.source_pcd_list (list): Stores the cropped and filtered point clouds.
        """   
        files = sorted(f for f in os.listdir(self.npy_file_path) if f.endswith('.npy')) # List all npy files in dir
        if not files: raise ValueError("No .npy files found in the specified directory.")
        
        npy_data_list = [np.load(os.path.join(self.npy_file_path, file)) for file in files] # Load each npy file into list

        # Convert numpy arrays to Open3D point clouds
        for i, data in enumerate(npy_data_list):
            print("-----------------------------------------------------------------------")
            print(f"Performing post-processing sequence on Point Cloud {i+1}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data) # Convert numpy arrays to o3d-compatible format
            
            # self.visualize(point_cloud=pcd)
            
            # with open("points_output.txt", 'w') as f:
            #     for point in pcd.points:
            #         # Write each point as a line of space-separated x, y, z coordinates
            #         f.write(f"{point[0]} {point[1]} {point[2]}\n")
            
            # Remove background point cloud (table, clipping plane)                        
            bb_min = self.scene_instance.obb_min - np.array([0.3, 0.3, 0.3]) #increase size of bb
            bb_max = self.scene_instance.obb_max + np.array([0.3, 0.3, 0.3])
            aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound = bb_min, max_bound = bb_max)
            
            pcd_masked = pcd.crop(aabb, invert = False)
            # self.visualize(point_cloud = pcd_masked)
            
            # Segment the largest plane
            _, inliers = pcd_masked.segment_plane(distance_threshold=0.005,  # Adjust based on your data scale
                                                            ransac_n=3,
                                                            num_iterations=1000)
            
            pcd_masked = pcd_masked.select_by_index(inliers, invert=True)
            
            # pcd_down = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
            
            pcd_down, fpfh = self.downsample_and_compute_fpfh(pcd_masked)
            
            logging.info(f"Masked point cloud size: {len(pcd_masked.points)}")
            logging.info(f"Downsampled point cloud size: {len(pcd_down.points)}")
            
            # Populate
            self.source_pcd_list.append(pcd_masked)
    
    def global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        """ 
        Performs global registration of two downsampled point clouds using RANSAC-based feature matching.
        Calculates the transformation matrix to align the source pcd to the target pcd by using FPFH for feature matching and
        RANSAC algorithm for estimation of the transformation between the point clouds.

        Args:
            source_down (open3d.geometry.PointCloud): The downsampled source point cloud to be aligned.
            target_down (open3d.geometry.PointCloud): The downsampled target point cloud for alignment.
            source_fpfh (open3d.registration.FPFHFeature): The FPFH feature descriptors of the source point cloud.
            target_fpfh (open3d.registration.FPFHFeature): The FPFH feature descriptors of the target point cloud.

        Returns:
            numpy.ndarray: A 4x4 transformation matrix that aligns the source point cloud with the target point cloud.
            
        Note: 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False) is set to False, as it only uses point positions
            to estimate the transformation, the algorithm will perform point-to-point matching without using the point normals: 
                1) Downsampled pcd have lower resolution of normal data (fewer points), hence complex surfaces and geometry details will not benefit 
                from the limited normal information.
                2) Improved computational efficiency for rough initial transformation using global registration, ICP will later be used for refined alignments.
        """
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, 
            mutual_filter = False,
            max_correspondence_distance = self.max_correspondence_dist,
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n = 3, 
            checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.max_correspondence_dist)], 
            criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 50000))
        
        logging.info(f"RANSAC result: fitness = {result.fitness}, inlier_rmse = {result.inlier_rmse}")
        
        return result.transformation    
    
    def point_to_plane_icp(self, source_pcd, target_pcd, transformation_matrix_RANSAC):
        """
        Refines the alignment of source and target pcd using the Point-to-Plane Iterative Closest Point (ICP) algorithm.
    
        The method assumes an initial rough alignment obtained from RANSAC algorithm and improves the registration by minimizing 
        the point-to-plane distance between corresponding points in the source and target point clouds.

        Args:
            source_pcd (open3d.geometry.PointCloud): The source point cloud to be aligned.
            target_pcd (open3d.geometry.PointCloud): The target point cloud to align the source point cloud against.
            transformation_matrix_RANSAC (numpy.ndarray): A 4x4 transformation matrix providing an initial rough alignment, obtained from RANSAC.

        Returns:
            numpy.ndarray: A refined 4x4 transformation matrix that aligns the source point cloud to the target point cloud.
        """
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            self.max_correspondence_dist, 
            transformation_matrix_RANSAC,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        logging.info(f"ICP result: fitness = {result.fitness}, inlier_rmse = {result.inlier_rmse}")
        
        return result.transformation
    
    
    def hierarchical_registration(self, pcd_list):
                
        def pairwise_merge(source_pcd, target_pcd):
            # logging.info("Starting pairwise merge")
            source_down, source_fpfh = self.downsample_and_compute_fpfh(source_pcd)
            target_down, target_fpfh = self.downsample_and_compute_fpfh(target_pcd)
            
            # Global registration
            transformation_matrix_RANSAC = self.global_registration(source_down, target_down, source_fpfh, target_fpfh)
            # logging.info("Global registration completed")

            # Point to plane ICP
            radius_normal = self.voxel_size * 2
            source_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30)
                )
            target_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30)
                )
            transformation_matrix_ICP = self.point_to_plane_icp(source_pcd, target_pcd, transformation_matrix_RANSAC)
            # logging.info("ICP registration completed")

            # Transform and merge source clouds
            source_pcd.transform(transformation_matrix_ICP)
            target_pcd += source_pcd        
            target_pcd.remove_duplicated_points()
            # logging.info("Pairwise merge completed")
                    
            return target_pcd
        
        current_layer = pcd_list
        logging.info("Starting hierarchical registration")

        layer = 1
        while len(current_layer) > 1:
            next_layer = []
            
            with ThreadPoolExecutor() as exe:
                # Pairwise merge point clouds at the current level
                futures = [
                    exe.submit(pairwise_merge, current_layer[i], current_layer[i + 1])
                    for i in range(0, len(current_layer) - 1, 2)
                ]
                for future in futures:
                    next_layer.append(future.result())
            
            # for i in range(0, len(current_layer) - 1, 2):
            #     merged_pcd = pairwise_merge(current_layer[i], current_layer[i + 1])
            #     next_layer.append(merged_pcd)
            
            # If there's an odd number of point clouds, carry the last one to the next level
            if len(current_layer) % 2 == 1:
                next_layer.append(current_layer[-1])
                                        
            current_layer = next_layer
            logging.info(f"Completed layer {layer}, {len(current_layer)} point clouds remaining")
            
            layer += 1
                
        merged_pcd = current_layer[0]
        logging.info("Hierarchical registration completed")
        
        self.visualize(merged_pcd)
        
        return merged_pcd
    
    
    def save_results(self, point_cloud: Optional[o3d.geometry.PointCloud], mesh: Optional[o3d.geometry.TriangleMesh]):
        """
        Save the merged point cloud/mesh to files.

        Args:
            point_cloud (Optional, o3d.geometry.PointCloud): The merged point cloud.
            mesh (Optional, o3d.geometry.TriangleMesh): The smoothed mesh.
        """
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'scans')

        pcd_filename = os.path.join(output_dir, "merged_point_cloud.ply")
        mesh_filename = os.path.join(output_dir, "merged_mesh.ply")
        
        if point_cloud is not None: 
            o3d.io.write_point_cloud(filename = pcd_filename, pointcloud = point_cloud) 
            logging.info(f"Point cloud saved to: {pcd_filename}")
        if mesh is not None: 
            o3d.io.write_triangle_mesh(filename = mesh_filename, mesh = mesh) 
            logging.info(f"Mesh saved to: {mesh_filename}")

    def draw_registration_result(self, source_pcd, target_pcd, transformation_matrix):
        """
        Visualizes the target and transformed source point clouds after applying an alignment transformation. Red: source pcd, Cyan: target pcd
        
        Args:
            source_pcd (open3d.geometry.PointCloud): The source pcd to be transformed.
            target_pcd (open3d.geometry.PointCloud): The target pcd.
            transform_matrix (numpy.ndarray or open3d.geometry.Transform): A 4x4 transformation matrix to align the source pcd with target pcd.
        """
        # Transform and merge source cloud
        source_temp = copy.deepcopy(source_pcd)        
        source_temp.transform(transformation_matrix)
        
        target_temp = copy.deepcopy(target_pcd)
        
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 1, 1])
        
        o3d.visualization.draw_geometries([source_temp, target_temp])
        
    def visualize(self, point_cloud = None, mesh = None):
        """
        Visualize point cloud or mesh.

        Args:
            point_cloud (o3d.geometry.PointCloud): The point cloud to visualize.
            mesh (o3d.geometry.TriangleMesh): The mesh to visualize.
        """

        if point_cloud:
            o3d.visualization.draw_geometries([point_cloud], window_name="Merged Point Cloud")
        if mesh:
            o3d.visualization.draw_geometries([mesh], window_name="Smoothed Mesh")
            
    def draw_bounding_box_in_pybullet(self, bbox_min, bbox_max):
        """
        Visualize the bounding box of a point cloud in PyBullet.
        """
        import pybullet as p
        # Define the 8 corners of the bounding box
        corners = [
            bbox_min,
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            bbox_max,
        ]

        # Draw lines between the corners to form the box
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]

        for edge in edges:
            p.addUserDebugLine(corners[edge[0]], corners[edge[1]], [1, 0, 0], lineWidth=2)
            