from typing import List, Tuple, Optional
import os
import numpy as np
import open3d as o3d
import copy

class PointCloudMerger:
    """
    Point cloud alignment process:
    1. Global Registration (RANSAC-based): Obtain rough initial alignment of the point clouds, performed on heavily downsampled pcd
    2. Local Registration (ICP): Using the rough alignment from the global registration, apply ICP to finetune the alignment on original pcd
    3. Tesselation using rolling-ball + Laplacian surface smoothing (for mesh generation)
    """
    
    def __init__(
        self,
        scene_instance,
        voxel_size: float = 0.01,
        max_correspondence_dist: float = 0.05,
        rolling_ball_radius: float = 0.1,
        smoothing_iter: int = 10,
        generate_mesh: bool = False
    ) -> None:
        """
        Initialize parameters for point cloud merging and processing.f

        Args:
            ----- Global Registration -----
            voxel_size (float): Size of voxel grid used for downsampling. 
                Recommended range:
                - High-density point clouds (millions of points): ~1% to 5% of the average point spacing.
                - Medium-density point clouds: ~0.02 to 0.05 times the point spacing.
                - Low-density point clouds: ~0.1 to 0.2 times the point spacing.
                
            ----- Local Registration (ICP) -----
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
        self.source_down_list: List[o3d.geometry.PointCloud] = [] # List of downsampled source pcd with normal
        self.source_fpfh_list: List[o3d.registration.Feature] = [] # List of FPFH descriptor (N, 33) for each pcd, where N = num of points in pcd, 33 = num of bins of the histogram in standard configuration
        self.target_pcd = o3d.geometry.PointCloud() # Initialize empty o3d for final pcd output
        
    def pcd_preprocessing(self):
        """
        1. Load and convert point_cloud.npy files to Open3D format.
        2. Downsample each point cloud with a specified voxel size.
        3. Estimate normals and compute the Fast Point Feature Histograms (FPFH) features.
        """
        
        def downsample_to_target(pcd, target_min=1000, target_max=3000, max_iter=50):
            """
            Downsample the point cloud to a size within the target range [target_min, target_max].
            """
            voxel_size = self.voxel_size
            downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            
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
        
        files = [f for f in os.listdir(self.npy_file_path) if f.endswith('.npy')] # List all npy files in dir
        if not files: raise ValueError("No .npy files found in the specified directory.")
        
        npy_data_list = [np.load(os.path.join(self.npy_file_path, file)) for file in files] # Load each npy file into list

        # Convert numpy arrays to Open3D point clouds
        for data in npy_data_list:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data) # Convert numpy arrays to o3d-compatible format
            self.visualize(point_cloud=pcd)
            
            # Segment and remove background (table, clipping plane)
            points = np.asarray(pcd.points)
            
            ###
            import pybullet as p
            import random
            # Randomly sample 100 points from the point cloud
            sampled_points = random.sample(points.tolist(), 10)

            for point in sampled_points:
                x, y, z = point
                p.addUserDebugLine(lineFromXYZ=[x, y, z], lineToXYZ=[x, y, z], lineColorRGB=[1, 0, 0], lineWidth=5)
            ###
                    
            print("Number of points before masking:", points.shape[0])
            
            #TODO since bounding box is axis aligned and doesnt account for rotations, when camera captures pcd from various angles, the bounding box will not be able to correctly segment the noise, try transforming all pcd data to local frame firsts
            
            obb_min = self.scene_instance.obb_min
            obb_max = self.scene_instance.obb_max
            mask = np.all((points >= obb_min) & (points <= obb_max), axis=1)
            print("Mask:", mask)
            segmented_pcd = pcd.select_by_index(np.where(mask)[0])
                                
            self.visualize(point_cloud = segmented_pcd)
            
            pcd_down = downsample_to_target(segmented_pcd)
            print("Downsampled point cloud size:", len(pcd_down.points))
                        
            # Estimate normals for downsampled pcd
            radius_normal = self.voxel_size * 2
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30)
                )
            
            # # Compute FPFH, which captures local geom properties of each point based on distribution of angles between the point's normal and neighbors' normals.
            # radius_feature = self.voxel_size * 5
            # pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            #     pcd_down, 
            #     o3d.geometry.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100)
            #     )
            
            # Populate
            self.source_pcd_list.append(pcd)
            self.source_down_list.append(pcd_down)
            # self.source_fpfh_list.append(pcd_fpfh)
            
        self.target_pcd = self.source_pcd_list[0]
    
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
                1) Downsampled pcd will have lower resolution of normal data, complex surfaces and geometry details will also not benefit 
                from the limited normal information.
                2) Computational efficiency for rough transformation using global registration, ICP for refined alignments.
        """
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, 
            mutual_filter = True,
            max_correspondence_distance = self.max_correspondence_dist,
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n = 3, 
            checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.max_correspondence_dist)], 
            criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500))
        
        return result.transformation    
    
    def point_to_plane_icp(self, source_pcd, target_pcd, trans_matrix):
        """
        Refines the alignment of source and target pcd using the Point-to-Plane Iterative Closest Point (ICP) algorithm.
    
        The method assumes an initial rough alignment obtained from RANSAC algorithm and improves the registration by minimizing 
        the point-to-plane distance between corresponding points in the source and target point clouds.

        Args:
            source_pcd (open3d.geometry.PointCloud): The source point cloud to be aligned.
            target_pcd (open3d.geometry.PointCloud): The target point cloud to align the source point cloud against.
            trans_matrix (numpy.ndarray): A 4x4 transformation matrix providing an initial rough alignment.

        Returns:
            numpy.ndarray: A refined 4x4 transformation matrix that aligns the source point cloud to the target point cloud.
        """
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            self.max_correspondence_dist, 
            trans_matrix,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return result.transformation
        
    def merge_point_clouds(self) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
        """
        Perform point cloud alignment process, then apply optional surface reconstruction and smoothing for mesh generation.

        Returns:
            o3d.geometry.PointCloud: Merged point cloud.
            o3d.geometry.TriangleMesh: Smoothed mesh (from surface reconstruction).
        """

        # Align each point cloud to the target using ICP
        for i in range(1, len(self.source_pcd_list)):
            target_pcd = self.target_pcd # First point cloud entry as the target/reference
            target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = self.voxel_size * 2, max_nn = 30)) # Only precomputed normals for target pcd are required for Point-to-plane ICP
            target_down = target_pcd.voxel_down_sample(self.voxel_size)
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = self.voxel_size * 2, max_nn = 30)) # Recalculate normals after voxel downsampling
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_down, o3d.geometry.KDTreeSearchParamHybrid(radius = self.voxel_size * 5, max_nn = 100))

            source_pcd = self.source_pcd_list[i]
            source_down = self.source_down_list[i]
            source_fpfh = self.source_fpfh_list[i]
            
            # RANSAC to obtain rough transformation matrix
            transformation_matrix_RANSAC = self.global_registration(source_down, target_down, source_fpfh, target_fpfh)
            
            # ICP for alignment
            transformation_matrix_ICP = self.point_to_plane_icp(source_pcd, target_pcd, transformation_matrix_RANSAC)
            
            # Transform and merge source cloud
            source_pcd.transform(transformation_matrix_ICP)
            
            self.draw_registration_result(source_pcd, target_pcd, transformation_matrix_ICP)
            
            self.target_pcd += source_pcd          
            self.target_pcd.remove_duplicated_points()
            
        self.visualize(self.target_pcd)
        
        mesh = None
        if self.generate_mesh: 
            # Surface reconstruction via rolling ball algorithm approximation
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                self.target_pcd,
                o3d.utility.DoubleVector([self.rolling_ball_radius, self.rolling_ball_radius * 2])
            )

            # Laplacian smoothing on the mesh
            for _ in range(self.smoothing_iter):
                mesh = mesh.filter_smooth_laplacian()

        return self.target_pcd, mesh

    def save_results(self, point_cloud: Optional[o3d.geometry.PointCloud], mesh: Optional[o3d.geometry.TriangleMesh]):
        """
        Save the merged point cloud and mesh to files.

        Args:
            point_cloud (Optional, o3d.geometry.PointCloud): The merged point cloud.
            mesh (Optional, o3d.geometry.TriangleMesh): The smoothed mesh.
            pcd_filename (str): Filename to save the point cloud.
            mesh_filename (str): Filename to save the mesh.
        """
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'scans')

        pcd_filename = os.path.join(output_dir, "merged_point_cloud.ply")
        mesh_filename = os.path.join(output_dir, "merged_mesh.ply")
        
        if point_cloud is not None: 
            o3d.io.write_point_cloud(filename = pcd_filename, pointcloud = point_cloud) 
            print(f"Point cloud saved to: {pcd_filename}")
        if mesh is not None: 
            o3d.io.write_triangle_mesh(filename = mesh_filename, mesh = mesh) 
            print(f"Mesh saved to: {mesh_filename}")

    def draw_registration_result(self, source, target, transform_matrix):
        """
        Visualizes the target and transformed source point clouds after applying an alignment transformation. Red: source pcd, Cyan: target pcd
        
        Args:
            source (open3d.geometry.PointCloud): The source pcd to be transformed.
            target (open3d.geometry.PointCloud): The target pcd.
            transform_matrix (numpy.ndarray or open3d.geometry.Transform): A 4x4 transformation matrix to align the source pcd with target pcd.
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 1, 1])
        source_temp.transform(transform_matrix)
        o3d.visualization.draw_geometries([source_temp, target_temp], 
                                          zoom=0.4459, 
                                          front=[0.9288, -0.2951, -0.2242], 
                                          lookat=[1.6784, 2.0612, 1.4451], 
                                          up=[-0.3402, -0.9189, -0.1996])
        
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
            
