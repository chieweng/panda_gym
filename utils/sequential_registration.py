import sys
import os
# Redirect stderr to os.devnull
# sys.stderr = open(os.devnull, 'w')
import open3d as o3d
import numpy as np
import time
import pymeshlab 
import argparse
from scipy.spatial import KDTree
import copy
import logging
import json
np.printoptions(precision=1, suppress=False)
float_formatter = "{:.2f}".format

np.set_printoptions(formatter={'float_kind': float_formatter})


dir = r"path_to_dir"
os.listdir(dir)
blade_xyz = open("scan2.xyz", "r")
open("scan2.txt", "w").close()
with open(os.path.join(dir, "scan2.txt"), "w") as blade_txt:
    for line in blade_xyz:
        blade_txt.write(line[:9] + "," + line[10:18] + "," + line[19:27] + ",1" + "\n")



def read_xyz_from_robotpose(path = 'scan2.txt'):
    # Initialize an empty list to hold the x, y, z values
    pointcloud_xyz = []

    # Open the file for reading
    with open(path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Remove the brackets and split by comma
            parts = line.strip("[]\n").split(",")
            # Extract x, y, z values and convert them to float
            xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            # Append the xyz list to the pointcloud_xyz list
            pointcloud_xyz.append(xyz)
    
    return pointcloud_xyz


def read_robotpose_from_json(path = '.temp\'temp_data.json'):

    robotPoses = []
    # Selectively read robotposes from json
    with open(path, "r") as json_file:
        data = json.load(json_file)
        robotPose = data["RobotPose"]
    
    # Access the 'RobotPose' data and slice the first three elements of each sublist
    robotPoses = [pose[:3] for pose in data['RobotPose']]

    return robotPoses


def merge_meshes_with_voxelization(mesh_list, voxel_size=0.005):
    # Convert all meshes to voxel grids
    voxel_grids = [o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size) for mesh in mesh_list]

    # Merge voxel grids by taking the union
    combined_voxel_grid = voxel_grids[0]
    for vg in voxel_grids[1:]:
        combined_voxel_grid = combined_voxel_grid + vg  # This is a simplistic way to combine; adjustments may be needed

    # Convert the combined voxel grid back to a mesh
    combined_mesh = o3d.geometry.TriangleMesh.create_from_voxel_grid(combined_voxel_grid)
    return combined_mesh


def translate_to_common_origin(pcd_list):
    # Determine a common origin based on the bounding boxes of the point clouds
    # For simplicity, we'll use the minimum corner of the first point cloud's bounding box
    common_origin = pcd_list[0].get_axis_aligned_bounding_box().get_min_bound()

    # Translate all point clouds to this common origin
    for pcd in pcd_list[1:]:
        bbox_min_corner = pcd.get_axis_aligned_bounding_box().get_min_bound()
        translation = common_origin - bbox_min_corner
        pcd.translate(translation)
    
    return pcd_list

'''
Method:

1. Read PointCloud (X,Y,Z,A) where A is the scan index
2. Read Robot Poses (X,Y,Z Only) --> for point normal estimation based on camera position
3. Select Convert pointcloud to meshlab data
FOR in RANGE LEN SCAN
    4. select Mesh[i]
    5. approximate normal[robotpose[i]]
    6. select Mesh[i+1]
    7. approximate normal [robotpose[i+1]]

8. Global align + ICP (small target distance)


FOR in RANGE LEN SCAN                  
    9.select Mesh[i]
    10. Merge close vertices[i][~1mm distance]

11. generate_by_merging_visible_meshes
12. run rolling ball algorithm on mesh
13. smoothing (if true)
14. close holes (if true)
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multi-Registration and Meshing")
    parser.add_argument('--input', '-i', required=False, help='input path', default = '.temp/')
    parser.add_argument('--alignsamplenum', '-asn', required=False, help='alignment sample number(int)', default = '50000')
    parser.add_argument('--alignmindistabs', '-amd', required=False, help='alignment minimum distance absolute(float)', default = '11')
    parser.add_argument('--alignmaxiter', '-asi', required=False, help='Maximum iteration (int)', default = '500')
    parser.add_argument('--aligntrgdistabs', '-atd' , required=False, help='alignment target distance(float)', default = '1.1')
    parser.add_argument('--smooth', '-s', required=False, help='Smooth Mesh (True/False)', default = 'True')
    parser.add_argument('--smoothlambda', '-sl', required=False, help='lambda parameter of taubin smoothing (float)', default = '0.600000')
    parser.add_argument('--smoothmu', '-sm', required=False, help='mu parameter of taubin smoothing(float)', default = '-0.540000')
    parser.add_argument('--stepsmoothnum', '-sn', required=False, help='number of iteration of taubin smoothing(int)', default = '5')
    parser.add_argument('--smoothlaplacian', '-sli', required=False, help='laplacian smoothing iteration', default = '5')
    parser.add_argument('--holefill', '-hf', required=False, help='Hole Fill (True/False)', default = 'True')
    parser.add_argument('--maxhole', '-mh', required=False, help='Max hole fill size(float)', default = '500')
    parser.add_argument('--rollingballsize', '-rbs', required=False, help='Multiplier of rolling ball size', default = '3')
    parser.add_argument('--voxdownsample', '-vds', required=False, help='downsample factor', default = '1')
    parser.add_argument('--repairs', '-rep', required=False, help='rep', default = 'True')
    parser.add_argument('--process', '-pro', required=False, help='process all or latest', default = 'ALL')

    #initialise logger
    log_filename = "MODULE.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('--------------STARTING SEQUENTIAL REGISTRATION MODULE--------------')

    args = parser.parse_args()

    try:
        path = args.input
        start_reg = time.process_time()
        logger.info('Processing robot movement history')
        #load the robot pose
        # robotpose = read_xyz_from_robotpose(path+ 'RobotPose.txt')
        # robotpose = read_robotpose_from_json(path+'temp_data.json')
        robotpose = np.array([[0.35430414997587173, 0.1957132814893931, 1.0872309432039695]])

        logger.info('loading pointcloud from temp file')
        data = np.loadtxt('scan2.txt', delimiter=',')
        points = data[:, :3]  # Extract x, y, z coordinates
        A_values = data[:, 3]  # Extract A values

    except Exception as e:
        logger.error(f"Error occured while reading file: {e}")

    try:

        #if process is "latest" , assign only the latest pointcloud and robotpose

        if args.process.upper() == "LATEST":
            #take the latest robot pose, but we still want to make this a list
            robotpose = [robotpose[-1]]
            scans = pymeshlab.MeshSet()
            logger.info(f'Assigning latest scan #{np.max(np.unique(A_values))}')
            # Get the points for this A value
            scan_group = points[A_values == np.max(np.unique(A_values))]
            m = pymeshlab.Mesh(scan_group)
            assert m.vertex_number() == len(np.asarray(scan_group))
            scans.add_mesh(m)
            logger.info(robotpose)

        if args.process.upper() == "ALL":
            scans = pymeshlab.MeshSet()
            for a in np.unique(A_values):
                logger.info(f'assigning scan {a}')
                
                # Get the points for this A value
                scan_group = points[A_values == a]

                # logger.info(scan_group)

                m = pymeshlab.Mesh(scan_group)
                assert m.vertex_number() == len(np.asarray(scan_group))
                scans.add_mesh(m)


        logger.info('computing normals')


        if scans.mesh_number() == 0:
            raise Exception('no mesh found')

        if scans.mesh_number() > 0:


            for i in range(scans.mesh_number()):
                logger.info(f'Registering scan {i} and scan {i+1}' )
                
                logger.info('   *computing normals')
                #ref mesh
                scans.set_current_mesh(i)
                
                scans.compute_normal_for_point_clouds(flipflag = True, viewpos = robotpose[i]) #to insert robot pose here


            for i in range(scans.mesh_number()):
                scans.set_current_mesh(i)
                m = scans.current_mesh()
                


            scans.compute_matrix_by_mesh_global_alignment(basemesh = 0, onlyvisiblemeshes = False, samplenum = int(args.alignsamplenum), mindistabs = float(args.alignmindistabs), trgdistabs = float(args.aligntrgdistabs), maxiternum = int(args.alignmaxiter))

            transformMatrix= []
            for i in range(scans.mesh_number()):
                scans.set_current_mesh(i)
                # scans.current_mesh().apply_matrix(scans.current_mesh().transform_matrix())
                m = scans.current_mesh()
                transformMatrix.append(scans.current_mesh().transform_matrix())


        #Convert to open3D format
        pcd_array = []
        pcd_normal_array = []
        for i in range(scans.mesh_number()):                
            scans.set_current_mesh(i)
            # scans.meshing_merge_close_vertices(threshold = pymeshlab.PureValue(1))
            # #CONVERT TO OPEN3D Format
            mesh = scans.current_mesh()

            
            # Extract vertices and faces from the PyMeshLab mesh.
            vertices = mesh.vertex_matrix()
            vertice_normal = mesh.vertex_normal_matrix()
            # Initialize an empty Open3D mesh
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            pcd.estimate_normals()
            # o3d.geometry.orient_normals_towards_camera_location(merged_pcd, orientation_reference=np.asarray(robotpose[0]))
            o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.asarray(robotpose[i]))
            



            logger.info(f'transforming by \n{transformMatrix[i]}')
            pcd_T = copy.deepcopy(pcd)
            pcd_T = pcd_T.transform(transformMatrix[i])
            

            pcd_array.append(pcd_T.points)
            pcd_normal_array.append(pcd_T.normals)


        all_points = pcd_array[0]
        all_normals = pcd_normal_array[0]
        if len(pcd_array) > 1:
            for i in range(len(pcd_array)-1):
                all_points = np.vstack((all_points, np.asarray(pcd_array[i+1])))
                logger.info(pcd_array[i+1])
                all_normals = np.vstack((all_normals, np.asarray(pcd_normal_array[i+1])))

        logger.info('Pointcloud is merged')
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(all_points)
        merged_pcd.normals = o3d.utility.Vector3dVector(all_normals)
        logger.info('Computing median radius')

        #calculate median on max 400,000 points    
        pointskipmultiplier = int(np.asarray(pcd.points).shape[0] /400000)

        if args.process.upper() == "LATEST":
            pointskipmultiplier = 10

        
        if pointskipmultiplier < 1:
            pointskipmultiplier = 1

        logger.info(f' there are # {np.asarray(pcd.points).shape[0]} points found, subsampling it {pointskipmultiplier}x for median calculation' )

        #compute radius
        kdtree = o3d.geometry.KDTreeFlann(merged_pcd)
            # Compute the distances to the nearest neighbors
        distances = []
        for i in range(int(np.asarray(pcd.points).shape[0]/ pointskipmultiplier)):
            _, idx, _ = kdtree.search_knn_vector_3d(merged_pcd.points[i], 2)
            distance = np.linalg.norm(np.asarray(merged_pcd.points)[i] - np.asarray(merged_pcd.points)[idx[1]])
            distances.append(distance)
        radius = np.median(distances)

        
        merged_pcd = merged_pcd.voxel_down_sample(radius*float(args.voxdownsample)) ###



        logger.info(f'Tesselation with ball pivoting with radius = {radius*float(args.voxdownsample)} and multiplier = {float(args.rollingballsize)}')

        start_tes = time.process_time()
        bpa_mesh_2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(merged_pcd, o3d.utility.DoubleVector([radius*float(args.voxdownsample)*float((args.rollingballsize)) , radius*float(args.voxdownsample)*2*float((args.rollingballsize))]))
        end_tes = time.process_time()
        logger.info(f'Completed Tesselation in {end_tes-start_tes:.6f} seconds')


        '''
        This region converts the data to pymeshlab and process it 
        '''

        vertices = np.asarray(bpa_mesh_2.vertices).astype(np.float64)  # Ensure data type is float64
        faces = np.asarray(bpa_mesh_2.triangles).astype(np.int32)  # Ensure data type is int32
        vertex_normals = np.asarray(bpa_mesh_2.vertex_normals).astype(np.float64)  # Ensure data type is float64
        m = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_normals_matrix=vertex_normals)

        ms_ = pymeshlab.MeshSet()
        ms_.add_mesh(m)
        
        #ms_.meshing_close_holes()
        logger.info('edge clean up')
        if args.process.upper() == "ALL":
            ms_.compute_selection_from_mesh_border()
            ms_.meshing_remove_selected_faces()




        logger.info('merging mesh')
        ms_.generate_by_merging_visible_meshes(mergevisible = True, deletelayer = True, mergevertices = True, alsounreferenced = False)
        
        ms_.compute_custom_radius_scalar_attribute_per_vertex(nbneighbors = 17)

        
        #BISMA

        # ms_.save_current_mesh(file_name = f"{path}check.obj")
        ###########################3
        
        
        ms = ms_
            

        
        logger.info('Optimisation:')

        if args.repairs.upper() == "TRUE":
            logger.info('              repair face normals')
            ms.apply_normal_normalization_per_face()
            logger.info('              repair vertex normals')
            ms.apply_normal_normalization_per_vertex()
            logger.info('              Remove T-Vertices')
            ms.meshing_remove_t_vertices()
            logger.info('              Repair Edges')
            ms.meshing_repair_non_manifold_edges()
            logger.info('              Repair non-manifold vertices')
            ms.meshing_repair_non_manifold_vertices()
            logger.info('              Repair duplicate faces')
            ms.meshing_remove_duplicate_faces()
            logger.info('              Repair duplicate vertices')
            ms.meshing_remove_duplicate_vertices()
            logger.info('              Repair folded faces')
            ms.meshing_remove_folded_faces()
        # logger.info('              snap mismatched borders')
            #ms.meshing_snap_mismatched_borders()





        if args.holefill.upper() == "TRUE":
            
            logger.info('              closing holes')
            ms.meshing_close_holes(maxholesize = int(args.maxhole))

        if args.smooth.upper() == "TRUE":
            if args.process.upper() == "ALL":
                logger.info('              performing taubin smoothing')
                ms.apply_coord_taubin_smoothing(lambda_ = float(args.smoothlambda), mu = float(args.smoothmu), stepsmoothnum = int(args.stepsmoothnum))
            logger.info('              performing laplacian smoothing')
            ms.apply_coord_laplacian_smoothing(stepsmoothnum = int(args.smoothlaplacian))
        


        logger.info('              performing scaling')
        #scale it to meter
        ms.compute_matrix_from_scaling_or_normalization(axisx = 0.001000, axisy = 0.001000, axisz = 0.001000)


        logger.info('              converting to unity frame')
        #Convert to unity coordinate frame 
        ms.compute_matrix_from_rotation(rotaxis = 0, angle = -90.000000) 
        ms.compute_matrix_from_rotation(rotaxis = 1, angle = -90.000000)

        logger.info('              performing mesh simplification')
        ms.meshing_decimation_quadric_edge_collapse()


        logger.info('Saving mesh')
        ms.save_current_mesh(file_name = f"output.obj", save_vertex_color=False, save_face_color=False)
        end_reg = time.process_time()



        logger.info(f'completed full registration in {end_reg - start_reg:.6f} seconds')


    except Exception as e:
        logger.error(f"Something went wrong. try disabling hole filling?: {e}")











