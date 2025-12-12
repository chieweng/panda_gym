import numpy as np
from env.franka_env import FrankaPandaCam, Scene
from utils import math_utils
from env.task import MultiScanTask
from utils.pcd_utils import PointCloudMerger
from pyb_class import PyBullet as p

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    sim = p(render_mode="human")
    scene = Scene(sim = sim, object_pos=np.array([0.5, 0.0, 0.0]))
    scene.import_object(mesh_name="YcbPowerDrill", base_position=scene.object_pos, base_orientation=math_utils.euler_to_quaternion(0,0,np.pi/2), scale = (0.75, 0.75, 0.75)) 
    robot = FrankaPandaCam(sim = sim)
    scene.robot.set_robot_transparency(0.8)
    
    scene.robot.set_joint_neutral()

    # Adjsut scan positions based on model complexity
    scan_positions= [
    np.array([0.5, 0.00, 0.45]),   
    np.array([0.3, 0.30, 0.35]),
    np.array([0.45, 0.40, 0.25]),
    np.array([0.625, 0.35, 0.25]),
    np.array([0.65, 0.30, 0.15]),
    np.array([0.675, 0.15, 0.20]),
    np.array([0.70, 0.00, 0.35]),
    np.array([0.675, -0.15, 0.20]),
    np.array([0.65, -0.30, 0.15]),
    np.array([0.625, -0.35, 0.25]),
    np.array([0.40, -0.40, 0.15]),
    np.array([0.20, -0.30, 0.10])
    ]
        
    multi_scan = MultiScanTask(robot, scan_positions)

    logging.info("Starting multi-scan...")
    multi_scan.perform_scan(point_to_cog = True, scene = scene)
    logging.info("Multi-scan completed.")
    
    sequential_reg = PointCloudMerger(scene_instance=scene)
    sequential_reg.pcd_preprocessing()
    
    # pcd_list = []
    # import open3d as o3d
    # for i, pcd in enumerate(sequential_reg.source_down_list): 
    #     pcd.paint_uniform_color(np.array([0.0, (i+1)*0.15, (i+1)*0.15]))
    #     pcd_list.append(pcd)
        
    # o3d.visualization.draw_geometries(pcd_list, point_show_normal = True)

    sequential_reg.hierarchical_registration(sequential_reg.source_pcd_list)
    
    # # Reset robot and close simulation
    # multi_scan.reset_robot()
    # multi_scan.close()

    try:
        while True:
            scene.sim.step()
            scene.sim.render() 
    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user.")
        scene.robot.close()

if __name__ == "__main__":
    main()