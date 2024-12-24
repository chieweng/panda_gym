import numpy as np
from env.franka_env import FrankaPandaCam, Scene
from utils import math_utils
from env.task import MultiScanTask
from utils.pcd_utils import PointCloudMerger
from pyb_class import PyBullet as p
import pybullet

def main():
    sim = p(render_mode="human")
    scene = Scene(sim = sim)
    scene.import_object(mesh_name="YcbPowerDrill", base_position=scene.object_pos, base_orientation=math_utils.euler_to_quaternion(0,0,np.pi/2))
    robot = FrankaPandaCam(sim = sim)
    scene.robot.set_robot_transparency(0.2)
    
    scene.robot.set_joint_neutral()

    scan_positions= [   
    np.array([0.55, 0.00, 0.45]),
    np.array([0.55, 0.40, 0.25]),
    np.array([0.55, -0.40, 0.25]),
    # np.array([0.65, 0.00, 0.25]),
    # np.array([0.45, 0.00, 0.25])
    ]
        
    multi_scan = MultiScanTask(robot, scan_positions)

    print("Starting multi-scan...")
    multi_scan.perform_scan(point_to_cog = True, scene = scene)
    print("Multi-scan completed.")
    
    sequential_reg = PointCloudMerger(scene_instance=scene)
    sequential_reg.pcd_preprocessing()
    
    pcd_list = []
    import open3d as o3d
    for pcd in sequential_reg.source_pcd_list: 
        rand_color = np.random.rand(3)
        pcd.paint_uniform_color(rand_color)
        pcd_list.append(pcd)
    
    o3d.visualization.draw_geometries(pcd_list)

    # sequential_reg.merge_point_clouds()
    
    # # Reset robot and close simulation
    # multi_scan.reset_robot()
    # multi_scan.close()

    try:
        while True:
            scene.sim.step()
            scene.sim.render() 
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        scene.robot.close()

if __name__ == "__main__":
    main()