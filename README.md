# Tabletop Robotic Manipulation using Composite AI techniques

## Overview
This project utilizes PyBullet to simulate a robotic arm equipped with a Intel RealSense L515 depth camera for multi-angle scanning of an object. 
The captured point clouds are registered using a hierarchical approach, leveraging global and local registration techniques to reconstruct a complete 3D point cloud of the object. 
For benchmarking purposes, models such as GPD and PointNet will be explored, but emphasis will be placed on transformer-based models due to their superior feature extraction capabilities and potential for improved performance in point cloud processing and grasp-pose detection.

## Features
- PyBullet Simulation: Uses a Franka Emika Panda robot with a mounted camera to capture depth images from multiple angles.

- Point Cloud Processing: Converts depth images into point clouds for further processing.

- Hierarchical Registration: Implements a pairwise merging method using RANSAC and ICP for accurate point cloud alignment.

- Parallel Processing: Supports multi-threaded registration to improve overall computational efficiency.

- Debugging & Visualization: Provides logging, error handling, and visualization of point clouds at each step.

## Contact
For questions or collaborations, feel free to reach out to Chie Weng: Email: chiekencoop@gmail.com