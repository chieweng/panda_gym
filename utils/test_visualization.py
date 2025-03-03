import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

"""MANUALLY SET BOUNDING BOX MIN AND MAX VALUES (bb_min, bb_max) BEFORE RUNNING VISUALIATION"""

# Load points from points_output.txt (assuming each line is x y z)
points = np.loadtxt("points_output.txt", delimiter = " ")

# Randomly select points
num_points_to_select = 5000
random_indices = random.sample(range(len(points)), num_points_to_select)
selected_points = points[random_indices]

# Given bounding box min and max
bb_min = np.array([0.51773, -0.068317, -0.004048])
bb_max = np.array([0.641862, 0.096624, 0.184448])

# Define the 8 corners of the bounding box
corners = np.array([
    [bb_min[0], bb_min[1], bb_min[2]],  # min
    [bb_min[0], bb_min[1], bb_max[2]],  # max Z
    [bb_min[0], bb_max[1], bb_min[2]],  # max Y
    [bb_min[0], bb_max[1], bb_max[2]],  # max Y and Z
    [bb_max[0], bb_min[1], bb_min[2]],  # max X
    [bb_max[0], bb_min[1], bb_max[2]],  # max X and Z
    [bb_max[0], bb_max[1], bb_min[2]],  # max X and Y
    [bb_max[0], bb_max[1], bb_max[2]]   # max (all)
])

# Define the edges connecting the corners of the bounding box
edges = [
    [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
    [4, 5], [5, 7], [7, 6], [6, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
]

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the bounding box
for edge in edges:
    ax.plot3D(*zip(*corners[edge]), color="b")

# Plot the XYZ axes at the origin
ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', label='X-axis')  # Red for X
ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', label='Y-axis')  # Green for Y
ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', label='Z-axis')  # Blue for Z

# Label the bb_min and bb_max points
ax.text(bb_min[0], bb_min[1], bb_min[2], 'bb_min', color='red', fontsize=12)
ax.text(bb_max[0], bb_max[1], bb_max[2], 'bb_max', color='green', fontsize=12)

# Plot the selected random points
ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], color="orange", label="Random Points")

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the limits based on bounding box extents
ax.set_xlim([bb_min[0] - 0.1, bb_max[0] + 0.1])
ax.set_ylim([bb_min[1] - 0.1, bb_max[1] + 0.1])
ax.set_zlim([bb_min[2] - 0.1, bb_max[2] + 0.1])

# Set the viewpoint from the origin
ax.view_init(elev=20, azim=30)

# Title and show
ax.set_title("Bounding Box with XYZ Axes and Random Points")
plt.show()
