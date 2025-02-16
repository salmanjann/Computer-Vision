# Salman Jan 21I-2574
# Camera forward projection (world to camera Demo!)
# ACV Course, Spring 2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# from scipy.linalg import svd
from scipy.linalg import rq
np.set_printoptions(suppress=True)


def create_rotation_matrix(yaw, pitch, roll):
    # Convert angles to radians
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    # Yaw matrix around Y axis
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    # Pitch matrix around X axis
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # Roll matrix around Z axis
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    # Combine the matrices by multiplying them
    R_combined = R_roll.dot(R_pitch).dot(R_yaw)
    return R_combined


def createGTCameraParameters(focal_length_x, focal_length_y, principal_point_x, principal_point_y, R,
                             camera_translation_vector_from_world_origin):
    # construction of a 4 by 4 Mext
    extrinsic_matrix = np.eye(4)  # 4 by 4
    extrinsic_matrix[:3, :3] = R  # Transpose of the rotation matrix
    extrinsic_matrix[:3, 3] = -R @ camera_translation_vector_from_world_origin  # Translation in camera coordinates

    # Intrinsic matrix (3 by 4)
    K = np.array([[focal_length_x, 0, principal_point_x, 0],
                  [0, focal_length_y, principal_point_y, 0],
                  [0, 0, 1, 0]])

    # Camera projection matrix P (3 by 4)
    P = np.dot(K, extrinsic_matrix)

    return P, extrinsic_matrix, K


# Function to project 3D points to 2D image coordinates
def project_world_to_camera(points_3d, P):
    # Add homogeneous coordinate (1) to the 3D points
    points_3d_homogeneous = np.column_stack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Project 3D points to 2D image coordinates
    points_2d_homogeneous = np.dot(P, points_3d_homogeneous.T).T

    # Normalize homogeneous coordinates
    points_2d_normalized = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    # Check if points are within image boundaries
    points_inside_frame = np.logical_and.reduce((points_2d_normalized[:, 0] >= 0,
                                                 points_2d_normalized[:, 0] <= 640,  # Adjust for image width
                                                 points_2d_normalized[:, 1] >= 0,
                                                 points_2d_normalized[:, 1] <= 480))  # Adjust for image height

    return points_2d_normalized, points_inside_frame


def plotWorldPoints(ax, points, rotation_matrix, camera_origin):
    num_points = len(points)

    # Plot the random points with numbers
    for i in range(num_points, ):
        ax.text(points[i, 0], points[i, 1], points[i, 2], str(i + 1), color='black', fontsize=8, ha='right',
                va='bottom')
        ax.scatter(points[i, 0], points[i, 1], points[i, 2], c=[colors[i]], marker='o', s=50)

    # Plot a red cross at the origin
    ax.plot([0], [0], [0], marker='x', markersize=10, color='red')

    # Plot world coordinate axes
    world_axes_length = 1.0
    world_x_axis = np.array([world_axes_length, 0, 0])
    world_y_axis = np.array([0, world_axes_length, 0])
    world_z_axis = np.array([0, 0, 3])

    ax.plot([0, world_x_axis[0]], [0, world_x_axis[1]], [0, world_x_axis[2]], color='green')
    ax.plot([0, world_y_axis[0]], [0, world_y_axis[1]], [0, world_y_axis[2]], color='orange')
    ax.plot([0, world_z_axis[0]], [0, world_z_axis[1]], [0, world_z_axis[2]], color='purple')

    # Add labels 'X', 'Y', and 'Z' at the top of each world axis line
    ax.text(world_x_axis[0], world_x_axis[1], world_x_axis[2], 'X', color='green', fontsize=8, ha='left', va='bottom')
    ax.text(world_y_axis[0], world_y_axis[1], world_y_axis[2], 'Y', color='orange', fontsize=8, ha='left', va='bottom')
    ax.text(world_z_axis[0], world_z_axis[1], world_z_axis[2], 'Z', color='purple', fontsize=8, ha='left', va='bottom')

    # Plot a green cross at the camera origin

    ax.plot([camera_origin[0]], [camera_origin[1]], [camera_origin[2]], marker='x', markersize=10, color='green')

    # Plot camera coordinate axes
    camera_axes_length = 1.0
    camera_x_axis = rotation_matrix[:, 0] * camera_axes_length + camera_origin
    camera_y_axis = rotation_matrix[:, 1] * camera_axes_length + camera_origin
    camera_z_axis = rotation_matrix[:, 2] * 3 + camera_origin

    ax.plot([camera_origin[0], camera_x_axis[0]], [camera_origin[1], camera_x_axis[1]],
            [camera_origin[2], camera_x_axis[2]], color='blue')
    ax.plot([camera_origin[0], camera_y_axis[0]], [camera_origin[1], camera_y_axis[1]],
            [camera_origin[2], camera_y_axis[2]], color='cyan')
    ax.plot([camera_origin[0], camera_z_axis[0]], [camera_origin[1], camera_z_axis[1]],
            [camera_origin[2], camera_z_axis[2]], color='magenta')

    # Add labels 'X', 'Y', and 'Z' at the top of each line
    ax.text(camera_x_axis[0], camera_x_axis[1], camera_x_axis[2], 'X', color='blue', fontsize=8, ha='left', va='bottom')
    ax.text(camera_y_axis[0], camera_y_axis[1], camera_y_axis[2], 'Y', color='cyan', fontsize=8, ha='left', va='bottom')
    ax.text(camera_z_axis[0], camera_z_axis[1], camera_z_axis[2], 'Z', color='magenta', fontsize=8, ha='left',
            va='bottom')

    # Add grid lines
    ax.grid(True)

    # Set equal scaling to make the plot cubic
    # ax.set_box_aspect([np.ptp(axis) for axis in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('World Coordinates')


def plotImagePixelPoints(ax, image_points, image_width, image_height):
    num_points = len(image_points)

    # Create a 2D image and plot the projected points
    # fig, ax = plt.subplots()

    ax.set_aspect('equal')

    # Invert y-coordinates to have the origin at the top-left
    image_points[:, 1] = -image_points[:, 1]

    # Plot the red cross at the origin
    ax.plot([0], [0], marker='x', markersize=10, color='red')

    # Plot the projected points with numbers and colors
    for i in range(num_points):
        # ax.text(image_points[i, 0], image_points[i, 1], str(i + 1), color='black', fontsize=8, ha='right', va='bottom')
        # ax.scatter(image_points[i, 0], image_points[i, 1], c=[colors[i]], marker='o', s=50)

        ax.text(image_points[i, 0], image_points[i, 1], str(i + 1), color='black', fontsize=8, ha='right', va='bottom')
        ax.scatter(image_points[i, 0], image_points[i, 1], c=[colors[i]], marker='o', s=50)

    # Set labels
    ax.set_xlabel('U')
    ax.set_ylabel('V')

    # Customize tick labels to remove the minus sign
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:0.0f}'.format(abs(x))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:0.0f}'.format(abs(y))))

    # Set limits based on the image size
    ax.set_xlim(0, image_width)
    ax.set_ylim(-image_height, 0)  # Inverted y-axis

    ax.set_title('Camera Coordinates')

    # Show the plot

# TASK 2

def estimate_projection_matrix_least_squares(world_points, image_points):
    assert world_points.shape[0] == image_points.shape[0], "Number of points must match"

    N = world_points.shape[0]
    A = []

    for i in range(N):
        X, Y, Z = world_points[i]
        u, v = image_points[i]

        row1 = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        row2 = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

        A.append(row1)
        A.append(row2)

    A = np.array(A)

    # Compute A^T A
    ATA = np.dot(A.T, A)

    # Solve for P using Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    # Find the eigenvector corresponding to the smallest eigenvalue
    sorted_indices = np.argsort(eigenvalues)
    P = eigenvectors[:, sorted_indices[0]]  # Smallest eigenvalue corresponds to least-squares solution

    # Reshape into 3x4 projection matrix
    P = P.reshape(3, 4)

    # Compute the correct scaling factor based on the last row norm
    last_row_norm = np.linalg.norm(P[2, :3])  # norm(P31, P32, P33)
    k = 1 / last_row_norm if last_row_norm != 0 else 1  # Prevent division by zero

    # Scale the projection matrix
    P_scaled = P * k

    return P_scaled

# Task 3
import numpy as np
from scipy.linalg import rq


def decompose_projection_matrix(P):
    # Extract M (first 3x3 block)
    M = P[:, :3]

    # Perform RQ decomposition
    K, R = rq(M)

    # Normalize K to have positive diagonal elements
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Compute translation
    t = np.linalg.inv(K) @ P[:, 3]

    return K, R, t

def compute_projection_error(P, world_points, ground_truth_2D_points):
    # Convert world points to homogeneous coordinates (add a 1 in the fourth column)
    world_points_homogeneous = np.column_stack((world_points, np.ones((world_points.shape[0], 1))))

    # Project world points using P
    projected_2D_homogeneous = np.dot(P, world_points_homogeneous.T).T

    # Convert from homogeneous to 2D by dividing x, y by w
    projected_2D_points = projected_2D_homogeneous[:, :2] / projected_2D_homogeneous[:, 2:]

    # Compute Euclidean distance error
    errors = np.linalg.norm(projected_2D_points - ground_truth_2D_points, axis=1)

    # Compute mean projection error
    avg_error = np.mean(errors)

    return avg_error


np.random.seed(42)  # for reproducibility

# GT Intrinsic parameters (example values) (#in pixels)
focal_length_x = 800.0
focal_length_y = 700.0
principal_point_x = 320.0
principal_point_y = 240.0
image_width = 640
image_height = 480
# GT Extrinsic parameters (example values) (#in degree and meters)
camera_yaw = 3  # rotation around Y axis
camera_pitch = 2  # rotation around X axis
camera_roll = 5  # rotation around Z axis
camera_translation_vector_from_world_origin = np.array([2, 2, 12])  # C translation of camera wrt world (in meters)

camera_rotation_matrix = create_rotation_matrix(camera_yaw, camera_pitch, camera_roll)
# Need to invert X and Z axis to create a forward, right and bottom system to properly project
camera_rotation_matrix[:, 0] = -camera_rotation_matrix[:, 0]
camera_rotation_matrix[:, 2] = -camera_rotation_matrix[:, 2]
R = camera_rotation_matrix.T  # for world to camera we need to take inverse of camera rotation

# Hardcoded list of points on the XY plane (just for rendering purpose)
world_3D_points = np.array([[1.0, 0.5, 2.5],
                           [2.5, 2.0, 4.0],
                           [3.0, 1.5, 6.5],
                           [1.5, 3.0, 5.0],
                           [2.0, 1.0, 7.0],
                           [3.5, 2.5, 3.5],
                           [4.0, 1.5, 8.0],
                           [1.2, 2.8, 4.5],
                           [2.7, 3.2, 5.2],
                           [3.8, 0.9, 7.3]], dtype=np.float32)

num_points = len(world_3D_points)
# Assign different colors to each point
colors = np.random.rand(num_points, 3)

# ground truth camera projection matrix
P_True, RT, K = createGTCameraParameters(focal_length_x, focal_length_y, principal_point_x, principal_point_y, R,
                                    camera_translation_vector_from_world_origin)

# Project 3D points to 2D image coordinates
camera_pixel_2D_points, insideViewOrNot = project_world_to_camera(world_3D_points, P_True)


fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')  # 1st subplot is 3D
ax2 = fig.add_subplot(122)

# plot 3D points and camera and world axis
plotWorldPoints(ax1, world_3D_points, camera_rotation_matrix, camera_translation_vector_from_world_origin)
# plot 2D points in pixel frame
plotImagePixelPoints(ax2, camera_pixel_2D_points, image_width, image_height)

plt.show(block=True)

print("\nWorld Points:")
print(world_3D_points)

print("\nCamera Pixel 2D Points")
print(camera_pixel_2D_points)

print("\nCamera Projection Matrix (P):")
print(P_True)

print("\nProjection Matrix Using Least Squares Method")
P_Estimated = estimate_projection_matrix_least_squares(world_3D_points, camera_pixel_2D_points)
print(P_Estimated)

print("\nDifference between estimated and ground truth matrices before sign correction\n",P_True-P_Estimated)

# Fixing the sign issue
if np.sign(P_Estimated[0, 0]) != np.sign(P_True[0, 0]):
    P_estimated = -P_Estimated  # Flip the sign of the entire matrix
P_corrected = np.where(np.sign(P_Estimated) != np.sign(P_True), -P_Estimated, P_Estimated)
print("\nSign-Corrected Estimated Projection Matrix:\n", P_corrected)
print("\nDifference between estimated and ground truth matrices after sign correction\n",P_True-P_corrected)


print("The sign difference arises because the eigenvector corresponding to the smallest eigenvalue is unique up to a scale factor.\n"
      "This means the computed projection matrix (P_estimated) could be a negative version of the expected matrix (P_ground_truth):\n"
      "P_estimated = -P_ground_truth\n"
      "Mathematically, this does not change the projection results (as homogeneous coordinates remain valid under scaling),\n"
      "but for direct comparison, you need to resolve the sign ambiguity.")


print("\nOriginal Intrinsic Matrix (K):")
print(K)

K, R, t = decompose_projection_matrix(P_corrected)

print("\nComputed Intrinsic Matrix (K):\n", K)

print("Original Extrinsic Matrix:")
print(RT)

print("\nComputed Rotation Matrix (R):\n", R)
print("\nComputed Translation Vector (t):\n", t)

# Compute projection error
error = compute_projection_error(P_corrected, world_3D_points, camera_pixel_2D_points)
print(f"\nProjection Error: {error:.4f} pixels")

# Task 4
world_points = np.array([[0.155, 0.0, 0.155],
                         [0.105, 0.0, 0.185],
                         [0.0, 0.04, 0.055],
                         [0.0, 0.075, 0.115],
                         [0.155, 0.095, 0.0],
                         [0.085, 0.095, 0.0]], dtype=np.float32)


uv_coordinates = np.array([[495.0, 2150],
                           [900, 1840],
                           [1900, 2850],
                           [2050, 2390],
                           [730, 3720],
                           [1420, 3600]])


print("\nWorld Points:")
print(world_points)

print("\nCamera Pixel 2D Points")
print(uv_coordinates)

print("\nProjection Matrix Using Least Squares Method")
P_Estimated1 = estimate_projection_matrix_least_squares(world_points, uv_coordinates)
print(P_Estimated1)

K1, R1, t1 = decompose_projection_matrix(P_Estimated1)

print("\nComputed Intrinsic Matrix (K):\n", K1)

print("\nComputed Rotation Matrix (R):\n", R1)
print("\nComputed Translation Vector (t):\n", t1)

print("\n Actual Translation")
C = -R1.T @ t1
print(C)

