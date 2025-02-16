# Assignment-2: Camera Projection and Estimation

## Introduction
This assignment focuses on understanding camera projection using forward projection and the Direct Linear Transform (DLT) method. The tasks involve implementing camera transformations, estimating projection matrices, and analyzing errors.

## Tasks and Solutions

### Task 1: Forward Projection
- **Objective:** Run the provided code with non-coplanar 3D points and small camera rotations (yaw = 3°, pitch = 2°, roll = 5°).
- **Steps:**
  1. Define non-coplanar 3D points.
  2. Apply small rotations.
  3. Generate and visualize the projection results.
- **Results:**
  - Plots of the projected points.
  - Ground truth projection matrix.


### Task 2: Estimating the Projection Matrix using DLT
- **Objective:** Implement the DLT algorithm to estimate the projection matrix.
- **Steps:**
  1. Use the given 3D world points and their corresponding 2D projections.
  2. Apply the DLT algorithm.
  3. Normalize the projection matrix.
  4. Compare the estimated and ground truth matrices.
  5. Discuss and fix any sign differences.

### Task 3: Decomposing the Projection Matrix
- **Objective:** Extract intrinsic and extrinsic parameters from the estimated projection matrix.
- **Steps:**
  1. Decompose the estimated matrix to retrieve:
     - Focal lengths (fx, fy)
     - Principal point (cx, cy)
     - Rotation matrix (R)
     - Translation vector (t)
     - Camera center (C)
  2. Compare the results with the ground truth values.
  3. Verify correctness.

### Task 4: Real-World Camera Parameter Estimation
- **Objective:** Capture real-world data and estimate camera parameters using the DLT method.
- **Steps:**
  1. Mark six points in a corner of a room (spread across three planes).
  2. Measure their 3D world coordinates.
  3. Capture an image and extract the 2D pixel coordinates.
  4. Estimate the camera parameters using the DLT method.
  5. Compare estimated camera position (Cx, Cy, Cz) with actual measurements.

## Summary
This assignment enhances understanding of camera projection techniques by implementing forward projection, the DLT algorithm, and real-world camera calibration. The results and discussions help in analyzing accuracy and errors in estimation.