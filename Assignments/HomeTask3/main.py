import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images
image1 = cv2.imread('stitch0.jpg')
image2 = cv2.imread('stitch1.jpg')
image3 = cv2.imread('stitch2.jpg')

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# Create SIFT object
sift = cv2.SIFT_create()

# Detect and compute descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

# Match keypoints
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

MIN_MATCH_COUNT = 10
if len(good_matches) >= MIN_MATCH_COUNT:
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    homography, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)

    # Calculate displacement and overlap for first stitch
    displacements = dst_pts[:, 0] - src_pts[:, 0]
    overlap_width1 = np.mean(displacements)
    width = int(image1.shape[1] + image2.shape[1] - int(abs(overlap_width1)))

    # Warp image
    warped_image1 = cv2.warpPerspective(image2, homography, (image1.shape[1] + image2.shape[1], image1.shape[0]))

    # Create a copy for the stitched result
    stitched = warped_image1.copy()

    # Calculate the overlapping region for first stitch
    overlap_start1 = max(0, image1.shape[1] - int(abs(overlap_width1)))
    overlap_end1 = min(image1.shape[1], stitched.shape[1])
    overlap_width1 = overlap_end1 - overlap_start1

    # Apply blending for first stitch
    if overlap_width1 > 0:
        # First copy the non-overlapping part of image1
        stitched[:image1.shape[0], :overlap_start1] = image1[:, :overlap_start1]

        # Handle the overlapping region with weighted blending
        for y in range(image1.shape[0]):
            for x in range(overlap_start1, overlap_end1):
                # Calculate weight based on position in overlap region
                alpha = (x - overlap_start1) / overlap_width1
                # Apply weighted average blending
                stitched[y, x] = (1 - alpha) * image1[y, x] + alpha * warped_image1[y, x]

        # Copy the non-overlapping part of the warped image
        if overlap_end1 < warped_image1.shape[1]:
            stitched[:image1.shape[0], overlap_end1:warped_image1.shape[1]] = warped_image1[:image1.shape[0],
                                                                              overlap_end1:warped_image1.shape[1]]
    else:
        # If no actual overlap, just copy the first image
        stitched[:image1.shape[0], :image1.shape[1]] = image1

    # Crop stitched image to remove black areas
    gray_stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_stitched, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        first_stitched_cropped = stitched[y:y + h, x:x + w]
    else:
        first_stitched_cropped = stitched[:, :width]

    # Convert to ensure proper format for SIFT
    first_stitched_cropped = first_stitched_cropped.astype(np.uint8)

    # Repeat for third image
    gray_stitched = cv2.cvtColor(first_stitched_cropped, cv2.COLOR_BGR2GRAY)
    kp_stitched, des_stitched = sift.detectAndCompute(gray_stitched, None)

    # Match with the third image
    matches_final = bf.knnMatch(des_stitched, des3, k=2)
    good_matches_final = [m for m, n in matches_final if m.distance < 0.75 * n.distance]

    if len(good_matches_final) >= MIN_MATCH_COUNT:
        # Extract matched keypoints for the final stitch
        src_pts_2 = np.float32([kp_stitched[m.queryIdx].pt for m in good_matches_final]).reshape(-1, 1, 2)
        dst_pts_2 = np.float32([kp3[m.trainIdx].pt for m in good_matches_final]).reshape(-1, 1, 2)

        # Calculate displacement for second stitch
        displacements2 = dst_pts_2[:, 0] - src_pts_2[:, 0]
        overlap_width2 = np.mean(displacements2)
        width_final = int(first_stitched_cropped.shape[1] + image3.shape[1] - int(abs(overlap_width2)))

        # Find homography for third image
        homography2, _ = cv2.findHomography(dst_pts_2, src_pts_2, cv2.RANSAC, 3.0)

        # Warp third image
        warped_image2 = cv2.warpPerspective(image3, homography2,
                                            (first_stitched_cropped.shape[1] + image3.shape[1],
                                             max(first_stitched_cropped.shape[0], image3.shape[0])))

        # Create a copy for the final stitched result
        final_stitched = warped_image2.copy()

        # Calculate the overlapping region for second stitch
        overlap_start2 = max(0, first_stitched_cropped.shape[1] - int(abs(overlap_width2)))
        overlap_end2 = min(first_stitched_cropped.shape[1], final_stitched.shape[1])
        overlap_width2 = overlap_end2 - overlap_start2

        # Apply blending for second stitch
        if overlap_width2 > 0:
            # First copy the non-overlapping part of first stitched image
            final_stitched[:first_stitched_cropped.shape[0], :overlap_start2] = first_stitched_cropped[:,
                                                                                :overlap_start2]

            # Handle the overlapping region with weighted blending
            for y in range(first_stitched_cropped.shape[0]):
                for x in range(overlap_start2, overlap_end2):
                    if y < first_stitched_cropped.shape[0] and x < first_stitched_cropped.shape[1]:
                        # Calculate weight based on position in overlap region
                        alpha = (x - overlap_start2) / overlap_width2
                        # Apply weighted average blending
                        final_stitched[y, x] = (1 - alpha) * first_stitched_cropped[y, x] + alpha * warped_image2[y, x]

            # Copy the non-overlapping part of the warped image
            if overlap_end2 < warped_image2.shape[1]:
                valid_height = min(first_stitched_cropped.shape[0], warped_image2.shape[0])
                final_stitched[:valid_height, overlap_end2:warped_image2.shape[1]] = warped_image2[:valid_height,
                                                                                     overlap_end2:warped_image2.shape[
                                                                                         1]]
        else:
            # If no actual overlap, just copy the first stitched image
            final_stitched[:first_stitched_cropped.shape[0], :first_stitched_cropped.shape[1]] = first_stitched_cropped

        # Crop final stitched image to remove black areas
        gray_final = cv2.cvtColor(final_stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_final, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            final_result = final_stitched[y:y + h, x:x + w]
        else:
            final_result = final_stitched[:, :width_final]

        # Display result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
        plt.title("Final Stitched Image with Improved Blending")
        plt.axis("off")
        plt.show()

        # Save the final image
        cv2.imwrite('panorama_result.jpg', final_result)

        print("Stitching completed successfully!")
    else:
        print("Not enough matches found for final stitching -", len(good_matches_final), "/", MIN_MATCH_COUNT)
else:
    print("Not enough matches found for first stitching -", len(good_matches), "/", MIN_MATCH_COUNT)