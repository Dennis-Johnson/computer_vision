# problem set 2
import os
import numpy as np
import cv2
from tqdm import trange

# Note: They may need to be scaled/shifted before saving to show results properly


def disparity_ssd(imgL, imgR):

    WINDOW_WIDTH = 9  # Use a window of size WINDOW_WIDTH x WINDOW_WIDTH
    MAX_OFFSET = 30   # Maximum number of neighbouring pixels to check

    # Number of cols and rows on the flanks of the centre pixel
    half_kernel = (WINDOW_WIDTH - 1) // 2

    rows, cols = imgL.shape
    disparity_matrix = np.zeros((rows, cols))

    # Using trange to log the completion percentage
    for i in trange(half_kernel, rows - half_kernel):
        for j in range(half_kernel, cols - half_kernel):
            prev_ssd = 65534
            best_match_index = j

            base_window = imgL[i - half_kernel:i +
                               half_kernel, j - half_kernel:j + half_kernel]

            # Limit the centers of the right_window to only MAX_OFFSET/2 pixels
            # on the left and right of the base_window's center.
            # Additional hard bounding between the image boundary
            # which is in the range [half_kernel, cols-half_kernel)
            window_center_left_bound = max(j - MAX_OFFSET // 2, half_kernel)
            window_center_right_bound = min(
                j + MAX_OFFSET // 2, cols - half_kernel)

            # Compute SSD between base_window of the left image and all windows
            # along the same line on the right image.
            for k in range(window_center_left_bound, window_center_right_bound):
                right_window = imgR[i - half_kernel:i +
                                    half_kernel, k - half_kernel:k + half_kernel]
                diff = base_window - right_window
                ssd = np.sum(np.square(diff))

                if (ssd < prev_ssd):
                    prev_ssd = ssd
                    best_match_index = k

            disparity_matrix[i][j] = j - best_match_index
    return disparity_matrix


def main():
    # Question 1 -----------------------------------------------------

    L1 = cv2.imread(os.path.join('input', 'pair0-L.png'), 0)
    R1 = cv2.imread(os.path.join('input', 'pair0-R.png'), 0)

    # Compute disparity matrices for left to right and right to left
    D_LR1 = disparity_ssd(L1, R1)
    D_RL1 = disparity_ssd(R1, L1)

    # Scale original disparity matrices to highlight the differences
    SCALE_FACTOR = 255 / np.max(np.abs(D_LR1))

    scaledLR1 = np.abs(D_LR1) * SCALE_FACTOR
    scaledRL1 = np.abs(D_RL1) * SCALE_FACTOR

    cv2.imwrite("./output/ps2-1-a-1.png", scaledLR1)
    cv2.imwrite("./output/ps2-1-a-2.png", scaledRL1)

    # Question 2 -------------------------------------------------------
    """
    # Load images as greyscale --> flag = 0
    L2 = cv2.imread(os.path.join('input', 'pair1-L.png'), 0)
    R2 = cv2.imread(os.path.join('input', 'pair1-R.png'), 0)

    D_LR2 = disparity_ssd(L2, R2)
    D_RL2 = disparity_ssd(R2, L2)

    SCALE_FACTOR = 255 / np.max(np.abs(D_LR2))
    scaledLR2 = np.abs(D_LR2) * SCALE_FACTOR
    scaledRL2 = np.abs(D_RL2) * SCALE_FACTOR

    cv2.imwrite("./output/ps2-1-a-1.png", scaledLR2)
    cv2.imwrite("./output/ps2-1-a-2.png", scaledRL2)
    """


if __name__ == "__main__":
    main()
