# problem set 2
import os
import numpy as np
import cv2
from tqdm import trange

# Note: They may need to be scaled/shifted before saving to show results properly


def disparity_ssd(imgL, imgR):

    WINDOW_WIDTH = 19  # Use a window of size WINDOW_WIDTH x WINDOW_WIDTH
    MAX_OFFSET = 10   # Maximum number of neighbouring pixels to check

    # Number of cols and rows on the flanks of the centre pixel
    half_kernel = (WINDOW_WIDTH - 1) // 2

    rows, cols = imgL.shape
    disparity_matrix = np.zeros((rows, cols))

    scale_factor = 1.2

    # Using trange to log the completion percentage
    for i in trange(half_kernel, rows - half_kernel):
        for j in range(half_kernel, cols - half_kernel):
            prev_ssd = 65534
            best_match_index = j

            base_window = imgL[i - half_kernel:i +
                               half_kernel, j - half_kernel:j + half_kernel]

            # Compute SSD between base_window of the left image and all windows
            # along the same line on the right image.
            for k in range(half_kernel, cols - half_kernel):
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
    # read left and right images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0)
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0)

    D_L = disparity_ssd(L, R)
    D_R = disparity_ssd(R, L)

    # Scale original disparity matrices to highlight the differences
    SCALE_FACTOR = 255 / np.max(np.abs(D_L))

    scaledLR = np.abs(D_L) * SCALE_FACTOR
    scaledRL = np.abs(D_R) * SCALE_FACTOR

    cv2.imwrite("./output/ps2-1-a-1.png", scaledLR)
    cv2.imwrite("./output/ps2-1-a-2.png", scaledRL)


if __name__ == "__main__":
    main()
