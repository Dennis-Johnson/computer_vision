# problem set 2
import os
import numpy as np
import cv2
from tqdm import trange


# TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly

def disparity_ssd(imgL, imgR):
    # Use a window of size WINDOW_WIDTH x WINDOW_WIDTH
    WINDOW_WIDTH = 9

    # Number of cols and rows on the flanks of the centre pixel
    left_out = (WINDOW_WIDTH - 1) // 2

    rows, cols = imgL.shape
    disparity_matrix = np.zeros((rows, cols))

    # Using trange to log the completion percentage
    for i in trange(left_out, rows - left_out):
        for j in range(left_out, cols - left_out):
            base_window = imgL[i - left_out:i +
                               left_out, j - left_out:j + left_out]

            line_disparity = np.zeros(cols)

            # Compute SSD between base_window of the left image and windows
            # along the same line on the right image.
            for k in range(left_out, cols - left_out):
                right_window = imgR[i - left_out:i +
                                    left_out, k - left_out:k + left_out]
                diff = right_window - base_window
                line_disparity[k] = np.sum(np.square(diff))

            disparity_matrix[i][j] = np.argmin(
                line_disparity[left_out:cols - left_out])

            if (disparity_matrix[i][j] != 0):
                print(disparity_matrix[i][j])

    return disparity_matrix


def main():
    # read left and right images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0)
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = disparity_ssd(L, R)
    print(D_L)
    # D_R = disparity_ssd(R, L)

    # cv2.imwrite("dispLR.png", D_L)
    # cv2.imwrite("dispRL.png", D_R)


if __name__ == "__main__":
    main()
