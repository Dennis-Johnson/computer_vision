import cv2 as cv
import numpy as np


def hough_acc(img_edges):
    # Taking resolution of rho as 1 pixel and theta as 1 degree
    height, width = img_edges.shape
    diagonal = np.sqrt(
        height ** 2 + width ** 2)

    # rho values range from  0 to 2 * diagonal length
    rho_axis = int(2 * np.ceil(diagonal))
    thetas = np.deg2rad(np.arange(0, 180, 1))
    theta_axis = len(thetas)

    # Compute cos and sin values for the range of thetas
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    accumulator = np.zeros((rho_axis, theta_axis), dtype=np.uint8)

    # Indices of edge pixels in the image (white pixels)
    y_indx, x_indx = np.where(img_edges >= 255)

    for i in range(len(x_indx)):
        x = x_indx[i]
        y = y_indx[i]

        for theta_indx in range(len(thetas)):
            rho = int(x * cos_t[theta_indx] +
                      y * sin_t[theta_indx])

            accumulator[rho][theta_indx] += 1
    return accumulator


def hough_lines(coordinates, img, img_path):
    for i in range(len(coordinates)):
        rho = coordinates[i][0]
        theta = np.deg2rad(coordinates[i][1])

        cos = np.cos(theta)
        sin = np.sin(theta)

        x0 = rho * cos
        y0 = rho * sin
        x1 = int(x0 + 1000*(-sin))
        y1 = int(y0 + 1000*(cos))
        x2 = int(x0 - 1000*(-sin))
        y2 = int(y0 - 1000 * (cos))

        cv.line(img, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)

    # Write final image with lines highlighted
    cv.imwrite('./output/' + img_path, img)


def main():
    # Find object boundaries
    img = cv.imread('./input/ps1-input2.png')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Blurred with SigmaX = SigmaY = 7
    blurred = cv.GaussianBlur(grey, (7, 7), 0)
    edge_img = cv.Canny(blurred, 60, 200)

    # Compute the hough accumulator for the edge image
    accumulator = hough_acc(edge_img)

    # Find peaks in the accumulator with votes more than the lower threshold
    peaks = np.where(accumulator > 110)
    coordinates = list(zip(peaks[0], peaks[1]))

    # Draw lines over the smoothed image
    hough_lines(coordinates, blurred, "ps1-6-a-1.png")


if __name__ == '__main__':
    main()
