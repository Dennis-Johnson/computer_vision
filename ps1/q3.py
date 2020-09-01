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


def highlight_hough_peaks(coordinates, accumulator):

    for i in range(len(coordinates)):
        y = coordinates[i][0]
        x = coordinates[i][1]

        cv.rectangle(accumulator, (x-2, y-2), (x+2, y+2), (0, 255, 255))

    cv.imwrite("./output/ps1-2-b-1.png", accumulator)


def hough_lines(coordinates):
    img = cv.imread('./input/ps1-input0.png')
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

    # Write final image with lines highlighted in yellow
    cv.imwrite('./output/ps1-3-c-2.png', img)


def main():
    # Find object boundaries
    img = cv.imread('./input/ps1-input0-noise.png')

    # Blurred with SigmaX = SigmaY = 5
    blurred = cv.GaussianBlur(img, (13, 13), 0)
    cv.imwrite('./output/ps1-3-a-1.png', blurred)

    edges_original = cv.Canny(img, 100, 200)
    edges_blurred = cv.Canny(blurred, 60, 90)

    cv.imwrite('./output/ps1-3-b-1.png', edges_original)
    cv.imwrite('./output/ps1-3-b-2.png', edges_blurred)

    # Compute the hough accumulator for the edge image
    accumulator = hough_acc(edges_blurred)

    # Find peaks in the accumulator with votes more than the lower threshold
    peaks = np.where(accumulator > 90)
    coordinates = list(zip(peaks[0], peaks[1]))

    # Draw yellow boxes over peaks in the accumulator image
    highlight_hough_peaks(coordinates, accumulator)

    # Draw lines over the original image
    hough_lines(coordinates)


if __name__ == '__main__':
    main()
