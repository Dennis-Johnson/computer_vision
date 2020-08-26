import cv2 as cv
import numpy as np


def main():
    # Find object boundaries
    img = cv.imread('./input/ps1-input0.png')
    squares = cv.Canny(img, 100, 200)
    cv.imwrite('./output/ps1-1-a-1.png', squares)

    # Hough transform
    hough_lines = hough_lines_acc(squares)
    lines = cv.imwrite('./test.png', hough_lines)


def hough_lines_acc(img_edges):
    # Taking resolution of rho as 1 pixel and theta as 1 degree

    diagonal = np.sqrt(
        (img_edges.shape[0] - 1) ** 2 + (img_edges.shape[1] - 1) ** 2)

    # rho values range from -diagonal to +diagonal
    rho_axis = int(2 * np.ceil(diagonal) + 1)
    theta_axis = 180
    shape = (rho_axis, theta_axis)
    accumulator = np.zeros((shape))

    for y in range(0, img_edges.shape[0]):
        for x in range(0, img_edges.shape[1]):
            # Check if current point is an edge
            if img_edges[y][x] >= 255:
                # Vote for all lines through x,y
                for theta in range(-90, 90):
                    rho = int(x * np.cos(to_radians(theta)) +
                              y * np.sin(to_radians(theta)))
                    accumulator[rho][theta+90] += 1

    return accumulator


def to_radians(deg):
    return deg * np.pi / 180


def hough_peaks(accumulator):
    return accumulator


if __name__ == '__main__':
    main()
