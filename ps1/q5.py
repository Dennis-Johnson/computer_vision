import cv2 as cv
import numpy as np


def hough_circles_acc(img_edges, radius):
    # Resolution of a, b is 1 pixel
    # Accumulator with the images' dimensions
    height, width = img_edges.shape
    accumulator = np.zeros((height + 10, width + 10), dtype=np.uint8)

    # Indices of edge pixels in the image (white pixels)
    y_indx, x_indx = np.where(img_edges >= 255)

    for i in range(len(x_indx)):
        x = x_indx[i]
        y = y_indx[i]

        # Vote for the points of the circle in hough space which has (x, y) as its center in cartesian space
        coordinates = get_circle_coordinates(x, y, radius, img_edges.shape)
        for point in coordinates:
            point_x = int(point[0])
            point_y = int(point[1])

            print((point_x, point_y))
            accumulator[point_x][point_y] += 1

    return accumulator


def get_circle_coordinates(a, b, radius, boundaries):
    # Returns list of x,y coordinates that satisfy the circle equation for the given parameters
    coordinates = np.zeros((360, 2))

    for theta in range(0, 360):
        x = a + round(radius * np.cos(theta))
        y = b + round(radius * np.sin(theta))

        # Ignore the points outside the image boundary
        if x < 0 or x > boundaries[0] or y < 0 or y > boundaries[1]:
            continue

        coordinates[theta][0] = x
        coordinates[theta][1] = y

    return coordinates


def hough_circles(coordinates, radius, img, img_name):
    for i in range(len(coordinates)):
        x = coordinates[i][0]
        y = coordinates[i][1]

        cv.circle(img, (x, y), radius, (255, 255, 0))
    cv.imwrite(img_name, img)


def main():
    # Find object boundaries
    img = cv.rotate(cv.imread('./input/ps1-input1.png'), cv.ROTATE_180)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Blurred with SigmaX = SigmaY = 7
    smoothed = cv.GaussianBlur(grey, (7, 7), 0)
    cv.imwrite('./output/ps1-5-a-1.png', smoothed)

    edge_img = cv.Canny(smoothed, 100, 200)
    cv.imwrite('./output/ps1-5-a-2.png', edge_img)

    # Compute the hough accumulator for the edge image
    radius = 20
    accumulator = hough_circles_acc(edge_img, radius)
    # cv.imwrite('circles.png', accumulator)

    # Find peaks in the accumulator with votes more than the lower threshold
    peaks = np.where(accumulator > 100)
    coordinates = list(zip(peaks[0], peaks[1]))

    # Draw yellow circles over probable circles
    hough_circles(coordinates, radius, grey, './output/ps1-5-a-3.png')


if __name__ == '__main__':
    main()
