import cv2
import numpy as np

img1 = cv2.imread('./input/k.jpg')
img2 = cv2.imread('./input/holi.jpg')

# 1 ---------------------------------------
cv2.imwrite("./output/ps0-1-a-1.jpg", img1)
cv2.imwrite("./output/ps0-1-a-2.jpg", img2)

# 2 ---------------------------------------
# Swap
holi_blue, holi_green, holi_red = cv2.split(img2)
swapped = cv2.merge((holi_red, holi_green, holi_blue))
cv2.imwrite("./output/ps0-2-a-1.png", swapped)

# Green Channel
cv2.imwrite("./output/ps0-2-b-1.png", img2[:, :, 1])

# Red Channel
cv2.imwrite("./output/ps0-2-c-1.png", img2[:, :, 2])


# 3 ---------------------------------------
# Replacement
center_one = img2[:, :, 0]
height, width = center_one.shape[:2]
x = int((width - 100) / 2)
y = int((height - 100) / 2)

center_one = center_one[x:x + 100, y:y + 100]
center_two = holi_green.copy()
center_two[x:x + 100, y:y + 100] = center_one
cv2.imwrite("ps0-3-a-1.png", center_two)
