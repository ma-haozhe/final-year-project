# draw concentric cirlces illustrating the target ring
import cv2
import numpy as np

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Draw concentric circles
center = (256, 256)
color = (36, 255, 12)
thickness = 2
radius_increment = 40

for i in range(5):
    radius = (i + 1) * radius_increment
    img = cv2.circle(img, center, radius, color, thickness)

# Draw the center dot
img = cv2.circle(img, center, 2, color, -1)

# Display the image
cv2.imshow('target', img)
cv2.waitKey(0)
cv2.destroyAllWindows()