import numpy as np
import cv2

# This code loads an image and converts it to the HSV color space.
# Then, it sets a range of colors to detect (in this case, yellow)
# and creates a binary mask that highlights the parts of the image that fall within that color range.
# The code then finds the contours in the mask and filters them based on the number of vertices.
# Finally, it draws the contours that meet the criteria on the original image and displays both
# the mask and the original image with the contours drawn.
# The resulting images are saved as "mask.png" and "original.png".

# Load the input image
image = cv2.imread('60cm/272.jpeg')
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# blue lower[100,50,50] upper[130,255,255]
# yellow lower[20,100,100] upper[40,255,255]
# red lower[0,50,50] upper[10,255,255]
# red is good.

# Define the color ranges for red and yellow
lower_red = np.array([0, 100, 100], dtype="uint8")
upper_red = np.array([10, 255, 255], dtype="uint8")
lower_yellow = np.array([20, 100, 100], dtype="uint8")
upper_yellow = np.array([40, 255, 255], dtype="uint8")

# Create binary masks for red and yellow
mask_red = cv2.inRange(image, lower_red, upper_red)
mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)

# Combine the masks using bitwise OR
mask = cv2.bitwise_or(mask_red, mask_yellow)

# Find contours
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Extract contours depending on OpenCV version
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
print('Number of contours found: {}'.format(len(cnts)))

# Create a kernel for morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
# Perform morphological closing on the contours
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the closed image
cnts_closing = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Extract contours depending on OpenCV version
cnts_closing = cnts_closing[0] if len(cnts_closing) == 2 else cnts_closing[1]
print('Number of contours found after closing: {}'.format(len(cnts_closing)))

contour_count = 0
contour_count_closing = 0
# Iterate through contours and filter by the number of vertices
for c in cnts:
    perimeter = cv2.arcLength(c, False)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, False)
    if len(approx) > 4:
        contour_count += 1
        cv2.drawContours(original, [c], -1, (36, 255, 12), -1)

for c in cnts_closing:
    perimeter = cv2.arcLength(c, False)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, False)
    if len(approx) > 4:
        contour_count_closing += 1
        cv2.drawContours(original, [c], -1, (36, 255, 12), -1)

print('Number of contours drawn: {}'.format(contour_count))
print('Number of contours drawn after closing: {}'.format(contour_count_closing))

cv2.imshow('mask', mask)
cv2.imshow('original', original)
cv2.imwrite('mask.png', mask)
cv2.imwrite('original.png', original)
cv2.waitKey()