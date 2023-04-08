# Haozhe Ma
# March 2023

import numpy as np
import cv2

# Convert the input image from BGR to HSV color space.
# Define the color ranges for red and yellow in HSV color space.
# Create binary masks for red and yellow colors by filtering the image with the defined color ranges.
# Combine the binary masks for red and yellow using a bitwise OR operation.
# Find contours in the combined mask.
# Perform morphological closing on the mask using an elliptical kernel.
# Find contours in the closed mask.
# Filter the contours based on a minimum area (3000 pixels).
# Iterate through the filtered contours and check if they have more than 4 vertices.(rectangle have 4 vertices)
# For the contours that meet the above condition, draw them on the original image and fit an ellipse around each contour.
#  Display the original image with the drawn contours and the combined mask.
# 14. Save the combined mask and the original image with drawn contours as 'mask.png' and 'original.png', respectively.
# 15. Wait for a key press to close the displayed images.


# Load the input image
image = cv2.imread('/Users/haozhema/Documents/Final year project/IMG_1026.jpg')
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#ask user to input the coordinates of the dots by typing in the terminal
# print("Please input the coordinates: ")
# coordinates_str = input()
# coordinates_str = "[1497, 1719], [1968, 2130], [1930, 1743]"
# coordinates = eval(coordinates_str)
# print(coordinates)
# for x, y in coordinates:
#     cv2.circle(original, (x, y), 5, (0, 0, 255), 100)
                   
# Define the color ranges for red and yellow
lower_red = np.array([0, 100, 100], dtype="uint8")
upper_red = np.array([10, 255, 255], dtype="uint8")
lower_yellow = np.array([20, 100, 100], dtype="uint8")
upper_yellow = np.array([40, 255, 255], dtype="uint8")
lower_blue = np.array([100, 100, 100], dtype="uint8")
upper_blue = np.array([120, 255, 255], dtype="uint8")

# Create binary masks for red and yellow
mask_red = cv2.inRange(image, lower_red, upper_red)
mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
mask_blue = cv2.inRange(image, lower_blue, upper_blue)

# Combine the masks using bitwise OR
mask = cv2.bitwise_or(mask_red, mask_yellow)
#mask = mask_blue

# Find contours
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
print('Number of contours found: {}'.format(len(cnts)))

# Create a kernel for morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
# Perform morphological closing on the contours
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the closed image
cnts_closing = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_closing = cnts_closing[0] if len(cnts_closing) == 2 else cnts_closing[1]
print('Number of contours found after closing: {}'.format(len(cnts_closing)))

# Define the minimum area for a contour to be considered
MIN_AREA = 3000

# Iterate over the contours found in the closed image
filtered_cnts = []
for c in cnts_closing:
    area = cv2.contourArea(c)
    if area >= MIN_AREA:
        filtered_cnts.append(c)

print('Number of contours found after filtering: {}'.format(len(filtered_cnts)))

contour_count = 0
contour_count_closing = 0
# Iterate through contours and filter by the number of vertices
#for c in cnts:
 #   perimeter = cv2.arcLength(c, False)
  #  approx = cv2.approxPolyDP(c, 0.04 * perimeter, False)
   # if len(approx) > 4:
    #    contour_count += 1
     #   cv2.drawContours(original, [c], -1, (36, 255, 12), -1)

#for c in cnts_closing:

for c in filtered_cnts:
    perimeter = cv2.arcLength(c, False)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, False)
    if len(approx) > 4:
        contour_count_closing += 1
        
        #cv2.drawContours(original, [c], -1, (36, 255, 12), -1)
            
        ellipse = cv2.fitEllipse(c)
        rectangle = cv2.minAreaRect(c)
        print(ellipse)
        cv2.ellipse(original, ellipse, (36, 255, 12), 2)
        ellipse_1 = (ellipse[0], (ellipse[1][0]*2.5, ellipse[1][1]*2.5), ellipse[2])
        ellipse_3 = (ellipse[0], (ellipse[1][0]*2, ellipse[1][1]*2), ellipse[2])
        ellipse_4 = (ellipse[0], (ellipse[1][0]*1.5, ellipse[1][1]*1.5), ellipse[2])
        ellipse_8 = (ellipse[0], (ellipse[1][0]*0.5, ellipse[1][1]*0.5), ellipse[2])

        cv2.ellipse(original, ellipse_1, (36, 255, 12), 4)
        cv2.ellipse(original, ellipse_3, (36, 255, 12), 4)
        cv2.ellipse(original, ellipse_4, (36, 255, 12), 4)
        cv2.ellipse(original, ellipse_8, (36, 255, 12), 4)
        cv2.rectangle(original, (int(rectangle[0][0] - rectangle[1][0] / 2), int(rectangle[0][1] - rectangle[1][1] / 2)), (int(rectangle[0][0] + rectangle[1][0] / 2), int(rectangle[0][1] + rectangle[1][1] / 2)), (36, 255, 12), 10)

        # Get the centroid of the ellipse (center of the fitted ellipse)
        cX, cY = int(ellipse[0][0]), int(ellipse[0][1])

        # Draw the centroid on the original image WITH A BLUE DOT.
        #cv2.circle(original, (cX, cY), 5, (255, 0, 0), -1)
        cv2.circle(original, (cX, cY), 5, (255, 0, 0), 20)

# SCORING




print('Number of contours drawn: {}'.format(contour_count))
print('Number of contours drawn after closing: {}'.format(contour_count_closing))

cv2.imshow('mask', mask)
cv2.imshow('original', original)
cv2.imwrite('mask.png', mask)
cv2.imwrite('original.png', original)
cv2.waitKey()