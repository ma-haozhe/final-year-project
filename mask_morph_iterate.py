# this file is for batch checking if the detection is correct. 
# not the main developing file. 

import numpy as np
import cv2
import os
import time

input_directory = '60cm/'
output_directory = 'output/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for i in range(1, 286):
    # Load the input image
    image = cv2.imread(input_directory + str(i) + '.jpeg')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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

    # change between these two lines to control detection using blue or red/yellow
    mask = cv2.bitwise_or(mask_red, mask_yellow)
    #mask = mask_blue

    # Find contours
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Create a kernel for morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # Perform morphological closing on the contours
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed image
    cnts_closing = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_closing = cnts_closing[0] if len(cnts_closing) == 2 else cnts_closing[1]

    # Define the minimum area for a contour to be considered
    MIN_AREA = 3000

    # Iterate over the contours found in the closed image
    filtered_cnts = []
    for c in cnts_closing:
        area = cv2.contourArea(c)
        if area >= MIN_AREA:
            filtered_cnts.append(c)

    # Iterate through filtered contours
    for c in filtered_cnts:
        perimeter = cv2.arcLength(c, False)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, False)
        if len(approx) > 4:
            
            cv2.drawContours(original, [c], -1, (36, 255, 12), -1)
            
            ellipse = cv2.fitEllipse(c)
            rectangle = cv2.minAreaRect(c)
            print(ellipse)
            cv2.ellipse(original, ellipse, (36, 255, 12), 2)
            cv2.rectangle(original, (int(rectangle[0][0] - rectangle[1][0] / 2), int(rectangle[0][1] - rectangle[1][1] / 2)), (int(rectangle[0][0] + rectangle[1][0] / 2), int(rectangle[0][1] + rectangle[1][1] / 2)), (36, 255, 12), 2)

            # Get the centroid of the ellipse (center of the fitted ellipse)
            cX, cY = int(ellipse[0][0]), int(ellipse[0][1])

            # Draw the centroid on the original image
            cv2.circle(original, (cX, cY), 5, (255, 0, 0), -1)

    # Save the combined mask and the original image with drawn contours
    #cv2.imwrite(output_directory + 'mask_' + str(i) + '.png', mask)
    #cv2.imwrite(output_directory + 'original_' + str(i) + '.png', original)

    # Display the combined mask and the original image with drawn contours
    cv2.imshow('mask', mask)
    cv2.imshow('original', original)

    # Show result in 5 seconds interval
    cv2.waitKey(3000)

cv2.destroyAllWindows()