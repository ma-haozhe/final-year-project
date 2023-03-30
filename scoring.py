# Haozhe Ma 
# March 2023

# This script run after detection and deskewing for the original image. 
# deskewing and detection should transform the same way to avoid messing up the coordinates. 

# two scenarios: we use blue or red/yellow for countour detection.


import csv
import numpy as np
import cv2
# functions from mask_and_morph_update_func.py
import mask_and_morph_update_func as mmuf


#read image file for testing
image_path = '60cm/27.jpeg'

# TESTING MODE, WE USE GROUNDTHROUGH FOR SCORING
#read file from groundtruth (gt_72.csv)
def read_gt(image_path):
    #read groundtruth file
    gt_file = open('gt_72.csv', 'r')
    gt_reader = csv.reader(gt_file)
    gt_list = list(gt_reader)
    #number of points in groundtruth
    num_points = 0
    #coordinates of points in groundtruth
    coords = []
    #read the csv and find the row that has the matching image name in first column
    for row in gt_list:
        if row[0] == image_path:
            gt_row = row
            num_points = gt_row[1]
            print(num_points, 'points in this image.')
            coords = gt_row[2]
            print(coords)
            break
    gt_file.close()
    return num_points, coords


def check_contour_within_ellipse(ellipse_outer, coords):


    threshold=0.75

def draw_filtered_contours(image, contours):
    """
    Iterate through the filtered contours and draw them on the input image.
    :param image: The input image to draw the contours on.
    :param contours: The list of filtered contours.
    :return: The input image with drawn contours.
    """
    for c in contours:
        perimeter = cv2.arcLength(c, False)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, False)
        ellipses=[]
        if len(approx) > 4:
            ellipse = cv2.fitEllipse(c)
            ellipses.append(ellipse)
            # COMMENT OUT THIS LINE TO REMOVE DRAWING CONTOURS ON ORIGINAL IMAGE.
            #cv2.drawContours(image, [c], -1, (36, 255, 12), -1)

            cv2.ellipse(image, ellipse, (36, 255, 12), 2)
            #expand elippse diameter to 2.5 times
            ellipse_outer = (ellipse[0], (ellipse[1][0]*2.5, ellipse[1][1]*2.5), ellipse[2])
            cv2.ellipse(image, ellipse_outer, (36, 255, 12), 2)
            # Get the centroid of the ellipse (center of the fitted ellipse)
            cX, cY = int(ellipse[0][0]), int(ellipse[0][1])
            # Draw the centroid on the original image
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
    return image, ellipses


def mask_and_morph(image_path):
    image = cv2.imread(image_path)
    original = image.copy()
    hsv_image = mmuf.preprocess_image(image_path)

    # Define the color ranges for red and yellow
    lower_red = np.array([0, 100, 100], dtype="uint8")
    upper_red = np.array([10, 255, 255], dtype="uint8")
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([40, 255, 255], dtype="uint8")
    lower_blue = np.array([100, 100, 100], dtype="uint8")
    upper_blue = np.array([120, 255, 255], dtype="uint8")

    # Create binary masks for red and yellow
    mask_red = mmuf.create_color_masks(hsv_image, lower_red, upper_red)
    mask_yellow = mmuf.create_color_masks(hsv_image, lower_yellow, upper_yellow)

    # Combine the masks using bitwise OR
    mask = mmuf.combine_masks(mask_red, mask_yellow)

    # Find contours
    cnts = mmuf.find_contours(mask)

    # Perform morphological closing on the mask
    closing = mmuf.morphological_closing(mask)

    # Find contours in the closed mask
    cnts_closing = mmuf.find_contours(closing)

    # Filter the contours based on a minimum area
    filtered_cnts = mmuf.filter_contours_by_area(cnts_closing)

    # Draw the filtered contours on the original image
    original_with_contours = mmuf.draw_filtered_contours(original, filtered_cnts)[0]
    ellipse = draw_filtered_contours(original, filtered_cnts)[1]
    print('the ellipse is ', ellipse)
    ellipse_outer = (ellipse[0], (ellipse[1][0]*2.5, ellipse[1][1]*2.5), ellipse[2])
    #if equal or more than 75% of coords are within the ellipse_outer, then choose that as only one ellipse
    filtered_ellipse =check_contour_within_ellipse(ellipse_outer, read_gt(image_path)[1])

    #draw the only one ellipse on the original image
    #original_with_ellipse = cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    cv2.imshow('mask', mask)
    cv2.imshow('original', original_with_contours)
    #cv2.imshow('original', original_with_ellipse)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original_with_contours)
    cv2.waitKey()

mask_and_morph(image_path)