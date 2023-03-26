# Haozhe Ma
# March 2023

"""
this code extracted functionalities to functions from
mask_morph_update.py
for easier calling from outside. 
"""

import numpy as np
import cv2

def preprocess_image(image_path):
    """
    Preprocess the input image by converting it to HSV color space.
    :param image_path: The path of the input image.
    :return: The preprocessed image in HSV color space.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image

def create_color_masks(image, lower_color, upper_color):
    """
    Create a binary mask for a given color range in the input image.
    :param image: The input image in HSV color space.
    :param lower_color: The lower bound of the color range.
    :param upper_color: The upper bound of the color range.
    :return: The binary mask for the given color range.
    """
    mask = cv2.inRange(image, lower_color, upper_color)
    return mask

def combine_masks(mask1, mask2):
    """
    Combine two binary masks using a bitwise OR operation.
    :param mask1: The first binary mask.
    :param mask2: The second binary mask.
    :return: The combined binary mask.
    """
    combined_mask = cv2.bitwise_or(mask1, mask2)
    return combined_mask

def find_contours(mask):
    """
    Find contours in a binary mask.
    :param mask: The binary mask.
    :return: A list of contours.
    """
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

def morphological_closing(mask, kernel_size=(1, 1)):
    """
    Perform morphological closing on a binary mask using an elliptical kernel.
    :param mask: The binary mask.
    :param kernel_size: The size of the elliptical kernel.
    :return: The closed binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closing

def filter_contours_by_area(contours, min_area=3000):
    """
    Filter contours based on a minimum area.
    :param contours: The input list of contours.
    :param min_area: The minimum area for a contour to be considered.
    :return: A list of filtered contours.
    """
    filtered_cnts = [c for c in contours if cv2.contourArea(c) >= min_area]
    return filtered_cnts

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
        if len(approx) > 4:
            ellipse = cv2.fitEllipse(c)

            # COMMENT OUT THIS LINE TO REMOVE DRAWING CONTOURS ON ORIGINAL IMAGE.
            #cv2.drawContours(image, [c], -1, (36, 255, 12), -1)

            cv2.ellipse(image, ellipse, (36, 255, 12), 2)
            # Get the centroid of the ellipse (center of the fitted ellipse)
            cX, cY = int(ellipse[0][0]), int(ellipse[0][1])
            # Draw the centroid on the original image
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
    return image

def main(image_path):
    image = cv2.imread(image_path)
    original = image.copy()
    hsv_image = preprocess_image(image_path)

    # Define the color ranges for red and yellow
    lower_red = np.array([0, 100, 100], dtype="uint8")
    upper_red = np.array([10, 255, 255], dtype="uint8")
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([40, 255, 255], dtype="uint8")
    lower_blue = np.array([100, 100, 100], dtype="uint8")
    upper_blue = np.array([120, 255, 255], dtype="uint8")

    # Create binary masks for red and yellow
    mask_red = create_color_masks(hsv_image, lower_red, upper_red)
    mask_yellow = create_color_masks(hsv_image, lower_yellow, upper_yellow)

    # Combine the masks using bitwise OR
    mask = combine_masks(mask_red, mask_yellow)

    # Find contours
    cnts = find_contours(mask)

    # Perform morphological closing on the mask
    closing = morphological_closing(mask)

    # Find contours in the closed mask
    cnts_closing = find_contours(closing)

    # Filter the contours based on a minimum area
    filtered_cnts = filter_contours_by_area(cnts_closing)

    # Draw the filtered contours on the original image
    original_with_contours = draw_filtered_contours(original, filtered_cnts)

    cv2.imshow('mask', mask)
    cv2.imshow('original', original_with_contours)
    cv2.imwrite('mask.png', mask)
    cv2.imwrite('original.png', original_with_contours)
    cv2.waitKey()

if __name__ == "__main__":
    image_path = '60cm/27.jpeg'
    main(image_path)
