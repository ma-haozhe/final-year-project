import cv2
import numpy as np

# Load the input image
img = cv2.imread('plot_test_1_to_10/1.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Set up the SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by area
params.filterByArea = True
params.minArea = 100

# Filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.5

params.filterByInertia = True
params.minInertiaRatio = 0.9

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the binary image
keypoints = detector.detect(thresh)

# Draw the detected blobs on the original image
if keypoints:
    for kp in keypoints:
        x, y = np.int32(kp.pt)
        r = np.int32(kp.size / 2)
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

# Create a new window to display the result
cv2.namedWindow('Target Image', cv2.WINDOW_NORMAL)

# Display the final result
cv2.imshow("Target Image", img)

# Wait for user input and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
