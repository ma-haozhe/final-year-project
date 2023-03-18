import cv2
import numpy as np

# Load the input image
img = cv2.imread('plot_test_1_to_10/2.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply HoughCircles method to detect circles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=300, param2=0.1, minRadius=20, maxRadius=0)

# Draw detected circles on the original image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 0, 255), 2)

# Create a new window to display the result
cv2.namedWindow('Target Image', cv2.WINDOW_NORMAL)

# Display the final result
cv2.imshow("Target Image", img)

# Wait for user input and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
