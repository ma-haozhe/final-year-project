import os
import cv2

# Define the path to your image directory
img_dir = '1_test'

# Define the path to the output directory
output_dir = '1_augmentation'

# Loop through all the files in the directory
for filename in os.listdir(img_dir):
    print(filename)
    if filename.endswith('.jpeg'):
        # Read in the image
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)

        # Loop through rotation angles in 2 degree increments
        for angle in range(0, 360, 2):
            # Rotate the image
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            # Construct the new filename
            new_filename = filename[:-5] + '-' + str(angle) + '.jpeg'
            new_img_path = os.path.join(output_dir, new_filename)

            # Save the rotated image with the new filename
            cv2.imwrite(new_img_path, rotated_img)