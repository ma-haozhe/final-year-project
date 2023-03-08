#this one not only parse json and write it in csv,
# but also write for the augmentation files

import json
import csv
import os
import cv2

# Define the path to your annotation JSON file
anno_file = "project-1-at-2023-03-08-20-49-3b9f161b.json"

# Define the path to your image directory
img_file_dir = "1_test/"

# Define the path to the output CSV file
csv_file = '/Users/haozhema/Documents/Final year project/gt_aug.csv'

# Open the CSV file in write mode
with open(csv_file, 'w') as f:

    # Create the CSV writer
    writer = csv.writer(f)
    writer.writerow(['filename', 'count', 'locations'])

    # Load the JSON data from the file
    with open(anno_file, 'r') as f:
        data = json.load(f)

    # Loop through each object in the JSON data
    for obj in data:
        # Extract the image file name
        img_file = obj['data']['img']
        img_file = img_file_dir + img_file.split("-")[1]

        # Read in the image
        img = cv2.imread(img_file)

        # Extract the x and y values for each dot and calculate the plot coordinates
        dots = []
        for annotation in obj['annotations']:
            for result in annotation['result']:
                x = result['value']['x'] * result['original_width']/100
                y = result['value']['y'] * result['original_height']/100
                dots.append((y, x))

        # Loop through rotation angles in 2 degree increments
        for angle in range(0, 360, 2):
            # Rotate the dots
            rotated_dots = []
            for dot in dots:
                # Rotate each dot around the image center
                center = (img.shape[1] / 2, img.shape[0] / 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
                dot = tuple(map(int, dot))
                dot = (dot[0], dot[1], 1)
                rotated_dot = tuple(map(int, (rot_mat.dot(dot))))
                rotated_dot = (rotated_dot[1], rotated_dot[0])
                rotated_dots.append(rotated_dot)

            # Construct the new image filename
            new_img_file = os.path.splitext(img_file)[0] + '-' + str(angle) + os.path.splitext(img_file)[1]

            # Write a row to the CSV file for the rotated image
            row = [new_img_file, len(rotated_dots), rotated_dots]
            writer.writerow(row)
