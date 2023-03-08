#this script parse annotated json file from label studio
#to training file for object locator from 2019 paper
# object localization without bounding boxes, to test 
#the effectiveness of their work on Archery targets.
# Haozhe Ma
# March 2023

import json
import csv

# open the file in the write mode
f = open('/Users/haozhema/Documents/Final year project/gt.csv', 'w')

# create the csv writer
writer = csv.writer(f)
writer.writerow(['filename', 'count', 'locations'])

# load the JSON data from the file
anno_file = "project-1-at-2023-03-08-20-49-3b9f161b.json"
img_file_dir = "data/60cm/"
with open(anno_file, 'r') as f:
    data = json.load(f)

for obj in data:
    # extract the image file name
    img_file = obj['data']['img']
    img_file = img_file_dir + img_file.split("-")[1]

    # extract the x and y values for each dot and calculate the plot coordinates
    dots = []
    for annotation in obj['annotations']:
        for result in annotation['result']:
            x = result['value']['x'] * result['original_width']/100
            y = result['value']['y'] * result['original_height']/100
            dots.append((y, x))
            print(x,', ',y)
    #!!!!!!!float might be a problem here!!!!!!!
    # also the 'localisation without bounding boxes' code requires the coordinates format to be (y, x)
    #!!!!!!!also might need to hard code for windows 60cm training set.

    #print(dots)
    row = [img_file,len(dots),dots]
    print(row)
    # write a row to the csv file
    writer.writerow(row)


# close the file
f.close()

    
