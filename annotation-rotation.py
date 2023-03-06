import json
import math

# generated by AI, need to modify to work with label-studio generated images.
# function to rotate a point around the center of an image


def rotate_point(x, y, img_width, img_height, degree):
    xc = img_width/2
    yc = img_height/2

    x_prime = (x-xc)*math.cos(degree) - (y-yc)*math.sin(degree) + xc
    y_prime = (x-xc)*math.sin(degree) + (y-yc)*math.cos(degree) + yc

    return x_prime, y_prime


# open and read the JSON file
with open('image_coordinates.json') as json_file:
    image_data = json.load(json_file)

# get the image width and height
img_width = image_data['image_width']
img_height = image_data['image_height']

# loop through the coordinates and rotate them
rotated_coords = []
for coord in image_data['coordinates']:
    x_prime, y_prime = rotate_point(
        coord[0], coord[1], img_width, img_height, image_data['degree'])
    rotated_coords.append([x_prime, y_prime])

# save the rotated coordinates to the JSON file
image_data['coordinates'] = rotated_coords

with open('image_coordinates.json', 'w') as outfile:
    json.dump(image_data, outfile)
