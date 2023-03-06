import json
import matplotlib.pyplot as plt

# load the JSON data from the file
anno_file = "project-1-at-2023-03-06-17-55-c451917c.json"
img_file_dir = "plot_test_1_to_10/"
with open(anno_file, 'r') as f:
    data = json.load(f)

for obj in data:
    # extract the image file name
    img_file = obj['data']['img']
    img_file = img_file_dir + img_file.split("-")[1]
    print(img_file)

    # extract the x and y values for each dot and calculate the plot coordinates
    dots = []
    for annotation in obj['annotations']:
        for result in annotation['result']:
            x = result['value']['x'] * result['original_width']/100
            y = result['value']['y'] * result['original_height']/100
            dots.append((x, y))
            print(x,', ',y)

    # plot the dots on the image
    img = plt.imread(img_file)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for x, y in dots:
        ax.scatter(x, y, color='red', s=10)
    plt.show()

    
