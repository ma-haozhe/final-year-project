# Rename the 195 files in batch starting from 286.jpeg

import os
import shutil

# Path to the folder with the files to be renamed
path = '60cm-batch2'

# Get the list of files in the directory
file_list = os.listdir(path)

# Loop through the files
for file_name in file_list:
    if file_name.endswith('.jpeg'):
        # Get the file name and extension
        file_name, file_ext = os.path.splitext(file_name)
        # Rename the file
        new_name = '{}{}'.format(285 + int(float(file_name)), file_ext)
        # Rename the file
        os.rename(os.path.join(path, file_name + file_ext), os.path.join(path, new_name))

