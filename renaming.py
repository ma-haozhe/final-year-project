import os

import sys

import fnmatch

import time


def progress_bar(value, bar_length=20):
    """Displays a progress bar in the console.

    Arguments:
    value -- a float between 0 and 1 representing the progress
    bar_length -- the length of the progress bar in characters (default 20)
    """
    if value < 0 or value > 1:
        raise ValueError("Value must be between 0 and 1.")

    progress = int(value * bar_length)
    bar = "█" * progress + "░" * (bar_length - progress)
    sys.stdout.write(f"\rProgress: [{bar}] {value * 100:.2f}%")
    sys.stdout.flush()


def rename_photos(directory):
    file_num_count = len(fnmatch.filter(os.listdir(directory), '*.*'))
    print('File Count in Directory: ', file_num_count)
    print('Batch renaming.')
    global_count = 0
    for filename in os.listdir(directory):
        global_count += 1
        progress_bar((global_count/file_num_count))
        if filename.endswith('.jpg'):
            count = 0
            add_counter = 0
            head = ""
            for char in reversed(filename):
                add_counter += 1
                if char == 'x':
                    count += 1
                if add_counter > 4 and add_counter <= len(filename):
                    head = (head+char).replace("x", "")
                else:
                    continue
            new_filename = head[::-1]+'-'+str(count*5) + '.jpg'
            os.rename(os.path.join(directory, filename),
                      os.path.join(directory, new_filename))

# Pass in the directory here


rename_photos('10-200')
