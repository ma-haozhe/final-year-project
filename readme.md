# final year project

Feb 3 basically right now there are two approaches for the rotating the images (maybe three)
One is to rotate-save, rotate-save, which will leave a artificial looking thing on four corners.
The other one is to rotate 5 deg, save and then revert to original, rotate 10 deg and save again, this creates more
realistic looking content-aware filling but sometimes it uses the target face to fill the corners, which can confuse the training I guess. 

It is using a Photoshop batch process to rotate each image for five degrees each time. 

Using a conda environment called "tensorflow" in both machines.