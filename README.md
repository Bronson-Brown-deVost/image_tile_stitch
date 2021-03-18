# image_tile_stitch

The goal of this project is to put together a robust program to stitch together the images from the Israel Antiquities Authority that were taken in tiled form.

## Installation

This program has only been tested on Python 3.9, but may also work on earlier versions of Python 3.

To install the dependencies run `pip install -r requirements.txt`.

For program usage run `python stitch.py -h`. 

Good results can be obtained by using the default settings (SIFT keypoints, AFFINE transformation, 4x size reduction, and background suppression). Using the `--p True` switch will allow you to see the stitching as it is happening in realtime and perhaps diagnose some obvious problems. If the matching of images is a bit poor, sometimes using `--a AKAZE --s 2` will produce more precise matches (at the expense of time and memory usage). If you run into real difficulties stitching a certain image using `--b False` to turn off the background suppression may have a positive effect in select cases (when the manuscript is not very brown in color). 

## Testing Data

The `images` folder contains several sets of images for testing.

These testing images contain examples that need to be stitching in long rows or that need to be joined as a matrix of multiple columns and rows.

The images are licensed under a Creative Commons Attribution-Non Commercial 4.0 International (CC BY-NC 4.0).
https://creativecommons.org/licenses/by-nc/4.0/

You are permitted to use images for non-commercial uses such as lectures, public presentations and other educational uses.  To license images for commercial uses such as reproduction, publications, displays, etc., please contact the Israel Antiquities Authority Visual Archive at VisualArchive@israntique.org.il.  

For more information on licensing please contact contact@deadseascrolls.org.il

## Current Code

The current working code can be run with the `stitch.py` file. The utility can take several command line options (see `python stitch.py -h`). Passing 'SIFT' (default) or 'SUPERGLUE' as the keypoint detection algorithm will tell the code either to use SIFT for finding corresponding keypoints or the [SUPERGLUE](https://github.com/magicleap/SuperGluePretrainedNetwork) neural network matcher. The Superglue code in the `models` folder is subject to the copyrights and permissions detailed in each file, please see their project site for further details: https://github.com/magicleap/SuperGluePretrainedNetwork. The result of each stitch operation is output to the output folder (default `./stitched_images`).

The matrix_stitch approach used here reads in a list of files from a 2D array that matches the tile orientation of the image series. The order is detected currently by `gather_iaa_image_files.py`, which reads the column and row orientation from the IAA image filenames. The image matching process starts by arranging the images of the first column. Then it aligns the first image of the next column with the top of the first column. Finally, it aligns all subsequent images in the second column with a diagonal match, comparing it both to the image directly above in its own column and with the directly adjacent portion of the preceding column. This process is repeated for all subsequent columns.

This code is still not optimized for memory usage, so 8GB or more will likely be needed to run it successfully. It does not keep a record of any failed stitching operations yet, you will have to verify for yourself. The program will output a JSON file corresponding to the stitched image. This file contains the technical details of the stitching operation (i.e., the transform matrix applied to each image). This file should be saved and can be made available to anyone who might need it for scientific reasons.

## Other Attempts

We tried using the image alignment functions in skimage.registration, but more than translate seems to be required to align these images. Even when we tried to use account for rotation as well, the images still seemed not to align perfectly.