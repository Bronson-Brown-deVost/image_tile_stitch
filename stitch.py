import sys
import copy
import os
import argparse
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

from stitching.gather_iaa_image_files import parse_image_files
from stitching.matrix_stitch import matrix_stitch
from stitching.write_combined_image import merge_images

colorama.init()

parser = argparse.ArgumentParser(description='A command line utility to stitch images together for the IAA')
parser.add_argument('--i', default='./images', type=str, help='Set the folder containing all images that should be stitched together (default: ./images)')
parser.add_argument('--o', default='./stitched_images', type=str, help='This is folder where all the stitched images and their stitching data will be saved (default: ./stitched_images)')
parser.add_argument('--e', default='./exclude_features', type=str, help='This folder contains images of objects that should be ignored by the image stitching algorithm (default: ./exclude_features)')
parser.add_argument('--a', default='SIFT', type=str, choices=['SIFT', 'SUPERGLUE', 'AKAZE'], help='Set the feature detection algorithm (default: SIFT)')
parser.add_argument('--t', default='AFFINE', type=str, choices=['AFFINE', 'PERSPECTIVE'], help='Set the transformation type (default: AFFINE)')
parser.add_argument('--s', default=4, type=int, help='Set the pre-stitch scaling ratio (default: 4)')
parser.add_argument('--b', default=True, type=bool, help='Perform partial background removal during detection (default: True)')
parser.add_argument('--p', default=False, type=bool, help='Preview the stitching progress (default: False)')

args = parser.parse_args()

def main():
    errors = []

    # Gather the command line options and prepare the initial settings
    out_path = args.o

    # Check the output folder
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if not os.path.isdir(out_path):
        raise Exception(f'The output folder `{out_path}` already exists as a file.')

    # Process the objects to be ignored by the keypoint matching algorithms
    remove_feature_path = args.e
    if not os.path.isdir(remove_feature_path):
        raise Exception(f'The folder containing images of objects to exclude `{remove_feature_path}` does not exist.')

    remove_features = []
    for file in os.listdir(remove_feature_path):
        if file.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            remove_features.append(os.path.join(remove_feature_path, file))

    # Arrange the image files to be stitched
    # These are grouped by plate/fragment and by image type (color, ir, etc.)
    image_path = args.i
    if not os.path.isdir(image_path):
        raise Exception(f'The image input folder `{image_path}` does not exist.')

    print(Fore.BLUE)
    print(f'Processing all images in `{image_path}` to be stitched together')
    print(f'All object found in images in the `{remove_feature_path}` will be ignored')
    print(f'The stitched images and their stitching info will be saved to `{out_path}`')
    print(Style.RESET_ALL)
    image_catalogue = parse_image_files(image_path)

    # Load the detection algorithm and transform type settings
    detection_algorithm = args.a
    transform_type = args.t
    print(Fore.GREEN)
    print(f'Beginning the stitching process using the {detection_algorithm} keypoint detection algorithm')
    print(f'The images will be stitched using {transform_type} transforms\n')
    if (args.b):
        print(f'The background will be suppressed while matching the images\n')

    print(f'There are {len(image_catalogue.keys())} Imaged Objects to be processed:')
    print(Style.RESET_ALL)
    # Begin the stitch process for each images object
    for image_series in tqdm(list(image_catalogue.keys()), desc=f"{Fore.LIGHTMAGENTA_EX}Processing Imaged Objects{Style.RESET_ALL}", position=0):
        # Find the color series of images for this imaged object
        # We currently check only for the words 'Color' or 'PSC'
        color_substrings = ['Color', 'PSC']
        color_key_candidates = [x for x in image_catalogue[image_series].keys() if any(map(x.__contains__, color_substrings))]
        if len(color_key_candidates) == 0:
            errors.append(f'Could not find any color images for {image_series}')
            continue

        color_key = color_key_candidates[0]
        image_sequence = [[y["file"] for y in image_catalogue[image_series][color_key]['rows'][x]] for x in image_catalogue[image_series][color_key]['rows'].keys()]

        # Transpose the input so that rows are represented by the first dimension and columns by the second
        image_sequence = list(zip(*image_sequence))
        image_adjustments = matrix_stitch(image_sequence, remove_features, image_series, detection_algorithm, transform_type, args.s, args.b, args.p)

        for image_type in tqdm(list(image_catalogue[image_series].keys()), desc=f"{Fore.CYAN}Stitching Image Series {image_series}{Style.RESET_ALL}", leave=False, position=1):
            # Transpose the sequence so that rows are represented by the first dimension and columns by the second
            image_sequence = list(zip(*[[y["file"] for y in image_catalogue[image_series][image_type]['rows'][x]] for x in image_catalogue[image_series][image_type]['rows'].keys()]))
            if len(image_adjustments) != len(image_sequence) and len(image_adjustments[0]) != len(image_sequence[0]):
                errors.append(f'The number of images for {image_series} {image_type} does not match that of the color series')
                continue

            adj_transforms = copy.deepcopy(image_adjustments)
            for col_1, col_2 in zip(adj_transforms, image_sequence):
                for row_1, row_2 in zip(col_1, col_2):
                    row_1.file = row_2
            merge_images(adj_transforms, transform_type, out_path, image_type)

    print(Fore.GREEN)
    print(f'Finished stitching {len(image_catalogue.keys())} Imaged Objects')
    print(f'The stitched images can be found in {out_path}')
    print(Style.RESET_ALL)

    if len(errors) > 0:
        error_file = 'stitch_errors.txt'
        with open(error_file, 'w') as out_file:
            out_file.write('\n'.join(errors))
        print(Fore.RED)
        print('Some images or imaged object series were not stitched.')
        print(f'See {error_file} for details.')


if __name__ == '__main__':
    print(f'{Fore.YELLOW}Running the IAA image stitching program...{Style.RESET_ALL}')
    main()
