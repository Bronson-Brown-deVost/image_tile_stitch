from stitching.stitch_image import Stitch_Image, Full_Image, Image_Transform
import cv2
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style


# We output the result of each stitching step in sequence (in grayscale).
# The plan is to store the series of homographies so they could be
# directly applied to other sequesnces from the image cube.
def matrix_stitch(image_matrix, filter_images, image_name, algorithm='SIFT', transform_type='AFFINE'):
    scale = 4
    i = 0
    prev_col = None
    first_col = True
    filter_images = [cv2.imread(x) for x in filter_images]
    filter_images = [cv2.resize(x, (int(x.shape[1] / scale), int(x.shape[0] / scale))) for x in filter_images]
    transforms = [[]]

    for idx, row in enumerate(tqdm(image_matrix, desc=f"{Fore.YELLOW}Detecting Alignment of Rows in {image_name}{Style.RESET_ALL}", leave=False, position=1)):
        new_col = True
        for idy, col in enumerate(tqdm(row, desc=f"{Fore.YELLOW}Detecting Alignment of Columns in {image_name}{Style.RESET_ALL}", leave=False, position=2)):
            if prev_col is not None and first_col:
                new_img = Stitch_Image(col, row=idy + 1, col=idx + 1, scale=scale)
                new_img.mask_images(filter_images)
                prev_col.add_image(new_img, 0, algorithm=algorithm, transform_type=transform_type)
                adj_matrix = np.copy(new_img.matrix)
                adj_matrix[:, 2] *= scale # Scale the matrix back to unscaled coordinates
                transforms[i].append(Image_Transform(col, new_img.col, new_img.row, adj_matrix))
                del new_img
            elif first_col:
                prev_col = Full_Image(col, scale=scale)
                prev_col.mask_images(filter_images)
                transforms[i].append(Image_Transform(col, 1, 1, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])))
            else:
                new_img = Stitch_Image(col, row=idy + 1, col=idx + 1, scale=scale)
                new_img.mask_images(filter_images)
                prev_col.add_image(new_img, 1 if new_col else 2, algorithm=algorithm, transform_type=transform_type)
                adj_matrix = np.copy(new_img.matrix)
                adj_matrix[:, 2] *= scale # Scale the matrix back to unscaled coordinates
                if new_col:
                    i += 1
                    transforms.append([])
                transforms[i].append(Image_Transform(col, new_img.col, new_img.row, adj_matrix))
                del new_img

            new_col = False

        first_col = False

    return transforms
