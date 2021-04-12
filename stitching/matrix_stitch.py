from stitching.stitch_image import Stitch_Image, Full_Image, Image_Transform
import cv2
import numpy as np
from tqdm.auto import tqdm
from colorama import Fore, Style
from gc import collect
import logging
logger = logging.getLogger(__name__)


# We output the result of each stitching step in sequence (in grayscale).
# The plan is to store the series of homographies so they could be
# directly applied to other sequences from the image cube.
def matrix_stitch(image_matrix, filter_images, image_name, algorithm='SIFT', transform_type='AFFINE', scale=4, suppress_background=False, preview=False):
    i = 0
    prev_col = None
    first_col = True
    filter_images = [cv2.imread(x) for x in filter_images]
    filter_images = [cv2.resize(x, (int(x.shape[1] / scale), int(x.shape[0] / scale))) for x in filter_images]
    transforms = [[]]
    first_image = ''

    def add_offset(transforms, x_offset, y_offset, scale):
        x_offset = x_offset * scale
        y_offset = y_offset * scale
        for col in transforms:
            for img in col:
                img.matrix[0][2] -= x_offset
                img.matrix[1][2] -= y_offset

        return transforms

    for idx, row in enumerate(tqdm(image_matrix, desc=f"{Fore.YELLOW}Detecting Alignment of Rows in {image_name}{Style.RESET_ALL}", leave=False)):
        new_col = True
        for idy, col in enumerate(tqdm(row, desc=f"{Fore.YELLOW}Detecting Alignment of Columns in {image_name}{Style.RESET_ALL}", leave=False)):
            if prev_col is not None and first_col:
                new_img = Stitch_Image(col, row=idy + 1, col=idx + 1, scale=scale, suppress_background=suppress_background)
                new_img.mask_images(filter_images)
                added = prev_col.add_image(new_img, 0, algorithm=algorithm, transform_type=transform_type)
                if added is None:
                    logger.error(f'Could not add image {col} to {first_image}')
                    return None

                _, x_offset, y_offset = added
                adj_matrix = np.copy(new_img.matrix)
                transforms = add_offset(transforms, x_offset, y_offset, scale)
                adj_matrix[0][2] -= x_offset
                adj_matrix[1][2] -= y_offset
                adj_matrix[0][2] *= scale  # Scale the matrix back to unscaled coordinates
                adj_matrix[1][2] *= scale  # Scale the matrix back to unscaled coordinates
                if transform_type == 'PERSPECTIVE':
                    adj_matrix[2][0] /= scale
                    adj_matrix[2][1] /= scale

                transforms[i].append(Image_Transform(col, new_img.col, new_img.row, adj_matrix))
                del new_img
                collect()
            elif first_col:
                prev_col = Full_Image(col, scale=scale, suppress_background=suppress_background, preview=preview)
                first_image = col
                prev_col.mask_images(filter_images)
                matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) if transform_type == 'AFFINE' else np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                transforms[i].append(Image_Transform(col, 1, 1, matrix))
            else:
                new_img = Stitch_Image(col, row=idy + 1, col=idx + 1, scale=scale, suppress_background=suppress_background)
                new_img.mask_images(filter_images)
                added = prev_col.add_image(new_img, 1 if new_col else 2, algorithm=algorithm, transform_type=transform_type)
                if added is None:
                    logger.error(f'Could not add image {col} to {first_image}')
                    return None

                _, x_offset, y_offset = added
                adj_matrix = np.copy(new_img.matrix)
                transforms = add_offset(transforms, x_offset, y_offset, scale)
                adj_matrix[0][2] -= x_offset
                adj_matrix[1][2] -= y_offset
                adj_matrix[0][2] *= scale  # Scale the matrix back to unscaled coordinates
                adj_matrix[1][2] *= scale  # Scale the matrix back to unscaled coordinates
                if transform_type == 'PERSPECTIVE':
                    adj_matrix[2][0] /= scale
                    adj_matrix[2][1] /= scale
                if new_col:
                    i += 1
                    transforms.append([])
                transforms[i].append(Image_Transform(col, new_img.col, new_img.row, adj_matrix))
                del new_img
                collect()

            new_col = False

        first_col = False

    return transforms
