import numpy as np
from numba import jit
from stitching.sift_stitch import get_corresponding_points_sift
import cv2

from stitching.superglue_stitch import get_corresponding_points_superglue


@jit(nopython=True, parallel=True)
def _combine_masks(x, y):
    return np.where(x == 0, x, y)


@jit(nopython=True, parallel=True)
def _combine_images(x, y):
    return np.where(x != 0, x, y)


@jit(nopython=True, parallel=True)
def _top_mask(mask, image_height, image_width, cols, rows, col_count, row_count):
    rows -= 1
    row_count -= 1
    row_height = image_height / rows
    col_width = image_width / cols
    y_start = int(row_height * row_count - int(row_height / 2))
    temp_mask = np.copy(mask)
    y_end, x_end = (y_start + int(row_height / 2), int(col_width * col_count))
    temp_mask[0:y_start, 0:x_end] = 0
    temp_mask[y_end:, 0:x_end] = 0

    return temp_mask


@jit(nopython=True, parallel=True)
def _left_mask(mask, image_height, image_width, cols, rows, col_count, row_count):
    cols -= 1
    col_count -= 1
    row_height = image_height / rows
    col_width = image_width / cols
    x_start = int(col_width * col_count - int(col_width / 2))
    y_start = int(row_height * row_count - int(row_height))
    y_end, x_end = (int(row_height * row_count), x_start + int(col_width / 2))
    temp_mask = np.copy(mask)
    temp_mask[:, 0:x_start] = 0
    temp_mask[:, x_end:] = 0
    temp_mask[0:y_start, :] = 0
    temp_mask[y_end:, :] = 0

    return temp_mask


@jit(nopython=True, parallel=True)
def _diag_mask(mask, image_height, image_width, cols, rows, col_count, row_count):
    row_height = image_height / rows
    col_width = image_width / cols
    y_start = max(0, int(
        row_height * row_count - row_height - int(
            row_height / 2)))
    x_start = max(0, int(
        col_width * col_count - col_width - int(
            col_width / 2)))
    y_end, x_end = (int(y_start + row_height), int(x_start + col_width))
    temp_mask = np.copy(mask)
    temp_mask[:, 0:x_start] = 0
    temp_mask[:, x_end:] = 0
    temp_mask[0:y_start, :] = 0
    temp_mask[y_end:, :] = 0

    return temp_mask


@jit(nopython=True, parallel=True)
def _join_images(img1, img2, y_offset, x_offset):
    y_end, x_end = (img1.shape[0] -y_offset, img1.shape[1] - x_offset)
    img2[-y_offset:y_end, -x_offset:x_end] = np.where(img1 != 0, img1, img2[-y_offset:y_end, -x_offset:x_end])
    return img2


@jit(nopython=True, parallel=True)
def _join_masks(img1, img2, y_offset, x_offset):
    y_end, x_end = (img1.shape[0] - y_offset, img1.shape[1] - x_offset)
    img2[-y_offset:y_end, -x_offset:x_end] = np.where(img1 == 255, img1, img2[-y_offset:y_end, -x_offset:x_end])
    return img2

class Image_Transform:
    def __init__(self, file, col, row, matrix):
        self.file = file
        self.col = col
        self.row = row
        self.matrix = matrix

    def to_dict(self):
        return {'file': self.file, 'col': self.col, 'row': self.row, 'matrix': self.matrix.tolist()}

class Stitch_Image:
    def __init__(self, image_address, col, row, scale=4):
        self.scale = scale
        temp_img = cv2.imread(image_address)
        if self.scale != 1:
            self.image = cv2.resize(temp_img, (int(temp_img.shape[1] / self.scale), int(temp_img.shape[0] / self.scale)))
            del temp_img
        else:
            self.image = temp_img
        self.col = col
        self.row = row

        half_height = int(self.image.shape[0] / 2)
        half_width = int(self.image.shape[1] / 2)
        self.__bottom_mask = np.concatenate((
            np.full((half_height, self.image.shape[1]), 255, dtype=np.ubyte),
            np.full((self.image.shape[0] - half_height, self.image.shape[1]), 0, dtype=np.ubyte)), axis=0)

        self.__right_mask = np.concatenate((
            np.full((self.image.shape[0], half_width), 255, dtype=np.ubyte),
            np.full((self.image.shape[0], self.image.shape[1] - half_width), 0, dtype=np.ubyte)), axis=1)

        self.mask = np.full((self.image.shape[0], self.image.shape[1]), 255, dtype=np.ubyte)
        self.matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def mask_images(self, image_list):
        for mask_image in image_list:
            loc = get_corresponding_points_sift(self.image,
                                                np.full((self.image.shape[0], self.image.shape[1]), 255,
                                                        dtype=np.ubyte),
                                                mask_image,
                                                np.full((mask_image.shape[0], mask_image.shape[1]), 255,
                                                        dtype=np.ubyte))
            if loc is not None:
                transform, mask = cv2.estimateAffinePartial2D(loc[1], loc[0], method=cv2.RANSAC,
                                                              ransacReprojThreshold=5)
                blank_mask = np.full((mask_image.shape[0], mask_image.shape[1]), 0, dtype=np.ubyte)
                trans_mask = cv2.warpAffine(blank_mask, transform, (self.image.shape[1], self.image.shape[0]), borderValue=(255,255,255))

                self.mask = _combine_masks(self.mask, trans_mask)


    def right_mask(self):
        return _combine_masks(self.mask, self.__right_mask)

    def bottom_mask(self):
        return _combine_masks(self.mask, self.__bottom_mask)

    def reset_mask(self):
        self.mask = np.full((self.image.shape[1], self.image.shape[0]), 255, dtype=np.ubyte)

    def set_matrix(self, matrix):
        self.matrix = matrix


class Full_Image:
    def __init__(self, image_address, scale=4):
        self.scale = scale
        temp_img = cv2.imread(image_address)
        if self.scale != 1:
            self.image = cv2.resize(temp_img, (int(temp_img.shape[1] / self.scale), int(temp_img.shape[0] / self.scale)))
            del temp_img
        else:
            self.image = temp_img

        self.init_width = self.image.shape[1]
        self.init_height = self.image.shape[0]
        self.col_count = 1
        self.row_count = 1
        self.mask = np.full((self.image.shape[0], self.image.shape[1]), 255, dtype=np.ubyte)

    def mask_images(self, image_list):
        for mask_image in image_list:
            loc = get_corresponding_points_sift(self.image,
                                                np.full((self.image.shape[0], self.image.shape[1]), 255,
                                                        dtype=np.ubyte),
                                                mask_image,
                                                np.full((mask_image.shape[0], mask_image.shape[1]), 255,
                                                        dtype=np.ubyte))
            if loc is not None:
                transform, mask = cv2.estimateAffinePartial2D(loc[1], loc[0], method=cv2.RANSAC,
                                                              ransacReprojThreshold=5)
                blank_mask = np.full((mask_image.shape[0], mask_image.shape[1]), 0, dtype=np.ubyte)
                trans_mask = cv2.warpAffine(blank_mask, transform, (self.image.shape[1], self.image.shape[0]), borderValue=(255,255,255))
                self.mask = _combine_masks(self.mask, trans_mask)


    def get_width(self):
        return self.image.shape[1]

    def get_height(self):
        return self.image.shape[0]

    def top_mask(self, row_count, col_count):
        if row_count > self.row_count:
            self.row_count = row_count
        if col_count > self.col_count:
            self.col_count = col_count

        return _top_mask(self.mask, self.image.shape[0], self.image.shape[1], self.col_count, self.row_count, col_count, row_count)

    def left_mask(self, row_count, col_count):
        if row_count > self.row_count:
            self.row_count = row_count
        if col_count > self.col_count:
            self.col_count = col_count

        return _left_mask(self.mask, self.image.shape[0], self.image.shape[1], self.col_count, self.row_count, col_count,
                         row_count)

    def diag_mask(self, row_count, col_count):
        if col_count > self.col_count:
            self.col_count = col_count
        if row_count > self.row_count:
            self.row_count = row_count

        return _diag_mask(self.mask, self.image.shape[0], self.image.shape[1], self.col_count, self.row_count, col_count,
                         row_count)

    def set_image(self, image):
        self.image = image

    def add_image(self, new_image, dir, algorithm='SUPERGLUE', transform_type='AFFINE'):
        row = new_image.row
        col = new_image.col
        if dir == 0:
            self.stitch_vertical(new_image, row, col, algorithm, transform_type)

        elif dir == 1:
            self.stitch_horizontal(new_image, row, col, algorithm, transform_type)

        elif dir == 2:
            self.stitch_diagonal(new_image, row, col, algorithm, transform_type)

        return self.image

    def stitch_full(self, img2, algorithm, transform_type='AFFINE'):
        matching_results = self.get_homography(np.full((self.image.shape[0], self.image.shape[1]), 255, dtype=np.ubyte), img2.image, np.full((img2.image.shape[0], img2.image.shape[1]), 255, dtype=np.ubyte), algorithm,
                                               transform_type)
        if matching_results is not None:
            img2.set_matrix(matching_results[0])
            self.join_images(img2, transform_type)
            return
        else:
            print("Not enough matches are found")

    def stitch_full_mask(self, img2, algorithm, transform_type='AFFINE'):
        matching_results = self.get_homography(self.mask, img2.image, img2.mask, algorithm,
                                               transform_type)
        if matching_results is not None:
            img2.set_matrix(matching_results[0])
            self.join_images(img2, transform_type)
            return
        else:
            # Make a last ditch attempt to match with no masking at all
            return self.stitch_full(img2, algorithm, transform_type)

    def stitch_vertical(self, img2, row, col, algorithm, transform_type='AFFINE'):
        # Note we compare the bottom half of the first image
        # with the top half of the second image
        match_results = self.get_homography(self.top_mask(row, col), img2.image, img2.bottom_mask(), algorithm,
                                            transform_type)
        if match_results is not None:
            img2.set_matrix(match_results[0])
            self.join_images(img2, transform_type)
            return

        else:
            # Try again to make a match with the whole image instead
            return self.stitch_full_mask(img2, algorithm, transform_type)

    def stitch_horizontal(self, img2, row, col, algorithm, transform_type='AFFINE'):
        # Note we compare the right half of the first image
        # with the left half of the second image

        matching_results = self.get_homography(self.left_mask(row, col), img2.image, img2.right_mask(), algorithm,
                                               transform_type)

        if matching_results is not None:
            img2.set_matrix(matching_results[0])
            self.join_images(img2, transform_type)
            return

        else:
            # Try again to make a match with the whole image instead
            return self.stitch_full_mask(img2, algorithm, transform_type)

    def stitch_diagonal(self, img2, row, col, algorithm, transform_type='AFFINE'):
        matching_results = self.get_homography(self.diag_mask(row, col), img2.image,
                                               np.full((img2.image.shape[0], img2.image.shape[1]), 255, dtype=np.ubyte),
                                               algorithm,
                                               transform_type)

        if matching_results is not None:
            img2.set_matrix(matching_results[0])
            self.join_images(img2, transform_type)
            return

        else:
            # Try again to make a match with the whole image instead
            return self.stitch_full_mask(img2, algorithm, transform_type)

    def get_homography(self, img1_mask, img2, img2_mask, algorithm, transform_type='AFFINE'):
        matches = self.get_corresponding_points(img1_mask, img2, img2_mask, algorithm)

        if matches is None:
            return matches

        src_pts = matches[0]
        dst_pts = matches[1]
        return self.get_homography_from_points(src_pts, dst_pts, transform_type)

    def get_corresponding_points(self, img1_mask, img2, img2_mask, algorithm):
        if algorithm == 'SUPERGLUE':
            return get_corresponding_points_superglue(self.image, img1_mask, img2, img2_mask)
        if algorithm == 'SIFT':
            return get_corresponding_points_sift(self.image, img1_mask, img2, img2_mask)

        return None

    def get_homography_from_points(self, src_pts, dst_pts, transform_type='AFFINE'):
        if transform_type == 'AFFINE':
            return cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5)
        if transform_type == 'PERSPECTIVE':
            return cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
        return None

    def join_images(self, img2, transform_type='AFFINE'):
        top_left = np.array([[[0, 0]]])
        top_right = np.array([[[img2.image.shape[1], 0]]])
        bottom_right = np.array([[[img2.image.shape[1], img2.image.shape[0]]]])
        bottom_left = np.array([[[0, img2.image.shape[0]]]])

        trans_top_left = cv2.transform(top_left, img2.matrix)
        trans_top_right = cv2.transform(top_right, img2.matrix)
        trans_bottom_right = cv2.transform(bottom_right, img2.matrix)
        trans_bottom_left = cv2.transform(bottom_left, img2.matrix)

        width = max(trans_top_left[0][0][0], trans_top_right[0][0][0], trans_bottom_right[0][0][0], trans_bottom_left[0][0][0], self.image.shape[1])
        height = max(trans_top_left[0][0][1], trans_top_right[0][0][1], trans_bottom_right[0][0][1], trans_bottom_left[0][0][1], self.image.shape[0])

        x_offset = min(trans_top_left[0][0][0], trans_top_right[0][0][0], trans_bottom_right[0][0][0],
                    trans_bottom_left[0][0][0], 0)
        y_offset = min(trans_top_left[0][0][1], trans_top_right[0][0][1], trans_bottom_right[0][0][1],
                     trans_bottom_left[0][0][1], 0)

        if transform_type == 'AFFINE':
            result = cv2.warpAffine(np.pad(img2.image, ((-y_offset, 0), (-x_offset, 0), (0, 0)), mode='constant', constant_values=0), img2.matrix, (width - x_offset, height - y_offset), flags=cv2.INTER_CUBIC)
            result_mask = cv2.warpAffine(np.pad(img2.mask, ((-y_offset, 0), (-x_offset, 0)), mode='constant', constant_values=0), img2.matrix, (width - x_offset, height - y_offset), flags=cv2.INTER_NEAREST)
        elif transform_type == 'PERSPECTIVE':
            result = cv2.warpPerspective(np.pad(img2.image, ((-y_offset, 0), (-x_offset, 0), (0, 0)), mode='constant', constant_values=0), img2.matrix, (width - x_offset, height - y_offset), flags=cv2.INTER_CUBIC)
            result_mask = cv2.warpPerspective(np.pad(img2.mask, ((-y_offset, 0), (-x_offset, 0)), mode='constant', constant_values=0), img2.matrix, (width - x_offset, height - y_offset), flags=cv2.INTER_NEAREST)

        # remove the antialiasing on the right edge of the transformed image
        result[np.arange(result.shape[0]), (result.shape[1] - 1 - (result[:, ::-1] != 0).argmax(1)[:, 0:1]).flatten()] = 0

        self.image = _join_images(self.image, result, y_offset, x_offset)
        self.mask = _join_masks(self.mask, result_mask, y_offset, x_offset)
