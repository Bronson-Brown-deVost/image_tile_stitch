import cv2
import numpy as np
from stitching.stitch_image import _join_images
import ntpath
import json
from tqdm import tqdm
from colorama import Fore, Style


def merge_images(image_merge_list, transform_type, out_path, image_type):
    if len(image_merge_list) == 0:
        raise Exception('The submitted array is empty.')

    if len(image_merge_list[0]) == 0:
        raise Exception('The submitted array is empty.')

    file_name = ntpath.basename(image_merge_list[0][0].file)
    file_name_base, file_name_extension = file_name.split('.')
    transform_list_json = json.dumps([y.to_dict() for x in image_merge_list for y in x], ensure_ascii=False)
    with open(f'{out_path}/{file_name_base}-transforms.json', 'w') as out_file:
        out_file.write(transform_list_json)

    # Resort the list to start with the bottom-right image
    #image_merge_list = [i[::-1] for i in image_merge_list[::-1]]

    first_image_info = image_merge_list[0][0]

    first_image = cv2.imread(first_image_info.file)
    top_left = np.array([[[0, 0]]])
    top_right = np.array([[[first_image.shape[1], 0]]])
    bottom_right = np.array([[[first_image.shape[1], first_image.shape[0]]]])
    bottom_left = np.array([[[0, first_image.shape[0]]]])

    trans_top_left = cv2.transform(top_left, first_image_info.matrix)
    trans_top_right = cv2.transform(top_right, first_image_info.matrix)
    trans_bottom_right = cv2.transform(bottom_right, first_image_info.matrix)
    trans_bottom_left = cv2.transform(bottom_left, first_image_info.matrix)

    width = max(trans_top_left[0][0][0], trans_top_right[0][0][0], trans_bottom_right[0][0][0],
                trans_bottom_left[0][0][0], first_image.shape[1])
    height = max(trans_top_left[0][0][1], trans_top_right[0][0][1], trans_bottom_right[0][0][1],
                 trans_bottom_left[0][0][1], first_image.shape[0])

    if transform_type == 'AFFINE':
        full_image = cv2.warpAffine(
            first_image,
            first_image_info.matrix, (width, height), flags=cv2.INTER_CUBIC)
    elif transform_type == 'PERSPECTIVE':
        full_image = cv2.warpPerspective(
            first_image,
            first_image_info.matrix, (width, height), flags=cv2.INTER_CUBIC)

    del first_image

    for col in tqdm(image_merge_list, desc=f"{Fore.MAGENTA}Stitching Cols of {image_type}{Style.RESET_ALL}", leave=False, position=2):
        for row in tqdm(col, desc=f"{Fore.MAGENTA}Stitching Rows of {image_type}{Style.RESET_ALL}", leave=False, position=3):
            img = cv2.imread(row.file)
            full_image = join_images(full_image, img, row.matrix, transform_type)
            del img

    cv2.imwrite(f'{out_path}/{file_name_base}-st.{file_name_extension}', full_image)

def join_images(full_img, img2, img2_matrix, transform_type):
    top_left = np.array([[[0, 0]]])
    top_right = np.array([[[img2.shape[1], 0]]])
    bottom_right = np.array([[[img2.shape[1], img2.shape[0]]]])
    bottom_left = np.array([[[0, img2.shape[0]]]])

    trans_top_left = cv2.transform(top_left, img2_matrix)
    trans_top_right = cv2.transform(top_right, img2_matrix)
    trans_bottom_right = cv2.transform(bottom_right, img2_matrix)
    trans_bottom_left = cv2.transform(bottom_left, img2_matrix)

    width = max(trans_top_left[0][0][0], trans_top_right[0][0][0], trans_bottom_right[0][0][0], trans_bottom_left[0][0][0], full_img.shape[1])
    height = max(trans_top_left[0][0][1], trans_top_right[0][0][1], trans_bottom_right[0][0][1], trans_bottom_left[0][0][1], full_img.shape[0])

    x_offset = min(trans_top_left[0][0][0], trans_top_right[0][0][0], trans_bottom_right[0][0][0],
                trans_bottom_left[0][0][0], 0)
    y_offset = min(trans_top_left[0][0][1], trans_top_right[0][0][1], trans_bottom_right[0][0][1],
                 trans_bottom_left[0][0][1], 0)

    if transform_type == 'AFFINE':
        result = cv2.warpAffine(np.pad(img2, ((-y_offset, 0), (-x_offset, 0), (0, 0)), mode='constant', constant_values=0), img2_matrix, (width - x_offset, height - y_offset), flags=cv2.INTER_CUBIC)
    elif transform_type == 'PERSPECTIVE':
        result = cv2.warpPerspective(np.pad(img2, ((-y_offset, 0), (-x_offset, 0), (0, 0)), mode='constant', constant_values=0), img2_matrix, (width - x_offset, height - y_offset), flags=cv2.INTER_CUBIC)

    # remove the antialiasing on the right edge of the transformed image
    result[np.arange(result.shape[0]), (result.shape[1] - 1 - (result[:, ::-1] != 0).argmax(1)[:, 0:1]).flatten()] = 0

    return _join_images(full_img, result, y_offset, x_offset)
