import cv2
from stitching.keypoints_detect import get_keypoints, match_keypoints


def get_corresponding_points_sift(img1, img1_mask, img2, img2_mask, scale=1, accuracy=0.5):
    # find the key points and descriptors with SIFT
    kp1, des1 = get_keypoints(cv2.SIFT_create, img1, img1_mask, scale)
    kp2, des2 = get_keypoints(cv2.SIFT_create, img2, img2_mask, scale)

    return match_keypoints(des1, des2, kp1, kp2, scale, accuracy)
