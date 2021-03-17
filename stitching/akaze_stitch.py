import cv2
import numpy as np


def get_corresponding_points_akaze(img1, img1_mask, img2, img2_mask, scale=1, accuracy=0.75):
    sift = cv2.AKAZE_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(cv2.resize(img1, (int(img1.shape[1] / scale), int(img1.shape[0] / scale))),
                                      mask=cv2.resize(img1_mask, (int(img1.shape[1] / scale), int(img1.shape[0] / scale))))
    kp2, des2 = sift.detectAndCompute(cv2.resize(img2, (int(img2.shape[1] / scale), int(img2.shape[0] / scale))),
                                      mask=cv2.resize(img2_mask, (int(img2.shape[1] / scale), int(img2.shape[0] / scale))))

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # match = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = match.knnMatch(des1, des2, k=2)
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < accuracy * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return (src_pts * scale, dst_pts * scale)
    else:
        return None