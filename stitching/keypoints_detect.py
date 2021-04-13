import cv2
import numpy as np
from numba import jit

def get_keypoints(analyzer, img, mask, scale):
    kp_detect = analyzer()
    if scale != 1:
        img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
        mask = cv2.resize(mask, (int(img.shape[1] / scale), int(img.shape[0] / scale)))

    return kp_detect.detectAndCompute(img, mask=mask)


def match_keypoints(des1, des2, kp1, kp2, scale, accuracy):
    if None in [des1, des2, kp1, kp2] or 0 in [len(des1), len(des2)]:
        return None
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # match = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = match.knnMatch(des1, des2, k=2)
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    return sort_matches(kp1, kp2, matches, accuracy, scale)


@jit(nopython=False, parallel=True, forceobj=True)
def sort_matches(kp1, kp2, matches, accuracy, scale):
    good = []
    for m, n in matches:
        if m.distance < accuracy * n.distance:
            good.append(m)

    min_match_count = 10
    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return src_pts * scale, dst_pts * scale
    else:
        return None
