# This code is adapted from the code in models/matching.py and models/utils.py

import cv2
import torch
from models.matching import Matching
torch.set_grad_enabled(False)


def __process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        with open('./errors.txt', 's') as errors_out:
            errors_out.write('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        with open('./errors.txt', 's') as errors_out:
            errors_out.write('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def __frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def __read_loaded_image(image, device):
    inp = __frame2tensor(image, device)
    return image, inp, (1, 1)


def get_corresponding_points_superglue(img1, img1_mask, img2, img2_mask, scale=1, min_error=-1):
    matching = __get_matcher()
    image0, inp0, scales0 = __read_loaded_image(
        cv2.resize(cv2.bitwise_and(img1, img1, mask=img1_mask)[:, :, 2], (int(img1.shape[1] / scale), int(img1.shape[0] / scale))), 'cpu')
    image1, inp1, scales1 = __read_loaded_image(
        cv2.resize(cv2.bitwise_and(img2, img2, mask=img2_mask)[:, :, 2], (int(img2.shape[1] / scale), int(img2.shape[0] / scale))), 'cpu')

    pred = matching({'image0': inp0,
                     'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # print('finding matches')
    valid = matches > min_error

    if len(valid) > 10:  # len(good) > MIN_MATCH_COUNT:
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        src_pts = mkpts0 * scale
        dst_pts = mkpts1 * scale
        if len(src_pts) > 0 and len(dst_pts) > 0:
            return (src_pts, dst_pts)

        return None
    else:
        return None


def __get_matcher():
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    return Matching(config).eval().to(device)