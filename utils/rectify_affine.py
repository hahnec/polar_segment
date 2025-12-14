import numpy as np
from skimage import io, color
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac


def rectify_affine(src_img, dst_img, n_keypoints=1000, keypoint_threshold=0.08):
    # grayscale
    src_gray = color.rgb2gray(src_img) if src_img.ndim == 3 else src_img
    dst_gray = color.rgb2gray(dst_img) if dst_img.ndim == 3 else dst_img

    # detect ORB features
    orb1 = ORB(n_keypoints=n_keypoints, fast_threshold=keypoint_threshold)
    orb2 = ORB(n_keypoints=n_keypoints, fast_threshold=keypoint_threshold)
    orb1.detect_and_extract(src_gray)
    orb2.detect_and_extract(dst_gray)
    desc1, desc2 = orb1.descriptors, orb2.descriptors
    keypoints1, keypoints2 = orb1.keypoints, orb2.keypoints

    # match descriptors
    matches12 = match_descriptors(desc1, desc2, cross_check=True)
    if matches12.shape[0] < 3:
        raise RuntimeError("Not enough matches to estimate affine transform")

    src_matches = keypoints1[matches12[:, 0]]
    dst_matches = keypoints2[matches12[:, 1]]

    # robustly estimate affine with RANSAC
    model_robust, inliers = ransac(
        (src_matches, dst_matches),
        AffineTransform,
        min_samples=3,
        residual_threshold=2,
        max_trials=1000
    )

    # warp source to destination coordinate frame (rectify)
    warped = warp(src_img, inverse_map=model_robust.inverse, output_shape=dst_img.shape[:2])

    return warped


def rectify_mm(mm_img, ref_idx=4):

    h, w, chs = mm_img.shape
    ref = mm_img[..., ref_idx].squeeze()
    transformed = []
    for i in range(0, chs):
        if i == ref_idx or i < 4:
            transformed.append(ref)
            continue
        img = mm_img[..., i].squeeze()

        aligned = rectify_affine(src_img=img, dst_img=ref)

        transformed.append(aligned)

    return np.stack(transformed, axis=-1)
