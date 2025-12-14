import random
import torch
from torch.utils.data import DataLoader
from skimage import io, transform, registration
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import io, color
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac

from polar_augment.rotation_raw import RandomPolarRotation
from utils.transforms_segment import *
from horao_dataset_2025 import HORAO
from scripts.anaglyph import make_anaglyph

def img_norm(x):
    return (x-x.min())/(x.max()-x.min())

def insert_index_before_ext(path, index, sep='_'):
    """
    Insert index before the file extension in `path`.
    `path` may be a str or pathlib.Path. Returns a pathlib.Path.
    """
    p = Path(path)
    stem = p.stem            # filename without suffix
    suffix = ''.join(p.suffixes)  # preserves multi-part suffixes like .tar.gz
    new_name = f"{stem}{sep}{index}{suffix}"
    return p.with_name(new_name)


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

    return warped #, model_robust, inliers

def rectify_phase_shifts(ref, img):

    shift, _, _ = registration.phase_cross_correlation(ref, img, upsample_factor=upsample_factor)
    print(shift)
    tform = transform.SimilarityTransform(translation=-shift[::-1])
    aligned = transform.warp(img.clone(), tform)

    return aligned

def rectify_transform(mm_img, ref_idx=4, upsample_factor=2, anaglyph_show=False, save_fpath=None):

    ref = img_norm(mm_img[:, ref_idx].squeeze())
    transformed = []
    for i in range(0, 16):
        if i == ref_idx or i < 4:
            transformed.append(ref.numpy())
            continue
        img = img_norm(mm_img[:, i].squeeze())

        #aligned = rectify_phase_shifts(ref, img)
        aligned = rectify_affine(src_img=img, dst_img=ref)

        curr_save_fpath = insert_index_before_ext(save_fpath, i) if save_fpath is not None else None
        anaglyph_before = make_anaglyph(ref, img, show=anaglyph_show, save_fpath=curr_save_fpath)
        anaglyph_after = make_anaglyph(ref, aligned, show=anaglyph_show, save_fpath=curr_save_fpath.parent / (curr_save_fpath.stem + '_aligned' + curr_save_fpath.suffix))
        transformed.append(aligned)

    return torch.tensor(transformed)[None]


if __name__ == '__main__':

    random.seed(3008)
    torch.manual_seed(3008)
    torch.cuda.manual_seed_all(3008)
    batch_size = 1
    patch_size = 4
    bg_opt = 0
    transforms = [ToTensor(), SwapDims()] #[ToTensor(), RandomPolarRotation(45, p=0.5, any=False), SwapDims()]
    base_dir = '/home/chris/Datasets/03_HORAO/BPD_16_09_2025'
    target_dir = Path('/home/chris/TechDocs/11_HORAO/22_registration_pt2/img/')
    valid_set = HORAO(base_dir, ['val_bpd_1609.txt'], bg_opt=bg_opt, class_num=2, wlens=[630], transforms=transforms)
    if False:
        base_dir = '/home/chris/Datasets/03_HORAO/BPD_17_09_2025'
        valid_set = HORAO(base_dir, ['val_bpd_1709.txt'], bg_opt=bg_opt, class_num=2, wlens=[630], transforms=transforms)

    img_list = []
    dataset = torch.utils.data.ConcatDataset([valid_set])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    from mm.models import MuellerMatrixSelector as MMM
    mm_model = MMM(mask_fun=None)

    from skimage import io, transform, registration
    import numpy as np

    bg_pixels = 0
    ht_pixels = 0
    tt_pixels = 0
    tumor_samples = 0
    healthy_samples = 0
    for img_idx, batch in enumerate(loader):
        imgs, labels, img_class, bg, metadata = batch

        # move feature dimension for consistency
        bg = bg.moveaxis(-1, 1)
        imgs = imgs.moveaxis(-1, 1)
        labels = labels.moveaxis(-1, 1)

        bg_pixels += (bg[:, 0]>0).sum().item()
        ht_pixels += (labels[:, 0+bg_opt]>0).sum().item()
        tt_pixels += (labels[:, 1+bg_opt]>0).sum().item()
        tumor_samples += (img_class==1).sum().item()
        healthy_samples += (img_class==0).sum().item()

        # compute mueller matrix
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        #imgs = mm_model(imgs)
        end.record()
        torch.cuda.synchronize()
        t_total = start.elapsed_time(end) / 1000
        print('MM processing time: %s' % str(t_total))

        fpath = target_dir / (valid_set.fg_paths[img_idx].parent.parent.name + '.png')

        if True:
            fig1, axs = plt.subplots(4, 4, figsize=(12, 12))
            fig2, ax = plt.subplots(1, 1)
            for row in range(4):
                for col in range(4):
                    axs[row, col].imshow(imgs[0,row*col+row], cmap='gray')
                    axs[row, col].set_title('(%s, %s)' % (str(row+1), str(col+1)))
                    ax.imshow(imgs[0,row*col+row], cmap='gray')
                    fig2.savefig(fpath.parent / (fpath.stem + '_mm' + str(row+1) + str(col+1) + '.png'))
            plt.tight_layout()
            fig1.savefig(fpath)
            #plt.show()

        integral_img_before = img_norm(imgs.mean(1).squeeze())
        for i in range(imgs.shape[1]):
            imgs[0, i] = img_norm(imgs[0, i])
        
        rect_imgs = rectify_transform(imgs, save_fpath=fpath)
        integral_img_after = img_norm(rect_imgs.mean(1).squeeze())

        if True:
            fig2, ax = plt.subplots(1, 1)
            for row in range(4):
                for col in range(4):
                    ax.imshow(rect_imgs[0,row*col+row], cmap='gray')
                    fig2.savefig(fpath.parent / (fpath.stem + '_mm-align' + str(row+1) + str(col+1) + '.png'))

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(integral_img_before, cmap='gray')
        axs[1].imshow(integral_img_after, cmap='gray')
        plt.tight_layout()
        plt.savefig(fpath.parent / (fpath.stem + '_before_vs_after' + fpath.suffix))
        #plt.show()

        img_list.append(imgs)

    class_balance = {
        'Background': bg_pixels,
        'Healthy tissue pixels': ht_pixels,
        'Tumor tissue pixels': tt_pixels,
                    }

    try:
        from utils.rowdict2textable import generate_latex_table
        generate_latex_table(class_balance)
    except:
        pass

    print('%s tumor samples' % str(tumor_samples))
    print('%s healthy samples' % str(healthy_samples))
    for k,v in class_balance.items():
        print(str(v), k)
