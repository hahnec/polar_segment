import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

from utils.rectify_affine import rectify_mm


class HORAO(Dataset):
    def __init__(
            self, 
            path, 
            cases_file, 
            transforms=[], 
            transforms_img=[], 
            bg_opt=0, 
            wlens=[630], 
            class_num=2,
            signal_opt = True,
            rectify_affine = True,
        ):

        self.base_dir = Path(path)

        self.transforms = transforms
        self.transforms_img = transforms_img
        self.wlens = wlens
        self.bg_opt = int(bool(bg_opt))
        self.cases_files = list(cases_file)
        self.signal_opt = signal_opt
        self.rectify_affine = rectify_affine
        self.iz_opt = True if class_num > 4 else False

        self.ids = []
        for cases_fn in self.cases_files:
            with open(Path(__file__).parent / 'cases' / cases_fn) as f:
                self.ids = self.ids + [line.rstrip('\n') for line in f if line.strip()]
        
        self.class_num = class_num
        self.get_filenames(class_num=class_num)

        # read metadata
        self.df = pd.read_csv(self.base_dir / 'clinical_data.csv')

    def get_filenames(self, class_num=2):

        self.fg_paths = []
        self.img_paths = []
        self.label_paths = []
        self.img_classes = []

        filename = (str(self.wlens[0]) + '_Intensite.npy')

        for id in self.ids:
            sample_dir = self.base_dir / Path(id).relative_to(self.base_dir.name)
            fg_fname = sample_dir / 'annotation' / 'signal.tif'
            assert fg_fname.exists(), f'No label found for the ID {id}: {fg_fname}'
            img_fname = sample_dir / 'raw_data' / filename
            assert img_fname.exists(), f'No image found for the ID {id}: {img_fname}'
            label_fname = sample_dir / 'labels' / 'labels.tif'
            assert label_fname.exists(), f'No label found for the ID {id}: {label_fname}'

            self.fg_paths.append(fg_fname)
            self.img_paths.append(img_fname)
            self.label_paths.append(label_fname)
            self.img_classes.append(0 if id.__contains__('HT') else 1)
    
    def get_metadata(self, part: str):

        row = self.df[(self.df['Measurement ID'] == part)]
        row_str = row['Clinical Diagnosis'].values[0] if not row.empty else 'unlabeled tumor'

        return row_str

    def __getitem__(self, i):

        fg_path = self.fg_paths[i]
        img_path = self.img_paths[i]
        label_fname = self.label_paths[i]
        img_class = self.img_classes[i]

        # load metadata
        metadata = self.get_metadata(img_path.parts[-3]) if img_class else 'healthy'

        # tumor/healthy label construction
        labels = np.array(Image.open(label_fname), dtype=bool)
        labels = labels[None].repeat(2, 0)
        labels[~img_class] = 0
        labels = labels.swapaxes(0, 1).swapaxes(1, 2)
        labels = labels.astype(np.float32)

        # add background class (optional)
        bg = ~np.array(Image.open(fg_path), dtype=bool)[..., None]
        if self.bg_opt:
            labels = np.concatenate((bg.astype(labels.dtype), labels), axis=-1)

        # consider valid signal labels (rejecting blood etc.)
        if self.signal_opt:
            signal_label_path = Path(img_path.parent.parent / 'annotation' / 'signal.tif')
            signal_label = ~np.array(Image.open(signal_label_path), dtype=bool)
            labels[signal_label, self.bg_opt:] = 0
            bg[signal_label, :] = True

        # iterate over wavelengths
        frames = []
        for wlen in self.wlens:
            # update wavelength path
            img_path = Path(str(img_path).replace(str(self.wlens[0]), str(wlen)))
            if img_path.name.endswith('npy'):
                # intensity
                frame = np.load(img_path)
                # clipping
                clip_detect = lambda img, th=254: np.any(img > th, axis=-1).astype(bool)
                clip_mask = clip_detect(frame)
                print(clip_mask.sum(), clip_mask.size, img_path.parent.parent.name)
                bg[clip_mask, :] = True    # merge clipped areas with background
                labels[clip_mask, :] = 0   # mask clipped areas in labels
                # rectify camera misalignment
                if self.rectify_affine:
                    frame = rectify_mm(frame)
                # calibration data
                with open(img_path.parent.parent / 'calib_folder.txt', 'rb') as f: calib_folder = Path(f.readline().strip().decode('utf-8'))
                c_path = self.base_dir / calib_folder / (str(wlen)+'nm')
                amat = np.load(c_path / 'A.npy')
                wmat = np.load(c_path / 'W.npy')
                frame = np.concatenate([frame, amat, wmat], axis=-1)
            else:
                raise NotImplementedError('File type not recognized')

            frames.append(frame)

        # stack wavelength frames
        frames = np.concatenate(frames, axis=-1)
        frames = frames.astype(np.float32)

        for transform in self.transforms_img:
            frames = transform(frames)

        labels = np.concatenate([labels, bg], axis=-1)
        for transform in self.transforms:
            frames, labels = transform(frames, label=labels)

        # split background from labels considering varying dimension order due to transforms
        if isinstance(labels, np.ndarray): labels = torch.tensor(labels)
        split_dim = [i for i, s in enumerate(labels.shape) if s == self.class_num+self.bg_opt+1][0]
        labels, bg = torch.split(labels, [self.bg_opt+self.class_num, 1], dim=split_dim)

        return frames, labels, img_class, bg, metadata

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':

    import random
    from torch.utils.data import DataLoader
    from polar_augment.rotation_raw import RandomPolarRotation
    from utils.transforms_segment import *

    random.seed(3008)
    torch.manual_seed(3008)
    torch.cuda.manual_seed_all(3008)
    batch_size = 1
    patch_size = 4
    bg_opt = 1
    base_dir = '/home/chris/Datasets/03_HORAO/BPD_16_09_2025'
    base_dir = '/home/chris/Datasets/03_HORAO/BPD_17_09_2025'
    feat_keys = ['intensity', 'azimuth', 'linr', 'totp']

    img_list = []
    transforms = [ToTensor(), RandomPolarRotation(45, p=0.5, any=False), SwapDims()]
    valid_set = HORAO(base_dir, ['val_bpd_1709.txt'], bg_opt=bg_opt, class_num=2, wlens=[630], transforms=transforms)
    dataset = torch.utils.data.ConcatDataset([valid_set])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    from mm.models import LuChipmanPyramid as MMM
    mm_model = MMM(feature_keys=feat_keys, perc=.95, levels=1, kernel_size=0, method='averaging', wnum=len(valid_set.wlens), filter_opt=False)

    from skimage import io, transform, registration
    import numpy as np

    bg_pixels = 0
    ht_pixels = 0
    tt_pixels = 0
    tumor_samples = 0
    healthy_samples = 0
    for batch in loader:
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
        imgs = mm_model(imgs)
        end.record()
        torch.cuda.synchronize()
        t_total = start.elapsed_time(end) / 1000
        print('MM processing time: %s' % str(t_total))

        if True:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, labels.shape[1]+1)
            axs[0].set_title(['healthy', 'tumor'][img_class[0]])
            axs[0].imshow(imgs[0][0], cmap='gray')
            for i in range(1, labels.shape[1]+1):
                axs[i].imshow(labels[0, i-1])
            plt.tight_layout()
            plt.show()

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
