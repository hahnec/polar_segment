import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd


class HORAO(Dataset):
    def __init__(
            self, 
            path, 
            cases_file, 
            transforms=[], 
            transforms_img=[], 
            bg_opt=0, 
            wlens=[550], 
            class_num=4,
            blood_opt = False,
        ):

        self.base_dir = Path(path)

        self.transforms = transforms
        self.transforms_img = transforms_img
        self.wlens = wlens
        self.bg_opt = int(bool(bg_opt))
        self.cases_files = list(cases_file)
        self.blood_opt = blood_opt

        self.ids = []
        for cases_fn in self.cases_files:
            with open(Path(__file__).parent / 'cases' / cases_fn) as f:
                self.ids = self.ids + [line.rstrip('\n') for line in f if line.strip()]
        
        self.class_num = class_num
        self.get_filenames(class_num=class_num)

        # read metadata
        self.df = pd.read_csv(self.base_dir / 'clinical_data.csv')

    def get_filenames(self, class_num=2):

        self.img_paths = []
        self.label_paths = []
        self.matter_paths = []
        self.img_classes = []

        filename = (str(self.wlens[0]) + '_Intensite.cod')

        for id in self.ids:
            sample_dir = self.base_dir / Path(id).relative_to('/NPP')
            img_fname = sample_dir / 'raw_data' / filename
            assert img_fname.exists(), f'No image found for the ID {id}: {img_fname}'
            label_fname = sample_dir / 'histology' / 'TCC_full.png'
            assert label_fname.exists(), f'No label found for the ID {id}: {label_fname}'
            matter_fname = sample_dir / 'histology' / 'GM_WM_full.png'
            assert label_fname.exists(), f'No label for WM/GM found for the ID {id}: {matter_fname}'

            self.img_paths.append(img_fname)
            self.label_paths.append(label_fname)
            self.matter_paths.append(matter_fname)
            self.img_classes.append(0 if id.__contains__('HT') else 1)

    @staticmethod
    def map_string(input_string):

        string_map = {
            'std': 'azimuth_local_var',
            'mueller': 'M11',
        }
        
        return string_map.get(input_string, input_string)

    @staticmethod
    def create_multilabels(labels, matter_labels, rearrange=True):
        if rearrange:
            hwm = labels[..., -2].astype(bool) & matter_labels[..., -1].astype(bool)
            hgm = labels[..., -2].astype(bool) & matter_labels[..., -2].astype(bool)
            twm = labels[..., -1].astype(bool) & matter_labels[..., -1].astype(bool)
            tgm = labels[..., -1].astype(bool) & matter_labels[..., -2].astype(bool)
            new = np.stack([hwm, hgm, twm, tgm], axis=-1).astype(float)
        else:
            new = np.stack([labels[..., -2], labels[..., -1], matter_labels[..., -2], matter_labels[..., -1]], axis=-1).astype(float)
        
        if labels.shape[-1] == 3:
            bg = (labels[..., 0].astype(bool) & matter_labels[..., 0].astype(bool))[..., None] # ~np.any(new, -1)[..., None]
            return np.concatenate([bg.astype(float), new], axis=-1)
        
        return new
    
    def get_metadata(self, part: str):

        row = self.df[(self.df['Sample ID'] == part)]
        row_str = row['Diagnosis'].values[0] if not row.empty else 'unlabeled tumor'

        return row_str

    def __getitem__(self, i):

        img_path = self.img_paths[i]
        label_fname = self.label_paths[i]
        matter_fname = self.matter_paths[i]
        img_class = self.img_classes[i]

        # load metadata
        part = img_path.parts[-3]
        metadata = self.get_metadata(part) if img_class else part

        # tumor/healthy label construction
        labels = np.array(Image.open(label_fname)).sum(-1) > 0
        labels = labels[None].repeat(2, 0)
        labels[~img_class] = 0
        labels = labels.swapaxes(0, 1).swapaxes(1, 2)
        labels = labels.astype(np.float32)
        if labels.max() > 1: labels /= 255
        # add background class (optional)
        bg = (np.array(Image.open(matter_fname)).sum(-1) == 0)[..., None]
        if self.bg_opt:
            labels = np.concatenate((bg.astype(labels.dtype), labels), axis=-1)

        # consider white matter / grey matter
        if matter_fname.exists(): 
            matter_labels = np.array(Image.open(matter_fname))[..., 1]
            matter_labels[matter_labels==77] = 1
            matter_labels[matter_labels==153] = 2
            # one-hot encoding
            oh_mat_labels = np.eye(3)[matter_labels.astype(int)]
            # rearrange H,T,WM,GM to HWM, HGM, TWM, TGM
            labels = self.create_multilabels(labels, oh_mat_labels)

        # consider blood labels
        if self.blood_opt:
            blood_label_path = Path(img_path.parent.parent / 'annotation' / 'blood.tif')
            blood_label = np.array(Image.open(blood_label_path), dtype=bool)
            labels[blood_label, ...] = 0
            bg[blood_label, :] = True

        # iterate over wavelengths
        frames = []
        for wlen in self.wlens:
            # update wavelength path
            img_path = Path(str(img_path).replace(str(self.wlens[0]), str(wlen)))

            if img_path.name.endswith('cod'):
                # intensity
                from mm.utils.cod import read_cod_data_X3D
                raw_flag = True if wlen == 550 else False
                frame = read_cod_data_X3D(img_path, raw_flag=raw_flag)
                # clipping
                clip_detect = lambda img, th=65530: np.any(img > th, axis=-1).astype(bool)
                clip_mask = clip_detect(frame.numpy())
                # Remove 0.0 pixel values in 600nm images
                if wlen == 600:
                    empty_mask = frame[:,:,0] == 0.0
                    #frame[empty_mask] = torch.eye(4, dtype=torch.float64).flatten() 
                    C = 30  # Border width
                    C_Top = 50
                    # Replace top border with mirrored values from below
                    for i in range(C_Top):
                        #Repair top border
                        frame[i, :, :] = frame[ C_Top - 1 - i,:, :]
                        empty_mask[i,:] = 1.0

                    for i in range(C):
                        #repair left border
                        frame[:, i, :] = frame[:, C - 1 - i, :]
                        empty_mask[:,i] = 1.0
                        #repair right border
                        frame[:,  -1 - i, :] = frame[:, -1 - C + i, :]
                        empty_mask[:,-1-i] = 1.0
                    # Merge clip mask with empty mask to add empty fields to bg and lables
                    clip_mask = clip_mask | empty_mask.numpy()
                    
                bg[clip_mask, :] = True    # merge clipped areas with background
                labels[clip_mask, :] = 0   # mask clipped areas in labels
                # calibration data
                with open(img_path.parent.parent / 'calib_folder.txt', 'rb') as f: calib_folder = Path(f.readline().strip().decode('utf-8'))
                c_path = self.base_dir / calib_folder / (str(wlen)+'nm')
                amat = read_cod_data_X3D(c_path / (str(wlen) + '_A.cod'))
                wmat = read_cod_data_X3D(c_path / (str(wlen) + '_W.cod'))
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
        split_dim = [i for i, s in enumerate(labels.shape) if s == self.class_num+1][0]
        labels, bg = torch.split(labels, [self.class_num, 1], dim=split_dim)

        return frames, labels, img_class, bg, metadata

    def __len__(self):
        return len(self.ids)

class PatchHORAO(HORAO):
    def __init__(self, *args, **kwargs):
        self.patch_size = kwargs.pop('patch_size', 50)
        super(PatchHORAO, self).__init__(*args, **kwargs)
        self.b = self.patch_size // 2

    def __getitem__(self, i):
        # run through conventional dataloader
        frames, labels, img_class, bg, metadata = super().__getitem__(i)
        
        # get arbitrary 2d coordinate pair from a labeled pixel
        binary_map_crop = torch.any(labels, dim=0, keepdim=False)[self.b:-self.b, self.b:-self.b]
        # get coordinate indices (preferably where labels are)
        non_zero_indices = torch.nonzero(binary_map_crop if binary_map_crop.sum() > 0 else ~binary_map_crop, as_tuple=False)
        random_index = torch.randint(0, non_zero_indices.size(0), (1,)).item()
        coords = non_zero_indices[random_index] + self.b

        # select patch based on 2d coordinate
        patch = frames[:, coords[0]-self.b:coords[0]+self.b, coords[1]-self.b:coords[1]+self.b]
        
        return patch, labels[:, coords[0], coords[1]], img_class, bg[:, coords[0], coords[1]], metadata

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
    base_dir = '/home/chris/Datasets/03_HORAO/NPP'
    feat_keys = ['intensity', 'azimuth', 'linr', 'totp']

    img_list = []
    transforms = [ToTensor(), RandomPolarRotation(180, p=0, any=False), SwapDims()]
    train_set = HORAO(base_dir, ['k1_b_npp_imbalance.txt', 'k2_b_npp_imbalance.txt'], bg_opt=bg_opt, class_num=4, wlens=[550], transforms=transforms)
    valid_set = HORAO(base_dir, ['val2_b_npp.txt'], bg_opt=bg_opt, class_num=4, wlens=[550], transforms=transforms)
    test_set = HORAO(base_dir, ['k3_b_npp_imbalance.txt'], bg_opt=bg_opt, class_num=4, wlens=[550], transforms=transforms)
    dataset = torch.utils.data.ConcatDataset([train_set, valid_set, test_set])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    from mm.models import LuChipmanPyramid as MMM
    mm_model = MMM(feature_keys=feat_keys, perc=.95, levels=1, kernel_size=0, method='averaging', wnum=len(train_set.wlens), filter_opt=False)

    bg_pixels = 0
    twm_pixels = 0
    hwm_pixels = 0
    hgm_pixels = 0
    tgm_pixels = 0
    tumor_samples = 0
    healthy_samples = 0
    for batch in loader:
        imgs, labels, img_class, bg, metadata = batch

        # move feature dimension for consistency
        imgs = imgs.moveaxis(-1, 1)
        labels = labels.moveaxis(-1, 1)

        bg_pixels += (labels[:, 0]>0).sum().item()
        hwm_pixels += (labels[:, 1]>0).sum().item()
        twm_pixels += (labels[:, 3]>0).sum().item()
        hgm_pixels += (labels[:, 2]>0).sum().item()
        tgm_pixels += (labels[:, 4]>0).sum().item()
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

        from utils.multi_loss import reduce_htgm
        labels = reduce_htgm(torch.zeros_like(labels), labels)[1]

        if False:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, imgs.shape[1])
            axs[0].set_title(['healthy', 'tumor'][img_class[0]])
            axs[0].imshow(imgs[0][0], cmap='gray')
            for i in range(1, imgs.shape[1]):
                axs[i].imshow(labels[0, i-1])
            plt.tight_layout()
            plt.show()

        img_list.append(imgs)

    class_balance = {
        'Background': bg_pixels,
        'Healthy white matter': hwm_pixels,
        'Tumor white matter': twm_pixels,
        'Healthy grey matter': hgm_pixels,
        'Tumor grey matter': tgm_pixels,
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
