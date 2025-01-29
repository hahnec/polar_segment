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
            benign_accumulate = False,
            wlens=[550], 
            data_subfolder='polarimetry',
            keys=['azimuth', 'std'],
            class_num=4,
            use_no_border = False,
        ):

        self.base_dir = Path(path)

        self.transforms = transforms
        self.transforms_img = transforms_img
        self.data_subfolder = data_subfolder
        self.wlens = wlens
        self.bg_opt = int(bool(bg_opt))
        self.benign_accumulate = bool(benign_accumulate)
        self.cases_files = list(cases_file)
        self.use_no_border = use_no_border
        
        if self.data_subfolder.__contains__('polarimetry'):
            self.keys = [self.map_string(k) for k in keys]

        self.ids = []
        for cases_fn in self.cases_files:
            with open(self.base_dir / 'cases' / cases_fn) as f:
                self.ids = self.ids + [line.rstrip('\n') for line in f if line.strip()]

        self.get_filenames(class_num=class_num)

        # read metadata
        self.df = pd.read_csv(self.base_dir / 'batch_processing.csv')
        self.df = self.df.drop('Unnamed: 7', axis=1)
        self.df = self.df.drop('Number of measurements kept', axis=1)
        self.df = self.df.drop('Clinical diagnosis', axis=1) # redundant as it exists in another column

    def get_filenames(self, class_num=2):

        filename = (str(self.wlens[0]) + '_Intensite.cod') if self.data_subfolder.__contains__('raw') else 'MM.npz'

        self.img_paths = []
        self.label_paths = []
        self.bglabel_paths = []
        self.img_classes = []
        self.matter_paths = []
        for id in self.ids:
            img_fname = self.base_dir / str(id) / self.data_subfolder / (str(self.wlens[0])+'nm') / filename
            self.img_paths.append(img_fname)
            assert img_fname.exists(), f'No image found for the ID {id}: {img_fname}'

            if id.startswith('HT'):
                label_fname = self.base_dir / str(id) / 'annotation' / ('merged_no_border.png' if self.use_no_border else 'merged.png')
                bglabel_fname = self.base_dir / str(id) / 'annotation' / 'BG_merged.png'
                img_class = 0
            else:
                label_fname = self.base_dir / str(id) / 'annotation' / 'FG.tif'
                bglabel_fname = self.base_dir / str(id) / 'annotation' / 'ROI.tif'
                img_class = 1
                if not label_fname.exists(): label_fname = bglabel_fname
            if class_num > 2:
                if id.startswith('HT'):
                    matter_fname = self.base_dir / str(id) / 'annotation' / ('merged_no_border.png' if self.use_no_border else 'merged.png')
                else:
                    fpath = self.base_dir / str(id) / 'histology'
                    fname = 'labels_augmented_GM_WM_masked_FG.png' if (fpath / 'labels_augmented_GM_WM_masked_FG.png').exists() else 'labels_augmented_GM_WM_masked.png' 
                    matter_fname = fpath / ('labels_augmented_GM_WM_masked_FG_no_border.png' if self.use_no_border else fname)
                
                self.matter_paths.append(matter_fname)

            self.label_paths.append(label_fname)
            self.bglabel_paths.append(bglabel_fname)
            self.img_classes.append(img_class)
            assert label_fname.exists(), f'No label found for the ID {id}: {label_fname}'

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
    
    def get_metadata(self, seq: int, section: str):

        row = self.df[(self.df['Sample Nr'] == seq) & (self.df['Section'] == section)]
        if not row.empty and 'Histology Tumor' in row.columns:
            row_str = '| '.join([str(el) for el in row.values[0].tolist()])
        else:
            row_str = 'unlabeled tumor'

        return row_str

    def __getitem__(self, i):

        img_path = self.img_paths[i]
        label_fname = self.label_paths[i]
        img_class = self.img_classes[i]

        # load metadata
        if img_class: seq, section = int(img_path.parts[-5]), img_path.parts[-4].split('_')[-2]
        metadata = self.get_metadata(seq, section) if img_class else 'healthy'

        # label construction
        labels = np.array(Image.open(label_fname))
        labels = labels[None].repeat(2, 0)
        labels[~img_class] = 0
        if img_class == 0 and self.benign_accumulate:
            labels = labels.astype(bool)
            labels[0] = True
            fnames = self.label_paths[i].parent.glob('BG_*.tif')
            for fname in fnames:
                labels[0] = labels[0] & (np.array(Image.open(fname)) == 0).astype(bool)
        bg = np.array(Image.open(self.bglabel_paths[i]), bool)[..., None]
        if img_class == 1: bg = ~bg
        if self.bg_opt:
            labels = np.concatenate((bg.swapaxes(2, 1).swapaxes(1, 0).astype(labels.dtype) * labels.max(), labels), axis=0)
        labels = labels.swapaxes(0, 1).swapaxes(1, 2)
        labels = labels.astype(np.float32)
        if labels.max() > 1: labels /= 255

        # consider white matter / grey matter
        if len(self.matter_paths) > 0: 
            matter_fname = self.matter_paths[i]
            matter_labels = np.array(Image.open(matter_fname))
            if img_class == 0:
                matter_labels[matter_labels==128] = 1
                matter_labels[matter_labels==255] = 2
            else:
                # WM/GM encoding is different for tumor data
                placeholder = np.zeros_like(matter_labels[..., 0])
                placeholder[matter_labels[..., 1]==77] = 1
                placeholder[matter_labels[..., 1]==153] = 2
                matter_labels = placeholder
            oh_mat_labels = np.eye(3)[matter_labels.astype(int)]
            # replace foreground/background mask with inverted GM/WM mask to use full GM/WM area
            if not self.use_no_border: labels[..., -2:][..., np.any(labels, axis=(0, 1))[-2:]] = 1-oh_mat_labels[..., 0][..., None].astype(labels.dtype)
            # rearrange H,T,WM,GM to HWM, HGM, TWM, TGM
            labels = self.create_multilabels(labels, oh_mat_labels)

        # iterate over wavelengths
        frames = []
        for wlen in self.wlens:
            # update wavelength path
            img_path = Path(str(img_path).replace(str(self.wlens[0]), str(wlen)))

            if img_path.name.endswith('cod'):
                # intensity
                from mm.utils.cod import read_cod_data_X3D
                intensity = read_cod_data_X3D(img_path, raw_flag=True)
                bruit = read_cod_data_X3D(str(img_path).replace('_Intensite.cod', '_Bruit.cod'), raw_flag=True)
                frame = intensity - bruit
                # clipping
                clip_detect = lambda img, th=65530: np.any(img > th, axis=-1).astype(bool)
                clip_mask = clip_detect(intensity.numpy())
                bg[clip_mask, :] = True    # merge clipped areas with background
                labels[clip_mask, :] = 0    # mask clipped areas
                # calibration
                amat = read_cod_data_X3D(str(img_path).replace('raw_data', 'calibration').replace('Intensite', 'A'))
                wmat = read_cod_data_X3D(str(img_path).replace('raw_data', 'calibration').replace('Intensite', 'W'))
                # realizability mask
                if False:
                    from mm.functions.mm import compute_mm
                    mm = compute_mm(amat, wmat, frame)
                    from mm.functions.mm_filter import charpoly
                    valid_mask = charpoly(mm)
                    bg[~clip_mask, :] = True    # merge infeasible areas with background
                    labels[~valid_mask, :] = 0    # mask infeasible areas

                frame = np.concatenate([frame, amat, wmat], axis=-1)
                #if frame.max() > 2**8-1: frame /= (2**16-1)
                #if frame.max() > 1: frame /= 255
            elif img_path.name.endswith('npz'):
                polarimetry_dict = np.load(img_path)
                polarimetry_dict = {k.lower(): v for k, v in polarimetry_dict.items()}    # enforce lower case keys in dictionary
                ims_to_stack = []
                for k in self.keys:
                    if k in polarimetry_dict.keys():
                        polarimetry_data = polarimetry_dict[k]
                    if k == 'azimuth_local_var':
                        percentile95 = np.percentile(polarimetry_data, 95)
                        polarimetry_data[polarimetry_data > percentile95] = percentile95
                    if k == 'intensity':
                        polarimetry_data = polarimetry_data.mean(-1)
                    ims_to_stack.append(polarimetry_data)
                frame = np.stack(ims_to_stack, 0)
                frame = frame.swapaxes(0, 1).swapaxes(1, 2).squeeze()
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
        labels, bg = labels[:-1], labels[-1][None]

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
    base_dir = '/home/chris/Datasets/03_HORAO/TumorMeasurementsCalib/'
    feat_keys = ['std', 'mask'] #'azimuth', 'linr', 'totp'] #

    img_list = []
    for data_type in ['polarimetry', 'raw_data']:
        transforms = [ToTensor(), RandomPolarRotation(180, p=0, any=False), SwapDims()] if data_type.__contains__('raw_data') else []
        train_set = HORAO(base_dir, ['k1_b_imbalance.txt', 'k2_b_imbalance.txt'], bg_opt=bg_opt, class_num=4, data_subfolder=data_type, keys=feat_keys, wlens=[550], transforms=transforms)
        valid_set = HORAO(base_dir, ['val2_b.txt'], bg_opt=bg_opt, class_num=4, data_subfolder=data_type, keys=feat_keys, wlens=[550], transforms=transforms)
        test_set = HORAO(base_dir, ['k3_b_imbalance.txt'], bg_opt=bg_opt, class_num=4, data_subfolder=data_type, keys=feat_keys, wlens=[550], transforms=transforms)
        dataset = torch.utils.data.ConcatDataset([train_set, valid_set, test_set])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        
        from mm.models import MuellerMatrixPyramid as MMM
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

            if train_set.data_subfolder.__contains__('raw'):
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
                fig, axs = plt.subplots(1, 4)
                axs[0].set_title(['healthy', 'tumor'][img_class[0]])
                axs[0].imshow(imgs[0][0])
                axs[1].imshow(masks[0, ..., 0])
                axs[2].imshow(masks[0, ..., -1])
                axs[3].imshow(masks[0, ..., -2])
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

    print('comparison')
    s = 1
    h = len(img_list) // 2

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, h//4)
    for i in range(h//4):
        axs[0, i].set_title('mm_torch')
        axs[1, i].set_title('libmpMuelMat')
        axs[2, i].set_title('log. difference')
        a = img_list[0+s+i][0, 0].detach().cpu()
        b = img_list[h+s+i][0, 0].detach().cpu()
        m = img_list[0+s+i][0, -1].detach().cpu()
        c = abs(a-b)
        c[m==0] = 0
        axs[0, i].imshow(a)
        axs[1, i].imshow(b)
        axs[2, i].imshow(np.log(c))
    plt.tight_layout()
    plt.show()

    diffs = []
    for i in range(h):
        mask = img_list[0+s+i][:, -1].bool()
        diff = img_list[i][:, 0][mask] - img_list[h+i][:, 0][mask]
        diffs.append(diff.abs().nanmean())
        print(diffs[-1])
    print('total diff: %s' % torch.tensor(diffs).nanmean())
