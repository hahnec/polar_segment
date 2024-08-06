import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image


class HORAO(Dataset):
    def __init__(
            self, 
            path, 
            cases_file, 
            transforms=[], 
            transforms_img=[], 
            bg_opt=0, 
            benign_accumulate = True,
            wlens=[550], 
            data_subfolder='polarimetry',
            keys=['azimuth', 'std'],
        ):

        self.base_dir = Path(path)

        self.transforms = transforms
        self.transforms_img = transforms_img
        self.data_subfolder = data_subfolder
        self.wlens = wlens
        self.bg_opt = int(bool(bg_opt))
        self.benign_accumulate = bool(benign_accumulate)
        
        if self.data_subfolder.__contains__('polarimetry'):
            self.keys = [self.map_string(k) for k in keys]

        with open(self.base_dir / 'cases' / cases_file) as f:
            self.ids = [line.rstrip('\n') for line in f]

        self.get_filenames()

    def get_filenames(self):

        filename = (str(self.wlens[0]) + '_Intensite.cod') if self.data_subfolder.__contains__('raw') else 'MM.npz'

        self.img_paths = []
        self.label_paths = []
        self.img_classes = []
        for id in self.ids:
            img_fname = self.base_dir / str(id) / self.data_subfolder / (str(self.wlens[0])+'nm') / filename
            self.img_paths.append(img_fname)
            assert img_fname.exists(), f'No image found for the ID {id}: {img_fname}'

            if id.startswith('HT'):
                label_fname = self.base_dir / str(id) / 'annotation' / 'WM_merged.png'
                img_class = 0
            else:
                label_fname = self.base_dir / str(id) / 'annotation' / 'ROI.tif'
                img_class = 1
            self.label_paths.append(label_fname)
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
    def create_multilabels(labels, matter_labels):
        hwm = np.equal(labels[..., -2], matter_labels[..., -1])
        hgm = np.equal(labels[..., -2], matter_labels[..., -2])
        twm = np.equal(labels[..., -1], matter_labels[..., -1])
        tgm = np.equal(labels[..., -1], matter_labels[..., -2])
        new = np.stack([hwm, hgm, twm, tgm], axis=-1).astype(float)
        if len(labels.shape) == 3:
            bg = ~np.any(new, -1)[..., None]
            return np.concatenate([bg.astype(float), new], axis=-1)
        
        return new

    def __getitem__(self, i):

        img_path = self.img_paths[i]
        label_fname = self.label_paths[i]
        img_class = self.img_classes[i]

        # label construction
        labels = np.array(Image.open(label_fname))#, dtype=np.float32)
        labels = labels[None].repeat(2, 0)
        labels[~img_class] = 0
        if img_class == 0 and self.benign_accumulate:
            labels = labels.astype(bool)
            labels[0] = True
            fnames = self.label_paths[i].parent.glob('BG_*.tif')
            for fname in fnames:
                labels[0] = labels[0] & (np.array(Image.open(fname)) == 0).astype(bool)
        if self.bg_opt:
            bg = (labels.sum(0) == 0).astype(labels.dtype)
            labels = np.concatenate((bg[None] * labels.max(), labels), axis=0)
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
                matter_labels = np.zeros_like(matter_labels[..., 0])
                matter_labels[matter_labels[..., 1]==77] = 1
                matter_labels[matter_labels[..., 1]==153] = 2
            oh_mat_labels = np.eye(3)[matter_labels.astype(int)]
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
                try:
                    from libmpMuelMat_dev_1_0.libmpMuelMat import removeReflections3D
                    frame, mask = removeReflections3D(np.array(frame), maxThr=65530-float(bruit.max()))
                    mask = ~mask
                except ImportError:
                    from utils.specular_removal_cv import specular_removal_cv
                    frame, mask = specular_removal_cv(np.array(intensity, dtype=np.float32), size=4, method='navier')
                    #from utils.specular_removal_torch import specular_removal_t
                    #frame = specular_removal_t(frame, intensity>65530)
                if self.cases_file.__contains__('train'): labels[mask.astype(bool), :] = 0    # mask clipped areas
                # calibration
                amat = read_cod_data_X3D(str(img_path).replace('raw_data', 'calibration').replace('Intensite', 'A'))
                wmat = read_cod_data_X3D(str(img_path).replace('raw_data', 'calibration').replace('Intensite', 'W'))
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

        for transform in self.transforms:
            frames, labels = transform(frames, label=labels)

        return frames, labels, img_class

    def __len__(self):
        return len(self.ids)

class PatchHORAO(HORAO):
    def __init__(self, *args, **kwargs):
        self.patch_size = kwargs.pop('patch_size', 50)
        super(PatchHORAO, self).__init__(*args, **kwargs)
        self.b = self.patch_size // 2

    def __getitem__(self, i):
        # run through conventional dataloader
        frames, labels, img_class = super().__getitem__(i)
        
        # get arbitrary 2d coordinate pair from a labeled pixel
        binary_map_crop = torch.any(labels, dim=0, keepdim=False)[self.b:-self.b, self.b:-self.b]
        non_zero_indices = torch.nonzero(binary_map_crop, as_tuple=False)
        try:
            random_index = torch.randint(0, non_zero_indices.size(0), (1,)).item()
        except:
            print(non_zero_indices.shape)
            random_index = torch.randint(0, binary_map_crop.numel(), (1,)).item()
        coords = non_zero_indices[random_index] + self.b

        # select patch based on 2d coordinate
        patch = frames[:, coords[0]-self.b:coords[0]+self.b, coords[1]-self.b:coords[1]+self.b]
        
        return patch, labels[:, coords[0], coords[1]], img_class

if __name__ == '__main__':

    import time
    import random
    from torch.utils.data import DataLoader
    from utils.mm_rotation import RawRandomMuellerRotation
    from utils.transforms_segment import *

    random.seed(3008)
    torch.manual_seed(3008)
    torch.cuda.manual_seed_all(3008)
    batch_size = 1
    patch_size = 4
    bg_opt = 1
    base_dir = '/media/chris/EB62-383C/TumorMeasurementsCalib/'
    feat_keys = ['azimuth'] #, 'linr', 'totp', 'std'] #

    img_list = []
    for data_type in ['raw_data', 'polarimetry']:
        transforms = [ToTensor(), RawRandomMuellerRotation(180, p=1, any=False), SwapDims()] if data_type.__contains__('raw_data') else []
        dataset = HORAO(base_dir, 'val1.txt', bg_opt=bg_opt, data_subfolder=data_type, keys=feat_keys, wlens=[550], transforms=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        
        from mm.models import MuellerMatrixPyramid as MMM
        mm_model = MMM(feature_keys=feat_keys, perc=.95, levels=1, kernel_size=0, method='averaging', wnum=len(dataset.wlens), mask_fun=None, filter_opt=True)

        bg_pixels = 0
        tumor_pixels = 0
        tumor_samples = 0
        healthy_pixels = 0
        healthy_samples = 0
        for batch in loader:
            imgs, masks, label = batch

            healthy_pixels += (masks[:, -2]>0).sum().item()
            tumor_pixels += (masks[:, -1]>0).sum().item()
            bg_pixels += (masks[:, 0]>0).sum().item()

            healthy_samples += (label==0).sum().item()
            tumor_samples += (label!=0).sum().item()

            # move feature dimension for consistency
            imgs = imgs.moveaxis(-1, 1)
            if dataset.data_subfolder.__contains__('raw'):
                t = time.perf_counter()
                imgs = mm_model(imgs)
                t_total = time.perf_counter() -t
                print('MM processing time: %s' % str(t_total))

            if True:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 4)
                axs[0].set_title(['healthy', 'tumor'][label[0]])
                axs[0].imshow(imgs[0][0])
                axs[1].imshow(masks[0, ..., 0])
                axs[2].imshow(masks[0, ..., -1])
                axs[3].imshow(masks[0, ..., -2])
                plt.tight_layout()
                plt.show()
            
            img_list.append(imgs)

            print('%s healthy pixels' % str(healthy_pixels))
            print('%s healthy samples' % str(healthy_samples))
            print('%s tumor pixels' % str(tumor_pixels))
            print('%s tumor samples' % str(tumor_samples))
            print('%s bg pixels' % str(bg_pixels))

    print('comparison')
    s = 0
    h = len(img_list) // 2

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, h//4)
    for i in range(h//4):
        #axs[0, i].set_title('polarimetry')
        #axs[1, i].set_title('raw_data')
        a = img_list[0+s+i][0, 0].detach().cpu()
        b = img_list[h+s+i][0, 0].detach().cpu()
        m = img_list[h+s+i][0, -1].detach().cpu()
        #c = abs(a-b)
        c[m==0] = 0
        axs[0, i].imshow(a)
        axs[1, i].imshow(b)
        axs[2, i].imshow(c)
    plt.tight_layout()
    plt.show()

    diffs = []
    for i in range(h):
        mask = img_list[h+i][:, -1].bool()
        diff = img_list[i][:, 0][mask] - img_list[h+i][:, 0][mask]
        diffs.append(diff.abs().mean())
        print(diffs[-1])
    print('total diff: %s' % torch.tensor(diffs).mean())
