import os
import torch
import random
import imageio
import logging
from pathlib import Path
from omegaconf import OmegaConf

from mm.models import init_mm_model
from utils.draw_segment_img import draw_segment_maps


def infer_dataloader(cfg, case, model, mm_model):

    # import lib required for dataloader appraoch
    from tqdm import tqdm
    from horao_dataset import HORAO
    from torch.utils.data import DataLoader
    from utils.transforms_segment import ToTensor

    # create dataset
    dataset = HORAO(cfg.data_dir, [case], transforms=[ToTensor()], bg_opt=cfg.bg_opt, wlens=cfg.wlens, class_num=cfg.class_num)

    # create data loader
    batch_size = 1
    num_workers = min(1, os.cpu_count())
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)

    with torch.no_grad():
        with tqdm(total=len(dataloader.dataset), unit='img') as pbar:
            for i, batch in enumerate(dataloader):
                # preprocess
                frame, truth, img_class, bg, text = batch
                frame = frame.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
                bg = bg.bool()
                # inference
                preds, frame_segment = iter(frame, model, mm_model, bg=bg)
                pbar.update(frame.shape[0])
                # write image
                imageio.imsave('frame_segment_%s.png' % str(i), frame_segment.permute(1,2,0).numpy().astype('uint8'))

def iter(raw_frame, model, mm_model=None, log_img=True, bg=None):

    # polarimetry
    frame = mm_model(raw_frame) if mm_model else raw_frame

    # segmentation
    preds = model(frame)

    # image results
    if log_img:
        imgs = raw_frame.mean(1) # grascale image
        frame_segment = draw_segment_maps(imgs, preds, bg=bg)

    return preds, frame_segment

if __name__ == '__main__':

    # load configuration
    cfg = OmegaConf.load('./configs/infer.yml')
    if not 'intensity' in cfg.feature_keys: cfg.feature_keys = cfg.feature_keys + ['intensity']

    # override loaded configuration with CLI arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    
    # for reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # mueller matrix model
    mm_model = init_mm_model(cfg, train_opt=False, filter_opt=False)
    if cfg.kernel_size > 0:
        ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.mm_model_file.split('_')[0])]
        state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)['state_dict']
        mm_model.load_state_dict(state_dict)
        logging.info(f'MM Model loaded from {cfg.mm_model_file}')

    # segmentation model
    if cfg.model == 'unet':
        from segment_models.unet import UNet
        model = UNet(n_channels=mm_model.ochs, n_classes=cfg.class_num+cfg.bg_opt, shallow=cfg.shallow)
    else:
        raise Exception('Model %s not recognized' % cfg.model)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=cfg.device)
    model.eval()

    if cfg.model_file is not None:
        ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('.')[0])]
        state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
        model.load_state_dict(state_dict) if cfg.model != 'resnet' else model.model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {cfg.model_file}')
    else:
        raise Exception('No model file provided')

    for case in cfg.cases:
        # iterate through case dataset
        infer_dataloader(cfg, case, model, mm_model)
