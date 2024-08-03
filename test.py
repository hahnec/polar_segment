import os
import logging
import random
import numpy as np
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from horao_dataset import HORAO
from utils.transforms_segment import *

from mm.models import init_mm_model
from train import epoch_iter


def test_main(cfg, dataset, model, mm_model, th=None):

    # create data loaders
    num_workers = min(2, os.cpu_count())
    loader_args = dict(batch_size=len(dataset), num_workers=num_workers, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)

    with torch.no_grad():
        preds, truth, metrics = epoch_iter(cfg, dataloader, model, mm_model, branch_type='test', th=th)

    # pixel-wise assessment
    n_channels = int(preds.shape[1])
    m = torch.any(truth, dim=1).flatten().cpu().numpy() if cfg.labeled_only else np.ones(truth[:, 0].shape, dtype=bool).flatten()
    y_true = truth.argmax(1).flatten().cpu().numpy()
    y_pred = preds.argmax(1).flatten().cpu().numpy()
    
    try:
        from sklearn.metrics import classification_report
        report = classification_report(y_true[m], y_pred[m], target_names=['bg', 'benign', 'malignant'][-n_channels:], digits=4, output_dict=bool(cfg.logging))
    except ValueError as e:
        print(e)

    if cfg.logging:
        # upload other metrics to wandb
        wandb.log(metrics)
        wandb.log({'accuracy': metrics['acc']})
        table_metrics = wandb.Table(columns=list(metrics.keys()), data=[list(metrics.values())])
        wandb.log({'metrics': table_metrics})
        # convert report to wandb table
        flat_report = flatten_dict_to_rows(report)
        table_report = wandb.Table(columns=['category', 'precision', 'recall', 'f1-score', 'support', 'accuracy'])
        for row in flat_report:
            table_report.add_data(row['category'], row.get('precision'), row.get('recall'), row.get('f1-score'), row.get('support'), row.get('accuracy'))
        wandb.log({'report': table_report})
    else: 
        with open('./results.txt', "a") as f:
            f.write(report)
            f.write('\n')
            print(report)
            print('\n')
            for k, v in metrics.items():
                f.write('%s: %s' % (k, str(v)))
                print('%s: %s' % (k, str(v)))

def flatten_dict_to_rows(d):
    rows = []
    for key, value in d.items():
        if isinstance(value, dict):
            row = {'category': key}
            row.update(value)
            rows.append(row)
        else:
            rows.append({'category': 'accuracy', 'precision': None, 'recall': None, 'f1-score': None, 'support': None, 'accuracy': value})
    return rows


if __name__ == '__main__':

    # load configuration
    cfg = OmegaConf.load('./configs/train_local.yml')

    # override loaded configuration with test config
    cfg = OmegaConf.merge(cfg, OmegaConf.load('./configs/test.yml'))
    if not 'intensity' in cfg.feature_keys: cfg.feature_keys = cfg.feature_keys + ['intensity']

    # override loaded configuration with CLI arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    
    # for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')

    # mueller matrix model
    if cfg.data_subfolder.__contains__('raw'):
        mm_model = init_mm_model(cfg, train_opt=False)
        if cfg.kernel_size > 0:
            ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.mm_model_file.split('_')[0])]
            state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)['state_dict']
            mm_model.load_state_dict(state_dict)
            logging.info(f'MM Model loaded from {cfg.mm_model_file}')
    else:
        mm_model = None

    # create dataset
    dataset = HORAO(cfg.data_dir, 'test.txt', transforms=[ToTensor()], bg_opt=cfg.bg_opt, data_subfolder=cfg.data_subfolder, keys=cfg.feature_keys, wlens=cfg.wlens)

    # model selection
    n_channels = mm_model.ochs if cfg.data_subfolder.__contains__('raw') else len([k for k in dataset.keys if k != 'intensity'])
    if cfg.model == 'mlp':
        from segment_models.mlp import MLP
        model = MLP(n_channels=n_channels, n_classes=2+dataset.bg_opt,)
    elif cfg.model == 'resnet':
        from horao_dataset import PatchHORAO as HORAO
        from segment_models.resnet import PatchResNet
        model = PatchResNet(n_channels=n_channels, n_classes=2+cfg.bg_opt, patch_size=50, testing=True)
    elif cfg.model == 'unet':
        from segment_models.unet import UNet
        model = UNet(n_channels=n_channels, n_classes=2+dataset.bg_opt, shallow=cfg.shallow)
    elif cfg.model == 'unetpp':
        from segment_models.unet_pp import get_pretrained_unet_pp
        model = get_pretrained_unet_pp(n_channels, out_channels=2+dataset.bg_opt)
    elif cfg.model == 'uctransnet':
        from segment_models.uctransnet.UCTransNet import UCTransNet
        from segment_models.uctransnet.Config import get_CTranS_config
        model = UCTransNet(n_channels=n_channels, n_classes=2+dataset.bg_opt, in_channels=64, img_size=cfg.crop, config=get_CTranS_config())
    else:
        raise Exception('Model name not recognized')

    model = model.to(memory_format=torch.channels_last)
    model.to(device=cfg.device)
    model.eval()

    if cfg.model_file is not None:
        ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('_')[0])]
        state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
        model.load_state_dict(state_dict) if cfg.model != 'resnet' else model.model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {cfg.model_file}')

    # instantiate logging
    if cfg.logging:
        wb = wandb.init(project='polar_segment_test', resume='allow', anonymous='must', config=dict(cfg), group='train')
        wb.config.update(dict(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.lr, val_fraction=cfg.val_fraction, amp=cfg.amp))

        logging.info(f'''Starting testing:
            Batch size:      {cfg.batch_size}
            Test size:       {len(dataset)}
            Device:          {cfg.device}
        ''')
    
    from utils.find_threshold import get_threshold
    th = get_threshold(cfg, dataset, model, mm_model) if not cfg.labeled_only else None
    test_main(cfg, dataset, model, mm_model, th=th)