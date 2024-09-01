import os
import time
import copy
import logging
import random
import numpy as np
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from omegaconf import OmegaConf
from monai import transforms

from horao_dataset import HORAO
from utils.multi_focal_loss import sigmoid_focal_loss_multiclass
from utils.transforms_segment import *
from utils.metrics import compute_dice_score, compute_iou, compute_accuracy
from utils.draw_segment_img import draw_segmentation_imgs, draw_heatmap
from polar_augment.augmentations.flip_raw import RandomPolarFlip
from polar_augment.augmentations.rotation_raw import RandomPolarRotation
from polar_augment.augmentations.batch_segment_shuffle import BatchSegmentShuffler
from mm.models import init_mm_model

def batch_preprocess(batch, cfg):
    
    # device
    frames, truth, img_class, bg = batch
    imgs = frames[:, :16].clone().mean(1) if cfg.data_subfolder.__contains__('raw') else frames[:, 0].clone()
    frames = frames.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
    truth = truth.to(device=cfg.device, dtype=frames.dtype)
    bg = bg.to(device='cpu', dtype=bool)

    if random.random() < cfg.shuffle_crop and frames.shape[0] > 1:
        frames, truth = BatchSegmentShuffler('mask')(frames, truth)

    return frames, truth, imgs, bg

def batch_iter(frames, truth, cfg, model, train_opt=0, criterion=None, optimizer=None, grad_scaler=None, gradient_clipping=1.0):
    
    # initialize label selection
    m = torch.any(truth, dim=1, keepdim=True).repeat(1, truth.shape[1], 1, 1) if cfg.labeled_only else torch.ones_like(truth)

    # remove the feasibility mask from the features
    if cfg.data_subfolder.__contains__('raw') and 'mask' in cfg.feature_keys:
        wnum = len(cfg.wlens)
        mask = frames[:, -wnum:]
        frames = frames[:, :-wnum]
        m = (m.float() * mask).bool()

    with torch.autocast(cfg.device if cfg.device != 'mps' else 'cpu', enabled=cfg.amp):
        t_s = time.perf_counter()
        preds = model(frames)
        t_s = time.perf_counter() - t_s
        loss = criterion(preds*m, truth*m) if criterion and len(preds) > 0 else torch.tensor(float('nan'))

    if train_opt and not torch.isnan(loss):
        if True:
            optimizer.zero_grad(set_to_none=True)
            scale = grad_scaler.get_scale()
            grad_scaler.update()
            skip_lr_schedule = scale > grad_scaler.get_scale()

            if not skip_lr_schedule:
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
        else:
            loss.backward()

    if cfg.class_num > 2:
        # reduce prediction to healthy/tumor white matter and gray matter classes
        from utils.multi_loss import reduce_htgm
        preds, truth = reduce_htgm(preds, truth)

    # metrics
    mask = torch.any(truth, dim=1)
    ious, accs, dices = [torch.zeros(truth.shape[0], dtype=torch.float) for _ in range(3)]
    for i in range(truth.shape[0]):
        preds_b = torch.nn.functional.one_hot(preds.argmax(1), num_classes=preds.shape[1]).float()
        if len(preds_b.shape) == 4: preds_b = preds_b.permute(0, 3, 1, 2)
        ious[i] = compute_iou(preds_b[i], truth[i], mask=mask[i]).detach()
        accs[i] = compute_accuracy(preds_b[i], truth[i], mask=mask[i]).detach()
        dices[i] = compute_dice_score(preds_b[i], truth[i], mask=mask[i]).detach()
    metrics = {'dice': dices, 'iou': ious, 'acc': accs, 't_s': torch.tensor([t_s/frames.size(0)])}

    return loss, preds, truth, metrics

def epoch_iter(cfg, dataloader, model, mm_model=None, branch_type='test', step=None, log_img=False, epoch=None, optimizer=None, grad_scaler=None):

    criterion = (lambda x, y: sigmoid_focal_loss_multiclass(x, y).mean()) if branch_type != 'test' else None
    if cfg.class_num > 3 and branch_type != 'test':
        from utils.multi_loss import multi_loss_aggregation
        criterion = lambda x, y: multi_loss_aggregation(x, y, loss_fun=lambda x, y: sigmoid_focal_loss_multiclass(x, y).mean())
    train_opt = 0 if optimizer is None else 1
    model.train() if train_opt else model.eval()
    batch_it = lambda f, t: batch_iter(f, t, cfg=cfg, model=model, train_opt=train_opt, criterion=criterion, optimizer=optimizer, grad_scaler=grad_scaler)
    desc = f'Steps {len(dataloader.dataset)}' if epoch is None else f'Epoch {epoch}/{cfg.epochs}'

    step = 0 if step is None else step
    epoch_loss = 0
    metrics_dict = {'dice': [], 'iou': [], 'acc': [], 't_mm': [], 't_s': []}
    best_score, best_frame_pred, best_frame_mask = 0, None, None
    poor_score, poor_frame_pred, poor_frame_mask = 1, None, None
    with tqdm(total=len(dataloader.dataset), desc=desc+' '+branch_type, unit='img') as pbar:
        for batch in dataloader:
            frames, truth, imgs, bg = batch_preprocess(batch, cfg)
            t = time.perf_counter()
            if cfg.data_subfolder.__contains__('raw'): frames = mm_model(frames)
            t_mm = time.perf_counter() - t
            loss, preds, truth, metrics = batch_it(frames, truth)
            metrics['t_mm'] = torch.tensor([t_mm/frames.size(0)])
            step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update(batch[0].shape[0])
            if cfg.logging:
                wandb.log({
                    branch_type+'_loss': loss.item(),
                    branch_type+'_dice': metrics['dice'].mean().item(),
                    branch_type+'_iou ': metrics['iou'].mean().item(),
                    branch_type+'_acc ': metrics['acc'].mean().item(),
                    branch_type+'_step': step,
                })

            score = metrics['acc']
            if torch.any(score > best_score) and 'intensity' in cfg.feature_keys and cfg.logging and log_img:
                bidx = score.argmax()
                best_score = score[bidx]
                best_frame_pred, best_frame_mask = draw_segmentation_imgs(imgs, preds, truth, bidx=bidx, bg_opt=cfg.bg_opt)
                if not cfg.bg_opt:
                    alpha = (~bg[bidx]).float()*255
                    best_frame_pred = torch.cat((best_frame_pred, alpha), dim=0)
                    best_frame_mask = torch.cat((best_frame_mask, alpha), dim=0)
            if torch.any(score < poor_score) and 'intensity' in cfg.feature_keys and cfg.logging and log_img:
                bidx = score.argmin()
                poor_score = score[bidx]
                poor_frame_pred, poor_frame_mask = draw_segmentation_imgs(imgs, preds, truth, bidx=bidx, bg_opt=cfg.bg_opt)
                if not cfg.bg_opt:
                    alpha = (~bg[bidx]).float()*255
                    poor_frame_pred = torch.cat((poor_frame_pred, alpha), dim=0)
                    poor_frame_mask = torch.cat((poor_frame_mask, alpha), dim=0)
            # log all test images
            if cfg.logging and branch_type == 'test':
                for bidx in range(truth.shape[0]):
                    frame_pred, frame_mask = draw_segmentation_imgs(imgs, preds, truth, bidx=bidx, bg_opt=cfg.bg_opt)
                    out_class = int(batch[2][bidx]) + int(cfg.bg_opt)
                    hmask = (preds[bidx].argmax(0) == 0) if cfg.bg_opt else None
                    heatmap = draw_heatmap(preds[bidx, out_class], img=imgs[bidx], mask=hmask)
                    if not cfg.bg_opt:
                        alpha = (~bg[bidx]).float()*255
                        frame_pred = torch.cat((frame_pred, alpha), dim=0)
                        frame_mask = torch.cat((frame_mask, alpha), dim=0)
                        heatmap = np.concatenate((heatmap, (~bg[bidx]).float().moveaxis(0, -1).cpu().numpy()), axis=-1)
                    wandb.log({
                        'img_pred_'+branch_type: wandb.Image(frame_pred.cpu(), caption=['benign', 'malignant'][int(batch[2][bidx])]),
                        'img_mask_'+branch_type: wandb.Image(frame_mask.cpu(), caption=['benign-GT', 'malignant-GT'][int(batch[2][bidx])]), 
                        'heatmap_'+branch_type: wandb.Image(heatmap, caption="heatmap " + ['benign', 'malignant'][int(batch[2][bidx])]), 
                        branch_type+'_step': step+bidx
                    })

            # metrics extension
            for k in metrics_dict.keys():
                metrics_dict[k].extend(metrics[k].detach().cpu().numpy())

    if cfg.logging and log_img:
        if best_frame_pred is not None: wandb.log({'best_img_pred_'+branch_type: wandb.Image(best_frame_pred.cpu(), caption="green: healthy; red: tumor; blue: GM;"), 'epoch': epoch})
        if best_frame_mask is not None: wandb.log({'best_img_mask_'+branch_type: wandb.Image(best_frame_mask.cpu(), caption="green: healthy-GT; red: tumor-GT; blue: GM;"), 'epoch': epoch})
        if poor_frame_pred is not None: wandb.log({'poor_img_pred_'+branch_type: wandb.Image(poor_frame_pred.cpu(), caption="green: healthy; red: tumor; blue: GM;"), 'epoch': epoch})
        if poor_frame_mask is not None: wandb.log({'poor_img_mask_'+branch_type: wandb.Image(poor_frame_mask.cpu(), caption="green: healthy-GT; red: tumor-GT; blue: GM;"), 'epoch': epoch})

    # consolidate metrics to one scalar value per key
    for k in metrics_dict.keys():
        metrics_dict[k] = float(np.array(metrics_dict[k]).mean())

    if branch_type == 'test':
        return preds, truth, metrics_dict
    else:
        return model, mm_model, metrics_dict, step, epoch_loss


if __name__ == '__main__':

    # load configuration
    cfg = OmegaConf.load('./configs/train_local.yml')

    # override loaded configuration with server settings
    if Path(cfg.ubx_dir).exists(): cfg = OmegaConf.merge(cfg, OmegaConf.load('./configs/train_server.yml'))

    # override loaded configuration with CLI arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    # for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)    # multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')

    # augmentation transforms
    raw_opt = True if cfg.data_subfolder.__contains__('raw') else False
    transforms = [
            ToTensor(), 
            RandomCrop(size=int(cfg.crop)) if cfg.crop > 0 else EmptyTransform(), 
            RandomPolarRotation(degrees=cfg.rotation, p=.5, fill=[0,0,0,0,1]) if cfg.rotation > 0 and raw_opt else EmptyTransform(),
            RandomPolarFlip(orientation=0, p=.5) if cfg.flips and raw_opt else EmptyTransform(),
            RandomPolarFlip(orientation=1, p=.5) if cfg.flips and raw_opt else EmptyTransform(),
            RandomPolarFlip(orientation=2, p=.5) if cfg.flips and raw_opt else EmptyTransform(),
            #transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
            #Normalize(mean=0, std=1), 
        ]

    # mueller matrix model
    mm_model = init_mm_model(cfg, filter_opt=False) if cfg.data_subfolder.__contains__('raw') else None

    # model selection
    n_channels = mm_model.ochs if cfg.data_subfolder.__contains__('raw') else len(cfg.feature_keys)
    if cfg.model == 'mlp':
        from segment_models.mlp import MLP
        model = MLP(n_channels=n_channels, n_classes=cfg.class_num+cfg.bg_opt)
    elif cfg.model == 'resnet':
        from horao_dataset import PatchHORAO as HORAO
        from segment_models.resnet import PatchResNet
        model = PatchResNet(n_channels=n_channels, n_classes=cfg.class_num+cfg.bg_opt, patch_size=50)
    elif cfg.model == 'unet':
        from segment_models.unet import UNet
        model = UNet(n_channels=n_channels, n_classes=cfg.class_num+cfg.bg_opt, shallow=cfg.shallow)
    elif cfg.model == 'unetpp':
        from segment_models.unet_pp import UnetPP
        model = UnetPP(n_channels, out_channels=cfg.class_num+cfg.bg_opt)
    elif cfg.model == 'uctransnet':
        from segment_models.uctransnet.UCTransNet import UCTransNet
        from segment_models.uctransnet.Config import get_CTranS_config
        model = UCTransNet(n_channels=n_channels, n_classes=cfg.class_num+cfg.bg_opt, in_channels=64, img_size=cfg.crop, config=get_CTranS_config())
    else:
        raise Exception('Model %s not recognized' % cfg.model)

    # model weights initialization
    from segment_models.weights_init import initialize_weights
    if cfg.model in ('mlp', 'unet', 'resnet', 'uctransnet'):
        model.apply(initialize_weights)
    if mm_model is not None:
        mm_model.apply(initialize_weights)

    model = model.to(memory_format=torch.channels_last)
    model.to(device=cfg.device)

    # create dataset
    kfold_names = ['k1.txt', 'k2.txt', 'k3.txt']
    splits = [(kfold_names[:i] + kfold_names[i+1:], [kfold_names[i]]) for i in range(len(kfold_names))]
    train_cases, test_cases = splits[cfg.k_select]
    dataset = HORAO(cfg.data_dir, train_cases, transforms=transforms, class_num=cfg.class_num, bg_opt=cfg.bg_opt, data_subfolder=cfg.data_subfolder, keys=cfg.feature_keys, wlens=cfg.wlens)
    if (Path(cfg.data_dir) / 'cases' / 'val2.txt').exists():
        val_set = HORAO(cfg.data_dir, ['val2.txt'], transforms=transforms, class_num=cfg.class_num, bg_opt=cfg.bg_opt, data_subfolder=cfg.data_subfolder, keys=cfg.feature_keys, wlens=cfg.wlens)
    else:
        # split into train and validation partitions (if needed)
        n_val = int(len(dataset) * cfg.val_fraction)
        n_train = len(dataset) - n_val
        dataset, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    # create data loaders
    num_workers = min(2, os.cpu_count())
    loader_args = dict(batch_size=cfg.batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, drop_last=False, **loader_args)
    valid_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    if cfg.model_file is not None:
        ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('_')[0])]
        state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
        if cfg.model == 'uctransnet':
            state_dict = state_dict['state_dict']
            state_dict['inc.conv.weight'] = state_dict['inc.conv.weight'][:, :2, :, :].repeat(1, n_channels//2, 1, 1)
            state_dict['outc.weight'] = state_dict['outc.weight'].expand(2+dataset.bg_opt, -1, -1, -1).clone()
            state_dict['outc.bias'] = state_dict['outc.bias'].expand(2+dataset.bg_opt).clone()
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {cfg.model_file}')

    # instantiate logging
    if cfg.logging:
        wb = wandb.init(project='polar_segment', resume='allow', anonymous='must', config=dict(cfg), group=cfg.group)
        wb.config.update(dict(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.lr, val_fraction=cfg.val_fraction, amp=cfg.amp))

        logging.info(f'''Starting training:
            cfg.epochs:      {cfg.epochs}
            Batch size:      {cfg.batch_size}
            Learning rate:   {cfg.lr}
            Training size:   {len(dataset)}
            Validation size: {len(val_set)}
            Device:          {cfg.device}
            Mixed Precision: {cfg.amp}
        ''')

    # set up the optimizer, the loss, the learning rate scheduler and the loss scaling
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) #, foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    train_step, valid_step = 0, 0
    best_model, best_mm_model, best_epoch_score = model, mm_model, 0
    for epoch in range(1, cfg.epochs+1):
        # training
        with torch.enable_grad():
            model, mm_model, metrics_dict, train_step, tloss = epoch_iter(cfg, train_loader, model, mm_model, branch_type='train', step=train_step, log_img=0, epoch=epoch, optimizer=optimizer, grad_scaler=grad_scaler)
        # validation
        with torch.no_grad():
            model, mm_model, metrics_dict, valid_step, vloss = epoch_iter(cfg, valid_loader, model, mm_model, branch_type='valid', step=valid_step, log_img=cfg.model!='resnet' and epoch==cfg.epochs, epoch=epoch)

        # best model selection
        epoch_score = vloss
        if best_epoch_score > epoch_score:
            best_epoch_score = epoch_score
            best_model = copy.deepcopy(model).eval()
            if cfg.data_subfolder.__contains__('raw') and cfg.kernel_size > 0:
                best_mm_model = copy.deepcopy(mm_model).eval()

        if cfg.logging:
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not torch.isinf(value).any() and not torch.isnan(value).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if value.grad is not None:
                    if not torch.isinf(value.grad).any() and not torch.isnan(value.grad).any():
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
        if cfg.logging:
            wb.log({
                **histograms,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            })

        scheduler.step()

    # save weights
    if cfg.logging:
        dir_checkpoint = Path('./ckpts/')
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        state_dict = best_model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / (wb.name+str('_ckpt_epoch{}.pth'.format(epoch)))))
        if cfg.data_subfolder.__contains__('raw') and cfg.kernel_size > 0:
            state_dict_mm = best_mm_model.state_dict()
            torch.save(state_dict_mm, str(dir_checkpoint / (wb.name+str('_mm_ckpt_epoch{}.pth'.format(epoch)))))
        logging.info(f'Checkpoint {epoch} saved!')

    # adjust settings for patch-wise ResNet model
    if cfg.model == 'resnet': best_model.testing = True
    from horao_dataset import HORAO # override patch-wise dataloader for ResNet

    # perform test
    from test import test_main
    test_set = HORAO(cfg.data_dir, test_cases, transforms=[ToTensor()], class_num=cfg.class_num, bg_opt=cfg.bg_opt, data_subfolder=cfg.data_subfolder, keys=cfg.feature_keys, wlens=cfg.wlens)
    test_main(cfg, test_set, best_model, best_mm_model)
