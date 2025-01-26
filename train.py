import os
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
from polar_augment.flip_raw import RandomPolarFlip
from polar_augment.rotation_raw import RandomPolarRotation
from polar_augment.noise import RandomGaussNoise
from polar_augment.gamma import GammaAugmentation
from polar_augment.batch_segment_shuffle import BatchSegmentShuffler
from utils.transforms_segment import RandomResizedCrop
from mm.models import init_mm_model
from utils.draw_fiber_img import plot_fiber
from utils.duplicate_checks import check_duplicate_rows
from utils.reproducibility import set_seed_and_deterministic


def batch_preprocess(batch, cfg):
    
    # device
    frames, truth, img_class, bg, text = batch
    frames = frames.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
    truth = truth.to(device=cfg.device, dtype=frames.dtype)
    bg = bg.to(device='cpu', dtype=bool)

    # perform batch-wise shuffle transform
    if random.random() < cfg.shuffle_crop and frames.shape[0] > 1:
        frames, truth = BatchSegmentShuffler('crop')(frames, truth)

    # extract intensity images for plots
    imgs = frames[:, :16].clone().mean(1) if cfg.data_subfolder.__contains__('raw') else frames[:, 0].clone()
    imgs.detach()

    return frames, truth, imgs, bg, text

def batch_iter(frames, truth, cfg, model, train_opt=0, criterion=None, optimizer=None, grad_scaler=None, grad_clip=1.0):
    
    # initialize label selection
    m = torch.any(truth, dim=1, keepdim=True) if cfg.labeled_only else torch.ones_like(truth)
    if m.shape != truth[:, 0:1, ...].shape: m = m.repeat(1, 1, 1, 1)
    #m = m.repeat_interleave(cfg.class_num, 1)

    # remove the realizability mask from the features
    if cfg.data_subfolder.__contains__('raw') and 'mask' in cfg.feature_keys:
        wnum = len(cfg.wlens)
        mask = frames[:, -wnum:]
        frames = frames[:, :-wnum]
        if len(mask.shape) > len(m.shape): mask = mask.mean(-1).mean(-1)
        m = (m.float() * mask).bool()

    # inference
    if not train_opt:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    preds = model(frames)
    if not train_opt:
        end.record()
        torch.cuda.synchronize()
    t_s = start.elapsed_time(end) / 1000 if not train_opt else torch.tensor([float('NaN')])

    # loss and back-propagation
    loss = None
    if criterion and preds.numel() > 0:
        loss = criterion(preds, truth)
        loss = loss * m.squeeze(1)
        loss = loss.sum() / (m.sum() + 1e-8)
    if train_opt and loss is not None and torch.isfinite(m).all() and m.any(): 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # reduce prediction to healthy/tumor white matter and gray matter classes
    if cfg.class_num > 2:
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

    criterion = torch.nn.CrossEntropyLoss() if not cfg.imbalance or branch_type == 'test' else (lambda x, y: sigmoid_focal_loss_multiclass(x, y, alpha=cfg.alpha, gamma=cfg.gamma).mean())
    if cfg.class_num > 3 and branch_type != 'test':
        from utils.multi_loss import multi_loss_aggregation
        criterion = torch.nn.CrossEntropyLoss(reduction='none')  if not cfg.imbalance or branch_type == 'test' else (lambda x, y: sigmoid_focal_loss_multiclass(x, y, alpha=cfg.alpha, gamma=cfg.gamma, reduction='none').mean())
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
            frames, truth, imgs, bg, text = batch_preprocess(batch, cfg)
            ## polarimetry
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            input = mm_model(frames) if cfg.data_subfolder.__contains__('raw') else frames
            end.record()
            torch.cuda.synchronize()
            t_mm = start.elapsed_time(end) / 1000
            # segmentation
            loss, preds, truth, metrics = batch_it(input, truth)
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
            
            # image results
            score = metrics['acc']
            if torch.any(score > best_score) and 'intensity' in cfg.feature_keys and cfg.logging and log_img:
                bidx = score.argmax()
                best_score = score[bidx]
                best_frame_pred, best_frame_mask = draw_segmentation_imgs(imgs, preds, truth, bidx=bidx, bg_opt=cfg.bg_opt)
                best_frame_text = text[bidx]
                if not cfg.bg_opt:
                    alpha = (~bg[bidx]).float()*255
                    best_frame_pred = torch.cat((best_frame_pred, alpha), dim=0)
                    best_frame_mask = torch.cat((best_frame_mask, alpha), dim=0)
            if torch.any(score < poor_score) and 'intensity' in cfg.feature_keys and cfg.logging and log_img:
                bidx = score.argmin()
                poor_score = score[bidx]
                poor_frame_pred, poor_frame_mask = draw_segmentation_imgs(imgs, preds, truth, bidx=bidx, bg_opt=cfg.bg_opt)
                poor_frame_text = text[bidx]
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
                        'img_pred_'+branch_type: wandb.Image(frame_pred.cpu(), caption=text[bidx]),
                        'img_mask_'+branch_type: wandb.Image(frame_mask.cpu(), caption=text[bidx]), 
                        'heatmap_'+branch_type: wandb.Image(heatmap, caption=text[bidx]), 
                        branch_type+'_step': step+bidx
                    })
                    # fiber tracts image
                    if cfg.data_subfolder.__contains__('raw'):
                        from mm.models import LuChipmanModel
                        azimuth_model = LuChipmanModel(feature_keys=['azimuth', 'linr'])
                        lc_feats = azimuth_model(frames)
                        masks = preds.argmax(1) == 0 # predicted healthy white matter mask
                        vars = [var[bidx].cpu().numpy() for var in [lc_feats, masks, imgs]]
                        mask = ~(vars[1] & ~bg[bidx, 0].numpy())
                        # tbd: adjust fiber plot, which fails after rectification providing correct linear retardance
                        fiber_img = plot_fiber(raw_azimuth=vars[0][0], linr=vars[0][1], mask=mask, intensity=vars[2])
                        wandb.log({
                            'img_fiber_'+branch_type: wandb.Image(fiber_img, caption=text[bidx]),
                            #branch_type+'_step': step+bidx
                        })

            # metrics extension
            for k in metrics_dict.keys():
                metrics_dict[k].extend(metrics[k].detach().cpu().numpy())

    if cfg.logging and log_img:
        if best_frame_pred is not None: wandb.log({'best_img_pred_'+branch_type: wandb.Image(best_frame_pred.cpu(), caption=best_frame_text), 'epoch': epoch})
        if best_frame_mask is not None: wandb.log({'best_img_mask_'+branch_type: wandb.Image(best_frame_mask.cpu(), caption="green: healthy-GT; red: tumor-GT; blue: GM;"), 'epoch': epoch})
        if poor_frame_pred is not None: wandb.log({'poor_img_pred_'+branch_type: wandb.Image(poor_frame_pred.cpu(), caption=poor_frame_text), 'epoch': epoch})
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

    # reproducibility
    set_seed_and_deterministic(seed=cfg.seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')

    # augmentation transforms
    raw_opt = True if cfg.data_subfolder.__contains__('raw') else False
    transforms = [
            ToTensor(), 
            RandomPolarRotation(degrees=cfg.rotation, p=.5, fill=[0]*int(cfg.class_num)+[1]) if cfg.rotation > 0 and raw_opt else EmptyTransform(),
            RandomPolarFlip(orientation=0, p=.5) if cfg.flips and raw_opt else EmptyTransform(),
            RandomPolarFlip(orientation=1, p=.5) if cfg.flips and raw_opt else EmptyTransform(),
            RandomPolarFlip(orientation=2, p=.5) if cfg.flips and raw_opt else EmptyTransform(),
            GammaAugmentation(gamma_range=(0.5, 2.0)) if cfg.gamma else EmptyTransform(),
            RandomGaussNoise(mean=0.0, std=0.05, p=0.5) if cfg.noise > 0 else EmptyTransform(),
        ]

    # mueller matrix model
    mm_model = init_mm_model(cfg, filter_opt=False) if cfg.data_subfolder.__contains__('raw') else None

    # model selection
    n_channels = mm_model.ochs if cfg.data_subfolder.__contains__('raw') else len(cfg.feature_keys)
    if cfg.model == 'mlp':
        from segment_models.mlp import MLP
        model = MLP(n_channels=n_channels, n_classes=cfg.class_num+cfg.bg_opt)
        if False: # image patch-wise data input
            from horao_dataset import PatchHORAO as HORAO
            from segment_models.mlp import PatchMLP
            model = PatchMLP(n_channels=n_channels, patch_size=50, n_classes=cfg.class_num+cfg.bg_opt)
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

    # create datasets
    if cfg.imbalance: cfg.cases = [fname.split('.txt')[0] + '_imbalance.txt' if not fname.startswith('val') else fname for fname in cfg.cases]
    check_duplicate_rows(Path(cfg.data_dir) / 'cases', cfg.cases)
    from utils.kfold_splits import get_nested_kfold_splits
    splits = get_nested_kfold_splits(cfg.cases)
    train_cases, test_cases, valid_cases = splits[cfg.k_select]
    dataset = HORAO(cfg.data_dir, train_cases, transforms=transforms, class_num=cfg.class_num, bg_opt=cfg.bg_opt, data_subfolder=cfg.data_subfolder, keys=cfg.feature_keys, wlens=cfg.wlens, use_no_border=False)
    val_set = HORAO(cfg.data_dir, valid_cases, transforms=[ToTensor()], class_num=cfg.class_num, bg_opt=cfg.bg_opt, data_subfolder=cfg.data_subfolder, keys=cfg.feature_keys, wlens=cfg.wlens, use_no_border=False)

    # reproducibility when using multiple workers
    g = torch.Generator().manual_seed(cfg.seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # create data loaders
    num_workers = min(2, os.cpu_count()) if cfg.num_workers is None else cfg.num_workers
    loader_args = dict(num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(dataset, shuffle=True, drop_last=False, batch_size=cfg.batch_size, **loader_args)
    valid_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size=len(dataset), **loader_args)

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
        wb = wandb.init(project='polar_segment_opex', resume='allow', anonymous='must', config=dict(cfg), group=cfg.group, entity='hahnec')
        wb.config.update(dict(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.lr, amp=cfg.amp))

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

    train_step, valid_step = (0, 0)
    best_model, best_mm_model = (model, mm_model)
    best_epoch_score, best_epoch, epochs_decline = (float('-inf'), -1, 0)
    for epoch in range(1, cfg.epochs+1):
        # training
        with torch.enable_grad():
            model, mm_model, lmetrics_dict, train_step, tloss = epoch_iter(cfg, train_loader, model, mm_model, branch_type='train', step=train_step, log_img=0, epoch=epoch, optimizer=optimizer, grad_scaler=grad_scaler)
        # validation
        with torch.no_grad():
            model, mm_model, vmetrics_dict, valid_step, vloss = epoch_iter(cfg, valid_loader, model, mm_model, branch_type='valid', step=valid_step, log_img=cfg.model not in ('resnet', 'mlp') and epoch==cfg.epochs, epoch=epoch)
        epoch_score = vmetrics_dict['dice']

        if cfg.logging:
            # logging of weights
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not torch.isinf(value).any() and not torch.isnan(value).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if value.grad is not None:
                    if not torch.isinf(value.grad).any() and not torch.isnan(value.grad).any():
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
            # logging of histograms
            wb.log({
                **histograms,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_score': epoch_score,
                'epoch': epoch,
            })
        scheduler.step()

        # best model selection
        if best_epoch_score < epoch_score:
            best_epoch_score = epoch_score
            best_model = copy.deepcopy(model).eval()
            best_epoch = epoch
            if cfg.data_subfolder.__contains__('raw') and cfg.kernel_size > 0:
                best_mm_model = copy.deepcopy(mm_model).eval()
            epochs_decline = 0
        else:
            epochs_decline += 1
            if cfg.patience is not None and epochs_decline >= cfg.patience:
                break

    # save weights
    if cfg.logging:
        dir_checkpoint = Path('./ckpts/')
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        state_dict = best_model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / (wb.name+str('_ckpt_epoch{}.pt'.format(best_epoch)))))
        if cfg.data_subfolder.__contains__('raw') and cfg.kernel_size > 0:
            state_dict_mm = best_mm_model.state_dict()
            torch.save(state_dict_mm, str(dir_checkpoint / (wb.name+str('_mm_ckpt_epoch{}.pt'.format(best_epoch)))))
        logging.info(f'Checkpoint {best_epoch} saved!')
        wb.log({'best_epoch': best_epoch})

    # adjust settings for patch-wise models
    if cfg.model in ('resnet', 'mlp'): best_model.testing = True
    from horao_dataset import HORAO # override patch-wise dataloader for ResNet

    # perform test
    #test_cases = test_cases + ['test_tumor_grade4.txt', 'test_tumor_grade3.txt', 'test_tumor_grade2.txt']
    from test import test_main
    test_set = HORAO(
        cfg.data_dir, 
        test_cases, 
        transforms=[ToTensor()], 
        class_num=cfg.class_num, 
        bg_opt=cfg.bg_opt, 
        data_subfolder=cfg.data_subfolder, 
        keys=cfg.feature_keys, 
        wlens=cfg.wlens,
        use_no_border=False,
        )
    test_main(cfg, test_set, best_model, best_mm_model)
