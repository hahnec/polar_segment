import torch
from torchvision.utils import draw_segmentation_masks


def draw_segmentation_imgs(imgs, preds, truth, bidx=0, th=None, alpha=0.3):

    n_channels = truth.shape[1]
    if th is None:
        preds_b = torch.nn.functional.one_hot(preds.argmax(1), num_classes=truth.shape[1]).permute(0, 3, 1, 2).bool()
        combined_masks = torch.stack((preds_b[bidx], truth[bidx]>0)).cpu()
    else:
        combined_masks = torch.stack((preds[bidx]>th[bidx], truth[bidx]>0)).cpu()
    img = (imgs[bidx][None].repeat(3, 1, 1)/imgs[bidx].max()*255).cpu().to(torch.uint8)
    frame_pred = draw_segmentation_masks(img, masks=combined_masks[0], alpha=alpha, colors=['cyan', 'blue', 'green', 'orange', 'red'][:n_channels])
    frame_mask = draw_segmentation_masks(img, masks=combined_masks[1], alpha=alpha, colors=['cyan', 'blue', 'green', 'orange', 'red'][:n_channels])

    return frame_pred, frame_mask


def draw_heatmap(pred, img=None, mask=None, alpha=0.3, colormap='jet'):

    if isinstance(pred, torch.Tensor): pred = pred.cpu().numpy()
    norm = (pred - pred.min()) / (pred.max() - pred.min())

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(norm)

    if mask is not None:
        if isinstance(mask, torch.Tensor): mask = mask.bool().cpu().numpy()
        heatmap[mask] = 0

    if img is not None:
        if isinstance(img, torch.Tensor): img = img.cpu().numpy()
        if len(img.shape) == 2: img = img[..., None]
        img = (img - img.min()) / (img.max() - img.min())
        heatmap = img * (1 - alpha) + heatmap[..., :3] * alpha
    
    return heatmap
