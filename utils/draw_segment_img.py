import torch
from torchvision.utils import draw_segmentation_masks


def draw_segmentation_imgs(imgs, preds, truth, bidx=0, th=None, alpha=0.5):

    if th is None:
        class1 = preds[bidx, -1] > preds[bidx, -2]
        class2 = preds[bidx, -2] > preds[bidx, -1]
        if preds.shape[1] == 3:
            # consider background predictions
            class1[preds[bidx, -1] < preds[bidx, 0]] = 0
            class2[preds[bidx, -2] < preds[bidx, 0]] = 0
        combined_masks = torch.stack((class1, class2, truth[bidx, -1]>0, truth[bidx, -2]>0)).cpu()
    else:
        combined_masks = torch.stack((preds[bidx, -1]>th[bidx][-1], preds[bidx, -2]>th[bidx][-2], truth[bidx, -1]>0, truth[bidx, -2]>0)).cpu()
    img = (imgs[bidx, 0][None].repeat(3, 1, 1)/imgs[bidx, 0].max()*255).cpu().to(torch.uint8)
    frame_pred = draw_segmentation_masks(img, masks=combined_masks[:2], alpha=alpha, colors=['orange', 'blue'])
    frame_mask = draw_segmentation_masks(img, masks=combined_masks[2:], alpha=alpha, colors=['red', 'green'])

    return frame_pred, frame_mask


def draw_heatmap(pred, img=None, mask=None, alpha=0.5, colormap='jet'):

    if isinstance(pred, torch.Tensor): pred = pred.cpu().numpy()
    norm = (pred - pred.min()) / (pred.max() - pred.min())

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(norm)

    if mask is not None:
        if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
        heatmap[mask==0] = 0

    if img is not None:
        if isinstance(img, torch.Tensor): img = img.cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        heatmap = img * (1 - alpha) + heatmap * alpha
    
    return heatmap
