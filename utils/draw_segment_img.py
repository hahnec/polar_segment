import torch
from torchvision.utils import draw_segmentation_masks


def draw_segmentation_imgs(imgs, preds, truth, bidx=0, th=0.5, alpha=0.5):
    
    img = (imgs[bidx, 0][None].repeat(3, 1, 1)/imgs[bidx, 0].max()*255).cpu().to(torch.uint8)
    combined_masks = torch.stack((preds[bidx, -1]>th[-1], preds[bidx, -2]>th[-2], truth[bidx, -1]>0, truth[bidx, -2]>0)).cpu()
    frame_pred = draw_segmentation_masks(img, masks=combined_masks[:2], alpha=alpha, colors=['blue', 'orange'])
    frame_mask = draw_segmentation_masks(img, masks=combined_masks[2:], alpha=alpha, colors=['green', 'red'])
    #imageio.imsave('./mask.png', (best_frame.permute(1,2,0)/best_frame.max()*255).cpu().numpy().astype(np.uint8))

    return frame_pred, frame_mask
