import torch

def compute_dice_score(preds, truth, mask):
    """
    Compute the Dice score for valid pixels.

    Parameters:
    - preds (torch.Tensor): Predictions of shape (batch_size, classes, height, width)
    - truth (torch.Tensor): Ground truth of shape (batch_size, classes, height, width)
    - mask (torch.Tensor): Validity mask of shape (batch_size, 1, height, width)

    Returns:
    - dice_score (float): Dice score for valid pixels
    """

    mask = mask.bool()

    # Flatten the tensors
    pred_flat = preds[mask].view(-1)
    target_flat = truth[mask].view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection) / (pred_flat.sum() + target_flat.sum())
    
    return dice_score

def compute_iou(preds, truth, mask):
    """
    Compute the IoU for valid pixels.

    Parameters:
    - preds (torch.Tensor): Predictions of shape (batch_size, classes, height, width)
    - truth (torch.Tensor): Ground truth of shape (batch_size, classes, height, width)
    - mask (torch.Tensor): Validity mask of shape (batch_size, 1, height, width)

    Returns:
    - iou (float): IoU for valid pixels
    """

    mask = mask.bool()

    # Flatten the tensors
    pred_flat = preds[mask].view(-1)
    target_flat = truth[mask].view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = intersection / union
    
    return iou
