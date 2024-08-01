import torch

def compute_dice_score(preds, truth, mask=None):
    """
    Compute the Dice score for valid pixels.

    Parameters:
    - preds (torch.Tensor): Predictions of shape (batch_size, classes, height, width)
    - truth (torch.Tensor): Ground truth of shape (batch_size, classes, height, width)
    - mask (torch.Tensor): Validity mask of shape (batch_size, 1, height, width)

    Returns:
    - dice_score (float): Dice score for valid pixels
    """

    if mask is not None: mask = mask.bool()

    # Flatten the tensors
    preds_flat = preds[mask].flatten() if mask is not None else preds.flatten()
    truth_flat = truth[mask].flatten() if mask is not None else truth.flatten()
    
    intersection = (preds_flat * truth_flat).sum()
    dice_score = (2. * intersection) / (preds_flat.sum() + truth_flat.sum())
    
    return dice_score

def compute_iou(preds, truth, mask=None):
    """
    Compute the IoU for valid pixels.

    Parameters:
    - preds (torch.Tensor): Predictions of shape (batch_size, classes, height, width)
    - truth (torch.Tensor): Ground truth of shape (batch_size, classes, height, width)
    - mask (torch.Tensor): Validity mask of shape (batch_size, 1, height, width)

    Returns:
    - iou (float): IoU for valid pixels
    """

    if mask is not None: mask = mask.bool()

    # Flatten the tensors
    preds_flat = preds[mask].flatten() if mask is not None else preds.flatten()
    truth_flat = truth[mask].flatten() if mask is not None else truth.flatten()
    
    intersection = (preds_flat * truth_flat).sum()
    union = preds_flat.sum() + truth_flat.sum() - intersection
    iou = intersection / union
    
    return iou

def compute_accuracy(preds, truth, mask=None):

    if mask is not None: mask = mask.bool()

    preds_flat = preds[mask].flatten() if mask is not None else preds.flatten()
    truth_flat = truth[mask].flatten() if mask is not None else truth.flatten()
    
    eq_map = torch.eq(preds_flat, truth_flat)
    acc = eq_map.sum() / len(eq_map)
    
    return acc