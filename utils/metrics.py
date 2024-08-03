import torch

def compute_dice_score(preds, truth, mask=None):
    """
    Compute the Dice score for valid pixels.

    Parameters:
    - preds (torch.Tensor): Predictions of shape (classes, height, width)
    - truth (torch.Tensor): Ground truth of shape (classes, height, width)
    - mask (torch.Tensor): Validity mask of shape (height, width)

    Returns:
    - dice_score (float): Dice score for valid pixels
    """

    if mask is not None: mask = mask.bool()

    # Flatten the tensors
    preds_flat = preds[:, mask] if mask is not None else preds
    truth_flat = truth[:, mask] if mask is not None else truth
    
    numer = 2. * (preds_flat * truth_flat).sum()
    denom = preds_flat.sum() + truth_flat.sum()
    score = numer / denom

    # Handle zero division
    if denom == 0:
        if truth_flat.sum() == 0:
            score = torch.tensor(1.0, device=preds.device)
        else:
            score = torch.tensor(0.0, device=preds.device)

    return score

def compute_iou(preds, truth, mask=None):
    """
    Compute the IoU for valid pixels.

    Parameters:
    - preds (torch.Tensor): Predictions of shape (classes, height, width)
    - truth (torch.Tensor): Ground truth of shape (classes, height, width)
    - mask (torch.Tensor): Validity mask of shape (height, width)

    Returns:
    - iou (float): IoU for valid pixels
    """

    if mask is not None: mask = mask.bool()

    # Flatten the tensors
    preds_flat = preds[:, mask] if mask is not None else preds
    truth_flat = truth[:, mask] if mask is not None else truth
    
    numer = (preds_flat * truth_flat).sum()
    denom = (preds_flat + truth_flat).sum() - numer
    score = numer / denom

    # Handle zero division
    if denom == 0:
        if truth_flat.sum() == 0:
            score = torch.tensor(1.0, device=preds.device)
        else:
            score = torch.tensor(0.0, device=preds.device)

    return score

def compute_accuracy(preds, truth, mask=None):

    if mask is not None: mask = mask.bool()

    preds_flat = preds[:, mask] if mask is not None else preds
    truth_flat = truth[:, mask] if mask is not None else truth
    
    eq_map = torch.eq(preds_flat, truth_flat)
    acc = eq_map.sum() / eq_map.numel()
    
    return acc