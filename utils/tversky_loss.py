import torch
import torch.nn.functional as F

epsilon = 1e-5
smooth = 1.0

def dsc(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    score = (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
    return score

def dice_loss(y_true, y_pred):
    return 1 - dsc(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def confusion(y_true, y_pred):
    y_pred_pos = torch.clamp(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.clamp(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = (y_pos * y_pred_pos).sum()
    fp = (y_neg * y_pred_pos).sum()
    fn = (y_pos * y_pred_neg).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall

def tp(y_true, y_pred):
    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pos = torch.round(torch.clamp(y_true, 0, 1))
    tp = (y_pos * y_pred_pos).sum() + smooth
    return tp / (y_pos.sum() + smooth)

def tn(y_true, y_pred):
    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.round(torch.clamp(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (y_neg * y_pred_neg).sum() + smooth
    return tn / (y_neg.sum() + smooth)

def tversky(y_true, y_pred, alpha=0.7):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    true_pos = (y_true_f * y_pred_f).sum()
    false_neg = (y_true_f * (1 - y_pred_f)).sum()
    false_pos = ((1 - y_true_f) * y_pred_f).sum()
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7):
    return 1 - tversky(y_true, y_pred, alpha)

def focal_tversky(y_true, y_pred, alpha=0.7, gamma=0.75):
    pt_1 = tversky(y_true, y_pred, alpha)
    return torch.pow((1 - pt_1), gamma)
