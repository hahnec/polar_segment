import numpy as np
from sklearn.metrics import roc_curve

def find_optimal_threshold(predictions, ground_truth, num_classes, thresholds=np.arange(0, 1.05, 0.05)):
    """
    Find the optimal threshold for each class based on the F1 score.
    
    Parameters:
    - predictions: numpy array of shape (num_samples, num_classes) with predicted probabilities.
    - ground_truth: numpy array of shape (num_samples, num_classes) with binary ground truth values.
    - num_classes: int, number of classes or output channels.
    - thresholds: numpy array, range of threshold values to evaluate.

    Returns:
    - optimal_thresholds: list of optimal thresholds for each class.
    """
    optimal_thresholds = []

    for class_idx in range(num_classes):
        fpr, tpr, thresholds = roc_curve(ground_truth[class_idx], predictions[class_idx])
        gmeans = (tpr * (1-fpr))**.5
        th_idx = np.argmax(gmeans)
        best_threshold = thresholds[th_idx]

        # catch borderline cases
        if np.isinf(best_threshold) or np.isnan(best_threshold) or best_threshold == 0: best_threshold = 0.5
        
        optimal_thresholds.append(best_threshold)

    return optimal_thresholds

def get_threshold(cfg, dataset, model, mm_model):

    import torch
    from torch.utils.data import DataLoader
    from train import batch_iter, batch_preprocess

    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)
    batch_it = lambda f, t: batch_iter(f, t, cfg=cfg, model=model, train_opt=0)

    preds_list, truth_list = [], []
    for batch in loader:
        frames, truth = batch_preprocess(batch, cfg)
        if cfg.data_subfolder.__contains__('raw'): frames = mm_model(frames)
        preds = batch_it(frames, truth)[1]
        truth_list.append(truth.detach())
        preds_list.append(preds.detach())
    truth = torch.cat(truth_list, dim=0)
    preds = torch.cat(preds_list, dim=0)

    y_pred = preds.moveaxis(0, -1).reshape(2+cfg.bg_opt, -1).detach().cpu().numpy()
    y_true = truth.moveaxis(0, -1).reshape(2+cfg.bg_opt, -1).detach().cpu().numpy()
    th = find_optimal_threshold(y_pred, y_true, num_classes=2+cfg.bg_opt)
    th = torch.tensor(th, device=preds.device)[None, :, None, None]

    return th
