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


if __name__ == '__main__':

    predictions = np.array([[0.1, 0.4], [0.6, 0.8], [0.5, 0.3]])
    ground_truth = np.array([[0, 0], [1, 1], [1, 0]])
    num_classes = 2

    optimal_thresholds = find_optimal_threshold(predictions, ground_truth, num_classes)
    print("Optimal thresholds:", optimal_thresholds)
