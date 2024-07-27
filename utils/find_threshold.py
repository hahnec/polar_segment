import numpy as np
from sklearn.metrics import f1_score

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
        best_threshold = 0
        best_f1 = 0

        for threshold in thresholds:
            # Apply threshold to predictions for the current class
            binary_predictions = (predictions[:, class_idx] >= threshold).astype(int)
            # Calculate F1 score
            f1 = f1_score(ground_truth[:, class_idx], binary_predictions)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds.append(best_threshold)
    
    return optimal_thresholds


if __name__ == '__main__':

    predictions = np.array([[0.1, 0.4], [0.6, 0.8], [0.5, 0.3]])
    ground_truth = np.array([[0, 0], [1, 1], [1, 0]])
    num_classes = 2

    optimal_thresholds = find_optimal_threshold(predictions, ground_truth, num_classes)
    print("Optimal thresholds:", optimal_thresholds)
