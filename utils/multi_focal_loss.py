import torch
from torchvision.ops import sigmoid_focal_loss


def sigmoid_focal_loss_multiclass(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Multi-class implementation of the sigmoid focal loss using one-hot encoded targets.

    Args:
        inputs (Tensor): A float tensor of shape (N, C, *) where C is the number of classes.
        targets (Tensor): A float tensor of the same shape as inputs. The targets should be one-hot encoded.
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default: 2.
        reduction (str): 'none' | 'mean' | 'sum'. Specifies the reduction to apply to the output: 'none' (no reduction), 'mean' (mean of the output), 'sum' (sum of the output). Default: 'none'.

    Returns:
        Loss tensor with the specified reduction applied.
    """
    # Ensure that inputs and targets are the same shape
    assert inputs.shape == targets.shape, "Inputs and targets must have the same shape"

    # Reshape to handle multi-class cases
    # inputs and targets are expected to have shape (N, C, *), where C is the number of classes
    num_classes = inputs.size(1)
    
    # Compute the loss for each class separately
    losses = []
    for c in range(num_classes):
        loss = sigmoid_focal_loss(
            inputs=inputs[:, c, ...],
            targets=targets[:, c, ...],
            alpha=alpha,
            gamma=gamma,
            reduction="none"
        )
        losses.append(loss)

    # Stack the losses for each class
    loss = torch.stack(losses, dim=1)
    #loss = loss.sum(dim=1)

    # Apply the specified reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
