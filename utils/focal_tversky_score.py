import torch
from torch import Tensor



ALPHA = 0.7
BETA = 0.3
GAMMA = 3/4

def focal_tversky_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        TP = torch.dot(input.reshape(-1), target.reshape(-1))
        FP = ((1-target) * input).sum()
        FN = (target * (1-input)).sum()

        # if sets_sum.item() == 0:
        #     sets_sum = 2 * TP

        Tversky = (TP + epsilon) / (TP + BETA*FP + ALPHA*FN + epsilon)
        Focal_Tversky = (1-Tversky)**GAMMA
        return Focal_Tversky

    else:
        # compute and average metric for each batch element
        focal_tversky = 0
        for i in range(input.shape[0]):
            focal_tversky += focal_tversky_coeff(input[i, ...], target[i, ...])
        return focal_tversky / input.shape[0]


def multiclass_focal_tversky_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    focal_tversky = 0
    for channel in range(input.shape[1]):
        focal_tversky += focal_tversky_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return focal_tversky / input.shape[1]


def focal_tversky_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_focal_tversky_coeff if multiclass else focal_tversky_coeff
    return fn(input, target, reduce_batch_first=True)
