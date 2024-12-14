import torch
from torch import Tensor



ALPHA = 0.7
BETA = 0.3


def tversky_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
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

        return (TP + epsilon) / (TP + BETA*FP + ALPHA*FN + epsilon)
    else:
        # compute and average metric for each batch element
        tversky = 0
        for i in range(input.shape[0]):
            tversky += tversky_coeff(input[i, ...], target[i, ...])
        return tversky / input.shape[0]


def multiclass_tversky_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    tversky = 0
    for channel in range(input.shape[1]):
        tversky += tversky_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return tversky / input.shape[1]


def tversky_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_tversky_coeff if multiclass else tversky_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
