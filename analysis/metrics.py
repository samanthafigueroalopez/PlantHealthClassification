import torch


def compute_iou(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()

    intersection = (pred_mask & true_mask).sum().float()
    union = (pred_mask | true_mask).sum().float()

    if union == 0:
        return 0.0

    return (intersection / union).item()


def compute_dice(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()

    intersection = (pred_mask & true_mask).sum().float()
    total = pred_mask.sum() + true_mask.sum()

    if total == 0:
        return 0.0

    return (2.0 * intersection / total).item()
