import torch
import torch.nn.functional as F

from analysis.metrics import compute_iou, compute_dice


def threshold_heatmap(heatmap, threshold=0.5):
    return (heatmap >= threshold).float()


def resize_mask(mask, target_size):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)

    mask = F.interpolate(
        mask.float(),
        size=target_size,
        mode="nearest"
    )

    return mask[0, 0]


def evaluate_gradcam_mask(
    heatmap,
    true_mask,
    threshold=0.5
):
    if heatmap.dim() != 2:
        raise ValueError("Heatmap must be a 2D tensor.")

    true_mask = resize_mask(true_mask, heatmap.shape)

    predicted_mask = threshold_heatmap(heatmap, threshold)

    true_mask = (true_mask > 0).float()

    iou = compute_iou(predicted_mask, true_mask)

    dice = compute_dice(predicted_mask, true_mask)

    return {
        "iou": iou,
        "dice": dice,
        "predicted_mask": predicted_mask,
        "true_mask": true_mask
    }
