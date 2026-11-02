import torch

from analysis.evaluate_plantseg import (
    threshold_heatmap,
    resize_mask,
    evaluate_gradcam_mask,
)


def test_threshold_heatmap():
    heatmap = torch.tensor([[0.2, 0.7],[0.9, 0.3]])

    result = threshold_heatmap(heatmap, threshold=0.5)

    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    assert torch.equal(result, expected)


def test_resize_mask():
    mask = torch.tensor([[1.0, 0.0],[0.0, 1.0]])

    resized = resize_mask(mask, (4, 4))

    assert resized.shape == (4, 4)


def test_evaluate_perfect_overlap():
    heatmap = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    true_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    result = evaluate_gradcam_mask(
        heatmap,
        true_mask,
        threshold=0.5
    )

    assert result["iou"] == 1.0
    assert result["dice"] == 1.0

