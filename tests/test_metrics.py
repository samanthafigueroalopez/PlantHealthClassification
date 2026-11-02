import torch

from analysis.metrics import compute_iou, compute_dice


def test_iou_perfect_overlap():
    pred = torch.tensor([[1, 0], [0, 1]])

    true = torch.tensor([[1, 0], [0, 1]])

    assert compute_iou(pred, true) == 1.0


def test_dice_perfect_overlap():
    pred = torch.tensor([[1, 0], [0, 1]])

    true = torch.tensor([[1, 0], [0, 1]])

    assert compute_dice(pred, true) == 1.0


def test_iou_partial_overlap():
    pred = torch.tensor([[1, 1], [0, 0]])

    true = torch.tensor([[1, 0], [1, 0]])

    result = compute_iou(pred, true)

    assert abs(result - (1 / 3)) < 1e-6


def test_dice_partial_overlap():
    pred = torch.tensor([[1, 1], [0, 0]])

    true = torch.tensor([[1, 0], [1, 0]])

    result = compute_dice(pred, true)

    assert abs(result - 0.5) < 1e-6
