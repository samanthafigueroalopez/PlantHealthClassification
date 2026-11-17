import torch

from models.custom_cnn_v2 import CustomCNNV2


def test_custom_cnn_v2_output_shape():
    model = CustomCNNV2(num_classes=89)

    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert output.shape == (2, 89)


def test_custom_cnn_v2_parameter_count_matches_saved_experiment():
    model = CustomCNNV2(num_classes=89)

    num_params = sum(p.numel() for p in model.parameters())

    assert num_params == 121241
