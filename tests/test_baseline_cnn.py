import torch
from models.baseline_cnn import BaselineCNN
from models.resnet18 import create_resnet18


def test_resnet18_output_shape():
    model = create_resnet18(num_classes=89, pretrained=False)

    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert output.shape == (2, 89)


def test_baseline_output_shape():
    model = BaselineCNN(num_classes=89)

    x = torch.randn(4, 3, 224, 224)
    output = model(x)

    assert output.shape == (4, 89)


def test_baseline_parameter_count():
    model = BaselineCNN(num_classes=89)

    num_params = sum(p.numel() for p in model.parameters())

    assert num_params == 105177
