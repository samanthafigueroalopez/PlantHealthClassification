import torch

from analysis.gradcam import GradCAM
from models.resnet18 import create_resnet18


def test_gradcam_output():
    model = create_resnet18(
        num_classes=89,
        pretrained=False
    )

    target_layer = model.layer4[-1]

    gradcam = GradCAM(model, target_layer)

    x = torch.randn(1, 3, 224, 224)

    heatmap, class_idx, logits = gradcam.generate(x)

    assert heatmap.shape == (224, 224)
    assert logits.shape == (1, 89)

    assert heatmap.min().item() >= 0.0
    assert heatmap.max().item() <= 1.0

    assert isinstance(class_idx, int)

    gradcam.close()
