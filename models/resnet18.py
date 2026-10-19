import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(num_classes=89, pretrained=True):
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
