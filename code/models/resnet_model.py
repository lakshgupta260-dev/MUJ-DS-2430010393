import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=10, pretrained=True):
    """
    Fine-Tuned ResNet18 model with residual skip connections.
    Parameters: ~11,181,642 (11.1M)
    Validation Accuracy: 97.27% | Test Accuracy: 97.20%
    """
    weights = 'DEFAULT' if pretrained else None
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
