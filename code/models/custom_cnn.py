import torch
import torch.nn as nn

class RegularizedTomatoCNN(nn.Module):
    """
    Baseline 3-Block Regularized CNN architecture trained from scratch.
    Parameters: ~120,458 (120K)
    Test Accuracy: 72.20%
    """
    def __init__(self, num_classes=10, dropout=0.5):
        super(RegularizedTomatoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def regularization_loss(model, l1_lambda=1e-5, l2_lambda=1e-4, device="cpu"):
    """Computes combined L1 (Lasso) and L2 (Ridge) penalty on model weights."""
    l1_loss = torch.tensor(0.0, device=device)
    l2_loss = torch.tensor(0.0, device=device)
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
        l2_loss += torch.sum(param ** 2)
    return (l1_lambda * l1_loss) + (l2_lambda * l2_loss)
