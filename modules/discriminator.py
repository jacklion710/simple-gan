# discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 1, kernel_size=3, stride=2, padding=0),
            nn.Flatten(),
            # Add an adaptive pooling layer here
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(1, 1)  # Adjusted to match the output of the adaptive pooling layer

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
        x = self.fc(x)
        return torch.sigmoid(x)
