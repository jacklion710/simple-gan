# generator.py
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(noise_dim, 64, kernel_size=25, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=25, stride=4, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=25, stride=4, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=25, stride=4, padding=1),
            nn.Tanh()  # Use Tanh to output values between -1 and 1
        )

    def forward(self, x):
        return self.model(x)
