import os
import torch
import torch.nn as nn
import torch.optim as optim
from modules.discriminator import Discriminator
from modules.generator import Generator
from torch.utils.data import DataLoader
from modules.AudioDataset import AudioDataset

# Parameters (you might need to adjust these)
noise_dim = 100
batch_size = 32
epochs = 100
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
discriminator = Discriminator().to(device)
generator = Generator(noise_dim).to(device)

# Optimizers
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Load data
dataset = AudioDataset(directory='data/train')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader):

        # TRAIN THE DISCRIMINATORE
        real_data = data.to(device)
        batch_size = real_data.size(0)

        # Generate labels for real data (1s)
        real_labels = torch.ones(batch_size, 1).to(device)
        # Train on real data
        discriminator.zero_grad()
        predictions_real = discriminator(real_data)
        loss_real = criterion(predictions_real, real_labels)

        # Generate fake data
        noise = torch.randn(batch_size, noise_dim, 1).to(device)
        fake_data = generator(noise)
        # Generate labels for fake data (0s)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # Train on fake data
        predictions_fake = discriminator(fake_data.detach())
        loss_fake = criterion(predictions_fake, fake_labels)

        # Update discriminator
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # TRAIN GENERATOR
        generator.zero_grad()
        # We want discriminator to mistake these as real (label as 1s)
        predictions = discriminator(fake_data)
        loss_generator = criterion(predictions, real_labels)

        # Update generator
        loss_generator.backward()
        optimizer_generator.step()

        # Logging or printing some information
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], ...")

# Save the generator model
model_path = 'models/generator_model.pth'
if os.path.exists(model_path):
    version = 1
    while os.path.exists(f'models/generator_model_v{version}.pth'):
        version += 1
    model_path = f'models/generator_model_v{version}.pth'

torch.save(generator.state_dict(), model_path)
print(f"Model saved to {model_path}")
