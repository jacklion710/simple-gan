import torch
from modules.generator import Generator

# Parameters
noise_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained generator model
generator = Generator(noise_dim).to(device)
generator.load_state_dict(torch.load('models/generator_model.pth'))  # Adjust path if using versioning
generator.eval()

# Generate new audio samples
with torch.no_grad():
    noise = torch.randn(1, noise_dim, 1).to(device)  # Adjust dimensions as needed
    generated_audio = generator(noise)

# Process the generated_audio tensor to convert it into an actual audio file
# This might involve saving it as a WAV file or any other format you're working with
