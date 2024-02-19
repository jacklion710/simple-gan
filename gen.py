import torch
import torchaudio
from modules.generator import Generator

# Parameters
noise_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sample_rate = 22050  # Define the sample rate for the audio file

# Load the trained generator model
generator = Generator(noise_dim).to(device)
generator.load_state_dict(torch.load('models/generator_model.pth'))  # Adjust path if using versioning
generator.eval()

# Generate new audio samples
with torch.no_grad():
    noise = torch.randn(1, noise_dim, 1, device=device)  # Adjust dimensions as needed
    generated_audio = generator(noise)

# Ensure the directory exists
import os
if not os.path.exists('audio_exports'):
    os.makedirs('audio_exports')

# Assuming the generated_audio tensor shape might be (1, N, 1) where N is the number of samples
# We need to remove the batch and any unnecessary dimensions so it becomes (1, N) for mono audio
generated_audio = generated_audio.cpu().squeeze()  # Move to CPU and remove unnecessary dimensions

# Check if we need to add a channel dimension
if generated_audio.ndim == 1:
    generated_audio = generated_audio.unsqueeze(0)  # Add channel dimension (1, N)

# Save as WAV file
output_path = 'audio_exports/generated_audio.wav'
torchaudio.save(output_path, generated_audio, sample_rate)

print(f"Generated audio saved to {output_path}")
