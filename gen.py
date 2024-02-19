import torch
import torchaudio
from modules.generator import Generator
import os

# Parameters
noise_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sample_rate = 44100
num_samples_to_generate = 50  # Number of samples to generate and concatenate

# Load the trained generator model
generator = Generator(noise_dim).to(device)
generator.load_state_dict(torch.load('models/generator_model.pth'))
generator.eval()

# Generate multiple audio samples and concatenate them
with torch.no_grad():
    generated_samples = []
    for _ in range(num_samples_to_generate):
        noise = torch.randn(1, noise_dim, 1, device=device)
        generated_audio = generator(noise).cpu().squeeze(0)  # Assuming output is (1, C, L)
        generated_samples.append(generated_audio)

    # Concatenate along the time axis (L dimension)
    long_generated_audio = torch.cat(generated_samples, dim=1)

# Ensure the directory exists
if not os.path.exists('audio_exports'):
    os.makedirs('audio_exports')

# Log the shape of the tensor before saving
print("Shape of long_generated_audio before unsqueeze:", long_generated_audio.shape)

# First, ensure it's 2D [channels, samples]. If it's already [samples], add the channels dimension
if long_generated_audio.ndim == 2:
    correct_shape_audio = long_generated_audio
elif long_generated_audio.ndim == 1:
    # If it's 1D, add the channel dimension
    correct_shape_audio = long_generated_audio.unsqueeze(0)
else:
    # Log unexpected dimensions
    print("Unexpected tensor dimensions:", long_generated_audio.ndim)
    correct_shape_audio = long_generated_audio.squeeze()  # Attempt to correct by squeezing

# Save the long generated audio
output_path = 'audio_exports/long_generated_audio.wav'
torchaudio.save(output_path, correct_shape_audio, sample_rate)
print(f"Long generated audio saved to {output_path}")

# Log the final shape before saving
print("Final shape of audio tensor to be saved:", correct_shape_audio.shape)