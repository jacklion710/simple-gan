from torch.utils.data import Dataset
import librosa
import os
import torch
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, directory, sample_rate=22050, transform=None):
        """
        Args:
            directory (str): Directory with all the audio files.
            sample_rate (int): Target sample rate for all audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.sample_rate = sample_rate
        self.transform = transform
        self.audio_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.wav')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Fixed length for all audio samples
        target_length = 22050  # For example, 1 second at 22050Hz

        # Padding or Trimming to target_length
        if len(audio) > target_length:
            # Trim longer samples
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad shorter samples
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        if self.transform:
            audio = self.transform(audio)
        
        audio = torch.tensor(audio, dtype=torch.float32)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)  # Ensure audio tensor is C x L
        return audio

