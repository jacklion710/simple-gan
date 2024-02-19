import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def plot_spectrogram(audio_path, sr=None, hop_length=512, n_fft=2048):
    """
    Plots a spectrogram for the audio file at the given path.

    Parameters:
    - audio_path: Path to the audio file.
    - sr: Sample rate to use for loading the audio file. If None, librosa's default will be used.
    - hop_length: Hop length for STFT.
    - n_fft: FFT window size.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute the spectrogram magnitude and phase
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.amplitude_to_db(np.abs(S_complex), ref=np.max)

    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()

# Example usage
audio_path = 'audio_exports/long_generated_audio.wav'  
plot_spectrogram(audio_path)
