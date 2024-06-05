# Simple-GAN for Audio Generation

This GAN demonstrates the implementation of a Generative Adversarial Network (GAN) for generating audio samples. It utilizes PyTorch, a powerful library for deep learning, to train both generator and discriminator models capable of producing and evaluating audio data, respectively.

## Why

I wanted to generate novel drum samples for producing music. Synths will often feature a randomization feature for obtaining unexpected results. Using deep learning to randomly generate new sounds may be inspiring and useful for music creation.

## Environment Setup

### PyTorch GPU Setup

To leverage GPU acceleration for faster training times, follow the instructions provided in [this detailed](https://mct-master.github.io/machine-learning/2023/04/25/olivegr-pytorch-gpu.html) guide.

### Initial Setup

1. **Create a Conda Environment**

   PyTorch requires Python version 3.7 or above. Create a new conda environment named `simple-gan` with Python 3.8:

   ```bash
   conda create -n simple-gan python=3.8
   ```

2. **Activate the Conda Environment**

Activate the newly created environment:

```bash
conda activate simple-gan
```

3. **Install PyTorch**

Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the installation command tailored to your platform, package manager, and CUDA version. The command for conda and pip installations typically look as follows:

* Conda:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=[CUDA_VERSION] -c pytorch -c nvidia
```

* Pip:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[CUDA_VERSION]
```

Replace `[CUDA_VERSION]` with the version compatible with your GPU drivers. Use `nvcc --version` to check the CUDA version installed on your system.

Note: `nvidia-smi` shows the GPU driver version, which is different from the CUDA runtime version. It's essential to install the PyTorch version corresponding to the CUDA runtime version for compatibility.

4. **Install Other Dependencies**

```bash
conda install -c conda-forge librosa
```

## Verifying GPU Setup

To check if PyTorch can access your GPU:

```py
import torch
print(torch.cuda.is_available())
```

This command should return True if a GPU is detected.

If you encounter issues with GPU detection:

```py
print(torch.zeros(1).cuda())
```

This will attempt to create a tensor on the GPU and may provide useful error messages for troubleshooting.

## Resetting the Environment

If you need to start over:

```bash
conda activate base
conda remove -n simple-gan --all
```

## GPU Information

After successful setup, you can use the following commands to get information about the GPUs:

```py
print(torch.cuda.current_device())    # The ID of the current GPU.
print(torch.cuda.get_device_name(0))  # The name of the first GPU.
print(torch.cuda.device_count())      # The number of GPUs available.
```

## Project Structure

This project includes Python scripts and modules for training the GAN, along with a dataset module for loading and processing audio files. Here's a brief overview:

* `train.py`: Main script for training the GAN.
* `modules/`: Contains the Python modules for the generator (`generator.py`), discriminator (`discriminator.py`), and audio dataset (`AudioDataset.py`).

## Training the GAN

To train the GAN, ensure you are in the project's root directory and activate the simple-gan environment. Then run:

```bash
python train.py
```

## Model Architecture

Details about the generator and discriminator architectures are provided in the `modules/` directory.

* Generator: Uses transposed convolutional layers to generate audio samples from noise.
* Discriminator: Consists of convolutional layers to classify audio samples as real or fake.

## Audio Dataset

The 'AudioDataset' class in `AudioDataset.py` handles loading and preprocessing of audio files for training. I utilized the training set downloadable from [FSDKaggle2018](https://zenodo.org/records/2552860#.XFD05fwo-V4) for our experiments. However, users can experiment with any audio data they want, as long as it's organized like this: `/data/train/*.wav` so that the AudioDataset class can properly connect to it.

## Additional Resources

For background information on GANs and their applications in audio generation, [this article](https://realpython.com/generative-adversarial-networks/) provides a comprehensive introduction.

### Contributing

If you'd like to contribute, please fork the repository and open a pull request to the `main` branch.
