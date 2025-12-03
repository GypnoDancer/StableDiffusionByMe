# Custom Image Generator

This project implements a diffusion model for generating images trained exclusively on your custom dataset. The model is based on Denoising Diffusion Probabilistic Models (DDPM) and can generate new images that follow the style and patterns of your dataset.

## Features

- Train on your own custom image dataset
- Advanced U-Net architecture with attention mechanisms
- Proper diffusion process with noise scheduling
- Easy to use training and generation interface

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- Pillow
- tqdm
- numpy
- matplotlib

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Dataset

Create a folder called `my_dataset` and place all your training images inside:

```bash
mkdir my_dataset
# Place your images (JPG, PNG, etc.) in the my_dataset folder
```

### 2. Training

Run the training script:

```bash
python diffusion_model.py
```

The script will:
- Load images from the `my_dataset` folder
- Train the diffusion model
- Save checkpoints periodically
- Generate sample images during training

### 3. Generating Images

After training, you can load the trained model and generate new images:

```python
from diffusion_model import DiffusionUNet, DiffusionTrainer

# Load the model
model = DiffusionUNet(image_channels=3, time_channels=128, base_channels=64)
trainer = DiffusionTrainer(model, dataset_path="./my_dataset")

# Load your trained model
trainer.load_model("final_model.pth")

# Generate new images
generated_images = trainer.generate_images(num_images=16)
```

## Model Architecture

The model uses a U-Net architecture with:
- Residual blocks for better gradient flow
- Attention mechanisms for capturing global dependencies
- Time embedding for the diffusion process
- Proper noise scheduling following DDPM methodology

## Configuration

You can modify these parameters in the `main()` function:
- `BATCH_SIZE`: Number of images per training batch
- `IMAGE_SIZE`: Size to resize images (squared)
- `LEARNING_RATE`: Learning rate for training
- `NUM_EPOCHS`: Number of training epochs

## Output

During training, the model will:
- Save checkpoints every 10 epochs
- Generate sample images periodically
- Save final model and samples when training completes

## Customization

You can customize the model by modifying:
- `DiffusionUNet`: Change the U-Net architecture
- `DiffusionTrainer`: Adjust training parameters
- Noise schedule parameters in `get_noise_schedule()`
- Image preprocessing in the transform pipeline

## Note

- For best results, use a dataset with at least 1000 images
- Training time depends on dataset size and number of epochs
- GPU is recommended for faster training
- Images are resized to 64x64 by default for memory efficiency