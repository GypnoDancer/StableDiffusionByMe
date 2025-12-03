"""
Example usage script for the AI Image Generator
This demonstrates how to train and use the model
"""

import os
import torch
from diffusion_model import DiffusionUNet, DiffusionTrainer

def train_model():
    """Example function to train the model"""
    print("Creating model...")
    
    # Create the model with smaller dimensions for example
    model = DiffusionUNet(
        image_channels=3,      # RGB images
        time_channels=64,      # Time embedding dimension
        base_channels=32       # Base number of channels in U-Net
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        dataset_path="./my_dataset",  # Folder containing your images
        batch_size=2,                 # Adjust based on your GPU memory
        image_size=64,                # Resize images to this size
        learning_rate=2e-4
    )
    
    print("Starting training...")
    print("Note: Actual training would continue for many epochs")
    print("For demo purposes, we'll just show the setup is correct")
    
    # In a real scenario, you would call:
    # trainer.train(num_epochs=100)
    
    return trainer

def generate_images(trainer):
    """Example function to generate images with a trained model"""
    print("\nGenerating sample images...")
    
    # Generate new images
    generated_images = trainer.generate_images(num_images=4)
    
    print(f"Generated images shape: {generated_images.shape}")
    print("Images generated successfully!")
    
    return generated_images

def main():
    """Main example function"""
    print("AI Image Generator - Example Usage")
    print("=" * 40)
    
    # Make sure dataset directory exists
    os.makedirs('./my_dataset', exist_ok=True)
    
    # Train model (or load a pre-trained one)
    trainer = train_model()
    
    # Generate images
    # generated = generate_images(trainer)  # Commented out for demo
    
    print("\nTo run actual training:")
    print("1. Place your images in ./my_dataset/")
    print("2. Run: python diffusion_model.py")
    print("\nTo generate images after training:")
    print("1. Run: python generate_images.py")

if __name__ == "__main__":
    main()