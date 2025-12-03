"""
Script to generate images using a trained diffusion model
"""
import torch
import torchvision.utils as vutils
from diffusion_model import DiffusionUNet, DiffusionTrainer


def generate_from_trained_model(model_path, num_images=16, output_path="generated_images.png"):
    """Generate images using a trained model"""
    
    # Create model (same architecture as training)
    model = DiffusionUNet(image_channels=3, time_channels=128, base_channels=64)
    
    # Create trainer with dummy dataset path for initialization
    trainer = DiffusionTrainer(
        model=model,
        dataset_path="./my_dataset",  # Only used for transforms setup
        batch_size=16,
        image_size=64,
        learning_rate=2e-4
    )
    
    # Load the trained model
    trainer.load_model(model_path)
    
    # Generate images
    print(f"Generating {num_images} images...")
    generated_images = trainer.generate_images(num_images=num_images)
    
    # Save generated images
    vutils.save_image(generated_images, output_path, nrow=4, padding=2, normalize=False)
    print(f"Generated images saved to {output_path}")


if __name__ == "__main__":
    import os
    
    # Check if a trained model exists
    if os.path.exists("final_model.pth"):
        generate_from_trained_model("final_model.pth", num_images=16)
    else:
        print("No trained model found!")
        print("Please train the model first using: python diffusion_model.py")
        print("Or specify a path to a trained model.")