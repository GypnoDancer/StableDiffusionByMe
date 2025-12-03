#!/usr/bin/env python
"""
Setup script to install dependencies and prepare the environment
"""
import subprocess
import sys
import os


def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True


def create_dataset_dir():
    """Create the dataset directory if it doesn't exist"""
    dataset_dir = "my_dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created dataset directory: {dataset_dir}")
    else:
        print(f"Dataset directory already exists: {dataset_dir}")
    
    return True


def main():
    print("Setting up the image generator environment...")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually using: pip install -r requirements.txt")
        return False
    
    # Create dataset directory
    create_dataset_dir()
    
    print("\nSetup completed successfully!")
    print("To train the model, add your images to the 'my_dataset' folder and run:")
    print("  python diffusion_model.py")
    
    return True


if __name__ == "__main__":
    main()