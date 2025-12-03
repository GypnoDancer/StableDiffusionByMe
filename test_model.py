"""
Simple test to verify the model architecture works correctly
"""
import torch
from diffusion_model import DiffusionUNet


def test_model():
    """Test the model architecture"""
    print("Testing model architecture...")
    
    # Create model
    model = DiffusionUNet(image_channels=3, time_channels=128, base_channels=64)
    print(f"Model created successfully")
    
    # Test forward pass
    batch_size = 2
    img_size = 64
    x = torch.randn(batch_size, 3, img_size, img_size)
    t = torch.randn(batch_size)  # Time embedding
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x, t)
    
    print(f"Output shape: {output.shape}")
    
    # Check shapes match
    assert x.shape == output.shape, f"Input and output shapes don't match: {x.shape} vs {output.shape}"
    
    print("Model test passed successfully!")
    

if __name__ == "__main__":
    test_model()