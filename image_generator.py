"""
Image Generator using Diffusion Model
Trains exclusively on your custom dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    """Custom dataset class to load images from your dataset folder"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        # Get all image files from the directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Loaded {len(self.image_paths)} images from {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class DiffusionModel(nn.Module):
    """A simple diffusion model implementation"""
    
    def __init__(self, image_channels=3, noise_dim=100):
        super(DiffusionModel, self).__init__()
        
        self.noise_dim = noise_dim
        self.image_channels = image_channels
        
        # Encoder: downsamples the image
        self.encoder = nn.Sequential(
            # Input: image_channels x 64 x 64
            nn.Conv2d(image_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Bottleneck to connect encoder and decoder
        self.bottleneck = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: upsamples to generate image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, -1)
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        bottleneck = bottleneck.view(batch_size, 512, 4, 4)
        
        # Decode
        decoded = self.decoder(bottleneck)
        return decoded


class DiffusionTrainer:
    """Trainer for the diffusion model"""
    
    def __init__(self, model, dataset_path, batch_size=32, image_size=64, learning_rate=0.0002):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Dataset and dataloader
        self.dataset = CustomImageDataset(dataset_path, transform=self.transform)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, real_images in enumerate(tqdm(self.dataloader, desc="Training")):
            real_images = real_images.to(self.device)
            
            # Forward pass
            reconstructed = self.model(real_images)
            loss = self.criterion(reconstructed, real_images)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss
    
    def generate_images(self, num_images=16):
        """Generate new images using the trained model"""
        self.model.eval()
        with torch.no_grad():
            # Create random noise
            noise = torch.randn(num_images, self.model.image_channels, self.image_size, self.image_size).to(self.device)
            generated = self.model(noise)
            
            # Denormalize for display
            generated = generated * 0.5 + 0.5
            generated = torch.clamp(generated, 0, 1)
            
        return generated
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    def train(self, num_epochs=100, save_interval=10, output_dir="output"):
        """Main training loop"""
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            avg_loss = self.train_epoch()
            print(f"Average Loss: {avg_loss:.4f}")
            
            # Save model periodically
            if (epoch + 1) % save_interval == 0:
                model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
                self.save_model(model_path)
                
                # Generate sample images
                samples = self.generate_images(16)
                sample_path = os.path.join(output_dir, f"sample_epoch_{epoch+1}.png")
                vutils.save_image(samples, sample_path, nrow=4, padding=2, normalize=False)
                print(f"Sample images saved to {sample_path}")


def main():
    # Configuration
    DATASET_PATH = "./my_dataset"  # Path to your custom dataset
    BATCH_SIZE = 32
    IMAGE_SIZE = 64  # Images will be resized to this size
    LEARNING_RATE = 0.0002
    NUM_EPOCHS = 100
    
    # Create model
    model = DiffusionModel(image_channels=3, noise_dim=100)
    
    # Check if dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset directory {DATASET_PATH} does not exist!")
        print("Please place your images in the 'my_dataset' folder.")
        return
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Start training
    print("Starting training...")
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # Final save
    trainer.save_model("final_model.pth")
    
    # Generate some final samples
    final_samples = trainer.generate_images(16)
    vutils.save_image(final_samples, "final_samples.png", nrow=4, padding=2, normalize=False)
    print("Final samples saved to final_samples.png")


if __name__ == "__main__":
    main()