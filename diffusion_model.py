"""
Advanced Diffusion Model for Image Generation
Trains exclusively on your custom dataset
Based on Denoising Diffusion Probabilistic Models (DDPM)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os
import math
from tqdm import tqdm
import numpy as np


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


def get_noise_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """Create noise schedule for diffusion process"""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
    
    # Required for diffusion computations
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # Posterior variance
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance
    }


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        
        self.relu = nn.ReLU()
        
        # Skip connection if channels don't match
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t):
        # First conv + batch norm
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        
        # Add time embedding - expand to match spatial dimensions
        t_emb = self.time_emb(t)
        # t_emb shape is [batch, out_channels] after time_emb
        # Need to expand to [batch, out_channels, height, width]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [batch, out_channels, 1, 1]
        t_emb = t_emb.expand(-1, -1, h.shape[-2], h.shape[-1])  # [batch, out_channels, height, width]
        h = h + t_emb
        
        # Second conv + batch norm
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        
        # Skip connection
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Attention mechanism for diffusion model"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.projection = nn.Linear(channels, channels)
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        self.output_projection = nn.Linear(channels, channels)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        h = self.group_norm(x)
        
        # Reshape to (batch_size, height*width, channels) for attention
        h = h.view(batch_size, channels, height * width).transpose(1, 2)
        
        # Compute attention
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        
        # Scaled dot-product attention
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.channels)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.bmm(attention_weights, v)
        
        # Output projection
        output = self.output_projection(attended_values)
        
        # Reshape back to (batch_size, channels, height, width)
        output = output.transpose(1, 2).view(batch_size, channels, height, width)
        
        return x + output


class DiffusionUNet(nn.Module):
    """U-Net architecture for diffusion model"""
    
    def __init__(self, image_channels=3, time_channels=128, base_channels=64):
        super().__init__()
        
        self.time_channels = time_channels
        self.base_channels = base_channels
        
        # Time embedding - input is scalar, so we start with a single value
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_channels),
            nn.ReLU(),
            nn.Linear(time_channels, time_channels)
        )
        
        # Encoder
        self.enc1 = ResidualBlock(image_channels, base_channels, time_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels*2, time_channels)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*2, time_channels)
        
        # Middle
        self.middle = nn.Sequential(
            ResidualBlock(base_channels*2, base_channels*4, time_channels),
            AttentionBlock(base_channels*4),
            ResidualBlock(base_channels*4, base_channels*4, time_channels)
        )
        
        # Decoder
        self.dec1 = ResidualBlock(base_channels*4 + base_channels*2, base_channels*2, time_channels)
        self.dec2 = ResidualBlock(base_channels*2 + base_channels*2, base_channels*2, time_channels)
        self.dec3 = ResidualBlock(base_channels*2 + base_channels, base_channels, time_channels)
        
        # Output layer
        self.out = nn.Conv2d(base_channels, image_channels, 3, padding=1)
        
        # Upsampling and downsampling
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, t):
        # Time embedding - reshape t to [batch_size, 1] before embedding
        if t.dim() == 0 or t.shape == torch.Size([]):  # scalar
            t = t.view(1, 1)
        elif t.dim() == 1:  # [batch_size]
            t = t.view(-1, 1)  # [batch_size, 1]
        
        t_emb = self.time_embed(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)  # [B, 64, H, W]
        h = self.downsample(e1)   # [B, 64, H/2, W/2]
        
        e2 = self.enc2(h, t_emb)  # [B, 128, H/2, W/2]
        h = self.downsample(e2)   # [B, 128, H/4, W/4]
        
        e3 = self.enc3(h, t_emb)  # [B, 128, H/4, W/4]
        h = self.downsample(e3)   # [B, 128, H/8, W/8]
        
        # Middle - apply residual blocks with time embedding and attention
        h = self.middle[0](h, t_emb)  # First residual block with time embedding
        h = self.middle[1](h)         # Attention block (no time embedding needed)
        h = self.middle[2](h, t_emb)  # Second residual block with time embedding
        
        # Decoder
        h = self.upsample(h)      # [B, 256, H/4, W/4]
        h = torch.cat([h, e3], dim=1)  # [B, 384, H/4, W/4]
        h = self.dec1(h, t_emb)   # [B, 128, H/4, W/4]
        
        h = self.upsample(h)      # [B, 128, H/2, W/2]
        h = torch.cat([h, e2], dim=1)  # [B, 256, H/2, W/2]
        h = self.dec2(h, t_emb)   # [B, 128, H/2, W/2]
        
        h = self.upsample(h)      # [B, 128, H, W]
        h = torch.cat([h, e1], dim=1)  # [B, 192, H, W]
        h = self.dec3(h, t_emb)   # [B, 64, H, W]
        
        # Output
        out = self.out(h)
        return out


class DiffusionTrainer:
    """Trainer for the diffusion model"""
    
    def __init__(self, model, dataset_path, batch_size=32, image_size=64, learning_rate=2e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        
        # Noise schedule
        self.noise_schedule = get_noise_schedule()
        for key, value in self.noise_schedule.items():
            self.noise_schedule[key] = value.to(self.device)
        
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
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def forward_diffusion(self, x_0, t):
        """Add noise to images according to the diffusion schedule"""
        sqrt_alphas_cumprod_t = self.noise_schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.noise_schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x_0)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def train_step(self, real_images):
        """Single training step"""
        self.model.train()
        batch_size = real_images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, len(self.noise_schedule['betas']), (batch_size,), device=self.device).long()
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion(real_images, t)
        
        # Predict noise
        # Normalize t to [0, 1] range for the model
        t_normalized = t.float() / len(self.noise_schedule['betas'])
        noise_pred = self.model(x_t, t_normalized)
        
        # Calculate loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch_idx, real_images in enumerate(tqdm(self.dataloader, desc="Training")):
            real_images = real_images.to(self.device)
            loss = self.train_step(real_images)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def sample_timestep(self, x, t):
        """Sample x_{t-1} from the model using the reverse diffusion process"""
        with torch.no_grad():
            betas_t = self.noise_schedule['betas'][t].view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.noise_schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
            sqrt_recip_alphas_t = self.noise_schedule['sqrt_recip_alphas'][t].view(-1, 1, 1, 1)
            
            # Predict noise using model
            # Normalize t to [0, 1] range for the model
            t_normalized = t.float() / len(self.noise_schedule['betas'])
            model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t_normalized) / sqrt_one_minus_alphas_cumprod_t)
            
            if t == 0:
                return model_mean
            else:
                posterior_variance_t = self.noise_schedule['posterior_variance'][t].view(-1, 1, 1, 1)
                noise = torch.randn_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def generate_images(self, num_images=16):
        """Generate new images using the trained model"""
        self.model.eval()
        
        # Start with random noise
        img = torch.randn(
            num_images, 
            3, 
            self.image_size, 
            self.image_size
        ).to(self.device)
        
        # Iteratively denoise
        for i in tqdm(range(len(self.noise_schedule['betas']) - 1, -1, -1), desc="Generating"):
            t = torch.full((num_images,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(img, t)
        
        # Denormalize for display
        img = img * 0.5 + 0.5
        img = torch.clamp(img, 0, 1)
        
        return img
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'noise_schedule': {k: v.cpu() for k, v in self.noise_schedule.items()}
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load noise schedule to device
        for k, v in checkpoint['noise_schedule'].items():
            self.noise_schedule[k] = v.to(self.device)
        
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
    BATCH_SIZE = 16  # Reduced for memory
    IMAGE_SIZE = 64  # Images will be resized to this size
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 100
    
    # Create model
    model = DiffusionUNet(image_channels=3, time_channels=128, base_channels=64)
    
    # Check if dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset directory {DATASET_PATH} does not exist!")
        print("Please place your images in the 'my_dataset' folder.")
        print("Creating a sample dataset directory for you...")
        os.makedirs(DATASET_PATH, exist_ok=True)
        print(f"Created directory: {DATASET_PATH}")
        print("Please add your images to this directory before running training.")
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