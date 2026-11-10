
import torch
from siamese_unet import SiameseUNet

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SiameseUNet().to(device)
model.eval()

# Create two sample satellite images
image_a = torch.randn(1, 3, 256, 256).to(device)
image_b = torch.randn(1, 3, 256, 256).to(device)

# Test prediction
with torch.no_grad():
    output = model(image_a, image_b)

print("Device:", device)
print("Image A shape:", image_a.shape)
print("Image B shape:", image_b.shape)
print("Change map shape:", output.shape)
print("Model test successful!")
