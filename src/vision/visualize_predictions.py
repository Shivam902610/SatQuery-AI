
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from levir_dataset import LEVIRDataset
from siamese_unet import SiameseUNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "data/LEVIR-CD/raw"
MODEL_PATH = "models/siamese_unet_best.pth"

# Load test dataset
dataset = LEVIRDataset(DATA_ROOT, split="test")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model
model = SiameseUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Display 4 predictions
fig, axes = plt.subplots(4, 4, figsize=(14, 12))

with torch.no_grad():
    for i, (image_a, image_b, mask) in enumerate(loader):

        if i == 4:
            break

        image_a = image_a.to(DEVICE)
        image_b = image_b.to(DEVICE)

        logits = model(image_a, image_b)
        prediction = (torch.sigmoid(logits) >= 0.5).float()

        # Convert tensors for display
        img_a = image_a[0].cpu().permute(1, 2, 0).numpy()
        img_b = image_b[0].cpu().permute(1, 2, 0).numpy()
        actual = mask[0].squeeze().numpy()
        predicted = prediction[0].squeeze().cpu().numpy()

        images = [img_a, img_b, actual, predicted]
        titles = ["Earlier Image", "Later Image", "Actual Change", "Predicted Change"]

        for j in range(4):
            axes[i, j].imshow(images[j], cmap="gray" if j >= 2 else None)
            axes[i, j].set_title(titles[j])
            axes[i, j].axis("off")

plt.tight_layout()
plt.savefig("evaluation/change_predictions.png", dpi=150)
plt.show()

print("Visualization saved to evaluation/change_predictions.png")
