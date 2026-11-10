
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from levir_dataset import LEVIRDataset
from siamese_unet import SiameseUNet


# Configuration
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "data/LEVIR-CD/raw"
MODEL_PATH = Path("models/siamese_unet_best.pth")


# Create datasets
train_dataset = LEVIRDataset(DATA_ROOT, split="train")
val_dataset = LEVIRDataset(DATA_ROOT, split="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# Initialize model
model = SiameseUNet().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")


# Training loop
print("Device:", DEVICE)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for image_a, image_b, masks in train_loader:

        image_a = image_a.to(DEVICE)
        image_b = image_b.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(image_a, image_b)
        loss = criterion(predictions, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for image_a, image_b, masks in val_loader:

            image_a = image_a.to(DEVICE)
            image_b = image_b.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(image_a, image_b)
            loss = criterion(predictions, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved!")

print("\nTraining completed!")
print("Best validation loss:", best_val_loss)
print("Model location:", MODEL_PATH)
