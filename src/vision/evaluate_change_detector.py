
import torch
from torch.utils.data import DataLoader

from levir_dataset import LEVIRDataset
from siamese_unet import SiameseUNet


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "data/LEVIR-CD/raw"
MODEL_PATH = "models/siamese_unet_best.pth"

BATCH_SIZE = 2


# Load test dataset
test_dataset = LEVIRDataset(DATA_ROOT, split="test")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Load trained model
model = SiameseUNet().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


# Metric counters
TP = 0
FP = 0
FN = 0
TN = 0


print("Device:", DEVICE)
print("Test samples:", len(test_dataset))


# Evaluate
with torch.no_grad():

    for image_a, image_b, masks in test_loader:

        image_a = image_a.to(DEVICE)
        image_b = image_b.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(image_a, image_b)

        predictions = (torch.sigmoid(logits) >= 0.5)
        actual = (masks >= 0.5)

        TP += (predictions & actual).sum().item()
        FP += (predictions & ~actual).sum().item()
        FN += (~predictions & actual).sum().item()
        TN += (~predictions & ~actual).sum().item()


# Calculate metrics
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)

f1 = 2 * precision * recall / (precision + recall + 1e-8)

iou = TP / (TP + FP + FN + 1e-8)

accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)


# Display results
print("\n========== TEST RESULTS ==========")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

print("\nEvaluation completed!")
