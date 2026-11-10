
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from siamese_unet import SiameseUNet


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/siamese_unet_best.pth")
IMAGE_SIZE = (256, 256)


def load_image(image_path):
    original = Image.open(image_path).convert("RGB")

    resized = original.resize(IMAGE_SIZE)

    image = np.array(resized, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)

    return image.unsqueeze(0).to(DEVICE), original


def predict_change(image_a_path, image_b_path, output_path):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load trained model
    model = SiameseUNet().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    # Load satellite images
    image_a, _ = load_image(image_a_path)
    image_b, original_b = load_image(image_b_path)

    # Predict changes
    with torch.no_grad():
        logits = model(image_a, image_b)
        prediction = torch.sigmoid(logits)
        prediction = (prediction >= 0.5).float()

    # Convert prediction to image
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    # Resize mask to original image dimensions
    mask_image = Image.fromarray(mask).resize(
        original_b.size,
        Image.Resampling.NEAREST
    )

    # Save binary change mask
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(output_path)

    # Create red overlay
    red_layer = Image.new("RGB", original_b.size, (255, 0, 0))

    blended = Image.blend(original_b, red_layer, 0.45)

    highlighted = Image.composite(
        blended,
        original_b,
        mask_image
    )

    # Save overlay
    overlay_path = output_path.parent / (
        output_path.stem + "_overlay.png"
    )

    highlighted.save(overlay_path)

    print("Device:", DEVICE)
    print("Original image size:", original_b.size)
    print("Prediction completed!")
    print("Change mask saved to:", output_path)
    print("Overlay saved to:", overlay_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Satellite image change detection"
    )

    parser.add_argument("--image_a", required=True)
    parser.add_argument("--image_b", required=True)

    parser.add_argument(
        "--output",
        default="evaluation/predicted_change.png"
    )

    args = parser.parse_args()

    predict_change(
        args.image_a,
        args.image_b,
        args.output
    )
