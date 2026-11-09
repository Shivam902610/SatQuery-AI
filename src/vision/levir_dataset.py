
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRDataset(Dataset):

    def __init__(self, root, split="train", image_size=256):
        self.root = Path(root) / split
        self.image_size = image_size

        self.image_a_dir = self.root / "A"
        self.image_b_dir = self.root / "B"
        self.label_dir = self.root / "label"

        self.files = sorted(
            p for p in self.image_a_dir.iterdir()
            if p.is_file()
        )

        if not self.files:
            raise RuntimeError(f"No images found in {self.image_a_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index].name

        image_a = Image.open(
            self.image_a_dir / filename
        ).convert("RGB")

        image_b = Image.open(
            self.image_b_dir / filename
        ).convert("RGB")

        mask = Image.open(
            self.label_dir / filename
        ).convert("L")

        image_a = image_a.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        image_b = image_b.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image_a = np.array(image_a, dtype=np.float32) / 255.0
        image_b = np.array(image_b, dtype=np.float32) / 255.0

        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        image_a = torch.from_numpy(image_a).permute(2, 0, 1)
        image_b = torch.from_numpy(image_b).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image_a, image_b, mask
