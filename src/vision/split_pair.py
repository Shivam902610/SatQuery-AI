
from PIL import Image
from pathlib import Path

input_path = Path("data/samples/test.jpg")
output_dir = Path("data/samples")

image = Image.open(input_path).convert("RGB")

width, height = image.size
midpoint = width // 2

left_image = image.crop((0, 0, midpoint, height))
right_image = image.crop((midpoint, 0, width, height))

left_image.save(output_dir / "image_2023.jpg")
right_image.save(output_dir / "image_later.jpg")

print("Images separated successfully!")
print("Left image:", left_image.size)
print("Right image:", right_image.size)
