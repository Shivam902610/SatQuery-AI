
from PIL import Image

image_2023 = Image.open(
    "data/samples/image_2023.jpg"
).convert("RGB")

image_later = Image.open(
    "data/samples/image_later.jpg"
).convert("RGB")

# Blend both images equally
overlay = Image.blend(image_2023, image_later, 0.5)

overlay.save("data/samples/alignment_check.jpg")

print("Alignment check saved!")
