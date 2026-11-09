
import cv2
import numpy as np

# Load reference and aligned images
reference = cv2.imread("data/samples/image_2023.jpg")
later = cv2.imread("data/samples/aligned_later.jpg")

if reference is None or later is None:
    raise FileNotFoundError("Images not found.")

# Calculate pixel differences
difference = cv2.absdiff(reference, later)

# Convert differences to grayscale
gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# Reduce minor image noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Highlight significant differences
_, mask = cv2.threshold(
    blurred, 30, 255, cv2.THRESH_BINARY
)

# Remove small isolated noise
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)

# Highlight detected differences in red
result = reference.copy()
result[mask > 0] = [0, 0, 255]

# Save outputs
cv2.imwrite("data/samples/aligned_difference.jpg", difference)
cv2.imwrite("data/samples/aligned_change_map.jpg", mask)
cv2.imwrite("data/samples/aligned_change_overlay.jpg", result)

print("Aligned change detection completed!")
print("Generated 3 output images.")
