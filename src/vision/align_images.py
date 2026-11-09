
import cv2
import numpy as np

# Load satellite images
reference = cv2.imread("data/samples/image_2023.jpg")
moving = cv2.imread("data/samples/image_later.jpg")

if reference is None or moving is None:
    raise FileNotFoundError("Satellite images not found.")

# Convert images to grayscale
gray1 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

# Detect image features
orb = cv2.ORB_create(nfeatures=5000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise RuntimeError("Could not detect enough features.")

# Match features between images
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = matcher.knnMatch(des2, des1, k=2)

good_matches = [
    m for m, n in matches
    if m.distance < 0.75 * n.distance
]

print("Good matches:", len(good_matches))

if len(good_matches) < 10:
    raise RuntimeError("Insufficient feature matches.")

# Calculate transformation matrix
src = np.float32(
    [kp2[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

dst = np.float32(
    [kp1[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

H, mask = cv2.findHomography(
    src, dst, cv2.RANSAC, 5.0
)

if H is None:
    raise RuntimeError("Image alignment failed.")

# Evaluate alignment quality
inliers = int(mask.sum())
total_matches = len(good_matches)

inlier_ratio = inliers / total_matches * 100

print("RANSAC inliers:", inliers)
print("Inlier ratio:", round(inlier_ratio, 2), "%")


# Check spatial distribution of inlier features
inlier_points = src[mask.ravel() == 1].reshape(-1, 2)

height, width = moving.shape[:2]

grid_size = 4
occupied_cells = set()

for x, y in inlier_points:
    col = min(int(x / width * grid_size), grid_size - 1)
    row = min(int(y / height * grid_size), grid_size - 1)
    occupied_cells.add((row, col))

print("Occupied grid cells:", len(occupied_cells), "/ 16")
print(
    "Spatial coverage:",
    round(len(occupied_cells) / 16 * 100, 2),
    "%"
)

# Visualise geometrically consistent matches
match_image = cv2.drawMatches(
    moving,
    kp2,
    reference,
    kp1,
    good_matches,
    None,
    matchesMask=mask.ravel().tolist(),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    "data/samples/feature_matches.jpg",
    match_image
)

print("Feature matching visualization saved!")

# Align the later image
height, width = reference.shape[:2]

aligned = cv2.warpPerspective(
    moving, H, (width, height)
)

# Generate overlay
overlay = cv2.addWeighted(
    reference, 0.5, aligned, 0.5, 0
)

# Save results
cv2.imwrite("data/samples/aligned_later.jpg", aligned)
cv2.imwrite("data/samples/aligned_overlay.jpg", overlay)

print("Alignment completed!")
