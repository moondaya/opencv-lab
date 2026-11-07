# task01_preprocess.py — Task 1: Automatic image preprocessing

# STEP 01
from pathlib import Path
import cv2
import numpy as np

# ---------- Settings ----------
BASE_DIR = Path(__file__).resolve().parent          # folder of this script (01/)
IMAGE_PATH = BASE_DIR / "data" / "task01_input.jpg"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)                    # create results/ if missing

print("OpenCV version:", cv2.__version__)

# ---------- Step 1: Load the image ----------
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(f"Could not read the image: {IMAGE_PATH}")

print("Image loaded:", IMAGE_PATH.name)

#STEP 02


height, width = image.shape[:2]
channels = image.shape[2] if image.ndim == 3 else 1

print("\n--- Image information ---")
print(f"Width:     {width} px")
print(f"Height:    {height} px")
print(f"Channels:  {channels}")
print(f"Data type: {image.dtype}")
print(f"Pixels:    {width * height:,}")
print(f"Aspect ratio (W/H): {width / height:.3f}")

#OpenCV was originally developed in an era when many systems 
# and image formats commonly represented color data in BGR byte order. 
# Since OpenCV's internal image representation was designed around these conventions,
#  BGR became its default. RGB is the common conceptual color order; 
# BGR is the historical storage convention used by OpenCV.
# therefore in order to display the image correctly using libraries like Matplotlib,
#which expect images in RGB format, we need to convert the image from BGR to RGB.
# Step 3: BGR -> RGB 
import matplotlib.pyplot as plt


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Proof that the channels were swapped: compare one pixel
y, x = height // 2, width // 2                     # centre pixel
print("\n--- BGR -> RGB check (centre pixel) ---")
print("BGR (original):", image[y, x])
print("RGB (converted):", image_rgb[y, x])
print("Shape unchanged:", image.shape == image_rgb.shape)

# Visual demonstration: what happens WITHOUT conversion
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(image)                              # wrong: BGR shown as if RGB
axes[0].set_title("Without conversion (BGR shown as RGB)")
axes[1].imshow(image_rgb)                          # correct
axes[1].set_title("After cv2.COLOR_BGR2RGB")
for ax in axes:
    ax.axis("off")
# plt.tight_layout()
# plt.savefig(RESULTS_DIR / "task01_step3_bgr_vs_rgb.png", dpi=150)
# plt.show()

#Step 4 — Create a grayscale version
# combine three colour channels into one brightness channel.
#  Grayscale images are one-third the data,
# ---------- Step 4: Grayscale ----------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("\n--- Grayscale check ---")
print("Gray shape:", gray.shape, "| ndim:", gray.ndim)     # 2D, no channel axis
b, g, r = image[y, x].astype(float)
manual = 0.299 * r + 0.587 * g + 0.114 * b
print(f"Centre pixel: OpenCV = {gray[y, x]}, manual formula = {manual:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(image_rgb)
axes[0].set_title("Original (RGB)")
axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Grayscale (cv2.COLOR_BGR2GRAY)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "task01_step4_grayscale.png", dpi=150)
plt.show()

cv2.imwrite(str(RESULTS_DIR / "task01_gray.png"), gray)    # saved for Task 2