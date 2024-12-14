import os
import numpy as np
import cv2
from tqdm import tqdm

# Input and output paths
images_dir = "resizedimages"  # Directory containing resized images
masks_dir = "resizedmasks"  # Directory containing resized masks
output_images_dir = "images_npy"  # Directory to save images as .npy
output_masks_dir = "masks_npy"  # Directory to save masks as .npy

# Create the output directories if they don't exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

def convert_images_to_npy():
    """
    Converts resized images to .npy format and saves them.
    """
    for image_file in tqdm(os.listdir(images_dir), desc="Converting images to .npy"):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read {image_file}")
                continue
            npy_filename = os.path.splitext(image_file)[0] + ".npy"
            np.save(os.path.join(output_images_dir, npy_filename), image)

def convert_masks_to_npy():
    """
    Converts resized masks to .npy format and saves them with the appropriate naming convention.
    """
    for mask_file in tqdm(os.listdir(masks_dir), desc="Converting masks to .npy"):
        if mask_file.endswith(('.png', '.jpg', '.jpeg')):
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Could not read {mask_file}")
                continue
            # Ensure naming convention: image name + '_mask.npy'
            base_name = os.path.splitext(mask_file)[0]
            npy_filename = base_name + "_mask.npy"
            np.save(os.path.join(output_masks_dir, npy_filename), mask)

# Run the conversion functions
convert_images_to_npy()
convert_masks_to_npy()

print("Conversion to .npy format complete! Images and masks are ready.")
