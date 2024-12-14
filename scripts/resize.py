import os
import cv2
from tqdm import tqdm

# Input and output paths
images_dir = "images"  # Directory containing the original images
masks_dir = "binarymasks"  # Directory containing the original masks
output_images_dir = "resizedimages"  # Directory to save resized images
output_masks_dir = "resizedmasks"  # Directory to save resized masks
resize_dim = (224, 224)  # Target dimension for resizing

# Create the output directories if they don't exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

def resize_images():
    """
    Resizes images using cubic interpolation and saves them.
    """
    for image_file in tqdm(os.listdir(images_dir), desc="Resizing images"):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read {image_file}")
                continue
            resized_image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_images_dir, image_file), resized_image)

def resize_masks():
    """
    Resizes masks using nearest-neighbor interpolation and saves them.
    """
    for mask_file in tqdm(os.listdir(masks_dir), desc="Resizing masks"):
        if mask_file.endswith(('.png', '.jpg', '.jpeg')):
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Could not read {mask_file}")
                continue
            resized_mask = cv2.resize(mask, resize_dim, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(output_masks_dir, mask_file), resized_mask)

# Run the resizing functions
resize_images()
resize_masks()

print("Resizing complete! Images and masks have been processed.")
