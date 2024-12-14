import os
import numpy as np
import cv2
from tqdm import tqdm

# Input and output paths
yolo_annotations_dir = "annotations"  # Directory containing YOLO .txt files
output_masks_dir = "binarymasks"  # Directory to save generated binary masks
img_width = 3027  # Image width
img_height = 1480  # Image height

# Create the output directory if it doesn't exist
os.makedirs(output_masks_dir, exist_ok=True)

def yolo_to_binary_mask(annotation_file, output_mask_path):
    """
    Converts a YOLO-style segmentation annotation to a binary mask.

    :param annotation_file: Path to the YOLO annotation .txt file
    :param output_mask_path: Path to save the binary mask image
    """
    # Initialize a blank binary mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Read the annotation file
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Parse the segmentation line (ignore the class index if present)
        coords = list(map(float, line.strip().split()[1:]))

        # Denormalize the coordinates
        polygon = np.array(
            [[int(x * img_width), int(y * img_height)] for x, y in zip(coords[::2], coords[1::2])],
            dtype=np.int32
        )

        # Draw the polygon on the mask (fill with 1)
        cv2.fillPoly(mask, [polygon], color=1)

    # Save the binary mask
    cv2.imwrite(output_mask_path, mask)  # Scale 1 to 255 for visualization

# Process all YOLO annotation files
for annotation_file in tqdm(os.listdir(yolo_annotations_dir), desc="Processing YOLO annotations"):
    if annotation_file.endswith(".txt"):
        annotation_path = os.path.join(yolo_annotations_dir, annotation_file)
        mask_output_path = os.path.join(output_masks_dir, annotation_file.replace(".txt", ".png"))
        yolo_to_binary_mask(annotation_path, mask_output_path)

print("Binary mask generation complete!")
