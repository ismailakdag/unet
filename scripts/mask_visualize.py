import matplotlib.pyplot as plt
import cv2
import os

# Input paths for images and masks
images_dir = "resizedimages"  # Directory containing the original images
masks_dir = "resizedmasks"  # Directory containing the binary masks

def visualize_image_and_mask(image_path, mask_path):
    """
    Visualizes an image and its corresponding mask side-by-side.
    
    :param image_path: Path to the input image
    :param mask_path: Path to the binary mask
    """
    # Load image and mask
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error loading {image_path} or {mask_path}")
        return

    # Resize mask to match image dimensions if needed (optional, remove if unnecessary)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Display side-by-side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Binary Mask")
    plt.imshow(mask_resized, cmap='gray')
    plt.axis('off')

    plt.show()

# Visualize a few samples
image_files = sorted(os.listdir(images_dir))
mask_files = sorted(os.listdir(masks_dir))

for img_file, mask_file in zip(image_files[:5], mask_files[:5]):  # Visualize the first 5 samples
    img_path = os.path.join(images_dir, img_file)
    mask_path = os.path.join(masks_dir, mask_file)
    visualize_image_and_mask(img_path, mask_path)
