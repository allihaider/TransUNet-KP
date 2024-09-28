import os
import random
import numpy as np
from matplotlib.image import imread, imsave
from skimage.measure import label, regionprops
from skimage.transform import resize

def extract_keratin_pearls(images_folder, masks_folder):
    print("Starting keratin pearl extraction...")
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('_labels.png')]

    matching_pairs = [
        (img, mask) for img in image_files
        for mask in mask_files
        if img.replace('.jpg', '') == mask.replace('_labels.png', '')
    ]

    print(f"Found {len(matching_pairs)} matching image-mask pairs.")

    keratin_pearls = []

    for idx, (img_file, mask_file) in enumerate(matching_pairs, 1):
        print(f"Processing pair {idx}/{len(matching_pairs)}: {img_file}")
        img = imread(os.path.join(images_folder, img_file))
        mask = imread(os.path.join(masks_folder, mask_file))

        print(f"  Image shape: {img.shape}, Mask shape: {mask.shape}")

        # Ensure mask is binary
        if mask.ndim == 3:
            mask = mask[..., 0] > 0  # Use the first channel if multi-channel
            print("  Converted multi-channel mask to binary")
        else:
            mask = mask > 0
            print("  Converted single-channel mask to binary")

        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        print(f"  Found {len(regions)} potential keratin pearls")

        for region_idx, region in enumerate(regions, 1):
            y_min, x_min, y_max, x_max = region.bbox
            region_mask = labeled_mask == region.label

            extracted_region = img[y_min:y_max, x_min:x_max].copy()
            mask_slice = region_mask[y_min:y_max, x_min:x_max]
            extracted_region[~mask_slice] = 0

            keratin_pearls.append({
                'image': extracted_region,
                'mask': mask_slice
            })

            if region_idx % 10 == 0 or region_idx == len(regions):
                print(f"    Processed {region_idx}/{len(regions)} regions")

    print(f"Extraction complete. Total keratin pearls extracted: {len(keratin_pearls)}")
    return keratin_pearls

def generate_synthetic_data(images_folder, masks_folder, output_images_folder, output_masks_folder, num_synthetic_images=10, max_pearls_per_image=5):
    print(f"Starting synthetic data generation...")
    print(f"Parameters: num_synthetic_images={num_synthetic_images}, max_pearls_per_image={max_pearls_per_image}")

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)
    print(f"Output directories created/verified: {output_images_folder}, {output_masks_folder}")

    # Extract keratin pearls
    keratin_pearls = extract_keratin_pearls(images_folder, masks_folder)

    # Get list of all images
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} source images")

    for i in range(num_synthetic_images):
        print(f"\nGenerating synthetic image {i+1}/{num_synthetic_images}")
        
        # Randomly select a target image
        target_img_file = random.choice(image_files)
        print(f"  Selected target image: {target_img_file}")
        
        target_img = imread(os.path.join(images_folder, target_img_file))
        target_mask_file = target_img_file.replace('.jpg', '_labels.png')
        target_mask = imread(os.path.join(masks_folder, target_mask_file))

        print(f"  Target image shape: {target_img.shape}, Target mask shape: {target_mask.shape}")

        # Ensure mask is binary
        if target_mask.ndim == 3:
            target_mask = target_mask[..., 0] > 0  # Use the first channel if multi-channel
            print("  Converted multi-channel target mask to binary")
        else:
            target_mask = target_mask > 0
            print("  Converted single-channel target mask to binary")

        # Create a copy of the target image and mask
        synthetic_img = target_img.copy()
        synthetic_mask = target_mask.copy().astype(np.uint8) * 255

        # Randomly select number of pearls to add
        num_pearls = random.randint(1, max_pearls_per_image)
        print(f"  Adding {num_pearls} keratin pearls")

        for pearl_idx in range(num_pearls):
            # Randomly select a keratin pearl
            pearl = random.choice(keratin_pearls)

            # Randomly select position to paste the pearl
            max_y = synthetic_img.shape[0] - pearl['image'].shape[0]
            max_x = synthetic_img.shape[1] - pearl['image'].shape[1]
            paste_y = random.randint(0, max_y)
            paste_x = random.randint(0, max_x)

            print(f"    Pearl {pearl_idx + 1}: Pasting at position ({paste_x}, {paste_y})")

            # Paste the pearl into the synthetic image
            pearl_height, pearl_width = pearl['image'].shape[:2]
            synthetic_img[paste_y:paste_y+pearl_height, paste_x:paste_x+pearl_width] = np.where(
                pearl['image'] != 0, pearl['image'], synthetic_img[paste_y:paste_y+pearl_height, paste_x:paste_x+pearl_width]
            )

            # Update the synthetic mask
            synthetic_mask[paste_y:paste_y+pearl_height, paste_x:paste_x+pearl_width] = np.where(
                pearl['mask'], 255, synthetic_mask[paste_y:paste_y+pearl_height, paste_x:paste_x+pearl_width]
            )

        # Save the synthetic image and mask
        output_img_file = f"synthetic_{i+1}.jpg"
        output_mask_file = f"synthetic_{i+1}_labels.png"
        
        imsave(os.path.join(output_images_folder, output_img_file), synthetic_img)
        imsave(os.path.join(output_masks_folder, output_mask_file), synthetic_mask, cmap='gray')
        
        print(f"  Saved synthetic image: {output_img_file}")
        print(f"  Saved synthetic mask: {output_mask_file}")

    print("\nSynthetic data generation complete.")

# Usage
images_folder = "./data/test_images"
masks_folder = "./data/test_masks"
output_images_folder = "./synthetic_data/cp_aug/images"
output_masks_folder = "./synthetic_data/cp_aug/masks"

print("Starting script execution...")
print(f"Input folders: images={images_folder}, masks={masks_folder}")
print(f"Output folders: images={output_images_folder}, masks={output_masks_folder}")

generate_synthetic_data(images_folder, masks_folder, output_images_folder, output_masks_folder, num_synthetic_images=5, max_pearls_per_image=15)

print("Script execution completed.")
