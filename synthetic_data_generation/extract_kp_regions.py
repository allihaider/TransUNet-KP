import os
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation

def compare_folders(images_folder, masks_folder):
    # Get lists of files in both folders
    image_files = set(os.listdir(images_folder))
    mask_files = set(os.listdir(masks_folder))
    
    # Find images without masks
    images_without_masks = [
        img for img in image_files
        if img.endswith('.jpg') and f"{os.path.splitext(img)[0]}_labels.png" not in mask_files
    ]
    
    # Find masks without images
    masks_without_images = [
        mask for mask in mask_files
        if mask.endswith('_labels.png') and f"{mask.replace('_labels.png', '')}.jpg" not in image_files
    ]
    
    # Print results
    print("Images without corresponding masks:")
    for img in images_without_masks:
        print(f"  {img}")
    
    print("\nMasks without corresponding images:")
    for mask in masks_without_images:
        print(f"  {mask}")
    
    return images_without_masks, masks_without_images

def display_superimposed_images_and_masks(images_folder, masks_folder, num_samples=3):
    # Get lists of files in both folders
    image_files = [f for f in os.listdir(images_folder)]
    mask_files = [f for f in os.listdir(masks_folder)]
    
    # Find matching pairs
    matching_pairs = [
        (img, mask) for img in image_files
        for mask in mask_files
        if img.replace('.jpg', '') == mask.replace('_labels.png', '')
    ]
    
    # Randomly select samples
    samples = random.sample(matching_pairs, min(num_samples, len(matching_pairs)))
    
    # Create subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    fig.suptitle("Images with Superimposed Masks", fontsize=16)
    
    for i, (img_file, mask_file) in enumerate(samples):
        # Read images
        img = imread(os.path.join(images_folder, img_file))
        mask = imread(os.path.join(masks_folder, mask_file))
        
        # Ensure mask is in the correct format and adjust alpha
        if mask.ndim == 3 and mask.shape[2] == 4:
            mask_rgba = mask.copy()
            mask_rgba[..., 3] = mask_rgba[..., 3] * 0.4  # Reduce opacity to 40%
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        
        # Display superimposed image
        axes[i].imshow(img)
        axes[i].imshow(mask_rgba)
        axes[i].set_title(f"{img_file}\n{mask_file}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_keratin_pearls(images_folder, masks_folder, output_folder="kp_extracts", misc_output_folder="boundary_kp_extracts"):
    # Create output directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(misc_output_folder, exist_ok=True)

    # Get lists of files in both folders
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('_labels.png')]

    # Find matching pairs
    matching_pairs = [
        (img, mask) for img in image_files
        for mask in mask_files
        if img.replace('.jpg', '') == mask.replace('_labels.png', '')
    ]

    print(f"Found {len(matching_pairs)} matching image-mask pairs.")

    for pair_index, (img_file, mask_file) in enumerate(matching_pairs, 1):
        print(f"Processing pair {pair_index}/{len(matching_pairs)}: {img_file}")

        # Read images
        img = imread(os.path.join(images_folder, img_file))
        mask = imread(os.path.join(masks_folder, mask_file))

        # Ensure mask is in the correct format
        if mask.ndim == 3 and mask.shape[2] == 4:
            # Use the alpha channel as the mask
            binary_mask = mask[..., 3] > 0
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        # Label connected regions in the mask
        labeled_mask = label(binary_mask)

        # Extract properties of labeled regions
        regions = regionprops(labeled_mask)

        print(f"  Found {len(regions)} potential keratin pearls.")

        # Counters for extracted pearls
        extracted_count = 0
        boundary_count = 0

        # Extract and save each keratin pearl
        for i, region in enumerate(regions):
            # Get the mask for this region
            region_mask = labeled_mask == region.label

            # Check if the region touches the image boundary
            touches_boundary = (np.any(region_mask[0, :]) or np.any(region_mask[-1, :]) or
                                np.any(region_mask[:, 0]) or np.any(region_mask[:, -1]))

            # Get bounding box
            y_min, x_min, y_max, x_max = region.bbox

            # Extract the region from the original image
            extracted_region = img[y_min:y_max, x_min:x_max].copy()

            # Apply the mask
            mask_slice = region_mask[y_min:y_max, x_min:x_max]
            extracted_region[~mask_slice] = 0

            # Determine the save location and update the appropriate counter
            if touches_boundary:
                save_folder = misc_output_folder
                boundary_count += 1
                count = boundary_count
            else:
                save_folder = output_folder
                extracted_count += 1
                count = extracted_count

            # Save the extracted region
            output_filename = f"{os.path.splitext(img_file)[0]}_pearl_{count}.png"
            plt.imsave(os.path.join(save_folder, output_filename), extracted_region)

            if (extracted_count + boundary_count) % 10 == 0 or (extracted_count + boundary_count) == len(regions):
                print(f"    Processed {extracted_count + boundary_count} keratin pearls "
                      f"({extracted_count} non-boundary, {boundary_count} boundary-touching)")

        print(f"  Completed processing {img_file}. "
              f"Extracted {extracted_count} non-boundary pearls and {boundary_count} boundary-touching pearls.")

    print(f"Extraction complete. "
          f"Non-boundary pearls saved to {output_folder}, "
          f"boundary-touching pearls saved to {misc_output_folder}")

# Usage
images_folder = "./synthetic_data/cp_aug/images"
masks_folder = "./synthetic_data/cp_aug/masks"

# Add boolean variables to toggle functions
run_comparison = False
run_display = True 
run_extraction = False 

if run_comparison:
    # Compare folders and print results
    images_without_masks, masks_without_images = compare_folders(images_folder, masks_folder)

if run_display:
    # Display sample images with superimposed masks
    display_superimposed_images_and_masks(images_folder, masks_folder, num_samples=1)

if run_extraction:
    # Extract keratin pearls
    extract_keratin_pearls(images_folder, masks_folder)
