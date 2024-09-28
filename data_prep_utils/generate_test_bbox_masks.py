import os
from lxml import etree
import numpy as np
from PIL import Image

def create_binary_image(xml_path, output_path):
    print(f"\nProcessing XML file: {xml_path}")
    
    # Parse the XML file
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    print(f"Image dimensions: {width}x{height}")

    # Create a blank image array
    img_array = np.zeros((height, width), dtype=np.uint8)

    # Draw bounding boxes
    bounding_box_count = 0
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        print(f"Drawing bounding box: ({xmin}, {ymin}) to ({xmax}, {ymax})")
        img_array[ymin:ymax, xmin:xmax] = 1
        bounding_box_count += 1

    print(f"Total bounding boxes drawn: {bounding_box_count}")

    # Sanity check
    unique_values = np.unique(img_array)
    print(f"Unique values in the image: {unique_values}")
    print(f"Min value: {img_array.min()}, Max value: {img_array.max()}")

    # Create a palette-based image
    img = Image.fromarray(img_array, mode='P')
    img.putpalette([0, 0, 0, 255, 255, 255])  # Black for 0, White for 1

    # Save as PNG
    img.save(output_path)
    print(f"Saved binary mask to: {output_path}")

    # Double-check the saved image
    saved_img = Image.open(output_path)
    saved_array = np.array(saved_img)
    saved_unique = np.unique(saved_array)
    print(f"Saved image mode: {saved_img.mode}")
    print(f"Unique values in saved image: {saved_unique}")
    if not np.array_equal(saved_unique, unique_values):
        print("WARNING: Saved image values differ from original array!")

# Paths
labels_folder = 'original_data/labels'
test_folder = 'original_data/test'
output_folder = 'original_data/test_bbox_masks'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder created/verified: {output_folder}")

# Process each test image
for test_image in os.listdir(test_folder):
    if test_image.endswith('.png'):
        print(f"\nProcessing test image: {test_image}")
        # Find corresponding XML file
        xml_name = test_image[:-4] + '.xml'
        xml_path = os.path.join(labels_folder, xml_name)
        
        if os.path.exists(xml_path):
            output_path = os.path.join(output_folder, test_image)
            create_binary_image(xml_path, output_path)
            print(f"Created binary mask for {test_image}")
        else:
            print(f"WARNING: XML file not found for {test_image}")

print("\nProcessing complete.")

# Final sanity check
print("\nFinal Sanity Check:")
print(f"Number of test images: {len([f for f in os.listdir(test_folder) if f.endswith('.png')])}")
print(f"Number of binary masks created: {len([f for f in os.listdir(output_folder) if f.endswith('.png')])}")

# Check all created masks
print("\nChecking all created masks:")
for mask_file in os.listdir(output_folder):
    if mask_file.endswith('.png'):
        mask_path = os.path.join(output_folder, mask_file)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        unique_values = np.unique(mask_array)
        print(f"{mask_file}:")
        print(f"  Mode: {mask.mode}")
        print(f"  Size: {mask.size}")
        print(f"  Shape: {mask_array.shape}")
        print(f"  Dtype: {mask_array.dtype}")
        print(f"  Unique values: {unique_values}")
        print(f"  Min value: {mask_array.min()}")
        print(f"  Max value: {mask_array.max()}")
        print(f"  Mean value: {mask_array.mean():.2f}")
        print(f"  Sample data (top-left 3x3 pixel values):")
        print(mask_array[:3, :3])
        print()

        if not np.array_equal(unique_values, [0, 1]):
            print(f"WARNING: Mask {mask_file} contains values other than 0 and 1!")
