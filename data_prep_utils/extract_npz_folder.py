import os
import numpy as np
from PIL import Image

def extract_npz_contents(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_processed = 0
    count_over_50 = 0
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.npz'):
            npz_path = os.path.join(input_folder, filename)
            
            # Load the NPZ file
            with np.load(npz_path) as data:
                # Extract image and label
                image = data['image']
                label = data['label']

                if len(np.unique(label)) >= 2:
                    total_processed += 1
                    # Generate base filename (without extension)
                    base_filename = os.path.splitext(filename)[0]

                    # Save image (RGB, 0-255)
                    image_filename = f"{base_filename}_image.png"
                    image_path = os.path.join(output_folder, image_filename)
                    # Image.fromarray(image.astype(np.uint8)).save(image_path)

                    # Save label (binary mask, 0 and 1)
                    label_filename = f"{base_filename}_label.png"
                    label_path = os.path.join(output_folder, label_filename)
                    # Convert binary mask to 0 and 255 for better visibility
                    label_image = Image.fromarray((label * 255).astype(np.uint8), mode='L')
                    label_image.save(label_path)

                    total_pixels = label.size
                    pixels_with_value_1 = np.sum(label == 1)
                    percentage = (pixels_with_value_1 / total_pixels) * 100
                    
                    if percentage >= 50:
                        count_over_50 += 1
                    
                    print(f"Percentage of label with value 1: {percentage:.2f}%")

                    # print(f"Extracted {image_filename} and {label_filename}")
    print(f"Proportion of images with over 50% area covered by labels: {count_over_50/total_processed * 100}")

if __name__ == "__main__":
    input_folder = "transformed_data/data/KeratinPearls/train_npz/"  # Replace with your input folder path
    output_folder = "extracted_contents/"  # Replace with your desired output folder path
    extract_npz_contents(input_folder, output_folder)
