import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_indexed_png(file_path):
    # Open the image
    img = Image.open(file_path)
    
    # Ensure it's an indexed image
    if img.mode != 'P':
        raise ValueError("This is not an indexed image")
    
    # Get the palette
    palette = np.array(img.getpalette()).reshape(-1, 3)
    
    # Convert image to numpy array (this will be the label/mask array)
    label_array = np.array(img)
    
    # Create an RGB representation
    rgb_array = palette[label_array]
    
    return label_array, rgb_array

def display_images(label_array, rgb_array):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display label/mask image
    im1 = ax1.imshow(label_array, cmap='tab20')  # Using tab20 for discrete colors
    ax1.set_title('Label/Mask Image')
    plt.colorbar(im1, ax=ax1)
    
    # Display RGB representation
    ax2.imshow(rgb_array)
    ax2.set_title('RGB Representation')
    
    plt.tight_layout()
    plt.show()

# Usage
file_path = 'original_data/test/29-17-IIDC.svs_[12000,8000,4000,4000]_27.png'
label_array, rgb_array = read_indexed_png(file_path)
display_images(label_array, rgb_array)
