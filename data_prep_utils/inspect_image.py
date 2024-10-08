import sys
from PIL import Image
import numpy as np

def inspect_png(file_path):
    try:
        # Open the PNG file
        with Image.open(file_path) as img:
            print(f"Contents of {file_path}:")
            print("-" * 50)
            
            # Basic image information
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            print(f"Shape: {img_array.shape}")
            print(f"Dtype: {img_array.dtype}")
            
            # Color channel information
            if len(img_array.shape) == 3:
                channels = img_array.shape[2]
                print(f"Number of channels: {channels}")
                
                for i in range(channels):
                    channel = img_array[:,:,i]
                    print(f"Channel {i}:")
                    print(f"  Min value: {channel.min()}")
                    print(f"  Max value: {channel.max()}")
                    print(f"  Mean value: {channel.mean():.2f}")

                    unique_values = np.unique(channel)
                    print(f"  Unique values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
                    print(f"  Number of unique values: {len(unique_values)}")
            else:
                print("Single channel (grayscale) image:")
                print(f"  Min value: {img_array.min()}")
                print(f"  Max value: {img_array.max()}")
                print(f"  Mean value: {img_array.mean():.2f}")

                unique_values = np.unique(img_array)
                print(f"  Unique values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
                print(f"  Number of unique values: {len(unique_values)}")
            
            # Print a small sample of the data
            print("Sample data (top-left 3x3 pixel values):")
            print(img_array[:3, :3])
            
            print("-" * 50)
    except Exception as e:
        print(f"Error occurred while inspecting the file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_png.py <path_to_png_file>")
    else:
        file_path = sys.argv[1]
        inspect_png(file_path)
