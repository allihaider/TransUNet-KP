import numpy as np
import sys

def inspect_npz(file_path):
    try:
        # Load the .npz file
        with np.load(file_path) as data:
            print(f"Contents of {file_path}:")
            print("-" * 50)
            
            # Iterate through all arrays in the file
            for key in data.files:
                array = data[key]
                print(f"Key: {key}")
                print(f"  Type: {array.dtype}")
                print(f"  Shape: {array.shape}")
                print(f"  Size: {array.size}")
                
                # Print some statistics about the array
                print(f"  Min value: {array.min()}")
                print(f"  Max value: {array.max()}")
                print(f"  Mean value: {array.mean()}")
                
                # Print a small sample of the data
                print("  Sample data:")
                if array.ndim == 1:
                    print(f"    {array[:5]} ...")
                elif array.ndim == 2:
                    print(f"    {array[:3, :3]} ...")
                else:
                    print(f"    {array.flatten()[:5]} ...")
                
                print("-" * 50)

    except Exception as e:
        print(f"Error occurred while inspecting the file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_npz.py <path_to_npz_file>")
    else:
        file_path = sys.argv[1]
        inspect_npz(file_path)
