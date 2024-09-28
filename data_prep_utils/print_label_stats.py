import os
import numpy as np
from tqdm import tqdm

def inspect_npz_labels(folder_path):
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    
    print(f"Found {len(npz_files)} NPZ files in the folder.")
    print("Inspecting label arrays...")

    total_labels = 0
    one_unique_count = 0
    two_unique_count = 0

    for npz_file in tqdm(npz_files, desc="Processing files"):
        file_path = os.path.join(folder_path, npz_file)
        
        try:
            with np.load(file_path) as data:
                if 'label' not in data:
                    print(f"\nWarning: {npz_file} does not contain a 'label' array.")
                    continue
                
                label_array = data['label']
                unique_values = np.unique(label_array)
                
                total_labels += 1
                
                if len(unique_values) == 1:
                    one_unique_count += 1
                elif len(unique_values) == 2:
                    two_unique_count += 1
                    print(f"\nFile with pearls: {npz_file}")
                else:
                    print(f"\nWarning: {npz_file} has {len(unique_values)} unique values, which is unexpected.")
                
        except Exception as e:
            print(f"\nError processing {npz_file}: {str(e)}")

    # Calculate statistics
    two_unique_percentage = (two_unique_count / total_labels) * 100 if total_labels > 0 else 0

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total number of labels: {total_labels}")
    print(f"Labels with 1 unique value: {one_unique_count}")
    print(f"Labels with 2 unique values: {two_unique_count}")
    print(f"Percentage of labels with 2 unique values: {two_unique_percentage:.2f}%")

if __name__ == "__main__":
    npz_folder_path = "transformed_data/data/KeratinPearls/train_npz/"
    inspect_npz_labels(npz_folder_path)
