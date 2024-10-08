import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
def split_image(image, chunk_size=400):
    width, height = image.size
    chunks = []
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            box = (j, i, j+chunk_size, i+chunk_size)
            chunk = image.crop(box)
            chunks.append(chunk)
    return chunks

def get_corresponding_mask_filename(test_filename):
    base_part = test_filename.split('.svs_')[0]
    coords = re.search(r'\[(\d+),(\d+),(\d+),(\d+)\]', test_filename)
    if coords:
        x, y, width, height = map(int, coords.groups())
        mask_pattern = f"{base_part}_\\(1.00,{y},{x},{height},{width}\\)_labels\\.png"
        return mask_pattern
    return None

def find_matching_files(test_folder, masks_folder):
    matches = {}
    unmatched_test_files = []
    for test_file in os.listdir(test_folder):
        mask_pattern = get_corresponding_mask_filename(test_file)
        if mask_pattern:
            found_match = False
            for mask_file in os.listdir(masks_folder):
                if re.match(mask_pattern, mask_file):
                    matches[test_file] = mask_file
                    found_match = True
                    break
            if not found_match:
                unmatched_test_files.append(test_file)
        else:
            unmatched_test_files.append(test_file)
    return matches, unmatched_test_files

def process_dataset(input_root, output_root):
    os.makedirs(os.path.join(output_root, "data", "KeratinPearls", "train_npz"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "data", "KeratinPearls", "test_npz"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "TransUNet", "lists", "lists_KeratinPearls"), exist_ok=True)

    all_slices = []
    train_slices = []
    test_slices = []

    # Process training data
    train_dir = os.path.join(input_root, "train")
    train_files = os.listdir(train_dir)
    print("Processing training data:")
    for img_name in tqdm(train_files, desc="Training"):
        case_name = '.'.join(img_name.split('.')[:-1])
        img_path = os.path.join(train_dir, img_name)
        mask_path = os.path.join(input_root, "masks", img_name.replace(".png", "_labels.png"))
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_name}. Skipping.")
            continue
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        mask_array = np.array(mask)
    
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0] 
    
        binary_mask = (mask_array > 127).astype(np.uint8)
    
        img_chunks = split_image(img)
        mask_chunks = split_image(Image.fromarray(binary_mask, mode='P'))
        
        for i, (img_chunk, mask_chunk) in enumerate(zip(img_chunks, mask_chunks)):
            slice_name = f"{case_name}_slice{i:03d}"
            npz_path = os.path.join(output_root, "data", "KeratinPearls", "train_npz", f"{slice_name}.npz")
            
            np.savez(npz_path, 
                     image=np.array(img_chunk), 
                     label=np.array(mask_chunk))
            
            train_slices.append(slice_name + '\n')
            all_slices.append(slice_name + '.npz\n')

    # Process test data
    test_dir = os.path.join(input_root, "test")
    test_files = os.listdir(test_dir)

    print("\nProcessing test data:")
    for img_name in tqdm(test_files, desc="Testing"):
        case_name = '.'.join(img_name.split('.')[:-1])
        img_path = os.path.join(test_dir, img_name)
        mask_path = os.path.join(input_root, "masks", img_name.replace(".png", "_labels.png"))
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_name}. Skipping.")
            continue
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        mask_array = np.array(mask)
    
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0] 
    
        binary_mask = (mask_array > 127).astype(np.uint8)
    
        img_chunks = split_image(img)
        mask_chunks = split_image(Image.fromarray(binary_mask, mode='P'))

        for i, (img_chunk, mask_chunk) in enumerate(zip(img_chunks, mask_chunks)):
            slice_name = f"{case_name}_slice{i:03d}"
            npz_path = os.path.join(output_root, "data", "KeratinPearls", "test_npz", f"{slice_name}.npz")
            
            np.savez(npz_path, 
                     image=np.array(img_chunk), 
                     label=np.array(mask_chunk))
            
            test_slices.append(slice_name + '\n')
            all_slices.append(slice_name + '.npz\n')

    print("\nWriting list files...")
    # Write list files
    with open(os.path.join(output_root, "TransUNet", "lists", "lists_KeratinPearls", "all.lst"), 'w') as f:
        f.writelines(all_slices)
    
    with open(os.path.join(output_root, "TransUNet", "lists", "lists_KeratinPearls", "train.txt"), 'w') as f:
        f.writelines(train_slices)
    
    with open(os.path.join(output_root, "TransUNet", "lists", "lists_KeratinPearls", "test.txt"), 'w') as f:
        f.writelines(test_slices)

    print("Dataset processing complete!")

# Usage
input_root = "./original_data/gaze_labelled/"
output_root = "./transformed_data/"
process_dataset(input_root, output_root)
