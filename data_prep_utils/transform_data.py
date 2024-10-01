import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def split_image(image, chunk_size=400):
    width, height = image.size
    chunks = []
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            box = (j, i, j+chunk_size, i+chunk_size)
            chunk = image.crop(box)
            chunks.append(chunk)
    return chunks

def process_dataset(input_root, output_root):
    # Create necessary directories
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
        mask_path = os.path.join(input_root, "masks", img_name.replace(".jpg", "_labels.png"))
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_name}. Skipping.")
            continue
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        img_chunks = split_image(img)
        mask_chunks = split_image(mask)
        
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
        mask_path = os.path.join(input_root, "test_bbox_masks", img_name)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_name}. Skipping.")
            continue
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img_chunks = split_image(img)
        mask_chunks = split_image(mask)
        
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
input_root = "./original_data/"
output_root = "./transformed_data/"
process_dataset(input_root, output_root)
