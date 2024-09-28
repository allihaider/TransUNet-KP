import os
import numpy as np
from tqdm import tqdm
import random

def analyze_and_balance_dataset(train_npz_folder, all_lst_path, train_txt_path):
    # Step 1: Analyze NPZ files
    single_label_files = []
    double_label_files = []

    for npz_file in tqdm(os.listdir(train_npz_folder), desc="Analyzing NPZ files"):
        if npz_file.endswith('.npz'):
            file_path = os.path.join(train_npz_folder, npz_file)
            with np.load(file_path) as data:
                if 'label' in data:
                    unique_values = np.unique(data['label'])
                    if len(unique_values) == 1:
                        single_label_files.append(npz_file)
                    elif len(unique_values) == 2:
                        double_label_files.append(npz_file)

    print(f"Files with 1 unique label: {len(single_label_files)}")
    print(f"Files with 2 unique labels: {len(double_label_files)}")

    # Step 2: Balance the dataset
    num_to_keep = len(double_label_files)
    single_label_files_to_keep = random.sample(single_label_files, num_to_keep)
    files_to_remove = set(single_label_files) - set(single_label_files_to_keep)

    print(f"Removing {len(files_to_remove)} files with 1 unique label")

    # Remove excess single-label NPZ files
    for file_to_remove in files_to_remove:
        os.remove(os.path.join(train_npz_folder, file_to_remove))

    # Step 3: Update all.lst and train.txt files
    def update_all_lst(file_path, files_to_remove):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = [line for line in lines if not any(file in line for file in files_to_remove)]
        
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)

    def update_train_txt(file_path, files_to_remove):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        files_to_remove_without_extension = [os.path.splitext(file)[0] for file in files_to_remove]
        updated_lines = [line for line in lines if not any(file in line for file in files_to_remove_without_extension)]
        
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)

    update_all_lst(all_lst_path, files_to_remove)
    update_train_txt(train_txt_path, files_to_remove)

    print("Dataset balanced and list files updated")

# Usage
train_npz_folder = "./transformed_data/data/KeratinPearls/train_npz/"
all_lst_path = "./transformed_data/TransUNet/lists/lists_KeratinPearls/all.lst"
train_txt_path = "./transformed_data/TransUNet/lists/lists_KeratinPearls/train.txt"

analyze_and_balance_dataset(train_npz_folder, all_lst_path, train_txt_path)
