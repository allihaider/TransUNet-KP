import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(0, 1), order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            zoom_factors = (self.output_size[0] / h, self.output_size[1] / w) + (1,) * (image.ndim - 2)
            image = zoom(image, zoom_factors, order=3)
            label = zoom(label, zoom_factors[:2], order=0)

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class KeratinPearls_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')

        if self.split == "train":
            data_path = os.path.join(self.data_dir, 'train_npz', f"{slice_name}.npz")
        else:
            data_path = os.path.join(self.data_dir, 'test_npz', f"{slice_name}.npz")
        
        data = np.load(data_path)
        image, label = data['image'], data['label']
  
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
            
        sample['case_name'] = slice_name
        return sample


