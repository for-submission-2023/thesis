"""
Oxford-IIIT Pet Dataset for Semantic Segmentation

Supports:
- Binary mode: background (0), cat (1), dog (2)
- Multiclass mode: background (0) + 37 breeds (1-37)

Original mask values: 1=foreground, 2=background, 3=unknown/border
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AnimalDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset.
    
    Args:
        image_paths: List of paths to images
        mask_paths: List of paths to masks
        transform: Albumentations transform (optional)
        mode: 'binary' (cat vs dog) or 'multiclass' (37 breeds)
    
    Returns:
        (image, mask)
    """
    
    BREED_NAMES = [
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx',
        'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
        'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
        'english_setter', 'german_shorthaired', 'great_pyrenees',
        'havanese', 'japanese_chin', 'keeshond', 'leonberger',
        'miniature_pinscher', 'newfoundland', 'pomeranian',
        'pug', 'saint_bernard', 'samoyed', 'scottish_terrier',
        'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier',
        'yorkshire_terrier'
    ]
    
    def __init__(self, image_paths, mask_paths, transform=None, mode='binary', return_path = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mode = mode
        
        if mode == 'binary':
            self.num_classes = 3
        else:
            self.num_classes = 38

        self.return_path = return_path
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        # Load image
        image = Image.open(self.image_paths[i])
        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
        
        # Load mask
        mask = Image.open(self.mask_paths[i])
        # Convert to single channel if needed
        if mask.mode != 'L':
            mask = mask.convert('L')
        mask = np.array(mask)
        
        mask[mask != 2] = 0
        mask = mask == 0
        mask = mask * 1
        
        if self.mode == 'binary':
            if self.image_paths[i].split(os.path.sep)[-1].split('.')[0][0].isupper():
                mask[mask == 1] = 1  # Cat
            else:
                mask[mask == 1] = 2  # Dog
        else:            
            breed = '_'.join(self.image_paths[i].split(os.sep)[-1].split('_')[:-1])
            breed_idx = self.BREED_NAMES.index(breed) + 1
            mask[mask == 1] = breed_idx
        
        # Apply transform
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert to tensor
        image = transforms.ToTensor()(image).to(torch.float)
        mask = torch.from_numpy(mask).long()
        
        if self.return_path:
            return image, mask, self.image_paths[i]	
        return image, mask
