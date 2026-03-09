"""
ISIC Skin Lesion Dataset for Segmentation

Original code structure follows isic.py from Datasets folder.

Binary segmentation: background (0), lesion (1)
Classification labels: NV (0) vs Others (1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ISICDataset(Dataset):
    """
    ISIC Skin Lesion Segmentation Dataset.
    
    Follows the same structure as SkinDataset in isic.py
    
    Args:
        image_paths: List of paths to images
        mask_paths: List of paths to masks (_segmentation.png)
        transform: Albumentations transform (optional)
    
    Returns:
        (image, mask)
        image: [3, H, W] float32 tensor
        mask: [H, W] long tensor
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, return_path=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.num_classes = 2

        self.return_path = return_path
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        # Load image using PIL (same as original)
        image = Image.open(self.image_paths[i])
        image = np.array(image)
        
        # Load mask using PIL (same as original)
        mask = Image.open(self.mask_paths[i])
        mask = np.array(mask)
        mask[mask == 255] = 1  # Convert 255 to 1
        
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
