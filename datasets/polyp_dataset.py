"""
Polyp Segmentation Dataset (MedAI 2021)

Binary segmentation: background (0), polyp (1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PolypDataset(Dataset):
    """
    MedAI 2021 Polyp Segmentation Dataset.
    Same structure as NoraDataset in polyp.py
    
    Args:
        image_paths: List of paths to images
        mask_paths: List of paths to masks
        transform: Albumentations transform (optional)
        thresh: Threshold for binarizing mask (default 127)
    
    Returns:
        (image, mask)
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, thresh=127, return_path=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.thresh = thresh
        self.num_classes = 2

        self.return_path = return_path
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        image = np.array(Image.open(self.image_paths[i]))
        mask = np.array(Image.open(self.mask_paths[i]))[:, :, 0]
        mask = (mask > self.thresh).astype(float)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        
        image = transforms.ToTensor()(image).to(torch.float)
        mask = torch.from_numpy(mask).long()
        
        if self.return_path:
            return image, mask, self.image_paths[i]	
        return image, mask
