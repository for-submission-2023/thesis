"""
COCO_2017 Dataset for Segmentation

Original code structure follows isic.py from Datasets folder.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TorchvisionTransformWrapper:
    """
    Wraps a torchvision transform to be compatible with Albumentations-style interface.
    
    Torchvision transforms (from weights.transforms()) expect PIL Images and return tensors.
    This wrapper adapts them to work with the (image=..., mask=...) interface.
    """
    def __init__(self, torchvision_transform, resize_mask=True):
        self.transform = torchvision_transform
        self.resize_mask = resize_mask
        
    def __call__(self, image, mask):
        # image is numpy array, mask is numpy array
        # Convert numpy to PIL for torchvision transform
        image_pil = Image.fromarray(image)
        
        # Apply torchvision transform to image
        transformed_image = self.transform(image_pil)
        
        # For mask: resize to match transformed image size
        mask_pil = Image.fromarray(mask)
        if self.resize_mask:
            # Get target size from transform or use image size
            target_size = (transformed_image.shape[1], transformed_image.shape[2])
            mask_pil = mask_pil.resize((target_size[1], target_size[0]), Image.NEAREST)
        
        # Convert mask back to numpy (ToTensor will be applied by dataset)
        transformed_mask = np.array(mask_pil)
        
        return {"image": transformed_image, "mask": transformed_mask}


class COCODataset(Dataset):
    """
    COCO Segmentation Dataset.
        
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
        
        # Apply transform
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert to tensor (handle both numpy arrays and tensors)
        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image).to(torch.float)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        if self.return_path:
            return image, mask, self.image_paths[i]	
        return image, mask
