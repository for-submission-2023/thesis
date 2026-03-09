import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import cv2

class SynapseDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform = None, return_path = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.return_path = return_path
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        #print(i)
        image = cv2.imread(self.image_paths[i], 0)
        mask = cv2.imread(self.mask_paths[i], 0)

        if self.transform is not None:
            transformed = self.transform(image = image, mask = mask)
            image, mask = transformed['image'], transformed['mask']

        image = transforms.ToTensor()(image).to(torch.float)
		
        image = image.repeat(3, 1, 1)  # Repeat along channel dimension
    
        mask = torch.from_numpy(mask).long()
		
        if self.return_path:
            return image, mask, self.image_paths[i]		
        return image, mask
