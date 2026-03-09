import matplotlib
matplotlib.use('TkAgg')

import os
import numpy as np
np.bool = np.bool_
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '/home/litisnouman/Desktop/Thesis_18_02_26')


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

EXPERIMENT_NAME = 'coco_2017'

# Dataset
NUM_CLASSES  = 21    # 20 COCO categories + background (PyTorch torchvision pretrained)
IMG_SIZE     = 224
IN_CHANNELS  = 3

# ==============================================================================
# DATASET PATHS
# ==============================================================================

ROOT         = '/home/litisnouman/Desktop/Thesis_18_02_26/datasets/COCO_2017'

images_paths = [os.path.join(ROOT, 'images', i) for i in os.listdir(os.path.join(ROOT, 'images')) if '.jpg' in i]
masks_paths  = [i.replace('images', 'masks').replace('.jpg', '.png') for i in images_paths]

train_images, valid_images, train_masks, valid_masks = train_test_split(images_paths, masks_paths, test_size=0.33, random_state=42)
test_images,  test_masks = valid_images[:], valid_masks[:]
print(f'Train: {len(train_images)} | Valid: {len(valid_images)} | Test: {len(test_images)}')
