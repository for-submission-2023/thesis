"""
Dataset Visualization Script
============================
Display sample images from all datasets with their labels and masks.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import AnimalDataset, ISICDataset, PolypDataset, InstrumentDataset


def load_animal_paths(root):
    """Load Animal dataset paths."""
    list_file = os.path.join(root, 'annotations', 'list.txt')
    with open(list_file, 'r') as f:
        lines = f.read().split('\n')[6:-1]
    
    image_dir = os.path.join(root, 'images')
    mask_dir = os.path.join(root, 'annotations', 'trimaps')
    
    TO_REMOVE = [
        'miniature_pinscher_14.jpg', 'leonberger_18.jpg', 'saint_bernard_108.jpg',
        'saint_bernard_78.jpg', 'Persian_259.jpg', 'wheaten_terrier_195.jpg',
        'saint_bernard_60.jpg', 'japanese_chin_199.jpg', 'Egyptian_Mau_165.jpg',
        'Egyptian_Mau_196.jpg', 'Egyptian_Mau_20.jpg', 'saint_bernard_15.jpg',
        'keeshond_7.jpg', 'Egyptian_Mau_162.jpg'
    ]
    
    image_paths = []
    mask_paths = []
    labels = []
    
    for line in lines:
        img_name = line.split(' ')[0] + '.jpg'
        if img_name in TO_REMOVE:
            continue
        
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, line.split(' ')[0] + '.png')
        
        image_paths.append(img_path)
        mask_paths.append(mask_path)
        # Binary label: 1=dog (lowercase), 2=cat (uppercase)
        is_cat = img_name[0].isupper()
        labels.append(2 if is_cat else 1)
    
    return image_paths, mask_paths, labels


def load_isic_paths(root):
    """Load ISIC dataset paths."""
    df = pd.read_csv(os.path.join(root, "GroundTruth.csv"))
    
    image_dir = os.path.join(root, 'images')
    mask_dir = os.path.join(root, 'masks')
    
    image_paths = []
    mask_paths = []
    labels = []
    
    for _, row in df.iterrows():
        img_name = row['image']
        img_path = os.path.join(image_dir, img_name + '.jpg')
        mask_path = os.path.join(mask_dir, img_name + '_segmentation.png')
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        # Filter border-touching lesions
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        if (mask[0, 0] == 255 or mask[h-1, 0] == 255 or 
            mask[0, w-1] == 255 or mask[h-1, w-1] == 255):
            continue
        
        image_paths.append(img_path)
        mask_paths.append(mask_path)
        
        # Binary classification: NV=0, Others=1
        values = row[["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]].values
        labels.append(0 if values[1] == 1 else 1)
    
    return image_paths, mask_paths, labels


def load_polyp_paths(root):
    """Load Polyp dataset paths."""
    image_dir = os.path.join(root, 'images')
    mask_dir = os.path.join(root, 'masks')
    
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith(('.jpg', '.png'))])
    
    image_paths = []
    mask_paths = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + '.png')
        
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    
    # All have same label (polyp=1)
    labels = [1] * len(image_paths)
    
    return image_paths, mask_paths, labels


def load_instrument_paths(root):
    """Load Instrument dataset paths."""
    image_dir = os.path.join(root, 'images')
    mask_dir = os.path.join(root, 'masks')
    
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.jpg')])
    
    image_paths = []
    mask_paths = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    
    # All have same label (instrument=1)
    labels = [1] * len(image_paths)
    
    return image_paths, mask_paths, labels


def visualize_dataset(name, image_paths, mask_paths, labels, num_samples=3, mode='binary'):
    """Visualize samples from a dataset."""
    
    # Select random samples
    indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Label dictionaries
    if name == 'Animal':
        if mode == 'binary':
            label_names = {0: 'Background', 1: 'Dog', 2: 'Cat'}
        else:
            label_names = {i: f'Breed_{i}' for i in range(38)}
    elif name == 'ISIC':
        label_names = {0: 'NV (Benign)', 1: 'Other (Malignant)'}
    elif name == 'Polyp':
        label_names = {0: 'Background', 1: 'Polyp'}
    elif name == 'Instrument':
        label_names = {0: 'Background', 1: 'Instrument'}
    
    for idx, sample_idx in enumerate(indices):
        # Load image and mask
        image = np.array(Image.open(image_paths[sample_idx]))
        
        if name == 'Animal':
            # Animal uses cv2-style mask loading
            mask = cv2.imread(mask_paths[sample_idx], cv2.IMREAD_GRAYSCALE)
            mask[mask == 2] = 0
            mask[mask == 3] = 0
            mask = (mask == 0).astype(np.uint8)
            if mode == 'binary':
                is_cat = image_paths[sample_idx].split(os.sep)[-1][0].isupper()
                mask[mask == 1] = 2 if is_cat else 1
        elif name in ['Polyp', 'Instrument']:
            # Polyp/Instrument use PIL with [:, :, 0]
            mask = np.array(Image.open(mask_paths[sample_idx]))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = (mask > 127).astype(np.uint8)
        else:  # ISIC
            mask = np.array(Image.open(mask_paths[sample_idx]))
            mask = (mask > 127).astype(np.uint8)
        
        label = labels[sample_idx]
        
        # Display
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Image: {os.path.basename(image_paths[sample_idx])[:20]}...')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title(f'Mask (Classes: {len(np.unique(mask))})')
        axes[idx, 1].axis('off')
        
        # Overlay
        overlay = image.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[mask > 0] = [255, 0, 0]  # Red for foreground
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Label: {label_names.get(label, label)}')
        axes[idx, 2].axis('off')
    
    plt.suptitle(f'{name} Dataset Samples ({mode} mode)', fontsize=16)
    plt.tight_layout()
    return fig


def main():
    """Main visualization function."""
    
    # Dataset roots
    ANIMAL_ROOT = '/home/litisnouman/Desktop/Datasets/Animal'
    ISIC_ROOT = '/home/litisnouman/Desktop/Datasets/ISIC'
    POLYP_ROOT = '/home/litisnouman/Desktop/Datasets/MedAI/MedAI_2021_Polyp_Segmentation_Development_Dataset'
    INSTRUMENT_ROOT = '/home/litisnouman/Desktop/Datasets/MedAI/MedAI_2021_Instrument_Segmentation_Development_Dataset'
    
    print("Loading datasets...")
    
    # Load paths
    datasets = {}
    
    if os.path.exists(ANIMAL_ROOT):
        print(f"  Loading Animal from {ANIMAL_ROOT}")
        img_paths, mask_paths, labels = load_animal_paths(ANIMAL_ROOT)
        datasets['Animal'] = (img_paths, mask_paths, labels)
        print(f"    Found {len(img_paths)} samples")
    
    if os.path.exists(ISIC_ROOT):
        print(f"  Loading ISIC from {ISIC_ROOT}")
        img_paths, mask_paths, labels = load_isic_paths(ISIC_ROOT)
        datasets['ISIC'] = (img_paths, mask_paths, labels)
        print(f"    Found {len(img_paths)} samples")
    
    if os.path.exists(POLYP_ROOT):
        print(f"  Loading Polyp from {POLYP_ROOT}")
        img_paths, mask_paths, labels = load_polyp_paths(POLYP_ROOT)
        datasets['Polyp'] = (img_paths, mask_paths, labels)
        print(f"    Found {len(img_paths)} samples")
    
    if os.path.exists(INSTRUMENT_ROOT):
        print(f"  Loading Instrument from {INSTRUMENT_ROOT}")
        img_paths, mask_paths, labels = load_instrument_paths(INSTRUMENT_ROOT)
        datasets['Instrument'] = (img_paths, mask_paths, labels)
        print(f"    Found {len(img_paths)} samples")
    
    print(f"\nVisualizing {len(datasets)} datasets...")
    
    # Visualize each dataset
    for name, (img_paths, mask_paths, labels) in datasets.items():
        print(f"\nVisualizing {name}...")
        
        if name == 'Animal':
            # Visualize both binary and multiclass
            # fig1 = visualize_dataset(name, img_paths, mask_paths, labels, num_samples=3, mode='binary')
            # fig1.savefig(f'animal_binary_samples.png', dpi=150, bbox_inches='tight')
            # print(f"  Saved: animal_binary_samples.png")
            fig1 = visualize_dataset(name, img_paths, mask_paths, labels, num_samples=3, mode='multi')
            fig1.savefig(f'animal_multi_samples.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: animal_multi_samples.png")
        else:
            mode = 'binary' if name != 'ISIC' else 'binary'
            fig = visualize_dataset(name, img_paths, mask_paths, labels, num_samples=3, mode=mode)
            fig.savefig(f'{name.lower()}_samples.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: {name.lower()}_samples.png")
    
    print("\n" + "="*60)
    print("Visualization complete! Check the saved PNG files.")
    print("="*60)
    
    # Show statistics
    print("\nDataset Statistics:")
    print("="*60)
    for name, (img_paths, mask_paths, labels) in datasets.items():
        print(f"\n{name}:")
        print(f"  Total samples: {len(img_paths)}")
        print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # Check image sizes
        sample_img = np.array(Image.open(img_paths[0]))
        print(f"  Image shape: {sample_img.shape}")
        print(f"  Label range: {min(labels)} - {max(labels)}")


if __name__ == '__main__':
    main()
