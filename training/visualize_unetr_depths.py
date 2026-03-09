#!/usr/bin/env python3
"""
Visualize UNETR Predictions with Different Depths
=================================================
Compare predictions from UNETR models with depths 1, 2, and 3 on FashionMNIST dataset.
Skip depths by passing 'none' or empty string.

Usage:
    python visualize_unetr_depths.py <path_d1> <path_d2> <path_d3> [options]
    
To skip a depth, pass 'none' or empty string (use empty quotes '' in bash).

Examples:
    # All 3 depths
    python visualize_unetr_depths.py d1.pth d2.pth d3.pth
    
    # Skip depth 2 (compare d1 and d3)
    python visualize_unetr_depths.py d1.pth none d3.pth
    
    # Only depth 3
    python visualize_unetr_depths.py none none d3.pth
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

# Add project root to path
sys.path.insert(0, '/home/litisnouman/Desktop/Thesis_18_02_26')

from datasets import FashionMNISTDataset
from models.models import UNETR_ViT

# ============================================================
# CONFIGURATION (Same as triangle_fashionmnist.py)
# ============================================================

labels = [1]  # [1, 2, 3]
not_labels = [1]  # [1, 2, 3, 5, 6, 7]
background_obj = 5
include_label = False
length = 500
jitter = 0

NUM_CLASSES = 2
IMG_SIZE = 224
IN_CHANNELS = 3

def is_valid_path(path):
    """Check if path is valid (not None, not empty, not 'none', and file exists)."""
    if path is None:
        return False
    path = path.strip().lower()
    if path in ('', 'none', 'null', 'skip'):
        return False
    return True

# ============================================================
# MAIN SCRIPT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize UNETR predictions with different depths. Pass "none" to skip a depth.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # All 3 depths
    python visualize_unetr_depths.py d1.pth d2.pth d3.pth
    
    # Skip depth 2
    python visualize_unetr_depths.py d1.pth none d3.pth
    
    # Only depth 1
    python visualize_unetr_depths.py d1.pth none none
        """
    )
    parser.add_argument('d1_path', type=str, nargs='?', default='none',
                        help='Path to UNETR depth 1 checkpoint (.pth), or "none" to skip')
    parser.add_argument('d2_path', type=str, nargs='?', default='none',
                        help='Path to UNETR depth 2 checkpoint (.pth), or "none" to skip')
    parser.add_argument('d3_path', type=str, nargs='?', default='none',
                        help='Path to UNETR depth 3 checkpoint (.pth), or "none" to skip')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize (default: 5)')
    parser.add_argument('--output', type=str, default='unetr_depth_comparison.png', help='Output filename')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    return parser.parse_args()


def load_model(checkpoint_path, depth, device):
    """Load UNETR model with specified depth and weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = UNETR_ViT(
        num_classes=NUM_CLASSES,
        depth=depth,
        pretrained=False,
        direct_upsample=True  # Models trained with direct upsampling
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"  ✓ Loaded depth {depth} model from: {checkpoint_path}")
    return model


def get_predictions(model, image, device):
    """Get model prediction for a single image."""
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        output = model(image_batch)
        pred = output[0].argmax(0).cpu().numpy()
    return pred


def denormalize_image(image_tensor):
    """Convert normalized image tensor to numpy for visualization."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def main():
    args = parse_args()
    
    # Collect paths and determine which depths to show
    all_paths = [args.d1_path, args.d2_path, args.d3_path]
    all_depths = [1, 2, 3]
    
    # Filter to only valid paths
    valid_models = []
    valid_depths = []
    valid_paths = []
    
    for i, (path, depth) in enumerate(zip(all_paths, all_depths), 1):
        if is_valid_path(path):
            if os.path.exists(path):
                valid_models.append(None)  # Placeholder, will load later
                valid_depths.append(depth)
                valid_paths.append(path)
            else:
                print(f"  ⚠ Warning: Depth {depth} checkpoint not found: {path} - skipping")
        else:
            print(f"  ℹ Skipping depth {depth} (no checkpoint provided)")
    
    if len(valid_depths) == 0:
        print("Error: No valid checkpoints provided!")
        sys.exit(1)
    
    # Calculate columns: image + mask + number of valid depths
    num_cols = 2 + len(valid_depths)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Active depths: {valid_depths}")
    print(f"{'='*60}\n")
    
    # ============================================================
    # Load Dataset
    # ============================================================
    print("Loading FashionMNIST dataset...")
    
    valid_dataset_base = datasets.FashionMNIST(root='./data', train=False, download=True)
    
    valid_dataset = FashionMNISTDataset(
        dataset=valid_dataset_base,
        transform=None,
        shape=IMG_SIZE,
        labels=labels,
        not_labels=not_labels,
        background_obj=background_obj,
        include_label=include_label,
        length=length // 2,
        jitter=jitter
    )
    
    print(f"  ✓ Dataset loaded: {len(valid_dataset)} samples")
    
    # ============================================================
    # Load Models
    # ============================================================
    print("\nLoading UNETR models...")
    
    for i, (depth, path) in enumerate(zip(valid_depths, valid_paths)):
        valid_models[i] = load_model(path, depth, device)
    
    # ============================================================
    # Select Random Samples
    # ============================================================
    np.random.seed(args.seed)
    num_samples = min(args.num_samples, len(valid_dataset))
    sample_indices = np.random.choice(len(valid_dataset), num_samples, replace=False)
    
    print(f"\nSelected {num_samples} random samples (seed={args.seed})")
    
    # ============================================================
    # Generate Predictions and Create Visualization
    # ============================================================
    print("\nGenerating predictions and creating visualization...")
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4*num_cols, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Build column titles
    col_titles = ['Original Image', 'Ground Truth']
    for d in valid_depths:
        col_titles.append(f'UNETR D{d}')
    
    for row, idx in enumerate(sample_indices):
        image, mask = valid_dataset[idx]
        
        # Column 0: Original Image
        img_display = denormalize_image(image)
        axes[row, 0].imshow(img_display)
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Column 1: Ground Truth Mask
        mask_display = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        axes[row, 1].imshow(mask_display, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
        if row == 0:
            axes[row, 1].set_title(col_titles[1], fontsize=12, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Remaining columns: Predictions from each loaded model
        for col, (model, depth) in enumerate(zip(valid_models, valid_depths), start=2):
            pred = get_predictions(model, image, device)
            axes[row, col].imshow(pred, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
        
        print(f"  Sample {row+1}/{num_samples} processed")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {args.output}")
    print(f"  Layout: {num_samples} rows x {num_cols} cols (depths {valid_depths})")
    
    # Cleanup
    for model in valid_models:
        del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
