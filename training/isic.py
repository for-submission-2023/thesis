import matplotlib
matplotlib.use('TkAgg')

import os
import gc
import sys
import matplotlib.pyplot as plt
import numpy as np
np.bool = np.bool_
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, '/home/litisnouman/Desktop/Thesis_18_02_26')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A

from datasets import ISICDataset
from models.models import UNET, UNETR_ViT, UNETR_SAM, TransUNet, CustomTorchVisionSegmentation, CustomSMP
from utils.train import train
from utils.metrics import evaluate


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

EXPERIMENT_NAME  = 'isic'

# Dataset
NUM_CLASSES      = 2
IMG_SIZE         = 224
IN_CHANNELS      = 3

# DataLoader
BATCH_SIZE       = 8
NUM_WORKERS      = 4

# Training
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS           = 15
LR               = 1e-4
POLY_LR          = False

LABEL_DICT       = None

SAVE_DIR         = '/home/litisnouman/Desktop/Thesis_18_02_26/saved_models'
PLOTS_DIR        = '/home/litisnouman/Desktop/Thesis_18_02_26/plots'

# Augmentation
AUG_PROB         = 0.5
ROTATION_LIMIT   = 180

# Model architecture depth defaults
UNETR_ViT_DEPTH  = 6
UNETR_SAM_DEPTH  = 6

# ==============================================================================
# DATASET PATHS
# ==============================================================================

ROOT              = '/home/litisnouman/Desktop/Thesis_18_02_26/datasets/ISIC'
images_paths_root = os.path.join(ROOT, 'images')
images_paths_all  = [os.path.join(images_paths_root, i) for i in os.listdir(images_paths_root) if '.jpg' in i]
masks_paths_all   = [i.replace('images', 'masks').replace('.jpg', '_segmentation.png') for i in images_paths_all]

# Filter: keep only images where a mask corner pixel is foreground
images_paths, masks_paths = [], []
for i, j in zip(images_paths_all, masks_paths_all):
    mask = np.array(Image.open(j))
    images_paths.append(i)
    masks_paths.append(j)

train_valid_images, test_images,  train_valid_masks, test_masks  = train_test_split(images_paths, masks_paths, test_size=0.33, random_state=42)
train_images,       valid_images, train_masks,       valid_masks = train_test_split(train_valid_images, train_valid_masks, test_size=0.33, random_state=42)
print(f'Train: {len(train_images)} | Valid: {len(valid_images)} | Test: {len(test_images)}')

# ==============================================================================
# TRANSFORMS
# ==============================================================================

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=AUG_PROB),
    A.VerticalFlip(p=AUG_PROB),
    A.Rotate(limit=ROTATION_LIMIT, p=AUG_PROB),
])

valid_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

# ==============================================================================
# DATASETS
# ==============================================================================

train_dataset = ISICDataset(train_images, train_masks, transform=train_transform, return_path=False)
valid_dataset = ISICDataset(valid_images, valid_masks, transform=valid_transform, return_path=False)

# ==============================================================================
# MODELS CONFIG
# ==============================================================================
# Each entry:
#   name          : display name used in logs, plots, and save filenames
#   model         : instantiated nn.Module (all args explicit for reproducibility)
#   model_path    : path to a .pth file to load before training/evaluation,
#                   or None to start from scratch
#   train_model   : if False, skips training entirely (useful with evaluate_only)
#   evaluate_only : if True, only runs evaluation on valid_dataset (requires model_path)
# ==============================================================================

SAVED = '/home/litisnouman/Desktop/Thesis_18_02_26/saved_models'

models_config = [

    # ── Vanilla UNet ────────────────────────────────────────────────────────
    {
        'name':          'UNET',
        'model':         UNET(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, depth=4),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },

    # ── UNETR-ViT ───────────────────────────────────────────────────────────
    {
        'name':          f'UNETR-ViT-D{UNETR_ViT_DEPTH}',
        'model':         UNETR_ViT(num_classes=NUM_CLASSES, depth=UNETR_ViT_DEPTH, pretrained=True),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },

    # ── UNETR-SAM ───────────────────────────────────────────────────────────
    {
        'name':          f'UNETR-SAM-D{UNETR_SAM_DEPTH}',
        'model':         UNETR_SAM(num_classes=NUM_CLASSES, depth=UNETR_SAM_DEPTH, pretrained=True),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },

    # ── DeepLabV3 / FCN ─────────────────────────────────────────────────────
    {
        'name':          'DeepLabV3-ResNet50',
        'model':         CustomTorchVisionSegmentation(model_type='deeplabv3', num_classes=NUM_CLASSES, pretrained=True),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },
    {
        'name':          'FCN-ResNet50',
        'model':         CustomTorchVisionSegmentation(model_type='fcn', num_classes=NUM_CLASSES, pretrained=True),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },

    # ── SMP ─────────────────────────────────────────────────────────────────
    {
        'name':          'SMP-Unet-ResNet50',
        'model':         CustomSMP(arch='Unet', encoder_name='resnet50', encoder_weights='imagenet', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },
    {
        'name':          'SMP-Unet-VGG16',
        'model':         CustomSMP(arch='Unet', encoder_name='vgg16', encoder_weights='imagenet', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },

    # ── TransUNet ───────────────────────────────────────────────────────────
    {
        'name':          'TransUNet',
        'model':         TransUNet(img_size=IMG_SIZE, num_classes=NUM_CLASSES),
        'model_path':    None,
        'train_model':   True,
        'evaluate_only': False,
    },
]

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f'\n{"="*60}')
print(f'EXPERIMENT : {EXPERIMENT_NAME}')
print(f'DEVICE     : {DEVICE}')
print(f'Epochs: {EPOCHS} | LR: {LR} | Batch: {BATCH_SIZE}')
print(f'Classes: {NUM_CLASSES} | Image: {IMG_SIZE}x{IMG_SIZE} | Channels: {IN_CHANNELS}')
print(f'{"="*60}\n')

results = {}

for cfg in models_config:
    model_name    = cfg['name']
    model         = cfg['model']
    model_path    = cfg.get('model_path',    None)
    do_train      = cfg.get('train_model',   True)
    evaluate_only = cfg.get('evaluate_only', False)

    print(f'\n{"="*60}')
    print(f'  {model_name}')
    print(f'{"="*60}')

    # Load weights if provided
    if model_path is not None:
        print(f'  -> Loading weights: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.to(DEVICE)

    # Evaluate-only branch
    if evaluate_only:
        if model_path is None:
            print(f'  -> WARNING: evaluate_only=True but model_path is None — skipping.')
            continue
        print(f'  -> Evaluate only')
        model.eval()
        avg_dice = evaluate(model=model, dataset=valid_dataset, num_classes=NUM_CLASSES, device=DEVICE, label_dict=LABEL_DICT)
        results[model_name] = {'mode': 'evaluate_only', 'best_dice': avg_dice}
        del model
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        continue

    # Skip training if flagged
    if not do_train:
        print(f'  -> Skipping training (train_model=False)')
        del model
        gc.collect()
        continue

    # DataLoaders (halve batch for TransUNet due to memory)
    bs = BATCH_SIZE // 2 if 'trans' in model_name.lower() else BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    if model_path is not None:
        print(f'  -> Resuming training from: {model_path}')

    train_losses, valid_losses, valid_dice, best_model = train(
        model=model,
        optimizer=optimizer,
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_data=valid_dataset,
        epochs=EPOCHS,
        device=DEVICE,
        label_dict=LABEL_DICT,
        poly_lr=POLY_LR,
    )

    best_dice  = max(valid_dice)
    best_epoch = valid_dice.index(best_dice) + 1
    model_slug = model_name.lower().replace('-', '_')
    dice_str   = f"{best_dice:.2f}".replace('.', '_')
    save_path  = os.path.join(SAVE_DIR, f'{EXPERIMENT_NAME}_{model_slug}_dice{dice_str}_epoch{best_epoch}.pth')
    torch.save(best_model.state_dict(), save_path)
    print(f'  -> Saved: {save_path}')

    # Loss plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{model_name} — Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{EXPERIMENT_NAME}_{model_slug}_losses.png'))
    plt.close()

    # Dice plot
    plt.figure()
    plt.plot(valid_dice, label='Valid Dice')
    plt.xlabel('Epoch'); plt.ylabel('Dice')
    plt.title(f'{model_name} — Dice')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{EXPERIMENT_NAME}_{model_slug}_dice.png'))
    plt.close()

    results[model_name] = {
        'best_dice':    best_dice,
        'best_epoch':   best_epoch,
        'save_path':    save_path,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_dice':   valid_dice,
    }

    del model, optimizer, best_model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f'  -> GPU cleared. Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')

# ==============================================================================
# SUMMARY
# ==============================================================================

print(f'\n{"="*60}')
print(f'{EXPERIMENT_NAME.upper()} — SUMMARY')
print(f'{"="*60}')
for name, res in results.items():
    if res.get('mode') == 'evaluate_only':
        print(f'  {name:35s}: Dice = {res["best_dice"]:.4f}  [eval only]')
    else:
        print(f'  {name:35s}: Dice = {res["best_dice"]:.4f}  (epoch {res["best_epoch"]})')
print(f'{"="*60}')
