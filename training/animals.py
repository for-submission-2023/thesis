import matplotlib
matplotlib.use('TkAgg')  # Necessary to run matplotlib

import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
np.bool = np.bool_
from collections import Counter
from sklearn.model_selection import train_test_split

base = '/home/litisnouman/Desktop/Thesis_18_02_26/datasets/Animals'

print(os.listdir(base))

images_paths = [os.path.join(base, 'images', i) for i in os.listdir(os.path.join(base, 'images')) if '.jpg' in i]


TO_REMOVE = ['miniature_pinscher_14.jpg',
            'leonberger_18.jpg',
            'saint_bernard_108.jpg',
            'saint_bernard_78.jpg',
            'Persian_259.jpg',
            'wheaten_terrier_195.jpg',
            'saint_bernard_60.jpg',
            'japanese_chin_199.jpg',
            'Egyptian_Mau_165.jpg',
            'Egyptian_Mau_196.jpg',
            'Egyptian_Mau_20.jpg',
            'saint_bernard_15.jpg',
            'keeshond_7.jpg',
            'Egyptian_Mau_162.jpg',
            'staffordshire_bull_terrier_22.jpg',  # Empty mask
            'Egyptian_Mau_167.jpg']  # Empty mask

TO_REMOVE = [os.path.join(base, 'images', i) for i in TO_REMOVE]

images_paths = [i for i in images_paths if i not in TO_REMOVE]

masks_paths = [i.replace('images', os.path.join('annotations', 'trimaps')).replace('jpg', 'png') for i in images_paths]
print(len(images_paths), len(masks_paths))
print(images_paths[:2])
print(masks_paths[:2])

labels = [1 if i.split(os.path.sep)[-1].split('.')[0][0].isupper() else 2 for i in images_paths]

print(Counter(labels))

train_valid_images, test_images, train_valid_masks, test_masks = train_test_split(images_paths, masks_paths, test_size=0.33, random_state=42, stratify = labels)

labels = [1 if i.split(os.path.sep)[-1].split('.')[0][0].isupper() else 2 for i in train_valid_images]

print(Counter(labels))


train_images, valid_images, train_masks, valid_masks = train_test_split(train_valid_images, train_valid_masks, test_size=0.33, random_state=42, stratify = labels)
print(len(train_images), len(valid_images), len(test_images), len(train_masks), len(valid_masks), len(test_masks))

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

experiment_name = 'animal_binary'
LABEL_DICT = None # {'Cat': 1, 'Dog': 2}

# Dataset parameters
NUM_CLASSES = 3
IMG_SIZE = 224
IN_CHANNELS = 3

# DataLoader parameters
BATCH_SIZE = 8
NUM_WORKERS = 4

# Augmentation parameters
AUG_PROB = 0.5
ROTATION_LIMIT = 180

# Model architecture parameters
UNETR_ViT_DEPTH = 6
UNETR_SAM_DEPTH = 6

import sys
sys.path.insert(0, '/home/litisnouman/Desktop/Thesis_18_02_26')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
import gc

from datasets import AnimalDataset
from models.models import UNET, UNETR_ViT, UNETR_SAM, TransUNet, CustomTorchVisionSegmentation, CustomSMP
from utils.train import train
# ▼▼▼ ADDED: import evaluate function ▼▼▼
from utils.metrics import evaluate
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# Transforms
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=AUG_PROB),
    A.VerticalFlip(p=AUG_PROB),
    A.Rotate(limit=ROTATION_LIMIT, p=AUG_PROB),
])

valid_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

# Return Image Path

return_path = False

# Create datasets
train_dataset = AnimalDataset(train_images, train_masks, mode='binary', transform=train_transform, return_path=return_path)
valid_dataset = AnimalDataset(valid_images, valid_masks, mode='binary', transform=valid_transform, return_path=return_path)

# ▼▼▼ ADDED: model_path, train_model, evaluate_only fields ▼▼▼
models_config = [
    {
        'name': 'UNET',
        'model': UNET(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, depth=4),
        'model_path': None,   # e.g. '/path/to/unet_best.pth' to resume or evaluate
        'train_model': True,  # set False to skip training
        'evaluate_only': False,  # set True to only run evaluation (requires model_path)
    },
    {
        'name': f'UNETR-ViT-D{UNETR_ViT_DEPTH}',
        'model': UNETR_ViT(num_classes=NUM_CLASSES, depth=UNETR_ViT_DEPTH, pretrained=True),
        'model_path': None,
        'train_model': True,
        'evaluate_only': False,
    },
    {
        'name': f'UNETR-SAM-D{UNETR_SAM_DEPTH}',
        'model': UNETR_SAM(num_classes=NUM_CLASSES, depth=UNETR_SAM_DEPTH, pretrained=True),
        'model_path': None,
        'train_model': True,
        'evaluate_only': False,
    },
    {
        'name': 'DeepLabV3-ResNet50',
        'model': CustomTorchVisionSegmentation(model_type='deeplabv3', num_classes=NUM_CLASSES, pretrained=True),
        'model_path': None,
        'train_model': True,
        'evaluate_only': False,
    },
    {
        'name': 'FCN-ResNet50',
        'model': CustomTorchVisionSegmentation(model_type='fcn', num_classes=NUM_CLASSES, pretrained=True),
        'model_path': None,
        'train_model': True,
        'evaluate_only': False,
    },
    {
        'name': 'SMP-Unet-ResNet50',
        'model': CustomSMP(arch='Unet', encoder_name='resnet50', encoder_weights='imagenet', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        'model_path': None,
        'train_model': True,
        'evaluate_only': False,
    },
    {
        'name': 'SMP-Unet-VGG16',
        'model': CustomSMP(arch='Unet', encoder_name='vgg16', encoder_weights='imagenet', num_classes=NUM_CLASSES, in_channels=IN_CHANNELS),
        'model_path': None,
        'train_model': True,
        'evaluate_only': False,
    },
    {
        'name': 'TransUNet',
        'model': TransUNet(img_size=IMG_SIZE, num_classes=NUM_CLASSES),
        'model_path': None,
        'train_model': True,
        'evaluate_only': False,
    },
]
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# Training parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 15
LR = 1e-4
POLY_LR = False

print(f'\n{"="*60}')
print(f'BINARY TRAINING ON {DEVICE}')
print(f'Epochs: {EPOCHS} | LR: {LR} | Batch Size: {BATCH_SIZE}')
print(f'Num Classes: {NUM_CLASSES} | Image Size: {IMG_SIZE} | In Channels: {IN_CHANNELS}')
print(f'Aug Prob: {AUG_PROB} | Rotation: {ROTATION_LIMIT}°')
print(f'UNETR-ViT Depth: {UNETR_ViT_DEPTH} | UNETR-SAM Depth: {UNETR_SAM_DEPTH}')
print(f'{"="*60}\n')

save_dir  = '/home/litisnouman/Desktop/Thesis_18_02_26/saved_models'
plots_dir = '/home/litisnouman/Desktop/Thesis_18_02_26/plots'
os.makedirs(save_dir,  exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

results = {}

for config in models_config:
    model_name    = config['name']
    model         = config['model']
    # ▼▼▼ ADDED: read new config fields ▼▼▼
    model_path    = config.get('model_path', None)
    do_train      = config.get('train_model', True)
    evaluate_only = config.get('evaluate_only', False)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    print(f'\n{"="*60}')
    print(f'Model: {model_name}')
    print(f'{"="*60}')

    # ▼▼▼ ADDED: load weights if model_path is provided ▼▼▼
    if model_path is not None:
        print(f'  -> Loading weights from: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    model.to(DEVICE)

    # ▼▼▼ ADDED: evaluate-only branch ▼▼▼
    if evaluate_only:
        if model_path is None:
            print(f'  -> WARNING: evaluate_only=True but no model_path given, skipping {model_name}')
            continue
        print(f'  -> Evaluate only mode')

        model.eval()
        avg_dice = evaluate(model=model, dataset=valid_dataset, num_classes=NUM_CLASSES, device=DEVICE, label_dict=LABEL_DICT)
        results[model_name] = {'mode': 'evaluate_only', 'best_dice': avg_dice}

        del model
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        continue
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # ▼▼▼ ADDED: skip training if do_train is False ▼▼▼
    if not do_train:
        print(f'  -> Skipping training for {model_name} (train_model=False)')
        del model
        gc.collect()
        continue
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    if 'trans' in model_name.lower():
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE//2, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE//2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ▼▼▼ MODIFIED: print resume message if loading and training ▼▼▼
    if model_path is not None:
        print(f'  -> Resuming training from: {model_path}')
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    train_losses, valid_losses, valid_dice, model_to_return = train(
        model=model,
        optimizer=optimizer,
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_data=valid_dataset,
        epochs=EPOCHS,
        device=DEVICE,
        label_dict=LABEL_DICT,
        poly_lr=POLY_LR
    )

    best_dice  = max(valid_dice)
    best_epoch = valid_dice.index(best_dice) + 1

    model_slug = model_name.lower().replace('-', '_')
    # save_path  = os.path.join(save_dir, f'{experiment_name}_{model_slug}_best.pth')
    save_path = os.path.join(save_dir, f'{experiment_name}_{model_slug}_dice{f"{best_dice:.2f}".replace(".", "_")}_epoch{best_epoch}.pth')
    torch.save(model_to_return.state_dict(), save_path)
    print(f'  -> Model saved to: {save_path}')

    loss_plot_path = os.path.join(plots_dir, f'{experiment_name}_{model_slug}_losses.png')
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} — Train vs Valid Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    dice_plot_path = os.path.join(plots_dir, f'{experiment_name}_{model_slug}_dice.png')
    plt.figure()
    plt.plot(valid_dice, label='Valid Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title(f'{model_name} — Validation Dice')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dice_plot_path)
    plt.close()

    results[model_name] = {
        'best_dice':    best_dice,
        'best_epoch':   best_epoch,
        'save_path':    save_path,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_dice':   valid_dice,
    }

    del model, optimizer, model_to_return
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f'  -> GPU memory cleared. Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB')

# Summary
print(f'\n{"="*60}')
print(f'{experiment_name.upper()} — SUMMARY')
print(f'{"="*60}')
for name, res in results.items():
    if res.get('mode') == 'evaluate_only':
        print(f'{name:25s}: Dice = {res["best_dice"]:.4f} (evaluate only)')
    else:
        print(f'{name:25s}: Best Dice = {res["best_dice"]:.4f} (epoch {res["best_epoch"]})')
print(f'{"="*60}')