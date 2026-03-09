"""
Training Utilities
==================
Main training loop with early stopping based on Dice score
"""

import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from .metrics import evaluate
from .metrics_synapse import evaluate_synapse


def train_synapse(model, optimizer, num_classes, train_loader=None, valid_loader=None,
                  valid_data=None, epochs=100, device='cpu', label_dict=None, poly_lr=False, transform=None):
    train_losses, valid_losses = [], []
    valid_dice = []

    max_dice = 0
    model_to_return = None  # initialised here to avoid NameError if no epoch improves Dice

    # Poly LR setup
    base_lr        = optimizer.param_groups[0]['lr']
    max_iterations = epochs * len(train_loader)
    iter_num       = 0

    # Combined loss: CrossEntropy for per-pixel classification +
    # Dice loss for segmentation overlap, background ignored in Dice
    loss_1 = nn.CrossEntropyLoss()
    loss_2 = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(1, epochs + 1)):

        # ── Training ──────────────────────────────────────────────────────────
        train_epoch_loss = []
        model.train()
        for batch in tqdm(train_loader):
            i, j = batch[0].to(device), batch[1].to(device)
            with torch.cuda.amp.autocast():
                out  = model(i)
                loss = loss_1(out, j) + loss_2(out, j)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if poly_lr:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                iter_num += 1

            train_epoch_loss.append(loss.item())

        train_losses.append(np.mean(train_epoch_loss))

        # ── Validation loss ───────────────────────────────────────────────────
        valid_epoch_loss = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                i, j = batch[0].to(device), batch[1].to(device)
                with torch.cuda.amp.autocast():
                    out  = model(i)
                    loss = loss_1(out, j) + loss_2(out, j)
                valid_epoch_loss.append(loss.item())

        valid_losses.append(np.mean(valid_epoch_loss))

        # ── Validation Dice ───────────────────────────────────────────────────
        print(f"\n── Epoch {epoch}/{epochs} ──────────────────────────────")
        print(f"  Train Loss: {np.mean(train_epoch_loss):.4f}")
        print(f"  Valid Loss: {np.mean(valid_epoch_loss):.4f}")
        print(f"  Validation Dice:")

        # evaluate_synapse() handles its own no_grad internally
        mean_dice = evaluate_synapse(model=model, image_paths=valid_data, num_classes=num_classes, transform=transform, device=device, label_dict=label_dict)
        valid_dice.append(mean_dice)

        # Save a deep copy of the best model so training can continue without
        # interruption while still preserving the best checkpoint
        if mean_dice > max_dice:
            max_dice        = mean_dice
            model_to_return = copy.deepcopy(model)
            print(f"  ✓ New best model saved (Dice: {max_dice:.4f})")

    print(f"\nTraining complete. Best Dice: {max_dice:.4f}")
    return train_losses, valid_losses, valid_dice, model_to_return

