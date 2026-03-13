"""
MiSuRe: Minimally Sufficient Region for Semantic Segmentation
===============================================================

Implementation of MiSuRe explanation method for semantic segmentation.
Learns a minimal mask that preserves the model's prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from medpy.metric.binary import dc
import segmentation_models_pytorch as smp
from tqdm import tqdm

from .xai_metrics import compute_paired_metric

def tv_norm(input, tv_beta):
    """
    Compute total variation norm of a 2D tensor.
    
    Parameters
    ----------
    input : torch.Tensor
        Input tensor with shape (1, 1, H, W).
    tv_beta : float
        Beta parameter for the total variation norm.
    
    Returns
    -------
    torch.Tensor
        Total variation norm.
    """
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def dilate_to_sufficient(image, model, target, device='cpu',
                          dice_threshold=0.9, filter_size=7):
    """
    Dilates the predicted segmentation mask until the model's prediction on the
    background-removed image has DICE >= dice_threshold with the original prediction.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int
        Target class index.
    device : str, default='cpu'
        Device to run the model on.
    dice_threshold : float, default=0.9
        DICE threshold to stop dilation.
    filter_size : int, default=7
        Size of the elliptical structuring element for dilation.

    Returns
    -------
    background_removed : torch.Tensor
        Background removed image tensor with shape (C, H, W).
    counter : int
        Number of dilation iterations performed.
    pred_orig : np.ndarray
        Original predicted segmentation mask with shape (H, W).
    """
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))[0]
    pred_orig = output.argmax(0).detach().cpu().numpy()

    mask_image = 1.0 * (pred_orig == target)
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    counter = 0

    while True:
        background_removed = image.to(device) * torch.tensor(mask_image).float().to(device)
        with torch.no_grad():
            output_bg = model(background_removed.unsqueeze(0).to(device))[0]
        pred_bg = output_bg.argmax(0).detach().cpu().numpy()
        
        # Use medpy's dc (Dice coefficient)
        gt_binary = (pred_orig == target).astype(np.uint8)
        pred_binary = (pred_bg == target).astype(np.uint8)
        dice_score = dc(gt_binary, pred_binary)

        if dice_score >= dice_threshold:
            break
        mask_image = cv2.dilate(mask_image, struct, iterations=1)
        counter += 1

    return background_removed, pred_bg, counter


def misure_1(image, model, target=None, device='cpu',
           lr=0.01, alpha_fg=1, alpha_bg=1, alpha_ce=0.0, l1_coeff=0.1,
           tv_coeff=0, tv_beta=3,
           iterations=100, remove_background=True, mask_init=True,
           value=0.5, mask_size=224, dice_threshold=0.9,
           filter_size=7, class_conditioning=True,
           sweep_thresholds=(0.0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
           sweep_tolerance=0.2,
           # ---- new parameters ----
           min_dice_threshold=0.5,      # fallback if final dice < this
           patience=3,               # stop early if dice below patience_dice_threshold for N consecutive checkpoints
           patience_dice_threshold=0.3):  # dice must stay below this to count toward patience

    # ------------------------------------------------------------------ #
    #  1. INITIAL FORWARD PASS                                             #
    # ------------------------------------------------------------------ #

    image = image.to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))[0]

    pred_orig = output.argmax(0).detach().cpu().numpy()  # [H, W]

    # ------------------------------------------------------------------ #
    #  2. TARGET CLASS RESOLUTION                                          #
    # ------------------------------------------------------------------ #

    if target is None:
        target = pred_orig.max().item()

    if target == 0:
        return None

    # ------------------------------------------------------------------ #
    #  3. BUILD THE MASKING TARGET                                         #
    # ------------------------------------------------------------------ #

    masking_target = torch.tensor(
        target * (pred_orig == target), dtype=torch.long
    ).unsqueeze(0).to(device)

    # ------------------------------------------------------------------ #
    #  4. BACKGROUND REMOVAL VIA ITERATIVE DILATION                       #
    # ------------------------------------------------------------------ #

    if remove_background:
        image, pred_bg, counter = dilate_to_sufficient(
            image, model, target, device, dice_threshold, filter_size
        )
    else:
        counter = 0

    # ------------------------------------------------------------------ #
    #  5. ASPECT-RATIO-AWARE MASK SIZE                                     #
    #     Shorter dim → mask_size; longer dim → scaled proportionally.    #
    # ------------------------------------------------------------------ #

    img_H, img_W = image.shape[-2], image.shape[-1]

    if img_H <= img_W:                      # height is the shorter dimension
        mask_H = mask_size
        mask_W = int(round(mask_size * img_W / img_H))
    else:                                   # width is the shorter dimension
        mask_W = mask_size
        mask_H = int(round(mask_size * img_H / img_W))

    # ------------------------------------------------------------------ #
    #  6. MASK INITIALISATION                                              #
    # ------------------------------------------------------------------ #

    mask = torch.ones((mask_H, mask_W), dtype=torch.float32).to(device)

    if mask_init:
        prediction_mask = cv2.resize(pred_orig, (mask_W, mask_H),
                                      interpolation=cv2.INTER_NEAREST)
        prediction_mask = torch.tensor(prediction_mask, dtype=torch.float32)
        mask[prediction_mask != target] = value

    # ------------------------------------------------------------------ #
    #  7. CONSTRAIN MASK TO NON-BACKGROUND PIXELS                         #
    # ------------------------------------------------------------------ #

    dil_mask = cv2.resize(
        image[0].detach().cpu().numpy(), (mask_W, mask_H),
        interpolation=cv2.INTER_NEAREST
    )
    dil_mask = torch.tensor(dil_mask, dtype=torch.float32).to(device) > 0
    mask = mask * dil_mask

    # ------------------------------------------------------------------ #
    #  8. PREPARE MASK AS OPTIMISABLE PARAMETER                           #
    # ------------------------------------------------------------------ #

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(mask.clone().detach(), requires_grad=True)
    optimizer = torch.optim.AdamW([mask], lr=lr)

    # ------------------------------------------------------------------ #
    #  10. LOSS FUNCTION SETUP                                             #
    # ------------------------------------------------------------------ #

    if class_conditioning:
        dice_loss_fg = smp.losses.DiceLoss(mode='multiclass', classes=[target])
        dice_loss_bg = smp.losses.DiceLoss(mode='multiclass', classes=[0, target])

        if alpha_ce > 0:
            num_classes = output.shape[0]
            ce_weight = torch.zeros(num_classes, device=device)
            ce_weight[target] = 1.0
            ce_loss_fn = torch.nn.CrossEntropyLoss(weight=ce_weight)
        else:
            ce_loss_fn = None
    else:
        dice_loss_fg = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
        dice_loss_bg = smp.losses.DiceLoss(mode='multiclass')
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0) if alpha_ce > 0 else None

    # ------------------------------------------------------------------ #
    #  11. FOREGROUND-SCALED L1 COEFFICIENT                               #
    # ------------------------------------------------------------------ #

    num_target_pixels = (masking_target == target).sum().item()
    H, W = image.shape[-2], image.shape[-1]
    fg_fraction = num_target_pixels / (H * W)
    l1_coeff = l1_coeff   * fg_fraction

    # ------------------------------------------------------------------ #
    #  12. PERIODIC EVALUATION SETUP                                       #
    #      Checkpoints at 0%, 10%, 20%, ..., 100% of iterations.          #
    # ------------------------------------------------------------------ #

    gt_bin = (pred_orig == target).astype(np.uint8)

    # Checkpoints: every 10% of total iterations, starting from 0
    checkpoint_iters = sorted(set(
        [0] + [int(round(p * (iterations - 1))) for p in
               [i / 10 for i in range(1, 11)]]
    ))

    last_sufficient_cam  = None   # most recent checkpoint with dice >= min_dice_threshold
    no_improve_count     = 0     # consecutive checkpoints with dice < patience_dice_threshold
    stop_early           = False

    # ------------------------------------------------------------------ #
    #  13. OPTIMISATION LOOP                                               #
    # ------------------------------------------------------------------ #

    print(f"\n{'Iter':<8} {'Dice':<10} {'SR':<10} {'Note'}")
    print("-" * 40)

    for iteration in tqdm(range(iterations), desc='MiSuRe optimization'):

        # ---------- periodic evaluation (before gradient step) ----------
        if iteration in checkpoint_iters:
            with torch.no_grad():
                m_s = mask.detach()
                up_chk = F.interpolate(m_s, size=(img_H, img_W), mode='bilinear', align_corners=False)
                cam_chk = up_chk[0, 0].detach().cpu().numpy()
            dice_chk, sr_chk = compute_paired_metric(image, model, cam_chk, target, threshold=0, device=device)
            note = ""

            if dice_chk >= min_dice_threshold:
                last_sufficient_cam = cam_chk.copy()
                note = "← saved (above threshold)"
            else:
                note = ""

            if dice_chk < patience_dice_threshold:
                no_improve_count += 1
                note += f" below-threshold {no_improve_count}/{patience}"
            else:
                no_improve_count = 0

            print(f"{iteration:<8} {dice_chk:<10.4f} {sr_chk:<10.4f} {note}")

            # early stopping check (skip at iteration 0 — not meaningful)
            if iteration > 0 and no_improve_count >= patience:
                print(f"\nEarly stopping at iteration {iteration} "
                      f"(Dice below {patience_dice_threshold} for {patience} consecutive checkpoints).")
                stop_early = True
                break

        # ---------- forward pass & loss --------------------------------

        if mask.shape[-2:] != image.shape[-2:]:
            upsampled_mask = F.interpolate(
                mask, size=(img_H, img_W),
                mode='bilinear', align_corners=False
            )
        else:
            upsampled_mask = mask

        upsampled_mask_expanded = upsampled_mask.expand(
            1, image.shape[0], upsampled_mask.shape[2], upsampled_mask.shape[3]
        )

        perturbated_input    = image * upsampled_mask_expanded
        prediction_perturbed = model(perturbated_input)

        loss_ce = (alpha_ce * ce_loss_fn(prediction_perturbed, masking_target)
                   if alpha_ce > 0 else 0)

        loss_l1 = torch.mean(torch.abs(upsampled_mask))
        loss_tv = tv_norm(upsampled_mask, tv_beta) if tv_coeff != 0 else 0

        loss_output = (
            alpha_fg * dice_loss_fg(prediction_perturbed, masking_target) +
            alpha_bg * dice_loss_bg(prediction_perturbed, masking_target) +
            loss_ce
        )

        perturbation_loss = l1_coeff * loss_l1 + tv_coeff * loss_tv + loss_output

        optimizer.zero_grad()
        perturbation_loss.backward()
        optimizer.step()

        if iteration < iterations - 1:
            mask.data.clamp_(0, 1)

    # ------------------------------------------------------------------ #
    #  15. FINAL EVALUATION & BEST-PERFORMER FALLBACK                     #
    # ------------------------------------------------------------------ #

    # Evaluate the optimizer's final result
    with torch.no_grad():
        m_s = mask.detach()
        up_final = F.interpolate(m_s, size=(img_H, img_W), mode='bilinear', align_corners=False)
        final_cam = up_final[0, 0].detach().cpu().numpy()
    final_dice, final_sr = compute_paired_metric(image, model, final_cam, target, threshold=0, device=device)
    print(f"\nFinal optimizer  →  Dice: {final_dice:.4f}  SR: {final_sr:.4f}")

    if final_dice >= min_dice_threshold:
        cam = final_cam
        print(f"Using final optimizer result (Dice {final_dice:.4f} ≥ threshold {min_dice_threshold}).")
    else:
        cam = last_sufficient_cam if last_sufficient_cam is not None else final_cam
        source = "last sufficient checkpoint" if last_sufficient_cam is not None else "final (no sufficient checkpoint found)"
        print(f"Final Dice {final_dice:.4f} < threshold {min_dice_threshold}. "
              f"Falling back to {source}.")

    # ------------------------------------------------------------------ #
    #  16. POST-OPTIMIZATION THRESHOLD SWEEP                              #
    # ------------------------------------------------------------------ #

    best_cam       = cam.copy()
    best_threshold = 0.0
    raw_dice       = None

    print(f"\n{'Threshold':<12} {'Dice':<8}")
    print("-" * 20)

    for t in sorted(sweep_thresholds):

        cam_ = cam.copy()
        cam_[cam_ < t] = 0
        dice_t, sr_t = compute_paired_metric(
            image, model, cam_, target, threshold=0, device=device
        )

        print(f"{t:<12} {dice_t:.4f} {sr_t}")

        if raw_dice is None:
            raw_dice = dice_t

        if dice_t >= raw_dice - sweep_tolerance:
            best_threshold = t
            best_cam       = cam.copy()
            best_cam[cam < t] = 0

    print(f"\nBest threshold : {best_threshold}")
    print(f"Raw Dice       : {raw_dice:.4f}")
    print(f"Floor          : {raw_dice - sweep_tolerance:.4f}")

    return best_cam, counter

