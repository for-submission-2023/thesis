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


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def dilate_to_sufficient(image, model, target, device='cpu',
                          dice_threshold=0.9, filter_size=7):
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))[0]
    pred_orig = output.argmax(0).detach().cpu().numpy()

    mask_image = 1.0 * (pred_orig == target)
    struct  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    counter = 0

    while True:
        background_removed = image.to(device) * torch.tensor(mask_image).float().to(device)
        with torch.no_grad():
            output_bg = model(background_removed.unsqueeze(0).to(device))[0]
        pred_bg = output_bg.argmax(0).detach().cpu().numpy()

        gt_binary   = (pred_orig == target).astype(np.uint8)
        pred_binary = (pred_bg   == target).astype(np.uint8)
        dice_score  = dc(gt_binary, pred_binary)

        if dice_score >= dice_threshold:
            break
        mask_image = cv2.dilate(mask_image, struct, iterations=1)
        counter += 1

    return background_removed, pred_bg, counter




def misure(image, model, target=None, device='cpu',
           lr=0.01, alpha_fg=1, alpha_bg=1, alpha_ce=0.0, l1_coeff=0.1,
           tv_coeff=0, tv_beta=3, gaussian_sigma=None,
           iterations=100, remove_background=True, mask_init=True,
           value=0.5, mask_size=224, softmax=True, dice_threshold=0.9,
           filter_size=7, class_conditioning=True,
           sweep_thresholds=(0.0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
           sweep_tolerance=0.1):

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
    #  5. MASK INITIALISATION                                              #
    # ------------------------------------------------------------------ #

    mask = torch.ones((mask_size, mask_size), dtype=torch.float32).to(device)

    if mask_init:
        prediction_mask = cv2.resize(pred_orig, (mask_size, mask_size),
                                      interpolation=cv2.INTER_NEAREST)
        prediction_mask = torch.tensor(prediction_mask, dtype=torch.float32)
        mask[prediction_mask != target] = value

    # ------------------------------------------------------------------ #
    #  6. CONSTRAIN MASK TO NON-BACKGROUND PIXELS                         #
    # ------------------------------------------------------------------ #

    dil_mask = cv2.resize(image[0].detach().cpu().numpy(), (mask_size, mask_size),
                           interpolation=cv2.INTER_NEAREST)
    dil_mask = torch.tensor(dil_mask, dtype=torch.float32).to(device) > 0
    mask = mask * dil_mask

    # ------------------------------------------------------------------ #
    #  7. PREPARE MASK AS OPTIMISABLE PARAMETER                           #
    # ------------------------------------------------------------------ #

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(mask.clone().detach(), requires_grad=True)
    optimizer = torch.optim.AdamW([mask], lr=lr)

    # ------------------------------------------------------------------ #
    #  8. OPTIONAL GAUSSIAN BLUR KERNEL                                    #
    # ------------------------------------------------------------------ #

    if gaussian_sigma is not None:
        kernel_size = int(6 * gaussian_sigma + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=gaussian_sigma)

    softmax_fn = torch.nn.Softmax(dim=1)

    # ------------------------------------------------------------------ #
    #  9. LOSS FUNCTION SETUP                                              #
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
    #  10. FOREGROUND-SCALED L1 COEFFICIENT                               #
    # ------------------------------------------------------------------ #

    num_target_pixels = (masking_target == target).sum().item()
    H, W = image.shape[-2], image.shape[-1]
    fg_fraction = num_target_pixels / (H * W)
    l1_coeff = l1_coeff#  * fg_fraction

    # ------------------------------------------------------------------ #
    #  11. OPTIMISATION LOOP (no clamping)                                 #
    # ------------------------------------------------------------------ #

    for iteration in tqdm(range(iterations), desc='MiSuRe optimization'):

        mask_smooth = gaussian_blur(mask) if gaussian_sigma is not None else mask

        if mask_smooth.shape[-2:] != image.shape[-2:]:
            upsampled_mask = F.interpolate(
                mask_smooth, size=(image.shape[-2], image.shape[-1]),
                mode='bilinear', align_corners=False
            )
        else:
            upsampled_mask = mask_smooth

        upsampled_mask_expanded = upsampled_mask.expand(
            1, image.shape[0], upsampled_mask.shape[2], upsampled_mask.shape[3]
        )

        perturbated_input = image * upsampled_mask_expanded
        prediction_perturbed = model(perturbated_input)

        loss_ce = alpha_ce * ce_loss_fn(prediction_perturbed, masking_target) if alpha_ce > 0 else 0

        if softmax:
            prediction_perturbed = softmax_fn(prediction_perturbed)

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
    #  12. POST-OPTIMIZATION THRESHOLD SWEEP                              #
    # ------------------------------------------------------------------ #

    cam = upsampled_mask[0][0].detach().cpu().numpy()  # [H, W]

    # Binary ground-truth for Dice: pixels where original prediction == target.
    gt_bin = (pred_orig == target).astype(np.uint8)

    best_cam = cam.copy()
    best_threshold = 0.0
    raw_dice = None

    print(f"\n{'Threshold':<12} {'Dice':<8}")
    print("-" * 20)

    for t in sorted(sweep_thresholds):

        # Zero image pixels where cam < t, keep the rest unchanged.
        img_t = image.clone()
        img_t[:, cam < t] = 0

        with torch.no_grad():
            pred_t = model(img_t.unsqueeze(0))[0].argmax(0)

        pred_bin = (pred_t.detach().cpu().numpy() == target).astype(np.uint8)

        if gt_bin.sum() == 0 or pred_bin.sum() == 0:
            dice = 0.0
        else:
            dice = float(dc(gt_bin, pred_bin))
        cam_ = cam.copy()
        cam_[cam_ < t] = 0
        dice, sr = compute_paired_metric(image, model, cam_, target, threshold=0, device=device)
        
        print(f"{t:<12} {dice:.4f} {sr}")

        # t=0.0 is the raw unthresholded cam — record as reference.
        if raw_dice is None:
            raw_dice = dice
        # Keep the highest threshold whose Dice stays within tolerance.
        if dice >= raw_dice - sweep_tolerance:
            best_threshold = t
            best_cam = cam.copy()
            best_cam[cam < t] = 0

    print(f"\nBest threshold : {best_threshold}")
    print(f"Raw Dice       : {raw_dice:.4f}")
    print(f"Floor          : {raw_dice - sweep_tolerance:.4f}")

    return best_cam, counter

def misure_1(image, model, target=None, device='cpu',
           lr=0.01, alpha_fg=1, alpha_bg=1, alpha_ce=0.0, l1_coeff=0.1,
           tv_coeff=0, tv_beta=3, gaussian_sigma=None,
           iterations=100, remove_background=True, mask_init=True,
           value=0.5, mask_size=224, softmax=True, dice_threshold=0.9,
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
    #  9. OPTIONAL GAUSSIAN BLUR KERNEL                                    #
    # ------------------------------------------------------------------ #

    if gaussian_sigma is not None:
        kernel_size = int(6 * gaussian_sigma + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=gaussian_sigma)

    softmax_fn = torch.nn.Softmax(dim=1)

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
                m_s = gaussian_blur(mask.detach()) if gaussian_sigma is not None else mask.detach()
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
        mask_smooth = gaussian_blur(mask) if gaussian_sigma is not None else mask

        if mask_smooth.shape[-2:] != image.shape[-2:]:
            upsampled_mask = F.interpolate(
                mask_smooth, size=(img_H, img_W),
                mode='bilinear', align_corners=False
            )
        else:
            upsampled_mask = mask_smooth

        upsampled_mask_expanded = upsampled_mask.expand(
            1, image.shape[0], upsampled_mask.shape[2], upsampled_mask.shape[3]
        )

        perturbated_input    = image * upsampled_mask_expanded
        prediction_perturbed = model(perturbated_input)

        loss_ce = (alpha_ce * ce_loss_fn(prediction_perturbed, masking_target)
                   if alpha_ce > 0 else 0)

        if softmax:
            prediction_perturbed = softmax_fn(prediction_perturbed)

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
        m_s = gaussian_blur(mask.detach()) if gaussian_sigma is not None else mask.detach()
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


'''def misure(image, model, target=None, device='cpu',
           lr=0.01, alpha_fg=1, alpha_bg=1, alpha_ce=0.0, l1_coeff=0.1,
           tv_coeff=0, tv_beta=3, gaussian_sigma=None,
           iterations=100, remove_background=True, mask_init=True,
           value=0.5, mask_size=224, softmax=True, dice_threshold=0.9,
           filter_size=7, class_conditioning=True,
           sweep_thresholds=(0.0, 0.0001, 0.001, 0.01, 0.1),
           sweep_tolerance=0.1):

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
    #  5. MASK INITIALISATION                                              #
    # ------------------------------------------------------------------ #

    mask = torch.ones((mask_size, mask_size), dtype=torch.float32).to(device)

    if mask_init:
        prediction_mask = cv2.resize(pred_orig, (mask_size, mask_size),
                                      interpolation=cv2.INTER_NEAREST)
        prediction_mask = torch.tensor(prediction_mask, dtype=torch.float32)
        mask[prediction_mask != target] = value

    # ------------------------------------------------------------------ #
    #  6. CONSTRAIN MASK TO NON-BACKGROUND PIXELS                         #
    # ------------------------------------------------------------------ #

    dil_mask = cv2.resize(image[0].detach().cpu().numpy(), (mask_size, mask_size),
                           interpolation=cv2.INTER_NEAREST)
    dil_mask = torch.tensor(dil_mask, dtype=torch.float32).to(device) > 0
    mask = mask * dil_mask

    # ------------------------------------------------------------------ #
    #  7. PREPARE MASK AS OPTIMISABLE PARAMETER                           #
    # ------------------------------------------------------------------ #

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(mask.clone().detach(), requires_grad=True)
    optimizer = torch.optim.AdamW([mask], lr=lr)

    # ------------------------------------------------------------------ #
    #  8. OPTIONAL GAUSSIAN BLUR KERNEL                                    #
    # ------------------------------------------------------------------ #

    if gaussian_sigma is not None:
        kernel_size = int(6 * gaussian_sigma + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=gaussian_sigma)

    softmax_fn = torch.nn.Softmax(dim=1)

    # ------------------------------------------------------------------ #
    #  9. LOSS FUNCTION SETUP                                              #
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
    #  10. FOREGROUND-SCALED L1 COEFFICIENT                               #
    # ------------------------------------------------------------------ #

    num_target_pixels = (masking_target == target).sum().item()
    H, W = image.shape[-2], image.shape[-1]
    fg_fraction = num_target_pixels / (H * W)
    l1_coeff = l1_coeff#  * fg_fraction

    # ------------------------------------------------------------------ #
    #  11. OPTIMISATION LOOP (no clamping)                                 #
    # ------------------------------------------------------------------ #

    for iteration in tqdm(range(iterations), desc='MiSuRe optimization'):

        mask_smooth = gaussian_blur(mask) if gaussian_sigma is not None else mask

        if mask_smooth.shape[-2:] != image.shape[-2:]:
            upsampled_mask = F.interpolate(
                mask_smooth, size=(image.shape[-2], image.shape[-1]),
                mode='bilinear', align_corners=False
            )
        else:
            upsampled_mask = mask_smooth

        upsampled_mask_expanded = upsampled_mask.expand(
            1, image.shape[0], upsampled_mask.shape[2], upsampled_mask.shape[3]
        )

        perturbated_input = image * upsampled_mask_expanded
        prediction_perturbed = model(perturbated_input)

        loss_ce = alpha_ce * ce_loss_fn(prediction_perturbed, masking_target) if alpha_ce > 0 else 0

        if softmax:
            prediction_perturbed = softmax_fn(prediction_perturbed)

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

        mask.data.clamp_(0, 1)

    # ------------------------------------------------------------------ #
    #  12. POST-OPTIMIZATION THRESHOLD SWEEP                              #
    # ------------------------------------------------------------------ #

    cam = upsampled_mask[0][0].detach().cpu().numpy()  # [H, W]

    # Binary ground-truth for Dice: pixels where original prediction == target.
    gt_bin = (pred_orig == target).astype(np.uint8)

    best_cam = cam.copy()
    best_threshold = 0.0
    raw_dice = None

    print(f"\n{'Threshold':<12} {'Dice':<8}")
    print("-" * 20)

    for t in sorted(sweep_thresholds):

        # Zero image pixels where cam < t, keep the rest unchanged.
        img_t = image.clone()
        img_t[:, cam < t] = 0

        with torch.no_grad():
            pred_t = model(img_t.unsqueeze(0))[0].argmax(0)

        pred_bin = (pred_t.detach().cpu().numpy() == target).astype(np.uint8)

        if gt_bin.sum() == 0 or pred_bin.sum() == 0:
            dice = 0.0
        else:
            dice = float(dc(gt_bin, pred_bin))
        cam_ = cam.copy()
        cam_[cam_ < t] = 0
        dice, sr = compute_paired_metric(image, model, cam_, target, threshold=0, device=device)
        
        print(f"{t:<12} {dice:.4f} {sr}")

        # t=0.0 is the raw unthresholded cam — record as reference.
        if raw_dice is None:
            raw_dice = dice

        # Keep the highest threshold whose Dice stays within tolerance.
        if dice >= raw_dice - sweep_tolerance:
            best_threshold = t
            best_cam = cam.copy()
            best_cam[cam < t] = 0

    print(f"\nBest threshold : {best_threshold}")
    print(f"Raw Dice       : {raw_dice:.4f}")
    print(f"Floor          : {raw_dice - sweep_tolerance:.4f}")

    return best_cam, counter

def misure_new_old(image, model, target=None, device='cpu',
           lr=0.01, alpha_fg=1, alpha_bg=1, alpha_ce=0.0, l1_coeff=0.1,
           tv_coeff=0, tv_beta=3, gaussian_sigma=None,
           iterations=100, remove_background=True, mask_init=True,
           value=0.5, mask_size=224, softmax=True, clamp_iteration=10,
           clamp_threshold=0.2, dice_threshold=0.9,
           filter_size=7, class_conditioning=True):
    """
    MiSuRe: Minimally Sufficient Region for semantic segmentation explanation.

    Learns a minimal mask that preserves the model's prediction for the target
    class by optimizing a combination of Dice loss, optional CE loss, and L1 regularization.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int, optional
        Target class index. If None, uses the dominant predicted class.
    device : str, default='cpu'
        Device to run the model on.
    lr : float, default=0.01
        Learning rate for the AdamW optimizer.
    alpha_fg : float, default=1
        Weight for the foreground Dice loss.
        If class_conditioning=True,  only the target class contributes.
        If class_conditioning=False, all non-background classes contribute
        (background pixels are ignored via ignore_index=0).
    alpha_bg : float, default=1
        Weight for the background Dice loss.
        If class_conditioning=True,  only background (0) and target contribute;
        other classes are don't-care.
        If class_conditioning=False, all classes contribute (no filtering).
    alpha_ce : float, default=0.0
        Weight for the cross entropy loss. 0 disables it entirely.
        CE operates on raw logits (before softmax) so it penalizes
        per-pixel confidence loss independently of Dice.
        If class_conditioning=True,  CE uses a weight vector that zeroes out
        all classes except background and target.
        If class_conditioning=False, CE uses ignore_index=0 (ignores background).
    l1_coeff : float, default=0.1
        Weight for the L1 regularization term.
    tv_coeff : float, default=0
        Weight for the total variation regularization term.
    tv_beta : float, default=3
        Beta parameter for the total variation norm.
    gaussian_sigma : float or None, default=None
        Sigma for Gaussian blur smoothing applied to the mask each iteration.
        If None, no Gaussian smoothing is applied.
    iterations : int, default=100
        Number of optimization iterations.
    remove_background : bool, default=True
        Whether to remove background via dilation before optimization.
    mask_init : bool, default=True
        Whether to initialize mask from predicted segmentation.
    value : float, default=0.5
        Initial mask value outside the predicted region.
    mask_size : int, default=224
        Spatial size of the optimized mask.
    softmax : bool, default=True
        Whether to apply softmax to perturbed prediction before Dice loss.
    clamp_iteration : int, default=10
        Iteration after which to start hard thresholding the mask.
    clamp_threshold : float, default=0.2
        Mask values below this threshold are snapped to 0.
    dice_threshold : float, default=0.9
        DICE threshold for stopping dilation in background removal.
    filter_size : int, default=7
        Structuring element size for dilation in background removal.
    class_conditioning : bool, default=True
        Whether to condition the Dice and CE losses on specific classes.
        If True  (new behaviour): Dice losses use classes=[target] and
        classes=[0, target]; CE uses a per-class weight vector that zeroes
        out all classes except background and target.
        If False (old behaviour): Dice losses use ignore_index=0 and no class
        filtering; CE uses ignore_index=0.

    Returns
    -------
    heatmap : np.ndarray
        Normalized mask with shape (H, W) in [0, 1].
    counter : int
        Number of dilation steps performed during background removal.
    """

    # ------------------------------------------------------------------ #
    #  1. INITIAL FORWARD PASS — get the model's unperturbed prediction   #
    # ------------------------------------------------------------------ #

    # Move the input image to the target device (CPU or GPU).
    image = image.to(device)

    # Run a single forward pass without gradients since we only need the
    # original prediction as a reference — no backprop required here.
    with torch.no_grad():
        # Add a batch dimension (unsqueeze(0): [C,H,W] -> [1,C,H,W]),
        # run the model, then index [0] to remove the batch dim again.
        output = model(image.unsqueeze(0))[0]   # shape: [num_classes, H, W]

    # Convert the raw logits to a hard class map by taking the argmax
    # over the class dimension. The result is a 2-D integer array [H, W]
    # where each pixel holds the predicted class index.
    pred_orig = output.argmax(0).detach().cpu().numpy()  # [H, W]

    # ------------------------------------------------------------------ #
    #  2. TARGET CLASS RESOLUTION                                          #
    # ------------------------------------------------------------------ #

    # If no target class was provided by the caller, automatically pick the
    # class with the highest index present in the prediction. This is a
    # simple heuristic: in many medical / satellite datasets class 0 is
    # background, so max() tends to return the most "foreground" class.
    if target is None:
        target = pred_orig.max().item()

    # If the dominant class is background (0) the image contains no
    # foreground object that is worth explaining — return early.
    if target == 0:
        return None

    # ------------------------------------------------------------------ #
    #  3. BUILD THE MASKING TARGET                                         #
    # ------------------------------------------------------------------ #

    # Construct the reference segmentation map used by all loss functions.
    # We keep only the pixels that belong to `target`; every other pixel
    # (including other foreground classes) is set to 0 (background).
    # This means the loss only rewards the model for correctly predicting
    # `target` pixels, ignoring unrelated foreground regions.
    #
    # Breakdown:
    #   pred_orig == target  -> boolean mask, True where model predicted target
    #   target * (...)       -> converts True->target, False->0  (integer map)
    #   torch.tensor(..., dtype=torch.long) -> required dtype for CE / Dice
    #   .unsqueeze(0)        -> adds batch dim: [H,W] -> [1,H,W]
    masking_target = torch.tensor(
        target * (pred_orig == target), dtype=torch.long
    ).unsqueeze(0).to(device)

    # ------------------------------------------------------------------ #
    #  4. BACKGROUND REMOVAL VIA ITERATIVE DILATION                       #
    # ------------------------------------------------------------------ #

    if remove_background:
        # Iteratively dilate the foreground region until the model's Dice
        # score for `target` on the dilated image crosses `dice_threshold`.
        # This removes irrelevant background context so the optimizer
        # focuses only on the object of interest.
        # Returns:
        #   image    -> the dilated/masked version of the input
        #   pred_bg  -> model prediction on the background-removed image
        #   counter  -> number of dilation steps performed
        image, pred_bg, counter = dilate_to_sufficient(
            image, model, target, device, dice_threshold, filter_size
        )
        # Keep a reference to the background-removed image (not used further
        # in this function but can be useful for debugging / inspection).
        bg_removed_image = image
    else:
        # No background removal — use the original image as-is.
        counter = 0
        bg_removed_image = None

    # ------------------------------------------------------------------ #
    #  5. MASK INITIALISATION                                              #
    # ------------------------------------------------------------------ #

    # Start with a mask of all ones at the working resolution (mask_size x mask_size).
    # A value of 1 means "keep this pixel", 0 means "suppress this pixel".
    mask = torch.ones((mask_size, mask_size), dtype=torch.float32).to(device)

    if mask_init:
        # Resize the original prediction map to the mask resolution using
        # nearest-neighbour interpolation to preserve hard class boundaries.
        prediction_mask = cv2.resize(pred_orig, (mask_size, mask_size),
                                      interpolation=cv2.INTER_NEAREST)
        # Convert to a float tensor so arithmetic operations work cleanly.
        prediction_mask = torch.tensor(prediction_mask, dtype=torch.float32)

        # For every pixel that the model did NOT predict as `target`, lower
        # the initial mask value to `value` (default 0.5) instead of 1.
        # This gives the optimizer a head-start: it already "knows" which
        # region contains the target class and can focus its effort there.
        mask[prediction_mask != target] = value

    # ------------------------------------------------------------------ #
    #  6. CONSTRAIN MASK TO NON-BACKGROUND PIXELS                         #
    # ------------------------------------------------------------------ #

    # Take the first channel of the (possibly dilated) image and resize it
    # to the mask resolution. We use this as a binary indicator of which
    # pixels are non-background (i.e. were not zeroed out by dilation).
    dil_mask = cv2.resize(image[0].detach().cpu().numpy(), (mask_size, mask_size),
                           interpolation=cv2.INTER_NEAREST)

    # Threshold at 0: any pixel > 0 is considered non-background.
    # The mask should never be non-zero in background pixels because those
    # areas carry no signal and would just waste mask budget.
    dil_mask = torch.tensor(dil_mask, dtype=torch.float32).to(device) > 0

    # Zero out the initial mask wherever the pixel is pure background.
    mask = mask * dil_mask

    # ------------------------------------------------------------------ #
    #  7. PREPARE MASK AS OPTIMISABLE PARAMETER                           #
    # ------------------------------------------------------------------ #

    # Add batch and channel dimensions: [mask_size, mask_size] -> [1,1,mask_size,mask_size].
    # This shape is required by F.interpolate and by the expand() call later.
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Detach from any existing computation graph, clone to get a fresh
    # tensor, and enable gradient tracking so AdamW can update it.
    mask = torch.tensor(mask.clone().detach(), requires_grad=True)

    # Instantiate the AdamW optimizer. Only `mask` is optimised; the model
    # weights remain frozen throughout.
    optimizer = torch.optim.AdamW([mask], lr=lr)

    # ------------------------------------------------------------------ #
    #  8. OPTIONAL GAUSSIAN BLUR KERNEL                                    #
    # ------------------------------------------------------------------ #

    if gaussian_sigma is not None:
        # Compute a kernel size that covers ±3σ on each side (standard rule).
        kernel_size = int(6 * gaussian_sigma + 1)
        # Ensure the kernel size is odd — required by GaussianBlur.
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        # Build the blur transform once; it will be applied every iteration.
        gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=gaussian_sigma)

    # Softmax function used to convert raw logits to class probabilities
    # before computing the Dice loss (when softmax=True).
    softmax_fn = torch.nn.Softmax(dim=1)

    # ------------------------------------------------------------------ #
    #  9. LOSS FUNCTION SETUP                                              #
    # ------------------------------------------------------------------ #

    if class_conditioning:
        # --- NEW BEHAVIOUR: class-conditioned losses ---

        # Foreground Dice: measures how well the model predicts `target`
        # on the masked image. Only `target` pixels contribute to the loss;
        # all other classes (including background) are treated as don't-care.
        dice_loss_fg = smp.losses.DiceLoss(mode='multiclass', classes=[target])

        # Background+Target Dice: ensures the model also correctly predicts
        # background (class 0) in non-target regions. By including both 0
        # and `target`, the loss penalises confusing background with target
        # and vice versa, while ignoring unrelated foreground classes.
        dice_loss_bg = smp.losses.DiceLoss(mode='multiclass', classes=[0, target])

        if alpha_ce > 0:
            # Build a per-class weight vector of length `num_classes`.
            # Setting weight=0 for all classes except 0 and `target` means
            # those classes contribute zero gradient to the CE loss, making
            # the CE equivalent in scope to the class-conditioned Dice above.
            num_classes = output.shape[0]
            ce_weight = torch.zeros(num_classes, device=device)
            ce_weight[0] = 1.0       # background class always counts
            ce_weight[target] = 1.0  # target class always counts
            ce_loss_fn = torch.nn.CrossEntropyLoss(weight=ce_weight)
        else:
            # CE is disabled — set to None so the loop can skip it cheaply.
            ce_loss_fn = None

    else:
        # --- OLD BEHAVIOUR: unfiltered losses ---

        # Foreground Dice: all non-background classes contribute.
        # ignore_index=0 simply skips background pixels when computing Dice,
        # but does not restrict which foreground classes are considered.
        dice_loss_fg = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)

        # Background Dice: all classes (including background) contribute with
        # equal weight — no class filtering whatsoever.
        dice_loss_bg = smp.losses.DiceLoss(mode='multiclass')

        # CE with ignore_index=0 skips background pixels but otherwise treats
        # all foreground classes equally, regardless of whether they are the
        # target class or not.
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0) if alpha_ce > 0 else None

    # ------------------------------------------------------------------ #
    #  10. OPTIMISATION LOOP                                               #
    # ------------------------------------------------------------------ #

    for iteration in tqdm(range(iterations), desc='MiSuRe optimization'):

        # -------------------------------------------------------------- #
        #  10a. OPTIONAL GAUSSIAN SMOOTHING OF THE MASK                   #
        # -------------------------------------------------------------- #

        if gaussian_sigma is not None:
            # Smooth the mask before applying it to the image. This acts as
            # a spatial regulariser that prevents the mask from becoming
            # overly noisy or producing isolated single-pixel artefacts.
            mask_smooth = gaussian_blur(mask)
        else:
            # No smoothing — use the raw optimisable mask directly.
            mask_smooth = mask

        # -------------------------------------------------------------- #
        #  10b. UPSAMPLE MASK TO IMAGE RESOLUTION                         #
        # -------------------------------------------------------------- #

        if mask_smooth.shape[-2:] != image.shape[-2:]:
            # If the mask resolution differs from the image resolution,
            # upsample with bilinear interpolation to produce smooth
            # gradients during backpropagation (nearest-neighbour would
            # create blocky, non-differentiable transitions).
            upsampled_mask = F.interpolate(
                mask_smooth, size=(image.shape[-2], image.shape[-1]),
                mode='bilinear', align_corners=False
            )
        else:
            # Mask already matches the image resolution — no resize needed.
            upsampled_mask = mask_smooth

        # -------------------------------------------------------------- #
        #  10c. APPLY MASK TO IMAGE (PERTURBATION)                        #
        # -------------------------------------------------------------- #

        # Expand the single-channel mask [1,1,H,W] to match the number of
        # image channels [1,C,H,W] by repeating it C times along dim=1.
        # This is a memory-efficient view — no data is copied.
        upsampled_mask_expanded = upsampled_mask.expand(
            1, image.shape[0], upsampled_mask.shape[2], upsampled_mask.shape[3]
        )

        # Element-wise multiply: pixels where mask≈1 are kept as-is,
        # pixels where mask≈0 are suppressed to zero (black / ablated).
        # This is the core perturbation that drives the optimisation.
        perturbated_input = image * upsampled_mask_expanded

        # Run the (frozen) model on the perturbed image to get the
        # prediction that the loss will compare against masking_target.
        prediction_perturbed = model(perturbated_input)

        # -------------------------------------------------------------- #
        #  10d. CROSS-ENTROPY LOSS (operates on RAW LOGITS)               #
        # -------------------------------------------------------------- #

        # CE must be computed BEFORE softmax because CrossEntropyLoss
        # internally applies log-softmax; applying softmax beforehand
        # would double-squash the probabilities and corrupt the gradients.
        loss_ce = alpha_ce * ce_loss_fn(prediction_perturbed, masking_target) if alpha_ce > 0 else 0

        # -------------------------------------------------------------- #
        #  10e. OPTIONAL SOFTMAX BEFORE DICE                              #
        # -------------------------------------------------------------- #

        if softmax:
            # Convert raw logits to class probabilities. The SMP DiceLoss
            # can accept either probabilities or logits depending on its
            # `from_logits` flag, but here we normalise explicitly for
            # consistency and interpretability.
            prediction_perturbed = softmax_fn(prediction_perturbed)

        # -------------------------------------------------------------- #
        #  10f. REGULARISATION LOSSES                                      #
        # -------------------------------------------------------------- #

        # L1 regularisation: penalises the total "area" of the mask
        # (average absolute value). This is the key minimality pressure —
        # it pushes the mask to be as small as possible while still
        # maintaining a good prediction, hence "Minimally Sufficient".
        loss_l1 = torch.mean(torch.abs(upsampled_mask))

        # Total Variation regularisation: penalises spatial roughness of
        # the mask, encouraging smoother, more contiguous regions.
        # Only computed when tv_coeff > 0 to save compute otherwise.
        loss_tv = tv_norm(upsampled_mask, tv_beta) if tv_coeff != 0 else 0

        # -------------------------------------------------------------- #
        #  10g. DICE LOSS (fidelity term)                                  #
        # -------------------------------------------------------------- #

        # Combine foreground and background Dice losses with their respective
        # weights, plus the (possibly zero) CE loss.
        # This term measures how faithfully the masked image preserves
        # the original model prediction — higher Dice = better preservation.
        loss_output = (
            alpha_fg * dice_loss_fg(prediction_perturbed, masking_target) +
            alpha_bg * dice_loss_bg(prediction_perturbed, masking_target) +
            loss_ce
        )

        # -------------------------------------------------------------- #
        #  10h. TOTAL LOSS AND GRADIENT STEP                              #
        # -------------------------------------------------------------- #

        # Combine fidelity (loss_output) and regularisation (L1 + TV).
        # The coefficients balance the trade-off between explanation
        # faithfulness (large mask → better prediction) and minimality
        # (small mask → simpler explanation).
        perturbation_loss = l1_coeff * loss_l1 + tv_coeff * loss_tv + loss_output

        # Zero out gradients from the previous iteration to prevent
        # accumulation (PyTorch does not reset gradients automatically).
        optimizer.zero_grad()

        # Backpropagate the total loss through the model and mask to
        # compute d(loss)/d(mask) — the gradient of each mask pixel.
        perturbation_loss.backward()

        # Update the mask values using the AdamW rule (gradient descent
        # with weight decay and adaptive moment estimates).
        optimizer.step()

        # -------------------------------------------------------------- #
        #  10i. HARD CLAMPING / SPARSIFICATION                            #
        # -------------------------------------------------------------- #

        if iteration > clamp_iteration:
            # After the warm-up phase, snap any mask value below
            # `clamp_threshold` to exactly 0. This progressively
            # binarises the mask, forcing the optimizer to commit to
            # a sparse, interpretable explanation rather than keeping
            # many small non-zero values that contribute little to fidelity.
            mask.data[mask.data < clamp_threshold] = 0

        # Clip the entire mask to [0, 1] after each gradient step to ensure
        # it stays within a valid probability-like range. Values > 1 would
        # amplify the image signal beyond its original range (undesired);
        # values < 0 have no physical meaning for a multiplicative mask.
        mask.data.clamp_(0, 1)

    # ------------------------------------------------------------------ #
    #  11. POST-PROCESSING AND OUTPUT                                      #
    # ------------------------------------------------------------------ #

    # Extract the final 2-D mask array from the 4-D tensor [1,1,H,W].
    # detach() removes it from the computation graph; .cpu() moves it
    # to CPU memory; .numpy() converts to a NumPy array.
    heatmap = upsampled_mask[0][0].detach().cpu().numpy()

    # Min-max normalise the heatmap to [0, 1] so that the brightest pixel
    # always reaches 1 regardless of the absolute mask values.
    # A small epsilon (1e-8) prevents division by zero for constant maps.
    # min_, max_ = heatmap.min(), heatmap.max()
    # heatmap = (heatmap - min_) / (max_ - min_ + 1e-8)

    # Keep references to the final perturbed image and its prediction for
    # optional downstream inspection (not included in the return value but
    # useful when debugging or visualising intermediate results).
    perturbed_image = perturbated_input[0].detach().cpu()
    perturbed_prediction = prediction_perturbed[0].argmax(0).detach().cpu().numpy()
    prediction = pred_orig

    # Return the normalised heatmap and the number of dilation steps that
    # were needed to produce a sufficient background-removed image.
    return heatmap, counter

def misure_old(image, model, target=None, device='cpu',
           lr=0.01, alpha_fg=1, alpha_bg=1, l1_coeff=0.1, 
           tv_coeff=0, tv_beta=3, gaussian_sigma=None,
           iterations=100, remove_background=True, mask_init=True,
           value=0.5, mask_size=224, softmax=True, clamp_iteration=10,
           clamp_threshold=0.2, dice_threshold=0.9,
           filter_size=7):
    """
    MiSuRe: Minimally Sufficient Region for semantic segmentation explanation.

    Learns a minimal mask that preserves the model's prediction for the target
    class by optimizing a combination of Dice loss and L1 regularization.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int, optional
        Target class index. If None, uses the dominant predicted class.
    device : str, default='cpu'
        Device to run the model on.
    lr : float, default=0.01
        Learning rate for the AdamW optimizer.
    alpha_fg : float, default=1
        Weight for the foreground Dice loss (ignore_index=0).
    alpha_bg : float, default=1
        Weight for the combined Dice loss (all classes).
    l1_coeff : float, default=0.1
        Weight for the L1 regularization term.
    tv_coeff : float, default=0
        Weight for the total variation regularization term.
    tv_beta : float, default=3
        Beta parameter for the total variation norm.
    gaussian_sigma : float or None, default=None
        Sigma for Gaussian blur smoothing applied to the mask each iteration.
        If None, no Gaussian smoothing is applied.
    iterations : int, default=100
        Number of optimization iterations.
    remove_background : bool, default=True
        Whether to remove background via dilation before optimization.
    mask_init : bool, default=True
        Whether to initialize mask from predicted segmentation.
    value : float, default=0.5
        Initial mask value outside the predicted region.
    mask_size : int, default=224
        Spatial size of the optimized mask.
    softmax : bool, default=True
        Whether to apply softmax to perturbed prediction before loss.
    clamp_iteration : int, default=10
        Iteration after which to start hard thresholding the mask.
    clamp_threshold : float, default=0.2
        Mask values below this threshold are snapped to 0.
    dice_threshold : float, default=0.9
        DICE threshold for stopping dilation in background removal.
    filter_size : int, default=7
        Structuring element size for dilation in background removal.

    Returns
    -------
    heatmap : np.ndarray
        Normalized mask with shape (H, W) in [0, 1].
    prediction : np.ndarray
        Original predicted segmentation mask with shape (H, W).
    bg_removed_image : torch.Tensor or None
        Background removed image with shape (C, H, W). None if remove_background=False.
    perturbed_image : torch.Tensor
        Final masked image with shape (C, H, W).
    perturbed_prediction : np.ndarray
        Predicted segmentation on perturbed image with shape (H, W).
    dice_list : list
        DICE score at each optimization iteration.
    useful_metrics : list
        [(counter, final_dice, non_zero_pred, non_zero_perturbed)]
    """
    image = image.to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))[0]

    pred_orig = output.argmax(0).detach().cpu().numpy()  # [H, W]

    # Default target: dominant predicted class
    if target is None:
        target = pred_orig.max().item()

    # Early exit: model predicts only background
    if target == 0:
        return None

    masking_target = torch.tensor(
        target * (pred_orig == target), dtype=torch.long
    ).unsqueeze(0).to(device)

    # Background removal
    if remove_background:
        image, pred_bg, counter = dilate_to_sufficient(
            image, model, target, device, dice_threshold, filter_size
        )
        bg_removed_image = image
    else:
        counter = 0
        bg_removed_image = None

    # Mask initialization
    mask = torch.ones((mask_size, mask_size), dtype=torch.float32).to(device)

    if mask_init:
        prediction_mask = cv2.resize(pred_orig, (mask_size, mask_size),
                                      interpolation=cv2.INTER_NEAREST)
        prediction_mask = torch.tensor(prediction_mask, dtype=torch.float32)
        mask[prediction_mask != target] = value

    # Constrain mask to non-background pixels
    dil_mask = cv2.resize(image[0].detach().cpu().numpy(), (mask_size, mask_size),
                           interpolation=cv2.INTER_NEAREST)
    dil_mask = torch.tensor(dil_mask, dtype=torch.float32).to(device) > 0
    mask = mask * dil_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.tensor(mask.clone().detach(), requires_grad=True)

    optimizer = torch.optim.AdamW([mask], lr=lr)

    # Precompute Gaussian kernel if needed
    if gaussian_sigma is not None:
        kernel_size = int(6 * gaussian_sigma + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=gaussian_sigma)

    dice_list = []


    softmax_fn = torch.nn.Softmax(dim=1)
    dice_loss_fg = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
    dice_loss_bg = smp.losses.DiceLoss(mode='multiclass')


    # Optimization loop
    for iteration in tqdm(range(iterations), desc='MiSuRe optimization'):

        # Apply Gaussian smoothing to mask before forward pass
        if gaussian_sigma is not None:
            mask_smooth = gaussian_blur(mask)
        else:
            mask_smooth = mask

        if mask_smooth.shape[-2:] != image.shape[-2:]:
            upsampled_mask = F.interpolate(
                mask_smooth, size=(image.shape[-2], image.shape[-1]),
                mode='bilinear', align_corners=False
            )
        else:
            upsampled_mask = mask_smooth

        upsampled_mask_expanded = upsampled_mask.expand(
            1, image.shape[0], upsampled_mask.shape[2], upsampled_mask.shape[3]
        )
        perturbated_input = image * upsampled_mask_expanded
        prediction_perturbed = model(perturbated_input)

        if softmax:
            prediction_perturbed = softmax_fn(prediction_perturbed)

        loss_l1 = torch.mean(torch.abs(upsampled_mask))

        loss_tv = tv_norm(upsampled_mask, tv_beta) if tv_coeff != 0 else 0

        loss_output = (
            alpha_fg * dice_loss_fg(
                prediction_perturbed, masking_target
            ) +
            alpha_bg * dice_loss_bg(
                prediction_perturbed, masking_target
            )
        )

        perturbation_loss = l1_coeff * loss_l1 + tv_coeff * loss_tv + loss_output

        optimizer.zero_grad()
        perturbation_loss.backward()
        optimizer.step()

        if iteration > clamp_iteration:
            mask.data[mask.data < clamp_threshold] = 0

        mask.data.clamp_(0, 1)

        # # Calculate dice score using medpy's dc
        # pred_perturbed_np = prediction_perturbed[0].argmax(0).detach().cpu().numpy()
        # gt_binary = (pred_orig == target).astype(np.uint8)
        # pred_binary = (pred_perturbed_np == target).astype(np.uint8)
        # dice_perturbed = dc(gt_binary, pred_binary)
        # dice_list.append(dice_perturbed)

    # Final outputs
    # non_zero_pred = (masking_target > 0).sum().item()
    # non_zero_perturbed = (upsampled_mask[0][0] > 0).sum().item()
    # useful_metrics = [(counter, dice_list[-1], non_zero_pred, non_zero_perturbed)]

    # Normalize heatmap to [0, 1]
    heatmap = upsampled_mask[0][0].detach().cpu().numpy()
    min_, max_ = heatmap.min(), heatmap.max()
    heatmap = (heatmap - min_) / (max_ - min_ + 1e-8)

    perturbed_image = perturbated_input[0].detach().cpu()
    perturbed_prediction = prediction_perturbed[0].argmax(0).detach().cpu().numpy()
    prediction = pred_orig

    return heatmap, counter
    # return (heatmap, prediction, bg_removed_image,
    #         perturbed_image, perturbed_prediction,
    #         dice_list, useful_metrics)'''









# def misure(image, model, target=None, device='cpu',
#            lr=0.01, alpha_fg=1, alpha_bg=1, l1_coeff=0.1, 
#            tv_coeff=0, tv_beta=3, gaussian_sigma=None,
#            iterations=100, remove_background=True, mask_init=True,
#            value=0.5, mask_size=224, softmax=True, clamp_iteration=10,
#            clamp_threshold=0.2, dice_threshold=0.9,
#            filter_size=7):
#     """
#     MiSuRe: Minimally Sufficient Region for semantic segmentation explanation.

#     Learns a minimal mask that preserves the model's prediction for the target
#     class by optimizing a combination of Dice loss and L1 regularization.

#     Parameters
#     ----------
#     image : torch.Tensor
#         Input image tensor with shape (C, H, W).
#     model : torch.nn.Module
#         The segmentation model.
#     target : int, optional
#         Target class index. If None, uses the dominant predicted class.
#     device : str, default='cpu'
#         Device to run the model on.
#     lr : float, default=0.01
#         Learning rate for the AdamW optimizer.
#     alpha_fg : float, default=1
#         Weight for the foreground Dice loss (ignore_index=0).
#     alpha_bg : float, default=1
#         Weight for the combined Dice loss (all classes).
#     l1_coeff : float, default=0.1
#         Weight for the L1 regularization term.
#     tv_coeff : float, default=0
#         Weight for the total variation regularization term.
#     tv_beta : float, default=3
#         Beta parameter for the total variation norm.
#     gaussian_sigma : float or None, default=None
#         Sigma for Gaussian blur smoothing applied to the mask each iteration.
#         If None, no Gaussian smoothing is applied.
#     iterations : int, default=100
#         Number of optimization iterations.
#     remove_background : bool, default=True
#         Whether to remove background via dilation before optimization.
#     mask_init : bool, default=True
#         Whether to initialize mask from predicted segmentation.
#     value : float, default=0.5
#         Initial mask value outside the predicted region (in sigmoid space).
#     mask_size : int, default=224
#         Spatial size of the optimized mask.
#     softmax : bool, default=True
#         Whether to apply softmax to perturbed prediction before loss.
#     clamp_iteration : int, default=10
#         Iteration after which to double the L1 penalty to encourage sparsity.
#     clamp_threshold : float, default=0.2
#         Mask values below this threshold are zeroed out in the final heatmap.
#     dice_threshold : float, default=0.9
#         DICE threshold for stopping dilation in background removal.
#     filter_size : int, default=7
#         Structuring element size for dilation in background removal.

#     Returns
#     -------
#     heatmap : np.ndarray
#         Normalized mask with shape (H, W) in [0, 1].
#     counter : int
#         Number of dilation steps performed during background removal.
#     """
#     image = image.to(device)

#     with torch.no_grad():
#         output = model(image.unsqueeze(0))[0]

#     pred_orig = output.argmax(0).detach().cpu().numpy()  # [H, W]

#     # Default target: dominant predicted class
#     if target is None:
#         target = pred_orig.max().item()

#     # Early exit: model predicts only background
#     if target == 0:
#         return None

#     masking_target = torch.tensor(
#         target * (pred_orig == target), dtype=torch.long
#     ).unsqueeze(0).to(device)

#     # Background removal
#     if remove_background:
#         image, pred_bg, counter = dilate_to_sufficient(
#             image, model, target, device, dice_threshold, filter_size
#         )
#         bg_removed_image = image
#     else:
#         counter = 0
#         bg_removed_image = None

#     # --- Mask initialization in latent (pre-sigmoid) space ---
#     # sigmoid(0) = 0.5  → neutral starting point
#     # To initialize a pixel to `value` in [0,1]: latent = logit(value)
#     # To initialize a pixel to ~1: use a large positive value (e.g. 4.0)
#     mask_latent = torch.zeros((1, 1, mask_size, mask_size),
#                                dtype=torch.float32, device=device)

#     if mask_init:
#         prediction_mask = cv2.resize(pred_orig, (mask_size, mask_size),
#                                       interpolation=cv2.INTER_NEAREST)
#         prediction_mask = torch.tensor(prediction_mask, dtype=torch.float32)

#         # Pixels belonging to target class → latent ≈ +4  (sigmoid → ~0.98)
#         # Pixels outside target class     → latent = logit(value)
#         logit_value = torch.log(torch.tensor(value) / (1 - torch.tensor(value) + 1e-8))
#         mask_latent[0, 0][prediction_mask == target] = 4.0
#         mask_latent[0, 0][prediction_mask != target] = logit_value.item()

#     # Constrain latent mask to non-background pixels
#     dil_mask = cv2.resize(image[0].detach().cpu().numpy(), (mask_size, mask_size),
#                            interpolation=cv2.INTER_NEAREST)
#     dil_mask = torch.tensor(dil_mask, dtype=torch.float32).to(device) > 0
#     # Zero out background in latent space → sigmoid(0) = 0.5,
#     # then we rely on the dil_mask applied to the sigmoid output each iteration
#     mask_latent = mask_latent.clone().detach().requires_grad_(True)

#     optimizer = torch.optim.AdamW([mask_latent], lr=lr)

#     # Precompute Gaussian kernel if needed
#     if gaussian_sigma is not None:
#         kernel_size = int(6 * gaussian_sigma + 1)
#         kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
#         gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=gaussian_sigma)

#     dice_list = []

#     # Optimization loop
#     for iteration in tqdm(range(iterations), desc='MiSuRe optimization'):

#         # Apply sigmoid to get mask in [0, 1] — replaces manual clamping
#         mask = torch.sigmoid(mask_latent)

#         # Enforce non-background constraint
#         mask = mask * dil_mask.unsqueeze(0).unsqueeze(0)

#         # Apply optional Gaussian smoothing
#         if gaussian_sigma is not None:
#             mask_smooth = gaussian_blur(mask)
#         else:
#             mask_smooth = mask

#         if mask_smooth.shape[-2:] != image.shape[-2:]:
#             upsampled_mask = F.interpolate(
#                 mask_smooth, size=(image.shape[-2], image.shape[-1]),
#                 mode='bilinear', align_corners=False
#             )
#         else:
#             upsampled_mask = mask_smooth

#         upsampled_mask_expanded = upsampled_mask.expand(
#             1, image.shape[0], upsampled_mask.shape[2], upsampled_mask.shape[3]
#         )
#         perturbated_input = image * upsampled_mask_expanded
#         prediction_perturbed = model(perturbated_input)

#         if softmax:
#             prediction_perturbed = torch.nn.Softmax(1)(prediction_perturbed)

#         loss_l1 = torch.mean(torch.abs(upsampled_mask))

#         loss_tv = tv_norm(upsampled_mask, tv_beta) if tv_coeff != 0 else 0

#         loss_output = (
#             alpha_fg * smp.losses.DiceLoss(mode='multiclass', ignore_index=0)(
#                 prediction_perturbed, masking_target
#             ) +
#             alpha_bg * smp.losses.DiceLoss(mode='multiclass')(
#                 prediction_perturbed, masking_target
#             )
#         )

#         # Double L1 penalty after clamp_iteration to encourage sparsity,
#         # mimicking the original hard-threshold behaviour
#         current_l1_coeff = l1_coeff if iteration < clamp_iteration else l1_coeff * 2

#         perturbation_loss = current_l1_coeff * loss_l1 + tv_coeff * loss_tv + loss_output

#         optimizer.zero_grad()
#         perturbation_loss.backward()
#         optimizer.step()

#     # Final outputs — threshold only at the very end
#     final_mask = torch.sigmoid(mask_latent)

#     # Upsample back to original image resolution (same as upsampled_mask in the loop)
#     if final_mask.shape[-2:] != image.shape[-2:]:
#         final_mask = F.interpolate(
#             final_mask, size=(image.shape[-2], image.shape[-1]),
#             mode='bilinear', align_corners=False
#         )

#     heatmap = final_mask.detach().cpu().numpy()[0, 0]

#     # Apply non-background mask (also upsampled to original image resolution)
#     dil_mask_full = F.interpolate(
#         dil_mask.float().unsqueeze(0).unsqueeze(0),
#         size=(image.shape[-2], image.shape[-1]),
#         mode='nearest'
#     ).squeeze().cpu().numpy()
#     heatmap = heatmap * dil_mask_full

#     # Hard threshold
#     heatmap[heatmap < clamp_threshold] = 0

#     # Normalize to [0, 1]
#     min_, max_ = heatmap.min(), heatmap.max()
#     heatmap = (heatmap - min_) / (max_ - min_ + 1e-8)

#     return heatmap, counter