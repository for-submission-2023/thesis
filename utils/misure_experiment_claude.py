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
import segmentation_models_pytorch as smp
from medpy.metric.binary import dc
from tqdm import tqdm

from .xai_metrics_claude import (
    blend_with_baseline,
    compute_paired_metric,
    compute_paired_metric_batch,
    get_baseline,
)


# ======================================================================= #
#  HELPERS                                                                  #
# ======================================================================= #

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
    img      = input[0, 0, :]
    row_grad = torch.mean(torch.abs(img[:-1, :] - img[1:,  :]).pow(tv_beta))
    col_grad = torch.mean(torch.abs(img[:,  :-1] - img[:, 1:]).pow(tv_beta))
    return row_grad + col_grad


# ======================================================================= #
#  DILATION                                                                 #
# ======================================================================= #

def dilate_to_sufficient(image, model, target, device='cpu',
                          dice_threshold=0.9, filter_size=7,
                          kernel_shape=0,
                          baseline_mode='zero',
                          baseline_mean=(0., 0., 0.),
                          baseline_std=(1., 1., 1.),
                          baseline_blur_sigma=51):
    """
    Dilates the predicted segmentation mask until the model's prediction on the
    baseline-filled image has DICE >= dice_threshold with the original prediction.

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
        Size of the structuring element for dilation.
    kernel_shape : int, default=0
        Shape of the structuring element:
          0 → ellipse  (cv2.MORPH_ELLIPSE) — default, rounds off corners
          1 → rect     (cv2.MORPH_RECT)    — full square kernel
          2 → cross    (cv2.MORPH_CROSS)   — plus-shaped kernel
    baseline_mode : str, default='zero'
        Passed to get_baseline — 'zero' or 'blur'.
    baseline_mean : tuple of float, default=(0., 0., 0.)
        Per-channel mean for normalised-zero baseline.
    baseline_std : tuple of float, default=(1., 1., 1.)
        Per-channel std for normalised-zero baseline.
    baseline_blur_sigma : float, default=51
        Gaussian sigma for blur baseline.

    Returns
    -------
    background_removed : torch.Tensor
        Baseline-filled image tensor with shape (C, H, W).
    pred_bg : np.ndarray
        Predicted segmentation on the returned image, shape (H, W).
    mask_image : np.ndarray
        Final binary dilation mask, shape (H, W).
    x_prime : torch.Tensor
        The baseline tensor computed from the input image, shape (C, H, W).
        Returned so callers can reuse the exact same tensor.
    stats : list
        [counter, dice_score] — number of dilation iterations performed and
        the final Dice score that satisfied the threshold.
    """
    _device_type = device.split(':')[0]

    x_prime = get_baseline(
        image,
        mode=baseline_mode,
        mean=baseline_mean,
        std=baseline_std,
        blur_sigma=baseline_blur_sigma,
    )

    with torch.no_grad():
        with torch.amp.autocast(device_type=_device_type, enabled=(_device_type != 'cpu')):
            output = model(image.unsqueeze(0).to(device))[0]
    pred_orig = output.argmax(0).detach().cpu().numpy()

    mask_image = 1.0 * (pred_orig == target)
    _MORPH = {0: cv2.MORPH_ELLIPSE, 1: cv2.MORPH_RECT, 2: cv2.MORPH_CROSS}
    struct     = cv2.getStructuringElement(_MORPH[kernel_shape], (filter_size, filter_size))
    counter    = 0

    while True:
        mask_t = torch.tensor(mask_image, dtype=torch.float32).to(device).unsqueeze(0)  # (1, H, W)

        background_removed = blend_with_baseline(image.to(device), mask_t, x_prime)

        with torch.no_grad():
            with torch.amp.autocast(device_type=_device_type, enabled=(_device_type != 'cpu')):
                output_bg = model(background_removed.unsqueeze(0))[0]
        pred_bg = output_bg.argmax(0).detach().cpu().numpy()

        gt_binary   = (pred_orig == target).astype(np.uint8)
        pred_binary = (pred_bg   == target).astype(np.uint8)
        dice_score  = dc(gt_binary, pred_binary)

        if dice_score >= dice_threshold:
            break
        mask_image = cv2.dilate(mask_image, struct, iterations=1)
        counter   += 1

    return background_removed, pred_bg, mask_image, x_prime, [counter, dice_score]


# ======================================================================= #
#  MAIN                                                                     #
# ======================================================================= #

def misure_1(image, model, target=None, device='cpu',
             lr=0.01, alpha_fg=1, alpha_bg=1, alpha_ce=0.0, l1_coeff=0.1,
             tv_coeff=0, tv_beta=3,
             iterations=100, remove_background=True,
             mask_size=224, dice_threshold=0.9,
             filter_size=7, kernel_shape=0, class_conditioning=True,
             min_dice_threshold=0.5,
             patience=3,
             patience_dice_threshold=0.3,
             mask_init=False, value=0.5,
             # ---- baseline parameters ----
             baseline_mode='zero',
             baseline_mean=(0., 0., 0.),
             baseline_std=(1., 1., 1.),
             baseline_blur_sigma=51):
    """
    MiSuRe optimisation.

    Learns a minimal mask M such that:
        model(image * M + (1 - M) * x_prime) ≈ model(image)

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        Semantic segmentation model.
    target : int or None
        Target class to explain. If None, the most-predicted class is used.
    device : str
        Torch device string.
    lr : float
        AdamW learning rate.
    alpha_fg : float
        Weight for foreground Dice loss.
    alpha_bg : float
        Weight for background+foreground Dice loss.
    alpha_ce : float
        Weight for cross-entropy loss (0 disables it).
    l1_coeff : float
        Base L1 sparsity coefficient (scaled by foreground fraction internally).
    tv_coeff : float
        Total variation coefficient (0 disables it).
    tv_beta : float
        Exponent for total variation norm.
    iterations : int
        Number of optimisation steps.
    remove_background : bool
        Whether to apply iterative dilation background removal.
    mask_size : int
        Shorter spatial dimension of the optimised mask.
    dice_threshold : float
        DICE threshold used in iterative dilation.
    filter_size : int
        Dilation kernel size.
    kernel_shape : int, default=0
        Shape of the dilation structuring element:
          0 → ellipse, 1 → rect, 2 → cross
    class_conditioning : bool
        If True, Dice loss is restricted to target and background classes.
    min_dice_threshold : float
        Minimum acceptable Dice for the final mask; triggers fallback otherwise.
    patience : int
        Early-stop after this many consecutive bad checkpoints.
    patience_dice_threshold : float
        Checkpoints below this Dice count toward patience.
    mask_init : bool, default=False
        If True, initialise the mask from the model's own prediction:
        pixels predicted as the target class start at 1.0, all others at
        ``value``.  This warm-starts optimisation from the predicted region.
    value : float, default=0.5
        Initial mask value for non-target pixels when mask_init=True.
    baseline_mode : str
        'zero' or 'blur' — how x_prime is constructed.
    baseline_mean : tuple of float
        Per-channel mean used when baseline_mode='zero'.
    baseline_std : tuple of float
        Per-channel std  used when baseline_mode='zero'.
    baseline_blur_sigma : float
        Gaussian sigma used when baseline_mode='blur'.

    Returns
    -------
    cam : np.ndarray
        Raw explanation heatmap with shape (H, W). Pass to sweep_thresholds()
        for post-hoc threshold selection.
    x_prime : torch.Tensor
        The baseline tensor used during optimisation, shape (C, H, W).
        Pass this exact tensor to sweep_thresholds() and metric functions to
        guarantee consistency.
    stats : list
        [counter, dil_dice, pred_nonzero, cam_nonzero]
        counter       — dilation iterations (0 if remove_background=False)
        dil_dice      — Dice after dilation (0.0 if remove_background=False)
        pred_nonzero  — non-zero pixels in original prediction
        cam_nonzero   — non-zero pixels in raw cam
    """

    image        = image.to(device)
    _device_type = device.split(':')[0]
    _autocast    = _device_type != 'cpu'

    model.eval()   # set once — not repeated inside loops

    # ------------------------------------------------------------------ #
    #  1. INITIAL FORWARD PASS                                             #
    # ------------------------------------------------------------------ #

    with torch.no_grad():
        with torch.amp.autocast(device_type=_device_type, enabled=_autocast):
            output = model(image.unsqueeze(0))[0]

    pred_orig = output.argmax(0).detach().cpu().numpy()     # (H, W)

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

    # precompute original prediction binary mask once — reused in all
    # compute_paired_metric / compute_paired_metric_batch calls
    pred_orig_binary   = (pred_orig == target).astype(np.uint8)
    pred_orig_nonzero  = int(pred_orig_binary.sum())

    # ------------------------------------------------------------------ #
    #  4. BASELINE AND BACKGROUND REMOVAL                                  #
    #  dilate_to_sufficient computes x_prime internally and returns it,   #
    #  so misure_1 reuses the exact same tensor throughout.               #
    #  When remove_background=False we call get_baseline directly.        #
    # ------------------------------------------------------------------ #

    if remove_background:
        image, pred_bg, dil_mask_np, x_prime, dil_stats = dilate_to_sufficient(
            image, model, target,
            device=device,
            dice_threshold=dice_threshold,
            filter_size=filter_size,
            kernel_shape=kernel_shape,
            baseline_mode=baseline_mode,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            baseline_blur_sigma=baseline_blur_sigma,
        )
        counter = dil_stats[0]
    else:
        x_prime     = get_baseline(
            image,
            mode=baseline_mode,
            mean=baseline_mean,
            std=baseline_std,
            blur_sigma=baseline_blur_sigma,
        )
        counter     = 0
        dil_mask_np = None

    # ------------------------------------------------------------------ #
    #  5. ASPECT-RATIO-AWARE MASK SIZE                                     #
    # ------------------------------------------------------------------ #

    img_H, img_W = image.shape[-2], image.shape[-1]

    if img_H <= img_W:
        mask_H = mask_size
        mask_W = int(round(mask_size * img_W / img_H))
    else:
        mask_W = mask_size
        mask_H = int(round(mask_size * img_H / img_W))

    # ------------------------------------------------------------------ #
    #  6. MASK INITIALISATION                                              #
    # ------------------------------------------------------------------ #

    mask = torch.ones((mask_H, mask_W), dtype=torch.float32).to(device)

    if mask_init:
        pred_resized = cv2.resize(pred_orig, (mask_W, mask_H),
                                  interpolation=cv2.INTER_NEAREST)
        pred_resized_t = torch.tensor(pred_resized, dtype=torch.float32)
        mask[pred_resized_t != target] = value

    # ------------------------------------------------------------------ #
    #  7. CONSTRAIN MASK TO DILATED REGION                                 #
    #     When remove_background=True, dil_mask_np is the final dilation  #
    #     mask returned by dilate_to_sufficient. We resize it and zero     #
    #     out mask pixels that fall outside it.                            #
    #     With non-zero baselines (e.g. blur), we cannot infer the region  #
    #     from pixel values, so we use the mask directly.                  #
    # ------------------------------------------------------------------ #

    if dil_mask_np is not None:
        dil_resized = cv2.resize(
            dil_mask_np.astype(np.float32), (mask_W, mask_H),
            interpolation=cv2.INTER_NEAREST
        )
        dil_mask_t = torch.tensor(dil_resized, dtype=torch.float32).to(device) > 0
        mask       = mask * dil_mask_t

    # ------------------------------------------------------------------ #
    #  8. PREPARE MASK AS OPTIMISABLE PARAMETER                           #
    # ------------------------------------------------------------------ #

    mask      = mask.unsqueeze(0).unsqueeze(0)                          # (1, 1, H, W)
    mask      = torch.tensor(mask.clone().detach(), requires_grad=True)
    optimizer = torch.optim.AdamW([mask], lr=lr)
    scaler    = torch.cuda.amp.GradScaler() if _autocast else None

    # ------------------------------------------------------------------ #
    #  9. LOSS FUNCTION SETUP                                              #
    # ------------------------------------------------------------------ #

    if class_conditioning:
        dice_loss_fg = smp.losses.DiceLoss(mode='multiclass', classes=[target])
        dice_loss_bg = smp.losses.DiceLoss(mode='multiclass', classes=[0, target])

        if alpha_ce > 0:
            num_classes = output.shape[0]
            ce_weight   = torch.zeros(num_classes, device=device)
            ce_weight[target] = 1.0
            ce_loss_fn  = torch.nn.CrossEntropyLoss(weight=ce_weight)
        else:
            ce_loss_fn = None
    else:
        dice_loss_fg = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
        dice_loss_bg = smp.losses.DiceLoss(mode='multiclass')
        ce_loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=0) if alpha_ce > 0 else None

    # ------------------------------------------------------------------ #
    #  10. FOREGROUND-SCALED L1 COEFFICIENT                               #
    # ------------------------------------------------------------------ #

    num_target_pixels = (masking_target == target).sum().item()
    H, W              = image.shape[-2], image.shape[-1]
    fg_fraction       = num_target_pixels / (H * W)
    l1_coeff          = l1_coeff * fg_fraction

    # ------------------------------------------------------------------ #
    #  11. PERIODIC EVALUATION SETUP                                       #
    # ------------------------------------------------------------------ #

    checkpoint_iters = sorted(set(
        [0] + [int(round(p * (iterations - 1))) for p in
               [i / 10 for i in range(1, 11)]]
    ))

    last_sufficient_cam = None
    no_improve_count    = 0

    # ------------------------------------------------------------------ #
    #  12. PRECOMPUTE LOOP CONSTANTS                                       #
    # ------------------------------------------------------------------ #

    image_batch      = image.unsqueeze(0)           # (1, C, H, W)
    x_prime_batch    = x_prime.unsqueeze(0)         # (1, C, H, W)
    needs_upsample   = (mask_H != img_H) or (mask_W != img_W)

    # ------------------------------------------------------------------ #
    #  13. OPTIMISATION LOOP                                               #
    # ------------------------------------------------------------------ #

    print(f"\n{'Iter':<8} {'Dice':<10} {'SR':<10} {'Note'}")
    print("-" * 40)

    for iteration in tqdm(range(iterations), desc='MiSuRe optimization'):

        # ---------- periodic evaluation (before gradient step) ----------
        if iteration in checkpoint_iters:
            with torch.no_grad():
                up_chk  = F.interpolate(mask.detach(), size=(img_H, img_W),
                                         mode='bilinear', align_corners=False)
                cam_chk = up_chk[0, 0].detach().cpu().numpy()

            dice_chk, sr_chk = compute_paired_metric(
                image, model, cam_chk, target,
                x_prime=x_prime,
                threshold=0,
                device=device,
                pred_original_binary=pred_orig_binary,
                pred_original_nonzero=pred_orig_nonzero,
            )
            note = ""

            if dice_chk >= min_dice_threshold:
                last_sufficient_cam = cam_chk.copy()
                note = "← saved (above threshold)"

            if dice_chk < patience_dice_threshold:
                no_improve_count += 1
                note += f" below-threshold {no_improve_count}/{patience}"
            else:
                no_improve_count = 0

            print(f"{iteration:<8} {dice_chk:<10.4f} {sr_chk:<10.4f} {note}")

            if iteration > 0 and no_improve_count >= patience:
                print(f"\nEarly stopping at iteration {iteration} "
                      f"(Dice below {patience_dice_threshold} "
                      f"for {patience} consecutive checkpoints).")
                break

        # ---------- forward pass & loss ---------------------------------

        if needs_upsample:
            upsampled_mask = F.interpolate(
                mask, size=(img_H, img_W),
                mode='bilinear', align_corners=False
            )
        else:
            upsampled_mask = mask

        upsampled_mask_expanded = upsampled_mask.expand(
            1, image.shape[0], img_H, img_W
        )

        perturbated_input = blend_with_baseline(
            image_batch,
            upsampled_mask_expanded,
            x_prime_batch,
        )

        with torch.amp.autocast(device_type=_device_type, enabled=_autocast):
            prediction_perturbed = model(perturbated_input)

            loss_ce = (alpha_ce * ce_loss_fn(prediction_perturbed, masking_target)
                       if alpha_ce > 0 else 0)

            loss_l1 = torch.mean(torch.abs(upsampled_mask))
            loss_tv = tv_norm(mask, tv_beta) if tv_coeff != 0 else 0

            loss_output = (
                alpha_fg * dice_loss_fg(prediction_perturbed, masking_target) +
                alpha_bg * dice_loss_bg(prediction_perturbed, masking_target) +
                loss_ce
            )

            perturbation_loss = l1_coeff * loss_l1 + tv_coeff * loss_tv + loss_output

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(perturbation_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            perturbation_loss.backward()
            optimizer.step()

        if iteration < iterations - 1:
            mask.data.clamp_(0, 1)

    # ------------------------------------------------------------------ #
    #  14. FINAL EVALUATION & BEST-PERFORMER FALLBACK                     #
    # ------------------------------------------------------------------ #

    with torch.no_grad():
        up_final  = F.interpolate(mask.detach(), size=(img_H, img_W),
                                   mode='bilinear', align_corners=False)
        final_cam = up_final[0, 0].detach().cpu().numpy()

    final_dice, final_sr = compute_paired_metric(
        image, model, final_cam, target,
        x_prime=x_prime,
        threshold=0,
        device=device,
        pred_original_binary=pred_orig_binary,
        pred_original_nonzero=pred_orig_nonzero,
    )
    print(f"\nFinal optimizer  →  Dice: {final_dice:.4f}  SR: {final_sr:.4f}")

    if final_dice >= min_dice_threshold:
        cam = final_cam
        print(f"Using final optimizer result "
              f"(Dice {final_dice:.4f} ≥ threshold {min_dice_threshold}).")
    else:
        cam    = last_sufficient_cam if last_sufficient_cam is not None else final_cam
        source = ("last sufficient checkpoint"
                  if last_sufficient_cam is not None
                  else "final (no sufficient checkpoint found)")
        print(f"Final Dice {final_dice:.4f} < threshold {min_dice_threshold}. "
              f"Falling back to {source}.")

    cam_nonzero = int((cam > 0).sum())
    return (
        cam,
        x_prime,
        [counter, dil_stats[1] if remove_background else 0.0,
         pred_orig_nonzero, cam_nonzero],
    )


    return best_cam, [best_threshold, raw_dice, best_dice, best_sr, cam_nonzero]