import cv2
import numpy as np
import torch
from medpy.metric.binary import dc


def get_baseline(image, mode='zero', mean=(0., 0., 0.), std=(1., 1., 1.), blur_sigma=51):
    """
    Compute the baseline image x_prime used to fill masked-out regions.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W). Should be the original,
        pre-dilation image so blur is not computed on a zeroed image.
    mode : str, default='zero'
        'zero' → constant tensor representing a normalised black pixel,
                 computed as (0 - mean) / std per channel.
        'blur' → heavily Gaussian-blurred version of the image.
    mean : tuple of float, default=(0., 0., 0.)
        Per-channel normalisation mean (used only for mode='zero').
    std : tuple of float, default=(1., 1., 1.)
        Per-channel normalisation std  (used only for mode='zero').
    blur_sigma : float, default=51
        Gaussian blur sigma (used only for mode='blur').

    Returns
    -------
    x_prime : torch.Tensor
        Baseline tensor with shape (C, H, W), on the same device as image.
    """
    if mode == 'zero':
        mean_t  = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_t   = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)
        x_prime = (-mean_t / std_t).expand_as(image).clone()

    elif mode == 'blur':
        img_np  = image.detach().cpu().numpy()          # (C, H, W)
        blurred = np.stack([
            cv2.GaussianBlur(img_np[c], (0, 0), blur_sigma)
            for c in range(img_np.shape[0])
        ], axis=0)
        # Zero-mask: wherever ALL channels of the original are 0 (true background,
        # e.g. CT scan borders), force the blurred baseline to 0 as well.
        # This prevents blur from leaking signal into empty background regions.
        zero_mask = (img_np.sum(axis=0, keepdims=True) == 0)   # (1, H, W) bool
        blurred[zero_mask.repeat(img_np.shape[0], axis=0)] = 0.
        x_prime = torch.tensor(blurred, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown baseline mode '{mode}'. Choose 'zero' or 'blur'.")

    return x_prime.to(image.device)


def blend_with_baseline(image, mask, x_prime):
    """
    Replace masked-out regions with baseline x_prime.

    Computes: image * mask + (1 - mask) * x_prime

    Parameters
    ----------
    image : torch.Tensor
        Shape (C, H, W) or (1, C, H, W).
    mask : torch.Tensor
        Shape broadcast-compatible with image, e.g. (1, H, W) or (1, C, H, W).
    x_prime : torch.Tensor
        Same shape as image.

    Returns
    -------
    torch.Tensor
        Blended tensor, same shape as image.
    """
    return image * mask + (1.0 - mask) * x_prime


def apply_heatmap_mask(image, heatmap, x_prime, threshold=0):
    """
    Apply a heatmap as a binary mask to an image, filling masked-out
    regions with x_prime instead of zero.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    heatmap : np.ndarray or torch.Tensor
        Attribution heatmap with shape (H, W), values in [0, 1].
    x_prime : torch.Tensor
        Baseline tensor with shape (C, H, W), same device as image.
    threshold : float, default=0
        Threshold value. Heatmap is binarized: (heatmap > threshold).

    Returns
    -------
    masked_image : torch.Tensor
        Blended image with same shape as input (C, H, W).
    binary_mask : np.ndarray
        The binary mask used, shape (H, W).
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    device      = image.device
    image_cpu   = image.detach().cpu()
    x_prime_cpu = x_prime.detach().cpu()

    binary_mask   = (heatmap > threshold).astype(np.float32)
    mask_expanded = torch.from_numpy(binary_mask).unsqueeze(0).float()    # (1, H, W)

    masked_image = blend_with_baseline(image_cpu, mask_expanded, x_prime_cpu)

    return masked_image.to(device), binary_mask


def compute_paired_metric(image, model, heatmap, target, x_prime,
                           threshold=0, device='cpu',
                           pred_original_binary=None,
                           pred_original_nonzero=None):
    """
    Compute paired XAI metrics: Dice Explained and Saliency Ratio.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    heatmap : np.ndarray
        Attribution heatmap with shape (H, W).
    target : int
        Target class index.
    x_prime : torch.Tensor
        Baseline tensor with shape (C, H, W).
    threshold : float, default=0
        Threshold for binarizing heatmap: (heatmap > threshold).
    device : str, default='cpu'
        Device to run the model on.
    pred_original_binary : np.ndarray or None
        Precomputed binary mask of original prediction for target class,
        shape (H, W), dtype uint8. If provided, skips the original forward pass.
    pred_original_nonzero : int or None
        Precomputed sum of pred_original_binary. If provided alongside
        pred_original_binary, avoids recomputing it.

    Returns
    -------
    dice_explained : float
    saliency_ratio : float
    """
    image   = image.to(device)
    x_prime = x_prime.to(device)
    _device_type = device.split(':')[0]

    model.eval()

    # --- original prediction (skipped if precomputed) ---
    if pred_original_binary is None:
        with torch.no_grad():
            with torch.amp.autocast(device_type=_device_type, enabled=(_device_type != 'cpu')):
                output_orig = model(image.unsqueeze(0))[0].detach().cpu()
        pred_original_binary = (output_orig.argmax(0).numpy() == target).astype(np.uint8)

    if pred_original_nonzero is None:
        pred_original_nonzero = int(pred_original_binary.sum())

    # --- masked prediction ---
    masked_image, binary_mask = apply_heatmap_mask(
        image, heatmap, x_prime, threshold=threshold
    )

    with torch.no_grad():
        with torch.amp.autocast(device_type=_device_type, enabled=(_device_type != 'cpu')):
            output_masked = model(masked_image.unsqueeze(0))[0].detach().cpu()
    pred_masked_binary = (output_masked.argmax(0).numpy() == target).astype(np.uint8)

    # --- dice ---
    if pred_original_nonzero == 0 and pred_masked_binary.sum() == 0:
        dice_explained = 1.0
    elif pred_original_nonzero == 0 or pred_masked_binary.sum() == 0:
        dice_explained = 0.0
    else:
        dice_explained = dc(pred_masked_binary, pred_original_binary)

    # --- saliency ratio ---
    heatmap_nonzero = binary_mask.sum()
    if pred_original_nonzero == 0:
        saliency_ratio = float('inf') if heatmap_nonzero > 0 else 0.0
    else:
        saliency_ratio = heatmap_nonzero / pred_original_nonzero

    return dice_explained, saliency_ratio


def compute_paired_metric_batch(image, model, heatmaps, target, x_prime,
                                 threshold=0, device='cpu',
                                 pred_original_binary=None,
                                 pred_original_nonzero=None,
                                 sweep_batch_size=4):
    """
    Batched version of compute_paired_metric over multiple heatmaps.
    Runs model inference in chunks to avoid OOM.

    Parameters
    ----------
    image : torch.Tensor
        Shape (C, H, W).
    model : torch.nn.Module
    heatmaps : list of np.ndarray
        Each with shape (H, W).
    target : int
    x_prime : torch.Tensor
        Shape (C, H, W).
    threshold : float, default=0
    device : str, default='cpu'
    pred_original_binary : np.ndarray or None
        Precomputed binary mask of original prediction. Skips original
        forward pass if provided.
    pred_original_nonzero : int or None
    sweep_batch_size : int, default=4
        Number of masked images to run through the model at once.
        Reduce if OOM errors occur.

    Returns
    -------
    results : list of (dice_explained, saliency_ratio) tuples, one per heatmap.
    """
    image        = image.to(device)
    x_prime      = x_prime.to(device)
    _device_type = device.split(':')[0]

    model.eval()

    # --- original prediction (skipped if precomputed) ---
    if pred_original_binary is None:
        with torch.no_grad():
            with torch.amp.autocast(device_type=_device_type, enabled=(_device_type != 'cpu')):
                output_orig = model(image.unsqueeze(0))[0].detach().cpu()
        pred_original_binary = (output_orig.argmax(0).numpy() == target).astype(np.uint8)

    if pred_original_nonzero is None:
        pred_original_nonzero = int(pred_original_binary.sum())

    # --- build all masked images and binary masks ---
    masked_images = []
    binary_masks  = []
    for heatmap in heatmaps:
        masked_img, bin_mask = apply_heatmap_mask(
            image, heatmap, x_prime, threshold=threshold
        )
        masked_images.append(masked_img)
        binary_masks.append(bin_mask)

    # --- run model in chunks ---
    all_preds = []
    for start in range(0, len(masked_images), sweep_batch_size):
        chunk = torch.stack(
            masked_images[start:start + sweep_batch_size], dim=0
        ).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=_device_type, enabled=(_device_type != 'cpu')):
                out = model(chunk).detach().cpu()           # (B, C, H, W)
        all_preds.append(out.argmax(1).numpy())             # (B, H, W)

    all_preds = np.concatenate(all_preds, axis=0)           # (N, H, W)

    # --- compute metrics per heatmap ---
    results = []
    for pred_masked, bin_mask in zip(all_preds, binary_masks):
        pred_masked_binary = (pred_masked == target).astype(np.uint8)

        if pred_original_nonzero == 0 and pred_masked_binary.sum() == 0:
            dice_explained = 1.0
        elif pred_original_nonzero == 0 or pred_masked_binary.sum() == 0:
            dice_explained = 0.0
        else:
            dice_explained = dc(pred_masked_binary, pred_original_binary)

        heatmap_nonzero = bin_mask.sum()
        if pred_original_nonzero == 0:
            saliency_ratio = float('inf') if heatmap_nonzero > 0 else 0.0
        else:
            saliency_ratio = heatmap_nonzero / pred_original_nonzero

        results.append((dice_explained, saliency_ratio))

    return results

# ======================================================================= #
#  THRESHOLD SWEEP                                                          #
# ======================================================================= #

def sweep_thresholds(cam, image, model, target,
                     x_prime=None,
                     thresholds=(0.0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
                     sweep_tolerance=0.2,
                     threshold=0,
                     device='cpu',
                     sweep_batch_size=4,
                     baseline_mode='zero',
                     baseline_mean=(0, 0, 0),
                     baseline_std=(1, 1, 1),
                     baseline_blur_sigma=51,
                     pred_original_binary=None,
                     pred_original_nonzero=None):
    """
    Post-hoc threshold sweep over any heatmap (MiSuRe, RISE, Seg-Grad-CAM, …).

    For each candidate threshold the heatmap is binarised at that value and
    the resulting masked image is evaluated with the model.  The highest
    threshold whose Dice is still within sweep_tolerance of the raw (t=0)
    Dice is selected as the final explanation.

    Compatible with any method that returns an (H, W) numpy heatmap in [0, 1]:
    - misure_1          → cam             (element 0 of return tuple)
    - rise_segmentation → aggregated_mask
    - seg_grad_cam      → grayscale_cam

    Parameters
    ----------
    cam : np.ndarray
        Raw heatmap, shape (H, W), values in [0, 1].
    image : torch.Tensor
        Input image tensor, shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int
        Target class index.
    x_prime : torch.Tensor or None, default=None
        Baseline tensor used to fill masked-out pixels during metric evaluation.
        If None, computed automatically from image using baseline_mode/mean/std/blur_sigma.
        For MiSuRe, pass the exact x_prime returned by misure_1.
    thresholds : tuple of float
        Candidate thresholds to evaluate, in any order.
    sweep_tolerance : float, default=0.2
        Maximum allowed Dice drop from raw_dice when selecting a threshold.
    threshold : float, default=0
        Threshold passed to compute_paired_metric_batch for mask binarisation
        inside the metric (normally 0 — cam is already zeroed below each t).
    device : str, default='cpu'
        Device to run model inference on.
    sweep_batch_size : int, default=4
        Batch size for model inference inside compute_paired_metric_batch.
    baseline_mode : str, default='zero'
        Passed to get_baseline when x_prime is None. 'zero' or 'blur'.
    baseline_mean, baseline_std : tuple of float
        Normalisation parameters for get_baseline (used when x_prime is None).
    baseline_blur_sigma : int, default=51
        Blur sigma for get_baseline when baseline_mode='blur' and x_prime is None.
    pred_original_binary : np.ndarray or None, default=None
        Precomputed binary prediction on the original image (H, W), dtype uint8.
        If provided (with pred_original_nonzero), skips a model forward pass.
    pred_original_nonzero : int or None, default=None
        Precomputed count of non-zero pixels in pred_original_binary.

    Returns
    -------
    best_cam : np.ndarray
        Thresholded heatmap at the selected threshold, shape (H, W).
    stats : list
        [best_threshold, raw_dice, best_dice, best_sr, cam_nonzero]
        best_threshold  — the selected threshold value
        raw_dice        — Dice at threshold=0 (the reference)
        best_dice       — Dice at the selected threshold
        best_sr         — Saliency Ratio at the selected threshold
        cam_nonzero     — number of non-zero pixels in best_cam
    """
    image = image.to(device)

    if x_prime is None:
        x_prime = get_baseline(image.cpu(), mode=baseline_mode,
                               mean=baseline_mean, std=baseline_std,
                               blur_sigma=baseline_blur_sigma)
    x_prime = x_prime.to(device)

    _device_type = device.split(':')[0]
    _autocast    = _device_type != 'cpu'
    if pred_original_binary is None or pred_original_nonzero is None:
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=_device_type, enabled=_autocast):
                output_orig = model(image.unsqueeze(0))[0].detach().cpu()
        pred_original_binary  = (output_orig.argmax(0).numpy() == target).astype(np.uint8)
        pred_original_nonzero = int(pred_original_binary.sum())

    sorted_t = sorted(thresholds)

    cams_sweep = []
    for t in sorted_t:
        cam_ = cam.copy()
        cam_[cam_ < t] = 0
        cams_sweep.append(cam_)

    sweep_results = compute_paired_metric_batch(
        image, model, cams_sweep, target,
        x_prime=x_prime,
        threshold=threshold,
        device=device,
        pred_original_binary=pred_original_binary,
        pred_original_nonzero=pred_original_nonzero,
        sweep_batch_size=sweep_batch_size,
    )

    best_cam       = cam.copy()
    best_threshold = sorted_t[0]
    best_dice      = sweep_results[0][0]
    best_sr        = sweep_results[0][1]
    raw_dice       = sweep_results[0][0]

    print(f"\n{'Threshold':<12} {'Dice':<8} {'SR'}")
    print("-" * 28)

    for t, cam_, (dice_t, sr_t) in zip(sorted_t, cams_sweep, sweep_results):
        print(f"{t:<12} {dice_t:.4f}   {sr_t:.4f}")
        if dice_t >= raw_dice - sweep_tolerance:
            best_threshold = t
            best_cam       = cam_
            best_dice      = dice_t
            best_sr        = sr_t

    print(f"\nBest threshold : {best_threshold}")
    print(f"Raw Dice       : {raw_dice:.4f}")
    print(f"Floor          : {raw_dice - sweep_tolerance:.4f}")

    cam_nonzero = int((best_cam > 0).sum())
    return best_cam, [best_threshold, raw_dice, best_dice, best_sr, cam_nonzero]
