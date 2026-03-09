"""
XAI (Explainable AI) Metrics for Segmentation
===============================================

Metrics for evaluating the quality of attribution methods.
"""

import gc
import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import dc
from tqdm import tqdm


def pixel_deletion_curve(image, model, heatmap, target=None, box=None, step_size=100,
                         batch_size=8, device='cpu'):
    """
    Compute the pixel deletion curve (also known as "most relevant first" or MoRF).
    
    Progressively deletes the most important pixels (according to the heatmap)
    and measures how quickly the model's performance degrades.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    heatmap : np.ndarray
        Attribution heatmap with shape (H, W). Higher values indicate more important pixels.
    target : int, optional
        Target class index. If None, uses the dominant class in the ROI.
    box : tuple, optional
        Bounding box ROI (y_start, y_end, x_start, x_end) for evaluation.
        If None, uses the entire image.
    step_size : int, default=100
        Number of pixels to delete at each step.
    batch_size : int, default=8
        Batch size for processing multiple modified images at once.
    device : str, default='cpu'
        Device to run the model on.
    
    Returns
    -------
    dice_scores : np.ndarray
        Array of Dice scores after each deletion step.
    
    Notes
    -----
    - A good attribution method should show a rapid drop in Dice score as important pixels are deleted.
    - The heatmap should have the same spatial dimensions as the input image.
    - Pixels are deleted by setting them to 0 (black).
    
    References
    ----------
    Samek et al. "Evaluating the Visualization of What a Deep Neural Network Has Learned"
    https://arxiv.org/abs/1509.06321
    """
    image = image.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image.unsqueeze(0))[0].detach().cpu()

    predicted_mask = output.argmax(0).numpy()  # [H, W]

    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[1], 0, image.shape[2]
    else:
        y_start, y_end, x_start, x_end = box

    if target is None:
        target = predicted_mask[y_start:y_end, x_start:x_end].max().item()

    output_a = (predicted_mask == target)[y_start:y_end, x_start:x_end]

    # Rank pixels by importance descending (most important first)
    flat_indices = np.argsort(heatmap.flatten())[::-1]

    # Pre-compute all row/col indices per step
    all_rows, all_cols = [], []
    for step in range(0, len(flat_indices), step_size):
        indices = flat_indices[step:step + step_size]
        rows, cols = np.unravel_index(indices, heatmap.shape)
        all_rows.append(rows)
        all_cols.append(cols)

    n_steps = len(all_rows)
    modified_image = image.clone()
    dice_scores = []
    batch = []

    for step_idx in tqdm(range(n_steps), desc='Pixel Deletion'):
        modified_image[:, all_rows[step_idx], all_cols[step_idx]] = 0
        batch.append(modified_image.clone())

        if len(batch) == batch_size or step_idx == n_steps - 1:
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)

            for output_modified in outputs:
                pred = output_modified.argmax(0).cpu().numpy()
                output_b = (pred == target)[y_start:y_end, x_start:x_end]
                dice_score = dc(output_b, output_a)
                dice_scores.append(dice_score)

            # Clear memory after each batch
            del outputs, batch_tensor
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

            batch = []

    return np.array(dice_scores)


def pixel_insertion_curve(image, model, heatmap, target=None, box=None, step_size=100,
                          baseline='black', batch_size=8, device='cpu'):
    """
    Compute the pixel insertion curve (also known as "least relevant first" or LeRF).
    
    Starts with a blurred/black image and progressively inserts the most important 
    pixels (according to the heatmap), measuring how quickly the model recovers.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    heatmap : np.ndarray
        Attribution heatmap with shape (H, W). Higher values indicate more important pixels.
    target : int, optional
        Target class index. If None, uses the dominant class in the ROI.
    box : tuple, optional
        Bounding box ROI (y_start, y_end, x_start, x_end) for evaluation.
        If None, uses the entire image.
    step_size : int, default=100
        Number of pixels to insert at each step.
    baseline : str or torch.Tensor, default='black'
        Baseline image to start from. 'black' for zeros, 'blur' for Gaussian blur,
        or a custom tensor.
    batch_size : int, default=8
        Batch size for processing multiple modified images at once.
    device : str, default='cpu'
        Device to run the model on.
    
    Returns
    -------
    dice_scores : np.ndarray
        Array of Dice scores after each insertion step.
    
    Notes
    -----
    - A good attribution method should show a rapid increase in Dice score as important pixels are inserted.
    - The heatmap should have the same spatial dimensions as the input image.
    
    References
    ----------
    Samek et al. "Evaluating the Visualization of What a Deep Neural Network Has Learned"
    https://arxiv.org/abs/1509.06321
    """
    image = image.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image.unsqueeze(0))[0].detach().cpu()

    predicted_mask = output.argmax(0).numpy()  # [H, W]

    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[1], 0, image.shape[2]
    else:
        y_start, y_end, x_start, x_end = box

    if target is None:
        target = predicted_mask[y_start:y_end, x_start:x_end].max().item()

    output_a = (predicted_mask == target)[y_start:y_end, x_start:x_end]

    # Create baseline image
    if baseline == 'black':
        modified_image = torch.zeros_like(image)
    elif baseline == 'blur':
        blurred = F.avg_pool2d(image.unsqueeze(0), kernel_size=15, stride=1, padding=7)
        modified_image = blurred[0]
    elif isinstance(baseline, torch.Tensor):
        modified_image = baseline.clone().to(device)
    else:
        modified_image = torch.zeros_like(image)

    # Rank pixels by importance descending (most important first)
    flat_indices = np.argsort(heatmap.flatten())[::-1]

    # Pre-compute all row/col indices per step
    all_rows, all_cols = [], []
    for step in range(0, len(flat_indices), step_size):
        indices = flat_indices[step:step + step_size]
        rows, cols = np.unravel_index(indices, heatmap.shape)
        all_rows.append(rows)
        all_cols.append(cols)

    n_steps = len(all_rows)
    dice_scores = []
    batch = []

    for step_idx in tqdm(range(n_steps), desc='Pixel Insertion'):
        modified_image[:, all_rows[step_idx], all_cols[step_idx]] = image[:, all_rows[step_idx], all_cols[step_idx]]
        batch.append(modified_image.clone())

        if len(batch) == batch_size or step_idx == n_steps - 1:
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)

            for output_modified in outputs:
                pred = output_modified.argmax(0).cpu().numpy()
                output_b = (pred == target)[y_start:y_end, x_start:x_end]
                dice_score = dc(output_b, output_a)
                dice_scores.append(dice_score)

            # Clear memory after each batch
            del outputs, batch_tensor
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

            batch = []

    return np.array(dice_scores)


def compute_auc(scores):
    """
    Compute the Area Under the Curve (AUC) for deletion/insertion curves.
    
    Parameters
    ----------
    scores : np.ndarray
        Array of scores (e.g., Dice scores from deletion or insertion curve).
    
    Returns
    -------
    auc : float
        Area under the curve, normalized by the number of steps.
        Lower is better for deletion, higher is better for insertion.
    """
    return np.trapz(scores) / len(scores)


def apply_heatmap_mask(image, heatmap, threshold=0):
    """
    Apply a heatmap as a binary mask to an image.
    
    Binarizes the heatmap using threshold and performs element-wise
    multiplication with the image.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    heatmap : np.ndarray
        Attribution heatmap with shape (H, W), values in [0, 1].
    threshold : float, default=0
        Threshold value. Heatmap is binarized: (heatmap > threshold).
    
    Returns
    -------
    masked_image : torch.Tensor
        Masked image with same shape as input (C, H, W).
    binary_mask : np.ndarray
        The binary mask used.
    """
    # Ensure heatmap is numpy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    
    # Ensure image is on CPU for processing
    device = image.device
    image_cpu = image.detach().cpu()
    
    # Always use binary mask
    binary_mask = (heatmap > threshold).astype(np.float32)
    
    # Expand mask to match image channels
    # mask is (H, W), image is (C, H, W)
    mask_expanded = torch.from_numpy(binary_mask).unsqueeze(0).float()  # (1, H, W)
    mask_expanded = mask_expanded.expand_as(image_cpu)  # (C, H, W)
    
    # Element-wise multiplication
    masked_image = image_cpu * mask_expanded
    
    # Move back to original device
    masked_image = masked_image.to(device)
    
    return masked_image, binary_mask


def compute_paired_metric(image, model, heatmap, target, threshold=0, device='cpu'):
    """
    Compute paired XAI metrics: Dice Explained and Saliency Ratio.
    
    Dice Explained: Dice between prediction on original image and prediction on
                    heatmap-masked image.
    Saliency Ratio: Ratio of non-zero pixels in thresholded heatmap
                    to non-zero pixels in the prediction for the target class.
    
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
    threshold : float, default=0
        Threshold for binarizing heatmap: (heatmap > threshold).
    device : str, default='cpu'
        Device to run the model on.
    
    Returns
    -------
    dice_explained : float
        Dice score between original prediction and masked prediction.
    saliency_ratio : float
        Ratio of heatmap non-zero pixels to prediction non-zero pixels.
    """
    image = image.to(device)
    model = model.to(device)
    model.eval()
    
    # Get original prediction
    with torch.no_grad():
        output_orig = model(image.unsqueeze(0))[0].detach().cpu()
    pred_original = output_orig.argmax(0).numpy()  # (H, W)
    
    # Create masked image using heatmap, get the binary mask used
    masked_image, binary_mask = apply_heatmap_mask(image, heatmap, threshold=threshold)
    
    # Get prediction on masked image
    with torch.no_grad():
        output_masked = model(masked_image.unsqueeze(0))[0].detach().cpu()
    pred_masked = output_masked.argmax(0).numpy()  # (H, W)
    
    # Compute Dice Explained: Dice between original and masked predictions
    pred_original_binary = (pred_original == target).astype(np.uint8)
    pred_masked_binary = (pred_masked == target).astype(np.uint8)
    
    # Avoid division by zero
    if pred_original_binary.sum() == 0 and pred_masked_binary.sum() == 0:
        dice_explained = 1.0  # Both empty = perfect match
    elif pred_original_binary.sum() == 0 or pred_masked_binary.sum() == 0:
        dice_explained = 0.0  # One empty, one not = no match
    else:
        dice_explained = dc(pred_masked_binary, pred_original_binary)
    
    # Compute Saliency Ratio: non-zero heatmap pixels / non-zero prediction pixels
    # Always use binary mask (heatmap > threshold)
    heatmap_nonzero = binary_mask.sum()
    
    # Count non-zero in original prediction for target class
    pred_nonzero = pred_original_binary.sum()
    
    if pred_nonzero == 0:
        saliency_ratio = float('inf') if heatmap_nonzero > 0 else 0.0
    else:
        saliency_ratio = heatmap_nonzero / pred_nonzero
    
    return dice_explained, saliency_ratio
