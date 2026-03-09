"""
RISE: Randomized Input Sampling for Explanation
================================================

Includes:
- generate_masks: Generate random masks for RISE
- rise_segmentation: RISE attribution for segmentation models
- rise_segmentation_batch: Batched RISE for memory efficiency
"""

import gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from medpy.metric.binary import dc


def generate_masks(n_masks, input_size, p1=0.1, initial_mask_size=(7, 7), binary=True):
    """
    Generate random masks for RISE (Randomized Input Sampling for Explanation).

    Parameters
    ----------
    n_masks : int
        Number of masks to generate.
    input_size : tuple
        Target size (H, W) for the output masks.
    p1 : float, default=0.1
        Probability of keeping a pixel (1 - probability of masking).
    initial_mask_size : tuple, default=(7, 7)
        Size of the initial low-resolution binary mask.
    binary : bool, default=True
        If True, use nearest neighbor interpolation (binary masks).
        If False, use bilinear interpolation (soft masks).

    Returns
    -------
    masks : torch.Tensor
        Generated masks with shape [N, 1, H, W].
    """
    Ch = int(np.ceil(input_size[0] / initial_mask_size[0]))
    Cw = int(np.ceil(input_size[1] / initial_mask_size[1]))
    resize_h = int((initial_mask_size[0] + 1) * Ch)
    resize_w = int((initial_mask_size[1] + 1) * Cw)
    masks = []
    for _ in range(n_masks):
        binary_mask = torch.randn(1, 1, initial_mask_size[0], initial_mask_size[1])
        binary_mask = (binary_mask < p1).float()
        if binary:
            mask = F.interpolate(binary_mask, (resize_h, resize_w), mode='nearest')
        else:
            mask = F.interpolate(binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)
        i = np.random.randint(0, Ch)
        j = np.random.randint(0, Cw)
        mask = mask[:, :, i:i+input_size[0], j:j+input_size[1]]
        masks.append(mask)
    masks = torch.cat(masks, dim=0)  # [N, 1, H, W]
    return masks


def rise_segmentation(image, model, target=None, box=None, device='cpu', 
                      n_masks=1000, p1=0.1, initial_mask_size=(7, 7), binary=True,
                      return_masks=False):
    """
    RISE (Randomized Input Sampling for Explanation) for semantic segmentation.

    RISE generates random masks, applies them to the input image, and aggregates
    the masks weighted by how much they affect the target class prediction.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int, optional
        Target class index. If None, uses the dominant class in the ROI.
    box : tuple, optional
        Bounding box ROI (y_start, y_end, x_start, x_end) for evaluation.
        If None, uses the entire image.
    device : str, default='cpu'
        Device to run the model on.
    n_masks : int, default=1000
        Number of random masks to generate.
    p1 : float, default=0.1
        Probability of keeping a pixel in the masks.
    initial_mask_size : tuple, default=(7, 7)
        Size of the initial low-resolution binary mask.
    binary : bool, default=True
        Whether to use binary masks or soft masks.
    return_masks : bool, default=False
        If True, also return the generated masks for interpretability.

    Returns
    -------
    aggregated_mask : np.ndarray
        The importance map with shape (H, W) normalized to [0, 1].
    weights : np.ndarray
        The dice scores for each mask with shape (n_masks,).
    masks : torch.Tensor, optional
        The generated masks with shape (n_masks, 1, H, W). Returned if return_masks=True.

    References
    ----------
    Petsiuk et al. "RISE: Randomized Input Sampling for Explanation of Black-box Models"
    https://arxiv.org/abs/1806.07421
    """
    image = image.to(device)
    input_size = (image.shape[1], image.shape[2])
    
    masks = generate_masks(n_masks, input_size, p1, initial_mask_size, binary)

    # Baseline prediction on unmasked image
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

    aggregated_mask = np.zeros(masks[0][0].shape)
    weights = []  # Store dice scores for each mask

    for mask in tqdm(masks):
        masked_input = image * mask.to(device)
        with torch.no_grad():
            output_masked = model(masked_input.unsqueeze(0))[0].detach().cpu()

        output_b = (output_masked.argmax(0).numpy() == target)[y_start:y_end, x_start:x_end]

        dice_score = dc(output_b, output_a)
        weights.append(dice_score)
        aggregated_mask += mask[0].cpu().numpy() * dice_score
        
        # Clear memory after each mask
        del masked_input, output_masked
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Normalize to [0, 1]
    min_, max_ = aggregated_mask.min(), aggregated_mask.max()
    aggregated_mask = (aggregated_mask - min_) / (max_ - min_ + 1e-8)

    weights = np.array(weights)
    
    if return_masks:
        return aggregated_mask, weights, masks
    return aggregated_mask, weights


def rise_segmentation_batch(image, model, target=None, box=None, device='cpu', 
                      n_masks=1000, p1=0.1, initial_mask_size=(7, 7), binary=True, 
                      batch_size=16, return_masks=False):
    """
    Batched RISE for semantic segmentation (memory-efficient).

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int, optional
        Target class index.
    box : tuple, optional
        Bounding box ROI.
    device : str, default='cpu'
        Device to run on.
    n_masks : int, default=1000
        Number of random masks.
    p1 : float, default=0.1
        Probability of keeping a pixel.
    initial_mask_size : tuple, default=(7, 7)
        Size of initial low-res mask.
    binary : bool, default=True
        Binary or soft masks.
    batch_size : int, default=16
        Batch size for processing masks.
    return_masks : bool, default=False
        If True, also return the generated masks.

    Returns
    -------
    aggregated_mask : np.ndarray
        Importance map [H, W].
    weights : np.ndarray
        Dice scores [n_masks,].
    masks : torch.Tensor, optional
        Generated masks [n_masks, 1, H, W] if return_masks=True.
    """
    image = image.to(device)
    input_size = (image.shape[1], image.shape[2])
    
    masks = generate_masks(n_masks, input_size, p1, initial_mask_size, binary)

    with torch.no_grad():
        output = model(image.unsqueeze(0))[0].detach().cpu()

    predicted_mask = output.argmax(0).numpy()

    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[1], 0, image.shape[2]
    else:
        y_start, y_end, x_start, x_end = box

    if target is None:
        target = predicted_mask[y_start:y_end, x_start:x_end].max().item()

    output_a = (predicted_mask == target)[y_start:y_end, x_start:x_end]
    aggregated_mask = np.zeros(masks[0][0].shape)
    weights = []  # Store dice scores for each mask

    for i in tqdm(range(0, n_masks, batch_size)):
        batch_masks = masks[i:i + batch_size].to(device)  # [B, 1, H, W]
        # image is [C, H, W], batch_masks is [B, 1, H, W] -> broadcasts to [B, C, H, W]
        masked_inputs = image.unsqueeze(0) * batch_masks

        with torch.no_grad():
            outputs = model(masked_inputs)  # [B, num_classes, H, W]

        for j, mask in enumerate(batch_masks):
            output_b = (outputs[j].argmax(0).cpu().numpy() == target)[y_start:y_end, x_start:x_end]
            dice_score = dc(output_b, output_a)
            weights.append(dice_score)
            aggregated_mask += mask[0].cpu().numpy() * dice_score
        
        # Clear memory after each batch
        del outputs, masked_inputs, batch_masks
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    min_, max_ = aggregated_mask.min(), aggregated_mask.max()
    aggregated_mask = (aggregated_mask - min_) / (max_ - min_ + 1e-8)

    weights = np.array(weights)
    
    if return_masks:
        return aggregated_mask, weights, masks
    return aggregated_mask, weights
