"""
CAM: Class Activation Mapping for Segmentation Models
======================================================

Includes:
- seg_grad_cam: Seg-Grad-CAM and Seg-XRes-CAM attribution
- ablation_cam: Ablation-CAM attribution (non-batched)
- ablation_cam_batch: Ablation-CAM attribution (batched)
"""

import gc
import numpy as np
import torch
import cv2
from tqdm import tqdm
import skimage.measure
import skimage.transform


def seg_grad_cam(image, model, target=None, target_layer=None, box=None, device='cpu', 
                 method='seg-grad-cam', pool_size=None, pool_mode=np.max, 
                 reshape_transformer=False, return_activation_maps=False):
    """
    Seg-Grad-CAM and Seg-XRes-CAM for semantic segmentation models.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int, optional
        Target class index. If None, uses the dominant predicted class.
    target_layer : nn.Module
        The layer to compute activations/gradients from.
    box : tuple, optional
        Bounding box ROI (y_start, y_end, x_start, x_end).
    device : str, default='cpu'
        Device to run the model on.
    method : str, default='seg-grad-cam'
        Attribution method: 'seg-grad-cam' or 'seg-xres-cam'.
    pool_size : int, optional
        Pooling size for Seg-XRes-CAM.
    pool_mode : callable, default=np.max
        Pooling function for Seg-XRes-CAM.
    reshape_transformer : bool, default=False
        Whether to reshape transformer activations from [seq_len, hidden_dim] to [C, H, W].
    return_activation_maps : bool, default=False
        If True, also return the activation maps.

    Returns
    -------
    grayscale_cam : np.ndarray
        The computed CAM with shape (H, W) normalized to [0, 1].
    weights : np.ndarray or None
        The weights used (only for seg-grad-cam method).
    activation_maps : np.ndarray, optional
        The activation maps with shape [C, H, W]. Returned if return_activation_maps=True.
    """
    activations = []
    gradients = []

    handle_1 = target_layer.register_forward_hook(
        lambda x, y, z: activations.append(z)
    )
    handle_2 = target_layer.register_full_backward_hook(
        lambda module, grad_input, grad_output: gradients.append(grad_output[0])
    )

    model.zero_grad()
    output = model(image.unsqueeze(0).to(device))

    # Default target: dominant predicted class
    if target is None:
        target = output[0].argmax(0).max().item()

    # Segmentation mask for target class
    mask = output[0].argmax(0).detach().cpu().numpy()
    mask_float = np.float32(mask == target)

    # Apply bounding box ROI if provided
    if box is not None:
        y_start, y_end, x_start, x_end = box
        roi_mask = np.zeros_like(mask_float)
        roi_mask[y_start:y_end, x_start:x_end] = 1
        mask_float = mask_float * roi_mask

    loss = (output[0, target, :, :] * torch.tensor(mask_float).to(device)).sum()
    loss.backward()

    handle_1.remove()
    handle_2.remove()

    act = activations[0][0].detach().cpu().numpy()  # [C, H, W]
    grad = gradients[0][0].detach().cpu().numpy()   # [C, H, W]

    # Reshape transformer activations if needed
    if reshape_transformer:
        # act shape: [seq_len, hidden_dim] or [seq_len+1, hidden_dim]
        seq_len = act.shape[0]
        hidden_dim = act.shape[1]
        # Drop CLS token if seq_len is odd (e.g. 197 -> 196)
        if seq_len % 2 != 0:
            act = act[1:]
            grad = grad[1:]
            seq_len -= 1
        spatial = int(np.sqrt(seq_len))
        act = act.reshape(spatial, spatial, hidden_dim).transpose(2, 0, 1)   # [C, H, W]
        grad = grad.reshape(spatial, spatial, hidden_dim).transpose(2, 0, 1) # [C, H, W]

    if method == 'seg-grad-cam':
        weights = grad.mean(axis=(1, 2))          # GAP over spatial dims [C]
        grayscale_cam = (weights[:, None, None] * act).sum(axis=0)

    elif method == 'seg-xres-cam':
        weights = None
        if pool_size is not None:
            pooled = skimage.measure.block_reduce(grad, (1, pool_size, pool_size), pool_mode)
            pooled = np.transpose(pooled, (1, 2, 0))
            grad = skimage.transform.resize(pooled, (act.shape[1], act.shape[2]), order=0)
            grad = np.transpose(grad, (2, 0, 1))
        grayscale_cam = (grad * act).sum(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'seg-grad-cam' or 'seg-xres-cam'.")

    # ReLU
    grayscale_cam = np.maximum(grayscale_cam, 0)

    # Resize to image spatial dims
    image_h, image_w = image.shape[1], image.shape[2]
    grayscale_cam = cv2.resize(grayscale_cam, (image_w, image_h))

    # Normalize to [0, 1]
    min_, max_ = grayscale_cam.min(), grayscale_cam.max()
    grayscale_cam = (grayscale_cam - min_) / (max_ - min_ + 1e-8)

    # Output: per-pixel predicted class map
    predicted_mask = mask  # [H, W] with class indices

    weights_out = weights if method == 'seg-grad-cam' else None
    
    if return_activation_maps:
        return grayscale_cam, weights_out, act
    return grayscale_cam, weights_out


def ablation_cam(image, model, target=None, target_layer=None, box=None, device='cpu',
                 reshape_transformer=False, return_activation_maps=False):
    """
    Ablation-CAM for semantic segmentation models (simple, non-batched version).

    Instead of using gradients, weights each channel by how much the prediction
    drops when that channel is zeroed out. Processes one channel at a time.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int, optional
        Target class index. If None, uses the dominant predicted class.
    target_layer : nn.Module
        The layer to compute activations from.
    box : tuple, optional
        Bounding box ROI (y_start, y_end, x_start, x_end).
    device : str, default='cpu'
        Device to run the model on.
    reshape_transformer : bool, default=False
        Whether to reshape transformer activations from [seq_len, hidden_dim] to [C, H, W].
    return_activation_maps : bool, default=False
        If True, also return the activation maps.

    Returns
    -------
    grayscale_cam : np.ndarray
        The computed CAM with shape (H, W) normalized to [0, 1].
    weights : np.ndarray
        Per-channel importance weights (score drop per channel).
    activation_maps : np.ndarray, optional
        The activation maps with shape [C, H, W]. Returned if return_activation_maps=True.
    """
    activations = []

    handle = target_layer.register_forward_hook(
        lambda x, y, z: activations.append(z)
    )

    # Baseline forward pass
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))

    handle.remove()

    # Default target: dominant predicted class
    if target is None:
        target = output[0].argmax(0).max().item()

    predicted_mask = output[0].argmax(0).detach().cpu().numpy()  # [H, W]
    mask_float = np.float32(predicted_mask == target)

    # Apply bounding box ROI if provided
    if box is not None:
        y_start, y_end, x_start, x_end = box
        roi_mask = np.zeros_like(mask_float)
        roi_mask[y_start:y_end, x_start:x_end] = 1
        mask_float = mask_float * roi_mask

    mask_tensor = torch.tensor(mask_float).to(device)

    # Baseline score
    baseline_score = (output[0, target, :, :] * mask_tensor).sum().item()

    act = activations[0].detach()  # [1, C, H, W]

    # Reshape transformer activations if needed
    if reshape_transformer:
        seq_len = act.shape[1]
        hidden_dim = act.shape[2]
        if seq_len % 2 != 0:
            act = act[:, 1:, :]
            seq_len -= 1
        spatial = int(np.sqrt(seq_len))
        act = act.reshape(1, spatial, spatial, hidden_dim).permute(0, 3, 1, 2)  # [1, C, H, W]

    n_channels = act.shape[1]
    channel_weights = np.zeros(n_channels)

    # Process one channel at a time
    for c in tqdm(range(n_channels), desc='Ablation-CAM'):
        # Create ablated activation (zero out channel c)
        act_ablated = act.clone()
        act_ablated[0, c] = 0

        # Inject ablated activation via hook
        def ablation_hook(module, input, output, a=act_ablated):
            return a

        h = target_layer.register_forward_hook(ablation_hook)
        with torch.no_grad():
            output_ablated = model(image.unsqueeze(0).to(device))
        h.remove()

        score = (output_ablated[0, target, :, :] * mask_tensor).sum().item()
        channel_weights[c] = baseline_score - score
        
        # Clear memory
        del output_ablated, act_ablated, h
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    act_np = act[0].cpu().numpy()  # [C, H, W]

    # Weighted sum of activations
    grayscale_cam = (channel_weights[:, None, None] * act_np).sum(axis=0)

    # ReLU
    grayscale_cam = np.maximum(grayscale_cam, 0)

    # Resize to image spatial dims
    image_h, image_w = image.shape[1], image.shape[2]
    grayscale_cam = cv2.resize(grayscale_cam, (image_w, image_h))

    # Normalize to [0, 1]
    min_, max_ = grayscale_cam.min(), grayscale_cam.max()
    grayscale_cam = (grayscale_cam - min_) / (max_ - min_ + 1e-8)

    if return_activation_maps:
        return grayscale_cam, channel_weights, act_np
    return grayscale_cam, channel_weights


def ablation_cam_batch(image, model, target=None, target_layer=None, box=None, device='cpu',
                       reshape_transformer=False, batch_size=16, return_activation_maps=False):
    """
    Ablation-CAM for semantic segmentation models (batched version).

    Instead of using gradients, weights each channel by how much the prediction
    drops when that channel is zeroed out. Processes multiple channels per forward pass.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W).
    model : torch.nn.Module
        The segmentation model.
    target : int, optional
        Target class index. If None, uses the dominant predicted class.
    target_layer : nn.Module
        The layer to compute activations from.
    box : tuple, optional
        Bounding box ROI (y_start, y_end, x_start, x_end).
    device : str, default='cpu'
        Device to run the model on.
    reshape_transformer : bool, default=False
        Whether to reshape transformer activations from [seq_len, hidden_dim] to [C, H, W].
    batch_size : int, default=16
        Number of channels to ablate per forward pass.
    return_activation_maps : bool, default=False
        If True, also return the activation maps.

    Returns
    -------
    grayscale_cam : np.ndarray
        The computed CAM with shape (H, W) normalized to [0, 1].
    weights : np.ndarray
        Per-channel importance weights (score drop per channel).
    activation_maps : np.ndarray, optional
        The activation maps with shape [C, H, W]. Returned if return_activation_maps=True.
    """
    activations = []

    handle = target_layer.register_forward_hook(
        lambda x, y, z: activations.append(z)
    )

    # Baseline forward pass
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))

    handle.remove()

    # Default target: dominant predicted class
    if target is None:
        target = output[0].argmax(0).max().item()

    predicted_mask = output[0].argmax(0).detach().cpu().numpy()  # [H, W]
    mask_float = np.float32(predicted_mask == target)

    # Apply bounding box ROI if provided
    if box is not None:
        y_start, y_end, x_start, x_end = box
        roi_mask = np.zeros_like(mask_float)
        roi_mask[y_start:y_end, x_start:x_end] = 1
        mask_float = mask_float * roi_mask

    mask_tensor = torch.tensor(mask_float).to(device)

    # Baseline score
    baseline_score = (output[0, target, :, :] * mask_tensor).sum().item()

    act = activations[0].detach()  # [1, C, H, W]

    # Reshape transformer activations if needed
    if reshape_transformer:
        seq_len = act.shape[1]
        hidden_dim = act.shape[2]
        if seq_len % 2 != 0:
            act = act[:, 1:, :]
            seq_len -= 1
        spatial = int(np.sqrt(seq_len))
        act = act.reshape(1, spatial, spatial, hidden_dim).permute(0, 3, 1, 2)  # [1, C, H, W]

    n_channels = act.shape[1]
    channel_weights = np.zeros(n_channels)
    image_batched = image.unsqueeze(0).to(device)  # [1, C, H, W]

    for batch_start in tqdm(range(0, n_channels, batch_size), desc='Ablation-CAM-Batch'):
        batch_end = min(batch_start + batch_size, n_channels)
        current_batch_size = batch_end - batch_start

        # Stack ablated activations: [current_batch_size, C, H, W]
        batch_acts = act.repeat(current_batch_size, 1, 1, 1)
        for i, c in enumerate(range(batch_start, batch_end)):
            batch_acts[i, c] = 0

        # Inject ablated activations via hook, return them for the whole batch
        def ablation_hook(module, input, output, a=batch_acts):
            return a

        h = target_layer.register_forward_hook(ablation_hook)
        with torch.no_grad():
            outputs = model(image_batched.repeat(current_batch_size, 1, 1, 1))
        h.remove()

        for i, c in enumerate(range(batch_start, batch_end)):
            score = (outputs[i, target, :, :] * mask_tensor).sum().item()
            channel_weights[c] = baseline_score - score
        
        # Clear memory after each batch
        del outputs, batch_acts, h
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    act_np = act[0].cpu().numpy()  # [C, H, W]

    # Weighted sum of activations
    grayscale_cam = (channel_weights[:, None, None] * act_np).sum(axis=0)

    # ReLU
    grayscale_cam = np.maximum(grayscale_cam, 0)

    # Resize to image spatial dims
    image_h, image_w = image.shape[1], image.shape[2]
    grayscale_cam = cv2.resize(grayscale_cam, (image_w, image_h))

    # Normalize to [0, 1]
    min_, max_ = grayscale_cam.min(), grayscale_cam.max()
    grayscale_cam = (grayscale_cam - min_) / (max_ - min_ + 1e-8)

    if return_activation_maps:
        return grayscale_cam, channel_weights, act_np
    return grayscale_cam, channel_weights
