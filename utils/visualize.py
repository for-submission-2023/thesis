"""
Visualization Utilities for Semantic Segmentation
=================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use('TkAgg')  # Necessary to run matplotlib


def _normalize_image(image_np):
    """
    Normalize image to [0, 1] range for visualization.
    """
    img = image_np.copy()
    
    # Normalize to [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    
    return img


def visualize(image, mask, model, label_dict=None, title=None, device='cpu', figsize=(15, 5)):
    """
    Visualize an input image, its ground truth mask, and model prediction.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape (C, H, W) or (1, C, H, W).
    mask : torch.Tensor
        Ground truth mask with shape (H, W) or (1, H, W).
    model : torch.nn.Module
        The segmentation model to generate predictions.
    label_dict : dict, optional
        Dictionary mapping class indices to class names (e.g., {1: 'class_a', 2: 'class_b'}).
        If provided, class names will be used in the title instead of raw indices.
    title : str, optional
        Additional title text to display.
    device : str, default='cpu'
        Device to run the model on ('cpu' or 'cuda').
    figsize : tuple, default=(15, 5)
        Figure size for the matplotlib plot (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axs : list of matplotlib.axes.Axes
        List of axes objects for the three subplots.
    """
    model.eval()
    
    # Ensure image has batch dimension: (1, C, H, W)
    if image.dim() == 3:
        image_batch = image.unsqueeze(0)
    else:
        image_batch = image
    
    # Generate prediction
    with torch.no_grad():
        output = model(image_batch.to(device))
    
    # Get prediction: argmax over class dimension -> (H, W)
    prediction = output.argmax(1)[0].detach().cpu().numpy()
    
    # Prepare image for display: (C, H, W) -> (H, W, C) or (1, H, W) -> (H, W)
    image_np = image_batch[0].cpu().numpy()
    if image_np.shape[0] == 3:  # RGB
        image_display = np.transpose(image_np, (1, 2, 0))
    elif image_np.shape[0] == 1:  # Grayscale
        image_display = image_np[0]
    else:
        image_display = image_np[0] if image_np.ndim == 3 else image_np
    
    # Normalize image to [0, 1] for visualization
    image_display = _normalize_image(image_display)
    
    # Prepare mask for display
    mask_np = mask.squeeze().cpu().numpy() if torch.is_tensor(mask) else np.squeeze(mask)
    
    # Get unique classes in prediction (excluding background 0)
    pred_classes = set(np.unique(prediction).tolist()) - {0}
    
    # Format class labels
    if label_dict and pred_classes:
        class_labels = [label_dict.get(c, f"Class {c}") for c in sorted(pred_classes)]
    elif pred_classes:
        class_labels = [f"Class {c}" for c in sorted(pred_classes)]
    else:
        class_labels = ["No foreground classes"]
    
    # Build title
    classes_str = ", ".join(class_labels)
    if title and classes_str:
        full_title = f"{title}\nPredicted: {classes_str}"
    elif title:
        full_title = title
    else:
        full_title = f"Predicted: {classes_str}"
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Plot image
    if image_display.ndim == 3:
        axs[0].imshow(image_display)
    else:
        axs[0].imshow(image_display, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    # Plot ground truth mask
    axs[1].imshow(mask_np, cmap='nipy_spectral', vmin=0, vmax=max(1, mask_np.max()))
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')
    
    # Plot prediction
    axs[2].imshow(prediction, cmap='nipy_spectral', vmin=0, vmax=max(1, prediction.max()))
    axs[2].set_title('Prediction')
    axs[2].axis('off')
    
    # Add overall title
    fig.suptitle(full_title, fontsize=12, y=0.98)
    plt.tight_layout()
    
    return fig, axs


def plot_images_grid(images, titles=None, figsize=None, cmap=None, filename=None):
    """
    Plot images in a multi-row grid layout.

    Parameters
    ----------
    images : list of list of array-like or torch.Tensor
        2D list of images to display. Each inner list represents a row.
        Each image can be:
        - torch.Tensor: (C, H, W) or (H, W) 
        - numpy array: (H, W, C) or (H, W)
        Example: [[img1, img2, img3], [img4, img5, img6]] creates a 2x3 grid.
    titles : list of list of str, optional
        2D list of titles matching the images structure. If None, no titles are shown.
        Example: [['A', 'B', 'C'], ['D', 'E', 'F']]
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None based on grid size.
    cmap : str, optional
        Colormap for grayscale images (e.g., 'gray', 'nipy_spectral', 'jet').
        If None, uses default colormap.
    filename : str, optional
        If provided, save the figure to this path instead of returning it.
        The file format is inferred from the extension (e.g., '.png', '.jpg', '.pdf').

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The matplotlib figure object, or None if filename is provided.
    axs : numpy.ndarray of matplotlib.axes.Axes or None
        2D array of axes objects, or None if filename is provided.

    Raises
    ------
    ValueError
        If images is empty or rows have inconsistent lengths.

    Examples
    --------
    >>> # Plot a 2x3 grid
    >>> images = [[img1, img2, img3], [img4, img5, img6]]
    >>> titles = [['Row1-A', 'Row1-B', 'Row1-C'], ['Row2-A', 'Row2-B', 'Row2-C']]
    >>> plot_images_grid(images, titles=titles)
    
    >>> # Plot with custom size and colormap
    >>> plot_images_grid([[img1, img2], [img3, img4]], figsize=(12, 10), cmap='jet')
    
    >>> # Save to file instead of displaying
    >>> plot_images_grid([[img1, img2], [img3, img4]], titles=[['A', 'B'], ['C', 'D']], 
    ...                  filename='grid_output.png')
    """
    # Validate input
    if not images or not isinstance(images, list):
        raise ValueError("images must be a non-empty list of lists")
    
    n_rows = len(images)
    if n_rows == 0:
        raise ValueError("images list cannot be empty")
    
    # Get number of columns from first row
    n_cols = len(images[0])
    if n_cols == 0:
        raise ValueError("First row of images cannot be empty")
    
    # Validate that all rows have the same number of columns
    for i, row in enumerate(images):
        if len(row) != n_cols:
            raise ValueError(f"Row {i} has {len(row)} images, but row 0 has {n_cols}. "
                           f"All rows must have the same number of images.")
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single row or single column case to ensure axs is 2D
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1:
        axs = axs.reshape(-1, 1)
    
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            img = images[row_idx][col_idx]
            ax = axs[row_idx, col_idx]
            
            # Convert torch tensor to numpy if needed
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            
            # Handle different shapes
            if img.ndim == 3:
                if img.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                if img.shape[-1] == 1:  # (H, W, 1) -> (H, W)
                    img = img.squeeze(-1)
            
            # Normalize image to [0, 1] for visualization
            img = _normalize_image(img)
            
            # Determine if we need a colormap
            is_grayscale = img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)
            img_cmap = cmap if (is_grayscale and cmap) else None
            
            # Display image
            ax.imshow(img, cmap=img_cmap)
            
            # Add title if provided
            if titles is not None:
                if (row_idx < len(titles) and 
                    col_idx < len(titles[row_idx]) and 
                    titles[row_idx][col_idx] is not None):
                    ax.set_title(titles[row_idx][col_idx])
            
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save to file if filename is provided
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, axs


def visualize_batch(images, masks, predictions=None, num_samples=4, label_dict=None, 
                    figsize=(16, 12), cmap_mask='nipy_spectral'):
    """
    Visualize a batch of images with their masks and optional predictions.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images with shape (B, C, H, W).
    masks : torch.Tensor
        Batch of ground truth masks with shape (B, H, W).
    predictions : torch.Tensor, optional
        Batch of predictions with shape (B, H, W). If None, only images and masks are shown.
    num_samples : int, default=4
        Number of samples to display from the batch.
    label_dict : dict, optional
        Dictionary mapping class indices to class names.
    figsize : tuple, default=(16, 12)
        Figure size for the matplotlib plot.
    cmap_mask : str, default='nipy_spectral'
        Colormap for masks and predictions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of axes objects.
    """
    num_samples = min(num_samples, images.shape[0])
    num_cols = 3 if predictions is not None else 2
    
    fig, axs = plt.subplots(num_samples, num_cols, figsize=figsize)
    
    # Handle single row case
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    for i in range(num_samples):
        # Image
        img = images[i].cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        else:
            img = img[0] if img.ndim == 3 else img
        
        # Normalize image to [0, 1] for visualization
        img = _normalize_image(img)
        
        axs[i, 0].imshow(img if img.ndim == 3 else img, cmap='gray' if img.ndim == 2 else None)
        axs[i, 0].set_title(f'Sample {i+1}: Image')
        axs[i, 0].axis('off')
        
        # Ground truth mask
        mask = masks[i].squeeze().cpu().numpy()
        axs[i, 1].imshow(mask, cmap=cmap_mask, vmin=0, vmax=max(1, mask.max()))
        axs[i, 1].set_title('Ground Truth')
        axs[i, 1].axis('off')
        
        # Prediction (if provided)
        if predictions is not None:
            pred = predictions[i].squeeze().cpu().numpy()
            axs[i, 2].imshow(pred, cmap=cmap_mask, vmin=0, vmax=max(1, pred.max()))
            
            # Get predicted classes
            pred_classes = set(np.unique(pred).tolist()) - {0}
            if label_dict and pred_classes:
                class_labels = [label_dict.get(c, str(c)) for c in sorted(pred_classes)]
            elif pred_classes:
                class_labels = [str(c) for c in sorted(pred_classes)]
            else:
                class_labels = ["None"]
            
            axs[i, 2].set_title(f'Prediction: {", ".join(class_labels)}')
            axs[i, 2].axis('off')
    
    plt.tight_layout()
    return fig, axs


def plot_images_row(images, titles=None, figsize=None, cmap=None, filename=None):
    """
    Plot a list of images in a single row.

    Parameters
    ----------
    images : list of array-like or torch.Tensor
        List of images to display. Each image can be:
        - torch.Tensor: (C, H, W) or (H, W) 
        - numpy array: (H, W, C) or (H, W)
    titles : list of str, optional
        List of titles for each image. If None, no titles are shown.
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.
    cmap : str, optional
        Colormap for grayscale images (e.g., 'gray', 'nipy_spectral').
        If None, uses default colormap.
    filename : str, optional
        If provided, save the figure to this path instead of returning it.
        The file format is inferred from the extension (e.g., '.png', '.jpg', '.pdf').

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The matplotlib figure object, or None if filename is provided.
    axs : numpy.ndarray of matplotlib.axes.Axes or None
        Array of axes objects, or None if filename is provided.

    Examples
    --------
    >>> # Plot 3 images side by side
    >>> plot_images_row([img1, img2, img3], titles=['Original', 'Mask', 'Prediction'])
    
    >>> # Plot with custom size and colormap
    >>> plot_images_row([img1, img2], titles=['Input', 'Output'], figsize=(10, 5), cmap='gray')
    
    >>> # Save to file instead of displaying
    >>> plot_images_row([img1, img2, img3], titles=['A', 'B', 'C'], filename='output.png')
    """
    n_images = len(images)
    if n_images == 0:
        raise ValueError("images list cannot be empty")
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (4 * n_images, 4)
    
    fig, axs = plt.subplots(1, n_images, figsize=figsize)
    
    # Handle single image case
    if n_images == 1:
        axs = [axs]
    
    for i, img in enumerate(images):
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        
        # Handle different shapes
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:  # (H, W, 1) -> (H, W)
                img = img.squeeze(-1)
        
        # Normalize image to [0, 1] for visualization
        img = _normalize_image(img)
        
        # Determine if we need a colormap
        is_grayscale = img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)
        img_cmap = cmap if (is_grayscale and cmap) else None
        
        # Display image
        axs[i].imshow(img, cmap=img_cmap)
        
        # Add title if provided
        if titles and i < len(titles) and titles[i]:
            axs[i].set_title(titles[i])
        
        axs[i].axis('off')
    
    plt.tight_layout()
    
    # Save to file if filename is provided
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None, None
    
    return fig, axs
