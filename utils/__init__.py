"""
Utilities for Training, Metrics, and Visualization
==================================================

Available:
- train: Main training loop function
- evaluate: Evaluation metrics function
- train_synapse: Training loop for Synapse dataset
- evaluate_synapse: Evaluation metrics for Synapse dataset
- visualize: Visualize image, mask, and model prediction
- visualize_batch: Visualize a batch of samples
- plot_images_row: Plot multiple images in a single row
- seg_grad_cam: Seg-Grad-CAM attribution for segmentation models
- ablation_cam: Ablation-CAM attribution for segmentation models
- ablation_cam_batch: Batched Ablation-CAM attribution
- rise_segmentation: RISE attribution for segmentation models
- rise_segmentation_batch: Batched RISE attribution
- generate_masks: Generate random masks for RISE
- pixel_deletion_curve: Evaluate attribution with pixel deletion
- pixel_insertion_curve: Evaluate attribution with pixel insertion
- compute_auc: Compute AUC for evaluation curves
- misure: Minimally Sufficient Region explanation method
- dilate_to_sufficient: Dilate mask until DICE threshold is reached
"""

from .train import train
from .metrics import evaluate
from .train_synapse import train_synapse
from .metrics_synapse import evaluate_synapse
from .visualize import visualize, visualize_batch, plot_images_row
from .cam import seg_grad_cam, ablation_cam, ablation_cam_batch
from .rise import rise_segmentation, rise_segmentation_batch, generate_masks
from .xai_metrics import pixel_deletion_curve, pixel_insertion_curve, compute_auc
from .misure import misure, dilate_to_sufficient, misure_1

__all__ = [
    'train',
    'evaluate',
    'train_synapse',
    'evaluate_synapse',
    'visualize',
    'visualize_batch',
    'plot_images_row',
    'seg_grad_cam',
    'ablation_cam',
    'ablation_cam_batch',
    'rise_segmentation',
    'rise_segmentation_batch',
    'generate_masks',
    'pixel_deletion_curve',
    'pixel_insertion_curve',
    'compute_auc',
    'misure',
    'dilate_to_sufficient',
    'misure_1'
]
