"""
Metrics for Semantic Segmentation
==================================
Based on polyp.py implementation
"""

import numpy as np
import torch
from tqdm import tqdm
from medpy import metric

import pprint

def evaluate(dataset, model, num_classes, device='cpu', label_dict=None):
    # Pre-initialize all foreground classes (excluding background 0) so that
    # classes with no appearances still show up in the per-class printout
    result_dict = {cls: [] for cls in range(1, num_classes)}
    
    for i in tqdm(range(len(dataset))):
        image, label = dataset[i]
        image, label = image.to(device), label.numpy()
        
        with torch.no_grad():
            prediction = model(image.unsqueeze(0))
        prediction = prediction[0].argmax(0).detach().cpu().numpy()
        
        # Union of GT and predicted labels, excluding background (0)
        # Using union ensures false positive predictions are penalised (Dice = 0)
        # rather than silently ignored as they would be if we only used GT labels
        gt_labels    = set(np.unique(label).tolist())
        pred_labels  = set(np.unique(prediction).tolist())
        union_labels = (gt_labels | pred_labels) - {0}
        
        for cls in union_labels:
            gt_binary   = (label      == cls).astype(np.uint8)
            pred_binary = (prediction == cls).astype(np.uint8)
            
            # Trivial case: class absent from both GT and pred — skip to avoid
            # a meaningless Dice of 1.0 inflating the mean (shouldn't occur
            # given the union above, but kept as a safety guard)
            if gt_binary.sum() == 0 and pred_binary.sum() == 0:
                continue
            
            result_dict[cls].append(metric.binary.dc(gt_binary, pred_binary))
    
    # Compute per-class mean Dice, but only for classes that actually appeared
    # in at least one image — classes with no appearances are excluded from the
    # mean so they don't artificially deflate the score with a 0.0 contribution
    avg_dice_class = {
        cls: sum(result_dict[cls]) / len(result_dict[cls])
        for cls in result_dict
        if result_dict[cls]  # skip classes with no appearances
    }
    
    # Macro average: mean of per-class Dice scores, weighting each class equally
    # regardless of how frequently it appears in the dataset (as opposed to a
    # frequency-weighted mean which would bias towards common classes)
    avg_dice = sum(avg_dice_class.values()) / len(avg_dice_class) if avg_dice_class else 0.0
    
    # Use label_dict for human-readable names if provided, otherwise fall back
    # to raw class indices as string keys
    avg_dice_class_named = {
        (label_dict[cls] if label_dict else cls): round(avg_dice_class[cls], 4)
        for cls in avg_dice_class
    }
    print("  Average Dice / Class:")
    pprint.pprint(avg_dice_class_named, indent=4, sort_dicts=False)
    print(f"  Average Dice: {avg_dice:.4f}")
    
    return avg_dice


