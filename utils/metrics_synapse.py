"""
Metrics for Semantic Segmentation (Synapse)
============================================
3-D case-level evaluation over volumetric CT slices
"""

import numpy as np
import torch
from tqdm import tqdm
from medpy import metric

import pprint
import os
import cv2
from skimage.transform import resize

def evaluate_synapse(model, image_paths, num_classes, transform=None, device='cpu', label_dict=None):

    # Group slices by case
    eval_path_dict = {}
    cases = [i.split(os.path.sep)[-1].split('.')[0].split('_')[0] for i in image_paths]
    for case in cases:
        eval_path_dict[case] = sorted([i for i in image_paths if case in i])
    for case in eval_path_dict:
        eval_path_dict[case] = (eval_path_dict[case], [i.replace('images', 'masks') for i in eval_path_dict[case]])

    result_dict = {cls: [] for cls in range(1, num_classes)}

    model.eval()
    with torch.no_grad():
        for case in tqdm(eval_path_dict):
            mask_3d, pred_3d = [], []

            for i, j in zip(eval_path_dict[case][0], eval_path_dict[case][1]):
                image, mask = cv2.imread(i, 0), cv2.imread(j, 0)
                resize_height, resize_width = mask.shape[0], mask.shape[1]
                if transform is not None:
                    image = transform(image)
                    image = image.to(device)
                image = image.repeat(3, 1, 1)
                with torch.cuda.amp.autocast():
                    pred = model(image.unsqueeze(0))
                pred = pred[0].argmax(0).detach().cpu().numpy()
                pred = resize(pred, (resize_height, resize_width), order=0)
                pred_3d.append(pred)
                mask_3d.append(mask)

            mask_3d, pred_3d = np.array(mask_3d), np.array(pred_3d)

            for cls in range(1, num_classes):
                if not np.any(mask_3d == cls) and not np.any(pred_3d == cls):
                    continue
                gt_binary   = (mask_3d == cls).astype(np.uint8)
                pred_binary = (pred_3d == cls).astype(np.uint8)
                result_dict[cls].append(metric.binary.dc(gt_binary, pred_binary))

    # Compute per-class mean Dice, but only for classes that actually appeared
    # in at least one case — classes with no appearances are excluded from the
    # mean so they don't artificially deflate the score with a 0.0 contribution
    avg_dice_class = {
        cls: sum(result_dict[cls]) / len(result_dict[cls])
        for cls in result_dict
        if result_dict[cls]
    }

    # Macro average: mean of per-class Dice scores, weighting each class equally
    avg_dice = sum(avg_dice_class.values()) / len(avg_dice_class) if avg_dice_class else 0.0

    avg_dice_class_named = {
        (label_dict[cls] if label_dict else cls): round(avg_dice_class[cls], 4)
        for cls in avg_dice_class
    }
    print("  Average Dice / Class:")
    pprint.pprint(avg_dice_class_named, indent=4, sort_dicts=False)
    print(f"  Average Dice: {avg_dice:.4f}")

    return avg_dice





















