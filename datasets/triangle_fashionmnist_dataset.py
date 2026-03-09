





import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import random


class FashionMNISTDataset(Dataset):
    def __init__(self, dataset, transform=None,
                 shape=224, labels=[1, 2, 3],
                 not_labels=[5, 6, 7], background_obj=3,
                 include_label=True, length=10000, jitter=None):
        assert shape >= 224, "shape must be >= 224"
        self.dataset = dataset
        self.transform = transform
        self.shape = shape
        self.labels = labels           # foreground classes to segment
        self.not_labels = not_labels   # distractor classes (placed but not masked)
        self.background_obj = background_obj  # number of distractor items to place
        self.include_label = include_label    # True: multi-class mask, False: binary mask
        self.len = length
        # scale the square spacing proportionally to canvas size (100px at 224)
        self.spacing = int(100 * shape / 224)
        # optional per-vertex jitter in pixels; None means a clean square
        self.jitter = jitter

    def random_fashion_mnist(self):
        # sample a random item from the underlying dataset and convert to tensor
        idx = np.random.randint(0, len(self.dataset))
        img, label = self.dataset[idx]
        img = transforms.ToTensor()(img)
        return img, label

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        half = 14  # half of Fashion MNIST item size (28 // 2), used to center items on points
        s = self.spacing

        # blank RGB canvas and corresponding segmentation mask
        image = torch.zeros((3, self.shape, self.shape))
        mask  = torch.zeros((self.shape, self.shape))

        # random anchor for the top-left corner of the square,
        # keeping enough margin so all corners fit within the canvas
        ax = random.randint(half, self.shape - s - half)
        ay = random.randint(half, self.shape - s - half)

        # four corners of the square in (x, y) order
        base_points = [(ax, ay), (ax + s, ay), (ax + s, ay + s), (ax, ay + s)]

        # optionally jitter each corner independently to break the perfect square
        if self.jitter is not None:
            j = self.jitter
            points = [
                (bx + random.randint(-j, j), by + random.randint(-j, j))
                for bx, by in base_points
            ]
        else:
            points = base_points

        # place one foreground item at each corner
        for pt in points:
            # reject-sample until we get an item from the foreground classes
            while True:
                img, label = self.random_fashion_mnist()
                if label in self.labels:
                    break

            # center the 28x28 item on the point, clamping to canvas bounds
            x_pos = max(0, min(pt[0] - half, self.shape - 28))
            y_pos = max(0, min(pt[1] - half, self.shape - 28))

            image[:, x_pos:x_pos+28, y_pos:y_pos+28] = img

            # include_label=True  → mask value is label+1 (multi-class)
            # include_label=False → mask value is 1 (binary foreground)
            # +1 in both cases reserves 0 for background
            mask[x_pos:x_pos+28, y_pos:y_pos+28] = self.include_label * label + 1

            # zero out mask pixels where the item itself is black (true background)
            mask[x_pos:x_pos+28, y_pos:y_pos+28][img[0] == 0] = 0

        # place distractor items (not added to mask, so model must learn to ignore them)
        for _ in range(self.background_obj):
            # reject-sample until we get an item from the distractor classes
            while True:
                img, label = self.random_fashion_mnist()
                if label in self.not_labels:
                    break

            # reject-sample a position that doesn't overlap any masked foreground pixel
            while True:
                bx = random.randint(0, self.shape - 28)
                by = random.randint(0, self.shape - 28)
                if (mask[bx:bx+28, by:by+28] * img[0]).sum().item() == 0:
                    break

            image[:, bx:bx+28, by:by+28] = img

        # apply albumentations-style transforms jointly to image and mask
        if self.transform is not None:
            image = image.permute(1, 2, 0).numpy()  # CHW -> HWC for albumentations
            mask  = mask.numpy()
            transformed = self.transform(image=image, mask=mask)
            image = transforms.ToTensor()(transformed["image"]).float()
            mask  = torch.from_numpy(transformed["mask"]).long()

        return image, mask