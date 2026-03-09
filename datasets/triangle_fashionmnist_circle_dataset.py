import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random


class FashionMNISTCircleDataset(Dataset):
    def __init__(self, dataset, transform=None,
                 shape=224, labels=[1, 2, 3],
                 not_labels=[5, 6, 7], background_obj=3,
                 include_label=True, length=10000, jitter=None,
                 circle=False):
        assert shape >= 224, "shape must be >= 224"
        self.dataset = dataset
        self.transform = transform
        self.shape = shape
        self.labels = labels
        self.not_labels = not_labels
        self.background_obj = background_obj
        self.include_label = include_label
        self.len = length
        self.spacing = int(100 * shape / 224)
        self.jitter = jitter
        self.circle = circle
        self.circle_radius = 14

    def random_fashion_mnist(self):
        idx = np.random.randint(0, len(self.dataset))
        img, label = self.dataset[idx]
        img = transforms.ToTensor()(img)
        return img, label

    def _draw_circle(self, image, mask, cx, cy):
        """Draw a filled circle on the image at (cx, cy), avoiding masked regions."""
        r = self.circle_radius
        for y in range(max(0, cy - r), min(self.shape, cy + r)):
            for x in range(max(0, cx - r), min(self.shape, cx + r)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    image[:, y, x] = 0.5  # gray circle, distinct from black/white items

    def _sample_circle_position(self, mask):
        """Sample circle center that doesn't overlap any foreground mask pixel."""
        r = self.circle_radius
        for _ in range(1000):  # max attempts
            cx = random.randint(r, self.shape - r)
            cy = random.randint(r, self.shape - r)
            # check bounding box of circle against mask
            region = mask[cy - r:cy + r, cx - r:cx + r]
            if region.sum().item() == 0:
                return cx, cy
        return None  # failed to place circle (rare)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        half = 14
        s = self.spacing

        image = torch.zeros((3, self.shape, self.shape))
        mask  = torch.zeros((self.shape, self.shape))

        ax = random.randint(half, self.shape - s - half)
        ay = random.randint(half, self.shape - s - half)

        base_points = [(ax, ay), (ax + s, ay), (ax + s, ay + s), (ax, ay + s)]

        if self.jitter is not None:
            j = self.jitter
            points = [
                (bx + random.randint(-j, j), by + random.randint(-j, j))
                for bx, by in base_points
            ]
        else:
            points = base_points

        for pt in points:
            while True:
                img, label = self.random_fashion_mnist()
                if label in self.labels:
                    break

            x_pos = max(0, min(pt[0] - half, self.shape - 28))
            y_pos = max(0, min(pt[1] - half, self.shape - 28))

            image[:, x_pos:x_pos+28, y_pos:y_pos+28] = img
            mask[x_pos:x_pos+28, y_pos:y_pos+28] = self.include_label * label + 1
            mask[x_pos:x_pos+28, y_pos:y_pos+28][img[0] == 0] = 0

        for _ in range(self.background_obj):
            while True:
                img, label = self.random_fashion_mnist()
                if label in self.not_labels:
                    break
            while True:
                bx = random.randint(0, self.shape - 28)
                by = random.randint(0, self.shape - 28)
                if (mask[bx:bx+28, by:by+28] * img[0]).sum().item() == 0:
                    break
            image[:, bx:bx+28, by:by+28] = img

        # circle logic
        if self.circle:
            show_circle = random.random() < 0.5
            if show_circle:
                pos = self._sample_circle_position(mask)
                if pos is not None:
                    self._draw_circle(image, mask, pos[0], pos[1])
                # mask stays as-is (foreground = 1)
            else:
                # no circle → zero out the mask
                mask = torch.zeros((self.shape, self.shape))

        if self.transform is not None:
            image = image.permute(1, 2, 0).numpy()
            mask  = mask.numpy()
            transformed = self.transform(image=image, mask=mask)
            image = transforms.ToTensor()(transformed["image"]).float()
            mask  = torch.from_numpy(transformed["mask"]).long()

        return image, mask