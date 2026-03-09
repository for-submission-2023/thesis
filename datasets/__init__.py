"""
Datasets for Semantic Segmentation

All datasets follow the same structure as the original code:
- Use PIL.Image for loading
- Return (image, mask) tuple or (image, mask, label) if labels provided
- Use transforms.ToTensor() for conversion

Available:
- AnimalDataset: Oxford-IIIT Pet (binary/multiclass)
- ISICDataset: Skin lesion segmentation
- PolypDataset: Polyp segmentation
- InstrumentDataset: Instrument segmentation
- FashionMNISTDataset: Triangle detection on FashionMNIST
- SynapseDataset: Synapse multi-organ segmentation
"""

from .animal_dataset import AnimalDataset
from .isic_dataset import ISICDataset
from .polyp_dataset import PolypDataset
from .instrument_dataset import InstrumentDataset
from .triangle_fashionmnist_dataset import FashionMNISTDataset
from .synapse2d_dataset import SynapseDataset
from .coco_2017_dataset import COCODataset, TorchvisionTransformWrapper
from .maze_dataset import BinaryMazeDataset
from .triangle_fashionmnist_circle_dataset import FashionMNISTCircleDataset

__all__ = [
    'AnimalDataset',
    'ISICDataset',
    'PolypDataset',
    'InstrumentDataset',
    'FashionMNISTDataset',
    'SynapseDataset',
    'COCODataset',
    'TorchvisionTransformWrapper',
    'BinaryMazeDataset',
    'FashionMNISTCircleDataset'
]
