"""
Models Information v2
=====================
Layer and channel index information for all models in models.py

This file documents the layer indices and channel dimensions for:
- Feature extraction points
- Skip connections
- Decoder/Encoder pathways

Note: Layer indices refer to the position in the feature list returned by the backbone.
For CNNs: typically corresponds to spatial resolution levels
For Transformers: corresponds to transformer block indices
"""

# ==============================================================================
# 1. UNET (Custom Implementation)
# ==============================================================================

UNET_INFO = {
    "description": "Custom U-Net with configurable depth",
    "backbone": "None (Custom CNN)",
    "bottleneck": {
        "type": "Deepest DoubleConv + MaxPool",
        "depth_1": {"channels": 128, "spatial": "1/2 of input"},
        "depth_2": {"channels": 256, "spatial": "1/4 of input"},
        "depth_3": {"channels": 512, "spatial": "1/8 of input"},
        "depth_4": {"channels": 1024, "spatial": "1/16 of input"},
    },
    "encoder_channels": {
        "depth_1": [64],
        "depth_2": [64, 128],
        "depth_3": [64, 128, 256],
        "depth_4": [64, 128, 256, 512],
    },
    "bottleneck_channels": {
        "depth_1": 128,
        "depth_2": 256,
        "depth_3": 512,
        "depth_4": 1024,
    },
    "decoder_channels": [512, 256, 128, 64],  # From bottleneck to output
    "skip_connection_indices": {
        # Maps decoder level to encoder level
        # For depth=4: d4->u1, d3->u2, d2->u3, d1->u4
        "depth_1": [],  # No skip connections
        "depth_2": [1],  # Skip from d2 to u3
        "depth_3": [2, 1],  # Skip from d3->u2, d2->u3
        "depth_4": [3, 2, 1],  # Full skip connections
    },
    "output_channels": "Configurable (default=4)",
    "layer_indices": {
        # Encoder feature extraction points for skip connections
        "d1": 0,  # 64 channels
        "d2": 1,  # 128 channels
        "d3": 2,  # 256 channels
        "d4": 3,  # 512 channels
    }
}

# ==============================================================================
# 2. UNETR-ViT (Vision Transformer-based UNETR)
# ==============================================================================

UNETR_VIT_INFO = {
    "description": "UNETR with Vision Transformer backbone",
    "backbone": "ViT (timm)",
    "bottleneck": {
        "type": "Deepest ViT feature (feats[-1])",
        "shape": "[B, embed_dim, H/patch, W/patch]",
        "example": "[1, 768, 14, 14] for 224x224, patch=16",
        "embed_dim": 768,
        "spatial_resolution": "14x14 (fixed by patch size)",
    },
    "available_models": [
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_large_patch16_224",
    ],
    "embed_dim": 768,  # For vit_base
    "patch_size": 16,
    "encoder_feature_indices": {
        # Transformer block indices for feature extraction
        "depth_1": [2],      # Early features
        "depth_2": [2],      # Early features
        "depth_3": [2],      # Early features
        "depth_6": [2, 5],   # Early + mid
        "depth_9": [2, 5, 8], # Early + mid + late
        "depth_12": [2, 5, 8, 11],  # Multi-scale features
    },
    "feature_channels": {
        # All ViT features have the same channel dimension
        "vit_base": 768,
        "vit_large": 1024,
    },
    "decoder_channels": {
        "standard": {
            "u11_out": 512,
            "u12_out": 256,
            "u13_out": 128,
            "du3_out": 64,
        },
        "direct_upsample": {
            # Single path: embed_dim -> 512 -> 256 -> 128 -> 64
            "path": [512, 256, 128, 64],
        }
    },
    "skip_connections": {
        "standard": {
            # Skip connections between transformer levels
            "depth_6": {
                0: {"from": "feats[0]", "channels": 768, "to": "u11"},
                1: {"from": "feats[1]", "channels": 768, "to": "u21"},
            },
            "depth_9": {
                0: {"from": "feats[0]", "channels": 768, "to": "u11"},
                1: {"from": "feats[1]", "channels": 768, "to": "u21"},
                2: {"from": "feats[2]", "channels": 768, "to": "u31"},
            },
            "depth_12": {
                0: {"from": "feats[0]", "channels": 768, "to": "u11"},
                1: {"from": "feats[1]", "channels": 768, "to": "u21"},
                2: {"from": "feats[2]", "channels": 768, "to": "u31"},
                3: {"from": "feats[3]", "channels": 768, "to": "u41"},
            },
        },
        "direct_upsample": {
            # No skip connections, direct upsampling from deepest feature
            "description": "Takes feats[-1] and upsamples directly",
            "input_channels": 768,
        }
    },
    "image_conv": {
        "description": "Processes input image for skip connection",
        "in_channels": 3,
        "out_channels": 64,
    },
    "segmentation_head": {
        "in_channels": 64,
        "out_channels": "num_classes",
    },
    "layer_indices_map": {
        # Maps depth to transformer block indices
        1: [2],
        2: [2],
        3: [2],
        6: [2, 5],
        9: [2, 5, 8],
        12: [2, 5, 8, 11],
    }
}

# ==============================================================================
# 3. UNETR-SAM (Segment Anything Model ViT backbone)
# ==============================================================================

UNETR_SAM_INFO = {
    "description": "UNETR with SAM ViT backbone",
    "backbone": "SAM ViT (samvit_base_patch16.sa1b)",
    "bottleneck": {
        "type": "Deepest SAM ViT feature (feats[-1])",
        "shape": "[B, embed_dim, H/patch, W/patch]",
        "example": "[1, 768, 64, 64] for 1024x1024, patch=16",
        "embed_dim": 768,
        "spatial_resolution": "64x64 (fixed by patch size)",
    },
    "embed_dim": 768,
    "patch_size": 16,
    "img_size": 1024,  # SAM requires 1024x1024
    "inherits_from": "UNETR_VIT_INFO",
    "differences": {
        "img_size": "Fixed at 1024",
        "model_name": "samvit_base_patch16.sa1b",
        "pretrained_source": "SA-1B dataset",
    },
    "encoder_feature_indices": UNETR_VIT_INFO["encoder_feature_indices"],
    "decoder_channels": UNETR_VIT_INFO["decoder_channels"],
    "layer_indices_map": UNETR_VIT_INFO["layer_indices_map"],
}

# ==============================================================================
# 4. TransUNet (R50-ViT Hybrid)
# ==============================================================================

TRANSUNET_INFO = {
    "description": "TransUNet with ResNet50-ViT hybrid backbone",
    "backbone": "ResNet50-ViT Hybrid",
    "bottleneck": {
        "type": "Transformer Encoder Output",
        "shape": "[B, N_patches, hidden_size]",
        "example": "[1, 196, 768] for 224x224 input",
        "components": {
            "N_patches": "(img_size // patch_size) ** 2 = (224/16)^2 = 196",
            "hidden_size": 768,
            "spatial_resolution": "14x14 (before decoder reshaping)",
        },
        "to_decoder": "Reshaped to [B, 768, 14, 14] then conv_more -> [B, 512, 14, 14]",
    },
    "config": {
        "hidden_size": 768,
        "transformer_layers": 12,
        "mlp_dim": 3072,
        "num_heads": 12,
        "patch_size": 16,
        "resnet_layers": (3, 4, 9),
    },
    "encoder": {
        "type": "Hybrid ResNet-ViT",
        "resnet_stages": {
            "root": {"out_channels": 64, "stride": 2},
            "block1": {"out_channels": 256, "units": 3},
            "block2": {"out_channels": 512, "units": 4},
            "block3": {"out_channels": 1024, "units": 9},
        },
        "vit_patches": {
            "grid": (16, 16),
            "n_patches": 256,  # For 224x224 input
        },
    },
    "encoder_channels": {
        "resnet_features": [64, 256, 512, 1024],  # For skip connections
        "transformer_out": 768,  # Final encoded representation
    },
    "skip_connections": {
        "enabled": True,
        "n_skip": 3,  # Number of skip connections
        "channels": [512, 256, 64, 16],  # Skip channel dimensions
        "resnet_indices": [0, 1, 2, 3],  # Indices in ResNet feature list
    },
    "decoder": {
        "channels": [256, 128, 64, 16],  # Decoder channel progression
        "head_channels": 512,  # First conv after transformer
        "upsample": "Bilinear2x",
    },
    "layer_indices": {
        "resnet_features": {
            0: {"name": "root", "channels": 64},
            1: {"name": "block1_out", "channels": 256},
            2: {"name": "block2_out", "channels": 512},
            3: {"name": "block3_out", "channels": 1024},
        },
        "transformer_blocks": list(range(12)),  # 0-11
    },
    "segmentation_head": {
        "in_channels": 16,
        "out_channels": "num_classes",
        "kernel_size": 3,
    },
}

# ==============================================================================
# 5. DeepLabV3 & FCN (TorchVision Models)
# ==============================================================================

TORCHVISION_SEG_INFO = {
    "description": "TorchVision segmentation models with ResNet50 backbone",
    "available_models": ["deeplabv3", "fcn"],
    "bottleneck": {
        "type": "ResNet50 layer4 output",
        "shape": "[B, 2048, H/16, W/16]",
        "example": "[1, 2048, 14, 14] for 224x224 input",
        "channels": 2048,
        "spatial_resolution": "1/16 of input",
        "note": "ASPP (DeepLabV3) or FCN Head processes this",
    },
    "backbone": "ResNet50",
    "encoder_channels": {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,  # Final backbone output
    },
    "classifier": {
        "deeplabv3": {
            "type": "ASPP (Atrous Spatial Pyramid Pooling)",
            "rates": [12, 24, 36],
            "in_channels": 2048,
            "out_channels": 256,
            "final_conv": {"in": 256, "out": "num_classes"},
        },
        "fcn": {
            "type": "FCN Head",
            "convs": [
                {"in": 2048, "out": 512, "kernel": 3},
                {"in": 512, "out": 512, "kernel": 3},
                {"in": 512, "out": "num_classes", "kernel": 1},
            ],
        },
    },
    "layer_indices": {
        "backbone": {
            0: {"name": "conv1", "channels": 64},
            1: {"name": "layer1", "channels": 256},
            2: {"name": "layer2", "channels": 512},
            3: {"name": "layer3", "channels": 1024},
            4: {"name": "layer4", "channels": 2048},
        }
    },
    "aux_classifier": "Removed in our implementation",
}

# ==============================================================================
# 6. SMP Models (segmentation_models_pytorch)
# ==============================================================================

SMP_MODELS_INFO = {
    "description": "Wrapper for segmentation_models_pytorch library",
    "bottleneck": {
        "type": "Encoder final stage output",
        "description": "Varies by encoder architecture",
        "examples": {
            "resnet34": {"channels": 512, "spatial": "1/32 of input"},
            "resnet50": {"channels": 2048, "spatial": "1/32 of input"},
            "vgg16": {"channels": 512, "spatial": "1/32 of input"},
        },
        "note": "Decoder reduces channels progressively with skip connections",
    },
    "available_architectures": [
        "Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3Plus",
        "FPN", "PSPNet", "Linknet", "PAN", "MAnet"
    ],
    "encoder_info": {
        "resnet34": {
            "channels": [64, 64, 128, 256, 512],
            "indices": [0, 1, 2, 3, 4],
        },
        "resnet50": {
            "channels": [64, 256, 512, 1024, 2048],
            "indices": [0, 1, 2, 3, 4],
        },
        "vgg16": {
            "channels": [64, 128, 256, 512, 512],
            "indices": [0, 1, 2, 3, 4],
        },
        "efficientnet-b0": {
            "channels": [32, 24, 40, 112, 320],
            "indices": [0, 1, 2, 3, 4],
        },
    },
    "architecture_specific": {
        "Unet": {
            "skip_connections": 4,
            "decoder_channels": [256, 128, 64, 32, 16],
        },
        "UnetPlusPlus": {
            "skip_connections": 4,
            "decoder_channels": [256, 128, 64, 32, 16],
            "nested_skip": True,
        },
        "DeepLabV3": {
            "skip_connections": 0,
            "aspp_channels": 256,
        },
        "DeepLabV3Plus": {
            "skip_connections": 1,
            "decoder_channels": 256,
        },
        "FPN": {
            "skip_connections": "Pyramid",
            "pyramid_channels": 256,
        },
        "PSPNet": {
            "skip_connections": 0,
            "pyramid_channels": [512, 256, 128, 64],
        },
    },
    "layer_indices": {
        # Standard layer indices for feature extraction
        "resnet": {
            0: {"name": "stem", "channels": "varies"},
            1: {"name": "layer1", "channels": "varies"},
            2: {"name": "layer2", "channels": "varies"},
            3: {"name": "layer3", "channels": "varies"},
            4: {"name": "layer4", "channels": "varies"},
        }
    },
}

# ==============================================================================
# 7. Helper Classes Channel Info
# ==============================================================================

HELPER_LAYERS_INFO = {
    "DoubleConv": {
        "description": "Double convolution with BatchNorm and ReLU",
        "structure": [
            "Conv2d(in_channels, out_channels, 3x3, padding=1)",
            "BatchNorm2d(out_channels)",
            "ReLU()",
            "Conv2d(out_channels, out_channels, 3x3, padding=1)",
            "BatchNorm2d(out_channels)",
            "ReLU()",
        ],
        "params": ["in_channels", "out_channels"],
    },
    "Deconv": {
        "description": "Deconvolution block for upsampling",
        "structure": [
            "ConvTranspose2d(in_channels, out_channels, 2x2, stride=2)",
            "Conv2d(out_channels, out_channels, 3x3, padding=1)",
            "BatchNorm2d(out_channels)",
            "ReLU()",
        ],
        "params": ["in_channels", "out_channels"],
        "upsample_factor": 2,
    },
    "DecoderBlock": {
        "description": "TransUNet decoder block",
        "structure": [
            "UpsamplingBilinear2d(scale_factor=2)",
            "Conv2dReLU(in_channels + skip_channels, out_channels)",
            "Conv2dReLU(out_channels, out_channels)",
        ],
        "params": ["in_channels", "out_channels", "skip_channels"],
    },
}

# ==============================================================================
# 8. Quick Reference: Model to Layer Index Mapping
# ==============================================================================

LAYER_INDEX_REFERENCE = {
    "UNET": {
        "encoder_levels": {0: 64, 1: 128, 2: 256, 3: 512},
        "bottleneck": 1024,
    },
    "UNETR_ViT": {
        "transformer_blocks": {0: 2, 1: 5, 2: 8, 3: 11},  # For depth=12
        "all_same_channels": 768,
    },
    "UNETR_SAM": {
        "transformer_blocks": {0: 2, 1: 5, 2: 8, 3: 11},  # For depth=12
        "all_same_channels": 768,
    },
    "TransUNet": {
        "resnet_stages": {0: 64, 1: 256, 2: 512, 3: 1024},
        "transformer_out": 768,
    },
    "DeepLabV3": {
        "resnet_stages": {0: 256, 1: 512, 2: 1024, 3: 2048},
        "aspp_out": 256,
    },
    "FCN": {
        "resnet_stages": {0: 256, 1: 512, 2: 1024, 3: 2048},
        "head_out": [512, 512],
    },
}

# ==============================================================================
# 9. Export Functions
# ==============================================================================

def get_model_info(model_name):
    """Get information dictionary for a specific model."""
    info_map = {
        "UNET": UNET_INFO,
        "UNETR_ViT": UNETR_VIT_INFO,
        "UNETR_SAM": UNETR_SAM_INFO,
        "TransUNet": TRANSUNET_INFO,
        "DeepLabV3": TORCHVISION_SEG_INFO,
        "FCN": TORCHVISION_SEG_INFO,
        "SMP": SMP_MODELS_INFO,
    }
    return info_map.get(model_name, None)


def get_layer_indices(model_name, depth=None):
    """Get layer indices for feature extraction.
    
    Args:
        model_name: Name of the model
        depth: Model depth (for models with configurable depth)
    
    Returns:
        Dictionary mapping layer indices to channels
    """
    if model_name == "UNETR_ViT" or model_name == "UNETR_SAM":
        if depth and depth in UNETR_VIT_INFO["layer_indices_map"]:
            indices = UNETR_VIT_INFO["layer_indices_map"][depth]
            return {i: UNETR_VIT_INFO["feature_channels"]["vit_base"] for i in indices}
        return UNETR_VIT_INFO["layer_indices_map"]
    
    elif model_name == "UNET":
        depth_key = f"depth_{depth}" if depth else "depth_4"
        return UNET_INFO["encoder_channels"].get(depth_key, UNET_INFO["encoder_channels"]["depth_4"])
    
    elif model_name == "TransUNet":
        return {
            "resnet": TRANSUNET_INFO["encoder_channels"]["resnet_features"],
            "transformer": TRANSUNET_INFO["encoder_channels"]["transformer_out"],
        }
    
    return None


def print_summary():
    """Print a summary of all models and their key features."""
    print("=" * 80)
    print("MODELS INFO SUMMARY")
    print("=" * 80)
    
    models = [
        ("UNET", "Custom CNN-based U-Net"),
        ("UNETR-ViT", "Vision Transformer-based UNETR"),
        ("UNETR-SAM", "SAM ViT backbone UNETR"),
        ("TransUNet", "ResNet50-ViT Hybrid"),
        ("DeepLabV3", "TorchVision DeepLabV3"),
        ("FCN", "TorchVision FCN"),
        ("SMP Models", "segmentation_models_pytorch wrapper"),
    ]
    
    for name, desc in models:
        print(f"\n{name}:")
        print(f"  {desc}")
        info = get_model_info(name.replace("-", "_"))
        if info and "backbone" in info:
            print(f"  Backbone: {info['backbone']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_summary()
