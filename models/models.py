from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

# Use CUDA if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 1. HELPER CLASSES & MODEL DEFINITIONS
# ==============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)

class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # 1. Upsample 2x
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 2. Refine features
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

#################################################################################

# UNet

#################################################################################


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, depth=4):
        super().__init__()
        self.depth = depth
        self.d1 = DoubleConv(in_channels, 64)
        if self.depth >= 2: self.d2 = DoubleConv(64, 128)
        if self.depth >= 3: self.d3 = DoubleConv(128, 256)
        if self.depth >= 4: self.d4 = DoubleConv(256, 512)

        if self.depth == 1: self.d5 = DoubleConv(64, 128)
        elif self.depth == 2: self.d5 = DoubleConv(128, 256)
        elif self.depth == 3: self.d5 = DoubleConv(256, 512)
        elif self.depth == 4: self.d5 = DoubleConv(512, 1024)

        if self.depth >= 4:
            self.u1 = nn.ConvTranspose2d(1024, 512, 2, 2)
            self.du1 = DoubleConv(1024, 512)
        if self.depth >= 3:
            self.u2 = nn.ConvTranspose2d(512, 256, 2, 2)
            self.du2 = DoubleConv(512, 256)
        if self.depth >= 2:
            self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.du3 = DoubleConv(256, 128)

        self.u4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.du4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        d1 = self.d1(x)
        curr = d1
        if self.depth >= 2: d2 = self.d2(self.pool(d1)); curr = d2
        if self.depth >= 3: d3 = self.d3(self.pool(d2)); curr = d3
        if self.depth >= 4: d4 = self.d4(self.pool(d3)); curr = d4
        curr = self.d5(self.pool(curr))
        if self.depth >= 4: u1 = self.u1(curr); curr = self.du1(torch.cat((d4, u1), 1))
        if self.depth >= 3: u2 = self.u2(curr); curr = self.du2(torch.cat((d3, u2), 1))
        if self.depth >= 2: u3 = self.u3(curr); curr = self.du3(torch.cat((d2, u3), 1))
        u4 = self.u4(curr); curr = self.du4(torch.cat((d1, u4), 1))
        return self.out(curr)

#################################################################################

# UNETR-ViT

#################################################################################


class UNETR_ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=4, depth=12,
                 direct_upsample=False, pretrained=True, dynamic_img_size=True):
        super().__init__()
        self.depth = depth
        self.direct = direct_upsample
        self.dynamic_img_size = dynamic_img_size

        # 1. Logic for indices: Remains the same
        if self.depth == 12: indices = [2, 5, 8, 11]
        elif self.depth == 9: indices = [2, 5, 8]
        elif self.depth == 6: indices = [2, 5]
        elif 1 <= self.depth <= 3: indices = [self.depth - 1]
        else: raise ValueError(f"Depth {self.depth} is not allowed.")

        if self.dynamic_img_size:
            self.vit = timm.create_model(model_name, pretrained=pretrained, features_only=True, 
                                         out_indices=indices, dynamic_img_size=self.dynamic_img_size)
        else:
            self.vit = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=indices)
        
        self.embed_dim = self.vit.feature_info[0]['num_chs']

        # CNN branch
        self.img_conv = nn.Sequential(
            DoubleConv(3, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )

        # Final Fusion and Head
        self.dc4 = DoubleConv(128 + 128, 128)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        if self.direct:
            self.up_direct = nn.Sequential(
                Deconv(self.embed_dim, 512), 
                Deconv(512, 256),            
                Deconv(256, 128),            
            )
        else:
            # --- Depth 1-3 Support ---
            if self.depth >= 1:
                self.u11 = Deconv(self.embed_dim, 512)
                self.u12 = Deconv(512, 256)
                self.u13 = Deconv(256, 128)
            
            # --- Depth 6 Support ---
            if self.depth >= 6:
                self.u21 = Deconv(self.embed_dim, 512)
                self.u22 = Deconv(512, 256)
                self.du2 = nn.ConvTranspose2d(256, 128, 2, 2)
                self.dc3 = DoubleConv(128 + 128, 128) # Fuses x_bot and x_top

            # --- Depth 9 Support ---
            if self.depth >= 9:
                self.u31 = Deconv(self.embed_dim, 512)
                self.du1 = nn.ConvTranspose2d(512, 256, 2, 2)
                self.dc2 = DoubleConv(256 + 256, 256) # Fuses x2 and x3_up

            # --- Depth 12 Support ---
            if self.depth == 12:
                # FIX: Use Deconv for u41 to match u31's architecture
                self.u41 = Deconv(self.embed_dim, 512) 
                self.dc1 = DoubleConv(512 + 512, 512) # Fuses x3 and x4

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.vit(x)
        img = self.img_conv(x)

        if self.direct:
            x_up = self.up_direct(feats[-1])

        elif self.depth <= 3:
            x_up = self.u13(self.u12(self.u11(feats[0])))

        elif self.depth == 6:
            x_bot = self.u13(self.u12(self.u11(feats[0])))
            x_top = self.du2(self.u22(self.u21(feats[1])))
            x_up = self.dc3(torch.cat((x_bot, x_top), 1))

        elif self.depth == 9:
            x1 = self.u13(self.u12(self.u11(feats[0])))
            x2 = self.u22(self.u21(feats[1]))
            x3 = self.du1(self.u31(feats[2])) # 512 -> 256
            x_up = self.dc3(torch.cat((x1, self.du2(self.dc2(torch.cat((x2, x3), 1)))), 1))

        elif self.depth == 12:
            x1 = self.u13(self.u12(self.u11(feats[0])))
            x2 = self.u22(self.u21(feats[1]))
            x3 = self.u31(feats[2]) # Size 28x28, 512ch
            x4 = self.u41(feats[3]) # Size 28x28, 512ch
            
            # Nesting: (x3+x4) -> dc1 -> du1 -> cat with x2 -> dc2 -> du2 -> cat with x1 -> dc3
            x34 = self.du1(self.dc1(torch.cat((x3, x4), 1))) # (512+512) -> 512 -> 256
            x_up = self.dc3(torch.cat((x1, self.du2(self.dc2(torch.cat((x2, x34), 1)))), 1))

        # Dynamic resizing for patch-size flexibility
        if x_up.shape[2:] != img.shape[2:]:
            x_up = nn.functional.interpolate(x_up, size=img.shape[2:], mode='bilinear', align_corners=False)

        out = self.segmentation_head(self.dc4(torch.cat((img, x_up), 1)))
        return nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
    
#################################################################################

# UNETR-SAM

#################################################################################

class UNETR_SAM(UNETR_ViT):
    def __init__(self, num_classes=4, depth=12, direct_upsample=False, pretrained=True):
        # super().__init__(model_name='samvit_base_patch16.sa1b', num_classes=num_classes, img_size=1024, depth=depth, direct_upsample=direct_upsample, pretrained=pretrained)
        super().__init__(model_name='samvit_base_patch16.sa1b', num_classes=num_classes, depth=depth, direct_upsample=direct_upsample, pretrained=pretrained, dynamic_img_size=False)


#################################################################################

# DeepLabv3-R50 and FCN-R50

#################################################################################

class CustomTorchVisionSegmentation(nn.Module):
    def __init__(self, model_type='deeplabv3', num_classes=None, pretrained=True):
        """
        Args:
            model_type (str): 'deeplabv3' or 'fcn'
            num_classes (int): Number of output classes. 
                               If None, keeps the original COCO head (21 classes).
            pretrained (bool): Whether to load ImageNet/COCO weights.
        """
        super().__init__()
        
        # 1. Load Model & Weights
        if model_type == 'deeplabv3':
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet50(weights=weights)
        elif model_type == 'fcn':
            weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = fcn_resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 2. Remove Auxiliary Branch (Saves compute/memory)
        self.model.aux_classifier = None
        
        # 3. Handle Classification Head
        # Default is 21 classes. If num_classes is provided and != 21, replace head.
        if num_classes is not None and num_classes != 21:
            # The classifier is a Sequential block. The last layer (projection) is at index 4.
            in_channels = self.model.classifier[4].in_channels
            self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Since aux_classifier is None, the model returns OrderedDict with only 'out'
        return self.model(x)['out']

#################################################################################

# UNet R50 and VGG16

#################################################################################

class CustomSMP(nn.Module):
    def __init__(self, arch='Unet', encoder_name='resnet34', encoder_weights='imagenet', num_classes=4, in_channels=3):
        """
        Args:
            arch (str): Architecture name. Options: 'Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'PAN', 'Unet++'.
            encoder_name (str): Backbone encoder (e.g., 'resnet34', 'resnet50', 'vgg16', 'efficientnet-b0', 'mit_b0').
            encoder_weights (str or None): Pretrained weights ('imagenet', 'ssl', 'swsl') or None.
            num_classes (int): Number of output classes.
            in_channels (int): Input channels (3 for RGB, 1 for grayscale).
        """
        super().__init__()
        
        # 1. Create the model using SMP's model creation factory
        # This is safer than smp.Unet() because it handles capitalization (Unet vs unet)
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )

    def forward(self, x):
        # SMP models return the tensor directly, so we just pass it through.
        return self.model(x)
    
#################################################################################

# VGG16 - Compressed

#################################################################################

class VGGUNet_Dynamic(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, depth=5, 
                 direct_upsample=False, pretrained=True):
        super().__init__()
        assert 1 <= depth <= 5, "VGG16 supports depths 1-5."
        self.depth = depth
        self.direct = direct_upsample

        # 1. Encoder
        self.enc = timm.create_model('vgg16', pretrained=pretrained, features_only=True, out_indices=tuple(range(depth)))
        vgg_ch = [64, 128, 256, 512, 512]
        mid_ch = vgg_ch[depth-1]
        
        # 2. Center Block (The Bridge)
        self.center = DoubleConv(mid_ch, mid_ch)

        # 3. Path Logic (Explicit If/Else)
        if self.direct:
            # DIRECT PATH: Straight upsampling from the deepest feature to H/2
            if depth == 5:
                self.up_direct = nn.Sequential(
                    Deconv(512, 256), # 7x7 -> 14x14
                    Deconv(256, 128), # 14x14 -> 28x28
                    Deconv(128, 64),  # 28x28 -> 56x56
                    Deconv(64, 32)    # 56x56 -> 112x112
                )
                self.final_ch = 32
            elif depth == 4:
                self.up_direct = nn.Sequential(
                    Deconv(512, 128), # 14x14 -> 28x28
                    Deconv(128, 64),  # 28x28 -> 56x56
                    Deconv(64, 32)    # 56x56 -> 112x112
                )
                self.final_ch = 32
            elif depth == 3:
                self.up_direct = nn.Sequential(
                    Deconv(256, 64),  # 28x28 -> 56x56
                    Deconv(64, 32)    # 56x56 -> 112x112
                )
                self.final_ch = 32
            elif depth == 2:
                self.up_direct = Deconv(128, 32) # 56x56 -> 112x112
                self.final_ch = 32
            else: # depth 1
                self.up_direct = nn.Identity()
                self.final_ch = mid_ch
        else:
            # STANDARD U-NET PATH: Using Skips
            if depth == 5:
                self.u1 = Deconv(512, 256); self.dc1 = DoubleConv(256 + 512, 256)
                self.u2 = Deconv(256, 128); self.dc2 = DoubleConv(128 + 256, 128)
                self.u3 = Deconv(128, 64);  self.dc3 = DoubleConv(64 + 128, 64)
                self.u4 = Deconv(64, 32);   self.dc4 = DoubleConv(32 + 64, 32)
                self.final_ch = 32
            elif depth == 4:
                self.u1 = Deconv(512, 128); self.dc1 = DoubleConv(128 + 256, 128)
                self.u2 = Deconv(128, 64);  self.dc2 = DoubleConv(64 + 128, 64)
                self.u3 = Deconv(64, 32);   self.dc3 = DoubleConv(32 + 64, 32)
                self.final_ch = 32
            elif depth == 3:
                self.u1 = Deconv(256, 64);  self.dc1 = DoubleConv(64 + 128, 64)
                self.u2 = Deconv(64, 32);   self.dc2 = DoubleConv(32 + 64, 32)
                self.final_ch = 32
            elif depth == 2:
                self.u1 = Deconv(128, 32);  self.dc1 = DoubleConv(32 + 64, 32)
                self.final_ch = 32
            else: # depth 1
                self.final_ch = mid_ch

        # 4. Final Upsample (H/2 -> H)
        self.final_upsample = Deconv(self.final_ch, 16)
        self.segmentation_head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.enc(x)
        curr = self.center(feats[-1])

        if self.direct:
            curr = self.up_direct(curr)
        else:
            if self.depth == 5:
                curr = self.dc1(torch.cat([feats[3], self.u1(curr)], 1))
                curr = self.dc2(torch.cat([feats[2], self.u2(curr)], 1))
                curr = self.dc3(torch.cat([feats[1], self.u3(curr)], 1))
                curr = self.dc4(torch.cat([feats[0], self.u4(curr)], 1))
            elif self.depth == 4:
                curr = self.dc1(torch.cat([feats[2], self.u1(curr)], 1))
                curr = self.dc2(torch.cat([feats[1], self.u2(curr)], 1))
                curr = self.dc3(torch.cat([feats[0], self.u3(curr)], 1))
            elif self.depth == 3:
                curr = self.dc1(torch.cat([feats[1], self.u1(curr)], 1))
                curr = self.dc2(torch.cat([feats[0], self.u2(curr)], 1))
            elif self.depth == 2:
                curr = self.dc1(torch.cat([feats[0], self.u1(curr)], 1))

        # Final scale recovery (H/2 -> H)
        curr = self.final_upsample(curr)
        out = self.segmentation_head(curr)

        # Handle non-32-divisible input sizes
        if out.shape[2:] != (H, W):
            out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

