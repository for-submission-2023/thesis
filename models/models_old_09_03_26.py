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
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x): return self.conv(x)

#################################################################################

# UNet

#################################################################################


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, depth=4):
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
    # def __init__(self, model_name='vit_base_patch16_224', num_classes=4, img_size=224, depth=12, direct_upsample=False, pretrained=True):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=4, depth=12, direct_upsample=False, pretrained=True):
        super().__init__()
        self.depth = depth
        self.direct = direct_upsample  # 1. SAVE THE ARGUMENT

        # Depth logic
        if self.depth == 12: indices = [2, 5, 8, 11]
        elif self.depth == 9: indices = [2, 5, 8]
        elif self.depth == 6: indices = [2, 5]
        elif 1 <= self.depth <= 3: indices = [self.depth - 1]
        else: raise ValueError(f"Depth {self.depth} is not allowed. Allowed depths are 1, 2, 3, 6, 9, 12")
        
        # self.vit = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, features_only=True, out_indices=indices)
        self.vit = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=indices)
        self.embed_dim = self.vit.feature_info[0]['num_chs']

        self.img_conv = DoubleConv(3, 64)
        self.dc4 = DoubleConv(64 + 64, 64)
        self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)

        # 2. ADD DIRECT UPSAMPLE LAYER DEFINITION
        if self.direct:
            self.up_direct = nn.Sequential(
                Deconv(self.embed_dim, 512),
                Deconv(512, 256),
                Deconv(256, 128),
                nn.ConvTranspose2d(128, 64, 2, 2)
            )
        else:
            # Standard UNETR Layers
            if self.depth >= 1:
                self.u11 = Deconv(self.embed_dim, 512); self.u12 = Deconv(512, 256); self.u13 = Deconv(256, 128); self.du3 = nn.ConvTranspose2d(128, 64, 2, 2)
                if self.depth >= 6: self.dc3 = DoubleConv(128 + 128, 128)
            if self.depth >= 6:
                self.u21 = Deconv(self.embed_dim, 512); self.u22 = Deconv(512, 256); self.du2 = nn.ConvTranspose2d(256, 128, 2, 2); self.dc2 = DoubleConv(256 + 256, 256)
            if self.depth >= 9:
                self.u31 = Deconv(self.embed_dim, 512); self.du1 = nn.ConvTranspose2d(512, 256, 2, 2); self.dc1 = DoubleConv(512 + 512, 512)
            if self.depth == 12:
                self.u41 = nn.ConvTranspose2d(self.embed_dim, 512, 2, 2)

    def forward(self, x):
        feats = self.vit(x)
        img = self.img_conv(x)

        # 3. ADD DIRECT UPSAMPLE FORWARD PATH
        if self.direct:
            # Take the deepest available feature (last in list)
            x_up = self.up_direct(feats[-1])
            # Concatenate with image features and output
            return self.segmentation_head(self.dc4(torch.cat((img, x_up), 1)))

        # Standard UNETR Forward Path
        if self.depth <= 3:
            x = self.du3(self.u13(self.u12(self.u11(feats[0]))))
            return self.segmentation_head(self.dc4(torch.cat((img, x), 1)))
        elif self.depth == 6:
            x_bot = self.u13(self.u12(self.u11(feats[0])))
            x_top = self.du2(self.u22(self.u21(feats[1])))
            return self.segmentation_head(self.dc4(torch.cat((img, self.du3(self.dc3(torch.cat((x_bot, x_top), 1)))), 1)))
        elif self.depth == 9:
            x1 = self.u13(self.u12(self.u11(feats[0])))
            x2 = self.u22(self.u21(feats[1]))
            x3 = self.du1(self.u31(feats[2]))
            x_mid = self.dc3(torch.cat((x1, self.du2(self.dc2(torch.cat((x2, x3), 1)))), 1))
            return self.segmentation_head(self.dc4(torch.cat((img, self.du3(x_mid)), 1)))
        elif self.depth == 12:
            x1 = self.u13(self.u12(self.u11(feats[0])))
            x2 = self.u22(self.u21(feats[1]))
            x3 = self.u31(feats[2]); x4 = self.u41(feats[3])
            x_mid = self.dc3(torch.cat((x1, self.du2(self.dc2(torch.cat((x2, self.du1(self.dc1(torch.cat((x3, x4), 1)))), 1)))), 1))
            return self.segmentation_head(self.dc4(torch.cat((img, self.du3(x_mid)), 1)))
        

class DoubleConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, dilation = 1):
		super().__init__()
		self.double_conv = nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class UNETR_ViT_prime(nn.Module):
    # def __init__(self, model_name='vit_base_patch16_224', num_classes=4, img_size=224, depth=12, direct_upsample=False, pretrained=True):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=4, depth=12, direct_upsample=False, pretrained=True):
        super().__init__()
        self.depth = depth
        self.direct = direct_upsample  # 1. SAVE THE ARGUMENT

        # Depth logic
        if self.depth == 12: indices = [2, 5, 8, 11]
        elif self.depth == 9: indices = [2, 5, 8]
        elif self.depth == 6: indices = [2, 5]
        elif 1 <= self.depth <= 3: indices = [self.depth - 1]
        else: raise ValueError(f"Depth {self.depth} is not allowed. Allowed depths are 1, 2, 3, 6, 9, 12")
        
        # self.vit = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, features_only=True, out_indices=indices)
        self.vit = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=indices)
        self.embed_dim = self.vit.feature_info[0]['num_chs']

        # self.img_conv = DoubleConv(3, 64)

        self.img_conv = nn.Sequential(DoubleConv(3, 64), DoubleConv(64, 128))
        self.dc4 = DoubleConv(128 + 64, 128)
        self.segmentation_head = nn.Conv2d(128, num_classes, 3, 1, 1)

        # self.dc4 = DoubleConv(64 + 64, 64)
        # self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)

        # 2. ADD DIRECT UPSAMPLE LAYER DEFINITION
        if self.direct:
            self.up_direct = nn.Sequential(
                Deconv(self.embed_dim, 512),
                Deconv(512, 256),
                Deconv(256, 128),
                nn.ConvTranspose2d(128, 64, 2, 2)
            )
        else:
            # Standard UNETR Layers
            if self.depth >= 1:
                self.u11 = Deconv(self.embed_dim, 512); self.u12 = Deconv(512, 256); self.u13 = Deconv(256, 128); self.du3 = nn.ConvTranspose2d(128, 64, 2, 2)
                if self.depth >= 6: self.dc3 = DoubleConv(128 + 128, 128)
            if self.depth >= 6:
                self.u21 = Deconv(self.embed_dim, 512); self.u22 = Deconv(512, 256); self.du2 = nn.ConvTranspose2d(256, 128, 2, 2); self.dc2 = DoubleConv(256 + 256, 256)
            if self.depth >= 9:
                self.u31 = Deconv(self.embed_dim, 512); self.du1 = nn.ConvTranspose2d(512, 256, 2, 2); self.dc1 = DoubleConv(512 + 512, 512)
            if self.depth == 12:
                self.u41 = nn.ConvTranspose2d(self.embed_dim, 512, 2, 2)

    def forward(self, x):
        feats = self.vit(x)
        img = self.img_conv(x)

        # 3. ADD DIRECT UPSAMPLE FORWARD PATH
        if self.direct:
            # Take the deepest available feature (last in list)
            x_up = self.up_direct(feats[-1])
            # Concatenate with image features and output
            return self.segmentation_head(self.dc4(torch.cat((img, x_up), 1)))

        # Standard UNETR Forward Path
        if self.depth <= 3:
            x = self.du3(self.u13(self.u12(self.u11(feats[0]))))
            return self.segmentation_head(self.dc4(torch.cat((img, x), 1)))
        elif self.depth == 6:
            x_bot = self.u13(self.u12(self.u11(feats[0])))
            x_top = self.du2(self.u22(self.u21(feats[1])))
            return self.segmentation_head(self.dc4(torch.cat((img, self.du3(self.dc3(torch.cat((x_bot, x_top), 1)))), 1)))
        elif self.depth == 9:
            x1 = self.u13(self.u12(self.u11(feats[0])))
            x2 = self.u22(self.u21(feats[1]))
            x3 = self.du1(self.u31(feats[2]))
            x_mid = self.dc3(torch.cat((x1, self.du2(self.dc2(torch.cat((x2, x3), 1)))), 1))
            return self.segmentation_head(self.dc4(torch.cat((img, self.du3(x_mid)), 1)))
        elif self.depth == 12:
            x1 = self.u13(self.u12(self.u11(feats[0])))
            x2 = self.u22(self.u21(feats[1]))
            x3 = self.u31(feats[2]); x4 = self.u41(feats[3])
            x_mid = self.dc3(torch.cat((x1, self.du2(self.dc2(torch.cat((x2, self.du1(self.dc1(torch.cat((x3, x4), 1)))), 1)))), 1))
            return self.segmentation_head(self.dc4(torch.cat((img, self.du3(x_mid)), 1)))

#################################################################################

# UNETR-SAM

#################################################################################

class UNETR_SAM(UNETR_ViT):
    def __init__(self, num_classes=4, depth=12, direct_upsample=False, pretrained=True):
        # super().__init__(model_name='samvit_base_patch16.sa1b', num_classes=num_classes, img_size=1024, depth=depth, direct_upsample=direct_upsample, pretrained=pretrained)
        super().__init__(model_name='samvit_base_patch16.sa1b', num_classes=num_classes, depth=depth, direct_upsample=direct_upsample, pretrained=pretrained)

class UNETR_SAM_prime(UNETR_ViT_prime):
    def __init__(self, num_classes=4, depth=12, direct_upsample=False, pretrained=True):
        # super().__init__(model_name='samvit_base_patch16.sa1b', num_classes=num_classes, img_size=1024, depth=depth, direct_upsample=direct_upsample, pretrained=pretrained)
        super().__init__(model_name='samvit_base_patch16.sa1b', num_classes=num_classes, depth=depth, direct_upsample=direct_upsample, pretrained=pretrained)


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

# UNet R50 and VGG16 - Compressed

#################################################################################


class CustomSMPCompressed(nn.Module):
    def __init__(
        self,
        arch='Unet',
        encoder_name='resnet34',
        encoder_weights='imagenet',
        num_classes=4,
        in_channels=3,
        encoder_depth=5,
        decoder_channels=None,
    ):
        """
        Args:
            arch (str): Architecture name. Options: 'Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'PAN'.
            encoder_name (str): Backbone encoder (e.g., 'resnet34', 'resnet50', 'vgg16', 'efficientnet-b0').
            encoder_weights (str or None): Pretrained weights ('imagenet', 'ssl', 'swsl') or None.
            num_classes (int): Number of output classes.
            in_channels (int): Input channels (3 for RGB, 1 for grayscale).
            encoder_depth (int): Depth of the encoder (3–5). Defaults to 5.
            decoder_channels (tuple or None): Decoder channels per stage. Must match encoder_depth.
                                              If None, defaults are chosen automatically based on depth.
        """
        super().__init__()

        # Default decoder channels per depth level
        default_decoder_channels = {
            3: (256, 128, 64),
            4: (256, 128, 64, 32),
            5: (256, 128, 64, 32, 16),
        }

        if decoder_channels is None:
            if encoder_depth not in default_decoder_channels:
                raise ValueError(f"encoder_depth must be 3, 4, or 5. Got {encoder_depth}.")
            decoder_channels = default_decoder_channels[encoder_depth]
        else:
            if len(decoder_channels) != encoder_depth:
                raise ValueError(
                    f"decoder_channels length ({len(decoder_channels)}) must match encoder_depth ({encoder_depth})."
                )

        # Some architectures don't support encoder_depth or decoder_channels
        # (e.g. DeepLabV3, PSPNet, PAN) — we handle that gracefully
        arch_supports_depth = arch.lower() not in ('deeplabv3', 'deeplabv3plus', 'pspnet', 'pan')

        if arch_supports_depth:
            self.model = smp.create_model(
                arch=arch,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                encoder_depth=encoder_depth,
                decoder_channels=decoder_channels,
            )
        else:
            print(f"[Warning] arch='{arch}' does not support encoder_depth/decoder_channels. Ignoring those args.")
            self.model = smp.create_model(
                arch=arch,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
            )

    def forward(self, x):
        return self.model(x)
    

#################################################################################

# TransUNet

#################################################################################


# 1. CONFIGURATION (Defaults for R50-ViT-B_16)
class TransUNetConfig:
    def __init__(self, img_size=224):
        
        self.patches = {'size': (16, 16), 'grid': (14, 14)}
        self.hidden_size = 768
        self.transformer = {
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'attention_dropout_rate': 0.0,
            'dropout_rate': 0.1,
        }
        self.resnet = {'num_layers': (3, 4, 9), 'width_factor': 1}
        self.decoder_channels = (256, 128, 64, 16)
        self.skip_channels = [512, 256, 64, 16]
        self.n_classes = 2
        self.n_skip = 3
        self.activation = 'softmax'
        self.classifier = 'seg'
        self.pretrained_path = None
        self.patch_size = 16

# 2. HELPER MODULES (StdConv, Activations)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

# 3. RESNET BACKBONE (Hybrid)

class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False) 
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]

# 4. TRANSFORMER MODULES

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(layer)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.config = config
        
        # ResNet Hybrid logic
        grid_size = config.patches["grid"]
        patch_size = (img_size // 16 // grid_size[0], img_size // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size // patch_size_real[0]) * (img_size // patch_size_real[1])  
        
        self.hybrid_model = ResNetV2(block_units=config.resnet['num_layers'], width_factor=config.resnet['width_factor'])
        in_channels = self.hybrid_model.width * 16
        
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)  
        x = x.flatten(2)
        x = x.transpose(-1, -2) 

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output) 
        return encoded, features

# 5. DECODER & MAIN MODEL

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels = config.skip_channels
        for i in range(4-config.n_skip): 
            skip_channels[3-i]=0

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size() 
        h, w = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x

class TransUNet(nn.Module):
    """
    TransUNet (R50-ViT-B_16) Implementation.
    """
    def __init__(self, img_size=224, num_classes=2):
        super(TransUNet, self).__init__()
        
        # 1. Initialize Configuration
        config = TransUNetConfig(img_size=img_size)
        config.n_classes = num_classes
        self.config = config
        
        # 2. Build Modules
        self.transformer = Transformer(config, img_size)
        self.decoder = DecoderCup(config)
        self.segmentation_head = nn.Conv2d(
            config.decoder_channels[-1],
            config.n_classes,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):        
        # Encoder
        x, features = self.transformer(x) 
        
        # Decoder
        x = self.decoder(x, features)
        
        # Head
        logits = self.segmentation_head(x)
        return logits