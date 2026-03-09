import torch
import torch.nn as nn
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

class UNETR_ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=4, img_size=224, depth=12, direct_upsample=False, pretrained=True):
        super().__init__()
        self.depth = depth
        if self.depth == 12: indices = [2, 5, 8, 11]
        elif self.depth == 9: indices = [2, 5, 8]
        elif self.depth == 6: indices = [2, 5]
        elif self.depth <= 3: indices = [self.depth - 1]
        else: print('Allowed depths are 1, 2, 3, 6, 9, 12'); return None
        
        self.vit = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, features_only=True, out_indices=indices)
        self.embed_dim = self.vit.feature_info[0]['num_chs']
        self.img_conv = DoubleConv(3, 64)
        self.dc4 = DoubleConv(64 + 64, 64)
        self.segmentation_head = nn.Conv2d(64, num_classes, 3, 1, 1)

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

class UNETR_SAM(UNETR_ViT):
    def __init__(self, num_classes=4, depth=12, direct_upsample=False, pretrained=True):
        super().__init__(model_name='samvit_base_patch16.sa1b', num_classes=num_classes, img_size=1024, depth=depth, direct_upsample=direct_upsample, pretrained=pretrained)


# ==============================================================================
# 2. CUSTOM UNET INFO
# ==============================================================================

model_unet_custom = UNET(in_channels=3, out_channels=4, depth=4).to(DEVICE)
model_unet_custom.eval();

model_unet_custom_layers_of_interest = {
    'd1': model_unet_custom.d1,
    'd2': model_unet_custom.d2,
    'd3': model_unet_custom.d3,
    'd4': model_unet_custom.d4,
    'd5': model_unet_custom.d5, # Bottleneck
    'du1': model_unet_custom.du1,
    'du2': model_unet_custom.du2,
    'du3': model_unet_custom.du3,
    'du4': model_unet_custom.du4,
    'out': model_unet_custom.out
}

model_unet_custom_target_filters = {
    'd1': range(64),
    'd2': range(128),
    'd3': range(256),
    'd4': range(512),
    'd5': range(1024),
    'du1': range(512),
    'du2': range(256),
    'du3': range(128),
    'du4': range(64),
    'out': range(4) # Number of classes
}

model_unet_custom_target_layer_indexes = [1] * 10

# ==============================================================================
# 3. UNETR (ViT) INFO
# ==============================================================================

model_unetr_vit = UNETR_ViT(model_name='vit_base_patch16_224', num_classes=4, depth=12).to(DEVICE)
model_unetr_vit.eval();

# Note: We focus on the Decoder blocks here as the Encoder is hidden in the timm wrapper
model_unetr_vit_layers_of_interest = {
    'u41': model_unetr_vit.u41,  # Layer 12 Upsample
    'dc1': model_unetr_vit.dc1,  # Merge 12+9
    'dc2': model_unetr_vit.dc2,  # Merge 9+6
    'dc3': model_unetr_vit.dc3,  # Merge 6+3
    'du3': model_unetr_vit.du3,  # Final upsample before img merge
    'dc4': model_unetr_vit.dc4,  # Final merge with Image
    'head': model_unetr_vit.segmentation_head
}

model_unetr_vit_target_filters = {
    'u41': range(512),
    'dc1': range(512),
    'dc2': range(256),
    'dc3': range(128),
    'du3': range(64),
    'dc4': range(64),
    'head': range(4)
}

model_unetr_vit_target_layer_indexes = [1] * 7

# ==============================================================================
# 4. UNETR (SAM) INFO
# ==============================================================================

model_unetr_sam = UNETR_SAM(num_classes=4, depth=12).to(DEVICE)
model_unetr_sam.eval();

model_unetr_sam_layers_of_interest = {
    'u41': model_unetr_sam.u41,
    'dc1': model_unetr_sam.dc1,
    'dc2': model_unetr_sam.dc2,
    'dc3': model_unetr_sam.dc3,
    'du3': model_unetr_sam.du3,
    'dc4': model_unetr_sam.dc4,
    'head': model_unetr_sam.segmentation_head
}

model_unetr_sam_target_filters = {
    'u41': range(512),
    'dc1': range(512),
    'dc2': range(256),
    'dc3': range(128),
    'du3': range(64),
    'dc4': range(64),
    'head': range(4)
}

model_unetr_sam_target_layer_indexes = [1] * 7

# ==============================================================================
# 5. SMP UNET (ResNet50)
# ==============================================================================

model_smp_unet_r50 = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=4).to(DEVICE)
model_smp_unet_r50.eval();

model_smp_unet_r50_layers_of_interest = {
    # Encoder
    'enc_layer1': model_smp_unet_r50.encoder.layer1,
    'enc_layer2': model_smp_unet_r50.encoder.layer2,
    'enc_layer3': model_smp_unet_r50.encoder.layer3,
    'enc_layer4': model_smp_unet_r50.encoder.layer4,
    # Decoder (SMP blocks are indexed 0->4, where 0 is deepest)
    'dec_block0': model_smp_unet_r50.decoder.blocks[0],
    'dec_block1': model_smp_unet_r50.decoder.blocks[1],
    'dec_block2': model_smp_unet_r50.decoder.blocks[2],
    'dec_block3': model_smp_unet_r50.decoder.blocks[3],
    'dec_block4': model_smp_unet_r50.decoder.blocks[4],
    'head': model_smp_unet_r50.segmentation_head
}

model_smp_unet_r50_target_filters = {
    'enc_layer1': range(256),
    'enc_layer2': range(512),
    'enc_layer3': range(1024),
    'enc_layer4': range(2048),
    'dec_block0': range(256),
    'dec_block1': range(128),
    'dec_block2': range(64),
    'dec_block3': range(32),
    'dec_block4': range(16),
    'head': range(4)
}

model_smp_unet_r50_target_layer_indexes = [1] * 10

# ==============================================================================
# 6. SMP UNET (VGG16)
# ==============================================================================

model_smp_unet_vgg = smp.Unet(encoder_name="vgg16", encoder_weights="imagenet", in_channels=3, classes=4).to(DEVICE)
model_smp_unet_vgg.eval();

model_smp_unet_vgg_layers_of_interest = {
    # VGG Features (indices correspond to MaxPool layers usually)
    'features_3': model_smp_unet_vgg.encoder.features[3],
    'features_8': model_smp_unet_vgg.encoder.features[8],
    'features_15': model_smp_unet_vgg.encoder.features[15],
    'features_22': model_smp_unet_vgg.encoder.features[22],
    'features_29': model_smp_unet_vgg.encoder.features[29],
    # Decoder
    'dec_block0': model_smp_unet_vgg.decoder.blocks[0],
    'dec_block1': model_smp_unet_vgg.decoder.blocks[1],
    'dec_block2': model_smp_unet_vgg.decoder.blocks[2],
    'dec_block3': model_smp_unet_vgg.decoder.blocks[3],
    'dec_block4': model_smp_unet_vgg.decoder.blocks[4],
    'head': model_smp_unet_vgg.segmentation_head
}

model_smp_unet_vgg_target_filters = {
    'features_3': range(64),
    'features_8': range(128),
    'features_15': range(256),
    'features_22': range(512),
    'features_29': range(512),
    'dec_block0': range(256),
    'dec_block1': range(128),
    'dec_block2': range(64),
    'dec_block3': range(32),
    'dec_block4': range(16),
    'head': range(4)
}

model_smp_unet_vgg_target_layer_indexes = [1] * 11

# FCN-R50

#################################################################################


class TorchSegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']

weights = FCN_ResNet50_Weights.DEFAULT
model_fcn_r50 = fcn_resnet50(weights=weights).to(DEVICE)
model_fcn_r50 = TorchSegmentationWrapper(model_fcn_r50)
model_fcn_r50 = model_fcn_r50.eval();

# print(model_fcn_r50)

model_fcn_r50_layers_of_interest = {
        'act1': model_fcn_r50.model.backbone.maxpool,
        'layer1': model_fcn_r50.model.backbone.layer1,
        'layer2': model_fcn_r50.model.backbone.layer2,
        'layer3': model_fcn_r50.model.backbone.layer3,
        'layer4': model_fcn_r50.model.backbone.layer4,
        'classifier2': model_fcn_r50.model.classifier[2],
        'classifier4': model_fcn_r50.model.classifier[4],
    }


# Define the target filters for each layer

model_fcn_r50_target_filters = {
        'act1': range(64),
        'layer1': range(256),
        'layer2': range(512),
        'layer3': range(1024), # 1024
        'layer4': range(2048), # 2048
        'classifier2': range(512),
        'classifier4': range(21),
    }

model_fcn_r50_target_layer_indexes = [1] * 7

#################################################################################

# DeepLabv3-R50

#################################################################################


class TorchSegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model_deeplabv3_r50 = deeplabv3_resnet50(weights=weights).to(DEVICE)
model_deeplabv3_r50 = TorchSegmentationWrapper(model_deeplabv3_r50)
model_deeplabv3_r50 = model_deeplabv3_r50.eval();

# print(model_deeplabv3_r50)

model_deeplabv3_r50_layers_of_interest = {
        'act1': model_deeplabv3_r50.model.backbone.relu,
        'layer1': model_deeplabv3_r50.model.backbone.layer1,
        'layer2': model_deeplabv3_r50.model.backbone.layer2,
        'layer3': model_deeplabv3_r50.model.backbone.layer3,
        'layer4': model_deeplabv3_r50.model.backbone.layer4,
        'classifier1': model_deeplabv3_r50.model.classifier[1],  # Conv2d(256, 256)
        'classifier4': model_deeplabv3_r50.model.classifier[4],  # Conv2d(256, 21)
    }

# Define the target filters for each layer

model_deeplabv3_r50_target_filters = {
        'act1': range(64),
        'layer1': range(256),
        'layer2': range(512),
        'layer3': range(1024), # 1024
        'layer4': range(2048), # 2048
        'classifier1': range(256),  # Conv2d(256, 256) at classifier[1]
        'classifier4': range(21),   # Conv2d(256, 21) at classifier[4]
    }

model_deeplabv3_r50_target_layer_indexes = [1] * 7