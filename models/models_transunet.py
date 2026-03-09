# coding=utf-8
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from os.path import join as pjoin
from scipy import ndimage

#################################################################################

# TransUNet

#################################################################################


# ── NPZ key constants (JAX/Flax naming inside the .npz) ──────────────────────
ATTENTION_Q    = "MultiHeadDotProductAttention_1/query"
ATTENTION_K    = "MultiHeadDotProductAttention_1/key"
ATTENTION_V    = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT  = "MultiHeadDotProductAttention_1/out"
FC_0           = "MlpBlock_3/Dense_0"
FC_1           = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM       = "LayerNorm_2"

def np2th(weights, conv=False):
    """NumPy array → PyTorch tensor, optionally transposing HWIO → OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────

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
        self.patch_size = 16


# ── 2. HELPER MODULES ────────────────────────────────────────────────────────

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


# ── 3. RESNET BACKBONE ───────────────────────────────────────────────────────

class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1    = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1  = conv1x1(cin, cmid, bias=False)
        self.gn2    = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2  = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3    = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3  = conv1x1(cmid, cout, bias=False)
        self.relu   = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj    = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        """Load weights for one bottleneck unit from the .npz file."""
        prefix = f"{n_block}/{n_unit}/"
        with torch.no_grad():
            self.conv1.weight.copy_(np2th(weights[prefix + "conv1/kernel"], conv=True))
            self.conv2.weight.copy_(np2th(weights[prefix + "conv2/kernel"], conv=True))
            self.conv3.weight.copy_(np2th(weights[prefix + "conv3/kernel"], conv=True))
            self.gn1.weight.copy_(np2th(weights[prefix + "gn1/scale"]).view(-1))
            self.gn1.bias.copy_(  np2th(weights[prefix + "gn1/bias"]).view(-1))
            self.gn2.weight.copy_(np2th(weights[prefix + "gn2/scale"]).view(-1))
            self.gn2.bias.copy_(  np2th(weights[prefix + "gn2/bias"]).view(-1))
            self.gn3.weight.copy_(np2th(weights[prefix + "gn3/scale"]).view(-1))
            self.gn3.bias.copy_(  np2th(weights[prefix + "gn3/bias"]).view(-1))
            if hasattr(self, 'downsample'):
                self.downsample.weight.copy_(np2th(weights[prefix + "conv_proj/kernel"], conv=True))
                self.gn_proj.weight.copy_(np2th(weights[prefix + "gn_proj/scale"]).view(-1))
                self.gn_proj.bias.copy_(  np2th(weights[prefix + "gn_proj/bias"]).view(-1))


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width      = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn',   nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width,    cout=width*4,  cmid=width))] +
                [(f'unit{i}', PreActBottleneck(cin=width*4,  cout=width*4,  cmid=width))  for i in range(2, block_units[0]+1)]))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4,  cout=width*8,  cmid=width*2, stride=2))] +
                [(f'unit{i}', PreActBottleneck(cin=width*8,  cout=width*8,  cmid=width*2)) for i in range(2, block_units[1]+1)]))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8,  cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2]+1)]))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, :x.size()[2], :x.size()[3]] = x
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ── 4. TRANSFORMER MODULES ───────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size        = self.num_attention_heads * self.attention_head_size

        self.query        = nn.Linear(config.hidden_size, self.all_head_size)
        self.key          = nn.Linear(config.hidden_size, self.all_head_size)
        self.value        = nn.Linear(config.hidden_size, self.all_head_size)
        self.out          = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax      = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs  = self.attn_dropout(self.softmax(scores))

        ctx   = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx   = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
        return self.proj_dropout(self.out(ctx))


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1     = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2     = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn  = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act_fn(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size    = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm       = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn  = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        x = x + self.attn(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def load_from(self, weights, n_block):
        """Load one transformer encoder block from the .npz file."""
        ROOT = f"Transformer/encoderblock_{n_block}"
        H    = self.hidden_size
        with torch.no_grad():
            # Attention weights
            self.attn.query.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_Q,   "kernel")]).view(H, H).t())
            self.attn.key.weight.copy_(  np2th(weights[pjoin(ROOT, ATTENTION_K,   "kernel")]).view(H, H).t())
            self.attn.value.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_V,   "kernel")]).view(H, H).t())
            self.attn.out.weight.copy_(  np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(H, H).t())
            self.attn.query.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_Q,   "bias")]).view(-1))
            self.attn.key.bias.copy_(  np2th(weights[pjoin(ROOT, ATTENTION_K,   "bias")]).view(-1))
            self.attn.value.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_V,   "bias")]).view(-1))
            self.attn.out.bias.copy_(  np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1))
            # MLP weights
            self.ffn.fc1.weight.copy_(np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t())
            self.ffn.fc2.weight.copy_(np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t())
            self.ffn.fc1.bias.copy_(  np2th(weights[pjoin(ROOT, FC_0, "bias")]))
            self.ffn.fc2.bias.copy_(  np2th(weights[pjoin(ROOT, FC_1, "bias")]))
            # LayerNorms
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(  np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(  np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer        = nn.ModuleList([Block(config) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        for block in self.layer:
            hidden_states = block(hidden_states)
        return self.encoder_norm(hidden_states)


class Embeddings(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config    = config
        grid_size      = config.patches["grid"]
        patch_size     = (img_size // 16 // grid_size[0], img_size // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches      = (img_size // patch_size_real[0]) * (img_size // patch_size_real[1])

        self.hybrid_model      = ResNetV2(block_units=config.resnet['num_layers'], width_factor=config.resnet['width_factor'])
        in_channels            = self.hybrid_model.width * 16
        self.patch_embeddings  = nn.Conv2d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout           = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings), features


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder    = Encoder(config)

    def forward(self, x):
        embeddings, features = self.embeddings(x)
        return self.encoder(embeddings), features


# ── 5. DECODER ───────────────────────────────────────────────────────────────

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels,                out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up    = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config    = config
        head_channels  = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)

        in_channels    = [head_channels] + list(config.decoder_channels[:-1])
        out_channels   = config.decoder_channels
        skip_channels  = list(config.skip_channels)
        for i in range(4 - config.n_skip):
            skip_channels[3 - i] = 0

        self.blocks = nn.ModuleList([
            DecoderBlock(ic, oc, sc) for ic, oc, sc in zip(in_channels, out_channels, skip_channels)
        ])

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(math.sqrt(n_patch))
        x = self.conv_more(hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w))
        for i, block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = block(x, skip=skip)
        return x


# ── 6. MAIN MODEL ────────────────────────────────────────────────────────────

class TransUNet(nn.Module):
    """
    TransUNet (R50-ViT-B_16).

    Args:
        img_size   : Input image size (default 224).
        num_classes: Number of segmentation output classes (default 2).
        pretrained : If True, loads R50+ViT-B_16.npz from the 'imagenet21k'
                     folder that sits in the same directory as this file.
                     Pass a string path to use a custom .npz location instead.
    """
    def __init__(self, img_size=224, num_classes=2, pretrained=True):
        super().__init__()

        config           = TransUNetConfig(img_size=img_size)
        config.n_classes = num_classes
        self.config      = config

        self.transformer      = Transformer(config, img_size)
        self.decoder          = DecoderCup(config)
        self.segmentation_head = nn.Conv2d(config.decoder_channels[-1], num_classes, kernel_size=3, padding=1)

        if pretrained:
            if isinstance(pretrained, str):
                npz_path = pretrained
            else:
                # Default: imagenet21k/ next to this file
                npz_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "imagenet21k",
                    "R50+ViT-B_16.npz",
                )
            self.load_from(npz_path)

    def forward(self, x):
        if x.size(1) == 1:               # accept single-channel inputs
            x = x.repeat(1, 3, 1, 1)
        x, features = self.transformer(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)

    def load_from(self, npz_path: str):
        """Load R50+ViT-B_16 ImageNet-21k pretrained weights from a .npz file."""
        print(f"Loading pretrained weights from: {npz_path}")
        weights = np.load(npz_path)

        with torch.no_grad():
            # ── Patch embeddings ───────────────────────────────────────────
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))

            # ── Encoder norm ───────────────────────────────────────────────
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            # ── Position embeddings (with interpolation if sizes differ) ───
            posemb     = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size(1) - 1 == posemb_new.size(1):
                # drop CLS token
                self.transformer.embeddings.position_embeddings.copy_(posemb[:, 1:])
            else:
                print("Resizing position embeddings: %s → %s" % (posemb.size(), posemb_new.size()))
                ntok_new     = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old       = int(np.sqrt(len(posemb_grid)))
                gs_new       = int(np.sqrt(ntok_new))
                posemb_grid  = posemb_grid.reshape(gs_old, gs_old, -1)
                posemb_grid  = ndimage.zoom(posemb_grid, (gs_new / gs_old, gs_new / gs_old, 1), order=1)
                posemb_grid  = posemb_grid.reshape(1, gs_new * gs_new, -1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb_grid))

            # ── Transformer encoder blocks ─────────────────────────────────
            for i, block in enumerate(self.transformer.encoder.layer):
                block.load_from(weights, n_block=i)

            # ── ResNet hybrid backbone ─────────────────────────────────────
            self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                np2th(weights["conv_root/kernel"], conv=True))
            self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(
                np2th(weights["gn_root/scale"]).view(-1))
            self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(
                np2th(weights["gn_root/bias"]).view(-1))

            for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)

        print("Pretrained weights loaded successfully.")
        print(f"✓ Loaded pretrained R50+ViT-B_16 weights from {npz_path}")