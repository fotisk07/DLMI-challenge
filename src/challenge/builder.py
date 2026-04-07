# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torchvision
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, _cfg
from torchvision import models

__all__ = [
    "vit_small",
    "vit_base",
    "vit_conv_small",
    "vit_conv_base",
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, pretext_token=True, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # inserting a new token
        self.num_prefix_tokens += 1 if pretext_token else 0
        self.pretext_token = nn.Parameter(torch.ones(1, 1, self.embed_dim)) if pretext_token else None
        embed_len = self.patch_embed.num_patches if self.no_embed_class else self.patch_embed.num_patches + 1
        embed_len += 1 if pretext_token else 0
        self.embed_len = embed_len

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pretext_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "pretext_token"}

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def _ref_embed(self, ref):
        B, C, H, W = ref.shape
        ref = self.patch_embed.proj(ref)
        if self.patch_embed.flatten:
            ref = ref.flatten(2).transpose(1, 2)  # BCHW -> BNC
        ref = self.patch_embed.norm(ref)
        return ref

    def _pos_embed_with_ref(self, x, ref):
        pretext_tokens = self.pretext_token.expand(x.shape[0], -1, -1) * 0 + ref
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, ref=None):
        x = self.patch_embed(x)
        if ref is None:
            x = self._pos_embed(x)
        else:
            ref = self._ref_embed(ref).mean(dim=1, keepdim=True)
            x = self._pos_embed_with_ref(x, ref)
        # x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 1]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, ref=None):
        x_out = self.forward_features(x, ref)
        x = self.forward_head(x_out)
        return x_out, x

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, (
            "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        )
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[
            None, :, :
        ]

        assert self.num_prefix_tokens == 2, "Assuming two and only two tokens, [pretext][cls]"
        pe_token = torch.zeros([1, 2, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


def PathoDuet(path, freeze_backbone, num_class=2):
    model = VisionTransformerMoCo(pretext_token=True, global_pool="avg")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.head = nn.Linear(768, num_class)
    return model


def kimiaNet(
    pretrained=False, freeze_backbone=False, num_classes=2, path="weights/KimiaNetPyTorchWeights.pth"
):
    class fully_connected(nn.Module):
        """docstring for BottleNeck"""

        def __init__(self, model, num_ftrs, num_classes):
            super().__init__()
            self.model = model
            self.fc_4 = nn.Linear(num_ftrs, num_classes)

        def forward(self, x):
            x = self.model(x)
            x = torch.flatten(x, 1)
            out_1 = x
            out_3 = self.fc_4(x)
            return out_3

    # model = torchvision.models.densenet121(pretrained=pretrained)
    model = torchvision.models.densenet121()
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    num_ftrs = model.classifier.in_features
    model_final = fully_connected(model.features, num_ftrs, 30)
    model = model
    model_final = model_final
    # model_final = nn.DataParallel(model_final)

    if pretrained:
        state_dict = torch.load(path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_final.load_state_dict(
            state_dict,
            strict=True,
        )

    model_final.fc_4 = nn.Linear(num_ftrs, num_classes)

    return model_final


def build_model(arch, pretrained) -> nn.Module:
    if arch == "kimianet":
        arch = kimiaNet(pretrained, freeze_backbone=True)
    else:
        weights = "IMAGENET1K_V1" if pretrained else None
        arch = getattr(models, arch)(weights=weights)

        for param in arch.parameters():
            param.requires_grad = False

        # Replace the final fully-connected layer with a binary head (single logit)
        in_features = arch.fc.in_features
        arch.fc = nn.Linear(in_features, 2)

    return arch
