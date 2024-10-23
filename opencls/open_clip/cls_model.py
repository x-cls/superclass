# --------------------------------------------------------
# Copyright (2024) Bytedance Ltd. and/or its affiliates 
# Licensed under the Apache License, Version 2.0 (the "License")
# SuperClass Project
# Written by Zilong Huang
# --------------------------------------------------------
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
    MixClsHead,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class ClassHeadCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    layers: int = 1


def _build_cls_head(
        width,
        clshead_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    clshead_cfg = ClassHeadCfg(**clshead_cfg) if isinstance(clshead_cfg, dict) else clshead_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    head = MixClsHead(
        width=width,
        layers=clshead_cfg.layers,
        mlp_ratio=clshead_cfg.mlp_ratio,
        act_layer=act_layer,
        norm_layer=norm_layer,
        output_dim=clshead_cfg.vocab_size,
    )

    return head


class Classifier(nn.Module):
    def __init__(
            self,
            embed_dim,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        clshead_cfg = ClassHeadCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        vocab_size = clshead_cfg.vocab_size

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_cls_head(
            embed_dim if embed_dim else vision_cfg.width,
            clshead_cfg=clshead_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.register_buffer("cap_fq", torch.zeros([1, vocab_size], dtype=torch.float64))
        self.register_buffer("num_samples", torch.zeros([1, 1], dtype=torch.float64))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        # self.text.set_grad_checkpointing(enable)
        # self.text_decoder.set_grad_checkpointing(enable)

    def forward(self, image, text, image_embs=None):
        if image_embs is None:
            image_embs = self.visual(image)

        logits = self.text_decoder(image_embs)
        labels = text.clone()

        return {
            "cap_fq": self.cap_fq,
            "num_samples": self.num_samples,
            "logits": logits,
            "labels": labels,
            "logit_scale": torch.ones([1]),
        }