#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models, build_swin_model,
    build_mocov3_model, build_mae_model
)
from .mlp import MLP
from ..utils import logging

logger = logging.get_logger("visual_prompt")


class ViT(nn.Module):
    """ViT 相关模型。"""

    def __init__(self, cfg, load_pretrain=False, vis=True):     #
        super(ViT, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
            self.num_prompt = cfg.MODEL.PROMPT.NUM_TOKENS
        else:
            prompt_cfg = None
            self.num_prompt = 0

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            self.froze_enc = True
        else:
            self.froze_enc = False

        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.setup_side()
        self.setup_head(cfg)


    def setup_side(self):
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim,
                                             bias=False)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg, load_pretrain, vis
        )

        if transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.pretrained_weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                     [cfg.DATA.NUMBER_CLASSES],  # noqa
            special_bias=True
        )

    def forward(self, x, only_attn=False, is_extract=False):
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x, attn_weights = self.enc(x, vis=True, is_extract=is_extract)

        if is_extract:
            x = x[:, self.num_prompt:, :]
            return x
        if only_attn:
            return attn_weights
        else:
            x = self.head(x)
            return x

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        x = self.enc(x)
        return x
