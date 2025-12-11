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

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:  # 如果配置中包含 "prompt"
            prompt_cfg = cfg.MODEL.PROMPT
            self.num_prompt = cfg.MODEL.PROMPT.NUM_TOKENS
        else:
            prompt_cfg = None
            self.num_prompt = 0

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:  #######
            # 如果传输类型不是 "end2end"，且不包含 "prompt"
            self.froze_enc = True  # 冻结编码器     ######
        else:
            # 否则
            self.froze_enc = False

        if cfg.MODEL.TRANSFER_TYPE == "adapter":  # 如果传输类型是 "adapter"
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)  # 构建骨干网络
        self.cfg = cfg
        self.setup_side()  # 设置辅助功能
        self.setup_head(cfg)  # 设置头部结构


    def setup_side(self):
        if self.cfg.MODEL.TRANSFER_TYPE != "side":  # 如果模型的传输类型不是“side”
            self.side = None  # 将self.side设置为None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))  # 初始化一个可学习的参数self.side_alpha
            m = models.alexnet(pretrained=True)  # 加载预训练的AlexNet模型
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),  # 包含AlexNet的特征提取部分
                ("avgpool", m.avgpool),  # 包含AlexNet的平均池化层
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim,
                                             bias=False)  # 定义一个线性层用于投影，输入维度为9216，输出维度为self.feat_dim，不使用偏置

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):  ##############################
        transfer_type = cfg.MODEL.TRANSFER_TYPE  # 获取传输类型
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg, load_pretrain, vis
        )  # 构建ViT模型，返回编码器和特征维度

        if transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.pretrained_weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False  # 只更新提示相关参数和嵌入层的权重和偏置

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False  # 只更新提示相关参数

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")  # 启用所有参数更新

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                     [cfg.DATA.NUMBER_CLASSES],  # noqa
            special_bias=True
        )  # 设置MLP头部网络结构，输入维度为self.feat_dim，输出维度为cfg.DATA.NUMBER_CLASSES

    def forward(self, x, only_attn=False, is_extract=False):
        if self.froze_enc and self.enc.training:  # 如果编码器被冻结且处于训练模式
            self.enc.eval()  # 将编码器设为评估模式
        x, attn_weights = self.enc(x, vis=True, is_extract=is_extract)  # 通过编码器获取特征表示，形状为(batch_size, self.feat_dim)

        if is_extract:
            x = x[:, self.num_prompt:, :]
            return x

        if only_attn:
            return attn_weights  # 返回注意力
        else:
            x = self.head(x)  # 将特征表示通过MLP头部网络进行分类
            return x  # 分类结果

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)  # 获取分类器层级的嵌入表示
        return cls_embeds  # 返回分类器层级的嵌入表示

    def get_features(self, x):
        """获取(batch_size, self.feat_dim)的特征表示"""
        x = self.enc(x)  # 通过编码器获取特征表示，形状为(batch_size, self.feat_dim)
        return x  # 返回特征表示
