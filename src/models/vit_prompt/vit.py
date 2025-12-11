#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")  # 获取名为 "visual_prompt" 的日志记录器



class PromptedTransformer(Transformer):           #在这里改，输入图像改为输入特征
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"  # 确保提示位置在前
        assert prompt_config.INITIATION == "random"  # 确保初始化方式为随机
        assert prompt_config.NUM_DEEP_LAYERS is None  # 确保没有深层共享
        assert not prompt_config.DEEP_SHARED  # 确保没有深层共享
        super(PromptedTransformer, self).__init__(  # 调用父类（Transformer）的初始化方法
            config, img_size, vis)

        self.prompt_config = prompt_config  # 存储提示配置
        self.vit_config = config  # 存储ViT（视觉Transformer）配置

        img_size = _pair(img_size)  # 将图像尺寸转换为元组形式，如果输入是单个整数，则转换为 (img_size, img_size)
        patch_size = _pair(config.patches["size"])  # 将patch尺寸转换为元组形式，如果输入是单个整数，则转换为 (patch_size, patch_size)

        num_tokens = self.prompt_config.NUM_TOKENS  # 获取提示配置中的token数量

        self.num_tokens = num_tokens  # 提示令牌的数量

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # 如果需要对提示嵌入进行投影
        if self.prompt_config.PROJECT > -1:
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)  # 为提示嵌入投影
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')  # 使用kaiming_normal_初始化
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()  # 否则使用身份矩阵作为投影

        # 初始化提示:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # 计算初始化值

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))  # 初始化提示嵌入参数
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)  # 使用均匀分布初始化

            if self.prompt_config.DEEP:  # 如果需要深度提示
                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))  # 初始化深度提示嵌入参数
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)  # 使用均匀分布初始化

        else:
            raise ValueError("Other initiation scheme is not supported")  # 不支持其他初始化方案

    def incorporate_prompt(self, x):
        # 将提示嵌入与图像块嵌入结合
        B = x.shape[0]  # 批量大小
        x = self.embeddings(x)  # 获取图像块嵌入 (batch_size, 1 + n_patches, hidden_dim)##################
        x = torch.cat((
            x[:, :1, :],  # 保留CLS令牌
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),  # 添加提示嵌入
            x[:, 1:, :]  # 添加图像块嵌入
        ), dim=1)

        return x

    def train(self, mode=True):
        # 设置此类的训练状态

        if mode:
            # 训练模式:

            self.encoder.eval()  # 禁用编码器
            self.embeddings.eval()  # 禁用嵌入层
            self.prompt_proj.train()  # 启用提示投影
            self.prompt_dropout.train()  # 启用提示丢弃
        else:
            # 评估模式:

            for module in self.children():
                module.train(mode)  # 设置所有子模块的训练模式

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []  # 存储注意力权重
        hidden_states = None  # 初始化隐藏状态
        weights = None  # 初始化权重
        B = embedding_output.shape[0]  # 获取批量大小
        num_layers = self.vit_config.transformer["num_layers"]  # 获取层数

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)  # 编码器的第一层
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:  # 检查是否有深度提示
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))  # 获取深度提示嵌入
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],  # 保留CLS令牌
                        deep_prompt_emb,  # 添加深度提示嵌入
                        hidden_states[:, (1 + self.num_tokens):, :]  # 保留剩余的隐藏状态
                    ), dim=1)

                hidden_states, weights = self.encoder.layer[i](hidden_states)  # 编码器的其他层

            if self.encoder.vis:
                attn_weights.append(weights)  # 将注意力权重存储起来

        encoded = self.encoder.encoder_norm(hidden_states)  # 对隐藏状态进行归一化处理
        return encoded, attn_weights  # 返回编码后的结果和注意力权重

    def forward(self, x):
        # 默认版本：
        embedding_output = self.incorporate_prompt(x)  # 将提示嵌入与输入结合

        if self.prompt_config.DEEP:  # 如果使用了深度提示
            encoded, attn_weights = self.forward_deep_prompt(embedding_output)  # 使用深度提示进行前向传播
        else:
            encoded, attn_weights = self.encoder(embedding_output)  # 使用普通的编码器进行前向传播

        return encoded, attn_weights  # 返回编码后的结果和注意力权重

#############################################
class PromptedVisionTransformer(VisionTransformer):   ##########继承父类VisionTransformer
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False          #########
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"  # 断言VIT池化类型为"original"
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)  # 调用父类的初始化方法 ####### 转到vit_backbones\vit.py VisionTransformer
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")  # 如果使用PromptedVisionTransformer，则prompt_cfg不能为None
        self.prompt_cfg = prompt_cfg  # 设置prompt_cfg属性
        vit_cfg = CONFIGS[model_type]  # 获取ViT模型的配置
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis=True)  # 构建PromptedTransformer模型

    def forward(self, x, vis=True, is_extract=False):
        x, attn_weights = self.transformer(x)  # 使用PromptedTransformer处理输入x，获取输出和注意力权重   x是Transformer 处理后的输出特征
        if is_extract:
            x = x[:, 1:, :]
            return x, attn_weights
        x = x[:, 0]  # 取输出的第一个位置的特征
        logits = self.head(x)  # 将特征通过头部网络得到分类结果
        if not vis:  # 如果不需要可视化
            return logits  # 返回分类结果
        return logits, attn_weights  # 返回分类结果和注意力权重

