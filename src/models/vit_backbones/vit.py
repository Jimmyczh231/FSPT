#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
models for vits, borrowed from
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling_resnet.py
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
"""
import copy
import logging
import math

from os.path import join as pjoin
from turtle import forward

import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from ...configs import vit_configs as configs


logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)  # 获取一个记录器对象，用于记录日志

CONFIGS = {
    # "sup_vitb8": configs.get_b16_config(),  # 获取 B16 配置，暂时注释掉
    "sup_vitb16_224": configs.get_b16_config(),  # 获取 B16 配置
    "sup_vitb16": configs.get_b16_config(),  # 获取 B16 配置
    "sup_vitl16_224": configs.get_l16_config(),  # 获取 L16 配置
    "sup_vitl16": configs.get_l16_config(),  # 获取 L16 配置
    "sup_vitb16_imagenet21k": configs.get_b16_config(),  # 获取 B16 配置，针对 ImageNet-21k
    "sup_vitl16_imagenet21k": configs.get_l16_config(),  # 获取 L16 配置，针对 ImageNet-21k
    "sup_vitl32_imagenet21k": configs.get_l32_config(),  # 获取 L32 配置，针对 ImageNet-21k
    'sup_vitb32_imagenet21k': configs.get_b32_config(),  # 获取 B32 配置，针对 ImageNet-21k
    'sup_vitb8_imagenet21k': configs.get_b8_config(),  # 获取 B8 配置，针对 ImageNet-21k
    'sup_vith14_imagenet21k': configs.get_h14_config(),  # 获取 H14 配置，针对 ImageNet-21k
    # 'R50-ViT-B_16': configs.get_r50_b16_config(),  # 获取 R50-B16 配置，暂时注释掉
}

ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"  # 多头点积注意力机制的查询权重路径
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"  # 多头点积注意力机制的键权重路径
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"  # 多头点积注意力机制的值权重路径
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"  # 多头点积注意力机制的输出权重路径
FC_0 = "MlpBlock_3/Dense_0/"  # MLP块中第一个全连接层的权重路径
FC_1 = "MlpBlock_3/Dense_1/"  # MLP块中第二个全连接层的权重路径
ATTENTION_NORM = "LayerNorm_0/"  # 第一个层归一化层的权重路径
MLP_NORM = "LayerNorm_2/"  # 第二个层归一化层的权重路径



def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:  # 如果是卷积层的权重
        weights = weights.transpose([3, 2, 0, 1])  # 将权重从 HWIO 格式转换为 OIHW 格式
    return torch.from_numpy(weights)  # 将 NumPy 数组转换为 PyTorch 张量

def swish(x):
    return x * torch.sigmoid(x)  # Swish 激活函数，定义为 x * sigmoid(x)

ACT2FN = {
    "gelu": torch.nn.functional.gelu,  # GELU 激活函数
    "relu": torch.nn.functional.relu,  # ReLU 激活函数
    "swish": swish  # Swish 激活函数
}



class Attention(nn.Module):  ###################
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis  # 是否可视化注意力权重
        self.num_attention_heads = config.transformer["num_heads"]  # 注意力头的数量
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有注意力头的总大小

        self.query = Linear(config.hidden_size, self.all_head_size)  # 查询向量的线性层
        self.key = Linear(config.hidden_size, self.all_head_size)  # 键向量的线性层
        self.value = Linear(config.hidden_size, self.all_head_size)  # 值向量的线性层

        self.out = Linear(config.hidden_size, config.hidden_size)  # 输出的线性层
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])  # 注意力层的 dropout
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])  # 输出投影的 dropout

        self.softmax = Softmax(dim=-1)  # 软最大化函数，用于计算注意力权重

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # 新的形状：批量大小, 序列长度, 注意力头数量, 每个注意力头的大小
        x = x.view(*new_x_shape)  # 调整张量形状
        return x.permute(0, 2, 1, 3)  # 置换张量维度以匹配注意力计算的需求

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # 计算查询向量：批量大小, 序列长度, 所有注意力头的总大小
        mixed_key_layer = self.key(hidden_states)  # 计算键向量
        mixed_value_layer = self.value(hidden_states)  # 计算值向量

        query_layer = self.transpose_for_scores(mixed_query_layer)  # 调整查询向量的形状：批量大小, 注意力头数量, 序列长度, 每个注意力头的大小
        key_layer = self.transpose_for_scores(mixed_key_layer)  # 调整键向量的形状
        value_layer = self.transpose_for_scores(mixed_value_layer)  # 调整值向量的形状

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # 计算注意力得分：批量大小, 注意力头数量, 序列长度, 序列长度
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 缩放注意力得分
        attention_probs = self.softmax(attention_scores)  # 计算注意力权重：批量大小, 注意力头数量, 序列长度 (查询), 序列长度 (键)
        weights = attention_probs if self.vis else None  # 如果需要可视化，则保存注意力权重
        attention_probs = self.attn_dropout(attention_probs)  # 对注意力权重应用 dropout

        context_layer = torch.matmul(attention_probs, value_layer)  # 计算上下文层：批量大小, 注意力头数量, 序列长度, 每个注意力头的大小
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 调整上下文层的形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 新的上下文层形状：批量大小, 序列长度, 所有注意力头的总大小
        context_layer = context_layer.view(*new_context_layer_shape)  # 调整张量形状
        attention_output = self.out(context_layer)  # 通过线性层计算最终输出
        attention_output = self.proj_dropout(attention_output)  # 对输出应用 dropout
        return attention_output, weights  # 返回注意力输出和可视化权重（如果需要）



class Mlp(nn.Module):   #########
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])  # 第一个全连接层，输入维度为 hidden_size，输出维度为 mlp_dim
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)  # 第二个全连接层，输入维度为 mlp_dim，输出维度为 hidden_size
        self.act_fn = ACT2FN["gelu"]  # 使用 GELU 激活函数
        self.dropout = Dropout(config.transformer["dropout_rate"])  # Dropout 层，使用指定的 dropout 率

        self._init_weights()  # 初始化权重

    def _init_weights(self):   ########
        nn.init.xavier_uniform_(self.fc1.weight)  # 使用 Xavier 均匀分布初始化 fc1 的权重
        nn.init.xavier_uniform_(self.fc2.weight)  # 使用 Xavier 均匀分布初始化 fc2 的权重
        nn.init.normal_(self.fc1.bias, std=1e-6)  # 使用标准差为 1e-6 的正态分布初始化 fc1 的偏置
        nn.init.normal_(self.fc2.bias, std=1e-6)  # 使用标准差为 1e-6 的正态分布初始化 fc2 的偏置

    def forward(self, x):
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.act_fn(x)  # 通过激活函数
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc2(x)  # 通过第二个全连接层
        x = self.dropout(x)  # 再次应用 Dropout
        return x  # 返回输出

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None  # 初始化 hybrid 模式为 None
        img_size = _pair(img_size)  # 将 img_size 转换为 (img_size, img_size)

        if config.patches.get("grid") is not None:  # 如果配置中定义了网格大小       ###########
            grid_size = config.patches["grid"]  # 获取网格大小
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])  # 计算每个 patch 的大小
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)  # 计算总的 patch 数量
            self.hybrid = True  # 设置 hybrid 模式为 True
        else:  # 如果配置中没有定义网格大小
            patch_size = _pair(config.patches["size"])  # 获取 patch 的大小  ####################
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 计算总的 patch 数量
            self.hybrid = False  # 设置 hybrid 模式为 False

        if self.hybrid:  # 如果是 hybrid 模式
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)  # 初始化 ResNetV2 模型
            in_channels = self.hybrid_model.width * 16  # 更新输入通道数


        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)  # 定义卷积层，用于生成 patch 嵌入#############
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))  # 位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))  # 分类 token

        self.dropout = Dropout(config.transformer["dropout_rate"])  # Dropout 层，使用指定的 dropout 率

    def forward(self, x):
        B = x.shape[0]  # 获取批量大小
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展分类 token，以匹配批量大小
        if self.hybrid:  # 如果是 hybrid 模式
            x = self.hybrid_model(x)  # 通过 hybrid 模型处理输入
        x = self.patch_embeddings(x)  # 通过 patch 嵌入层           #################################
        x = x.flatten(2)  # 展平为二维
        x = x.transpose(-1, -2)  # 转置张量，以匹配后续操作
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类 token 连接到 patch 嵌入之前

        embeddings = x + self.position_embeddings  # 添加位置嵌入
        embeddings = self.dropout(embeddings)  # 应用 Dropout
        return embeddings  # 返回嵌入

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)  # 注意力层的 LayerNorm
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)  # 前馈网络的 LayerNorm
        self.ffn = Mlp(config)  # 前馈网络
        self.attn = Attention(config, vis)  # 注意力层          ############

    def forward(self, x):
        h = x  # 保存输入 x
        x = self.attention_norm(x)  # 对 x 应用 LayerNorm
        x, weights = self.attn(x)  # 通过注意力层，返回 x 和注意力权重
        x = x + h  # 残差连接

        h = x  # 保存当前 x
        x = self.ffn_norm(x)  # 对 x 应用 LayerNorm
        x = self.ffn(x)  # 通过前馈网络
        x = x + h  # 残差连接
        return x, weights  # 返回输出 x 和注意力权重

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"  # 定义根路径
        with torch.no_grad():  # 不计算梯度
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()  # 加载并转置查询权重
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()  # 加载并转置键权重
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()  # 加载并转置值权重
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()  # 加载并转置输出权重

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)  # 加载查询偏置
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)  # 加载键偏置
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)  # 加载值偏置
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)  # 加载输出偏置

            self.attn.query.weight.copy_(query_weight)  # 复制查询权重
            self.attn.key.weight.copy_(key_weight)  # 复制键权重
            self.attn.value.weight.copy_(value_weight)  # 复制值权重
            self.attn.out.weight.copy_(out_weight)  # 复制输出权重
            self.attn.query.bias.copy_(query_bias)  # 复制查询偏置
            self.attn.key.bias.copy_(key_bias)  # 复制键偏置
            self.attn.value.bias.copy_(value_bias)  # 复制值偏置
            self.attn.out.bias.copy_(out_bias)  # 复制输出偏置

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()  # 加载并转置 MLP 的第一个全连接层权重
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()  # 加载并转置 MLP 的第二个全连接层权重
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()  # 加载 MLP 的第一个全连接层偏置
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()  # 加载 MLP 的第二个全连接层偏置

            self.ffn.fc1.weight.copy_(mlp_weight_0)  # 复制 MLP 的第一个全连接层权重
            self.ffn.fc2.weight.copy_(mlp_weight_1)  # 复制 MLP 的第二个全连接层权重
            self.ffn.fc1.bias.copy_(mlp_bias_0)  # 复制 MLP 的第一个全连接层偏置
            self.ffn.fc2.bias.copy_(mlp_bias_1)  # 复制 MLP 的第二个全连接层偏置

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))  # 复制注意力层的 LayerNorm 权重
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))  # 复制注意力层的 LayerNorm 偏置
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))  # 复制前馈网络的 LayerNorm 权重
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))  # 复制前馈网络的 LayerNorm 偏置


class Encoder(nn.Module):  ########################################
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis  # 可视化标志
        self.layer = nn.ModuleList()  # 存放各层的模块列表
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)  # LayerNorm 层，用于对输出进行归一化
        for _ in range(config.transformer["num_layers"]):  # 根据配置中的层数构建每一层
            layer = Block(config, vis)  # 创建一个 Block 实例
            self.layer.append(copy.deepcopy(layer))  # 深复制并添加到模块列表中

    def forward(self, hidden_states):
        attn_weights = []  # 初始化注意力权重列表
        for layer_block in self.layer:  # 遍历每一层
            hidden_states, weights = layer_block(hidden_states)  # 通过每一层，返回更新后的 hidden_states 和注意力权重
            if self.vis:  # 如果需要可视化
                attn_weights.append(weights)  # 将权重添加到列表中
        encoded = self.encoder_norm(hidden_states)  # 对最后的 hidden_states 进行归一化
        return encoded, attn_weights  # 返回编码后的表示和注意力权重

    def forward_cls_layerwise(self, hidden_states):
        # hidden_states: B, 1+n_patches, dim  # 输入的 hidden_states 尺寸说明

        if hidden_states.size(0) != 1:  # 检查 batch size 是否为 1
            raise ValueError('not support batch-wise cls forward yet')  # 如果不是，抛出异常

        cls_embeds = []  # 初始化分类嵌入列表
        cls_embeds.append(hidden_states[0][0])  # 添加初始的分类 token 嵌入
        for i, layer_block in enumerate(self.layer):  # 遍历每一层
            hidden_states, _ = layer_block(hidden_states)  # 通过每一层，更新 hidden_states
            if i < len(self.layer) - 1:  # 如果不是最后一层
                cls_embeds.append(hidden_states[0][0])  # 添加当前层的分类 token 嵌入
        encoded = self.encoder_norm(hidden_states)  # 对最后的 hidden_states 进行归一化
        cls_embeds.append(hidden_states[0][0])  # 添加最终的分类 token 嵌入

        cls_embeds = torch.stack(cls_embeds)  # 将所有分类嵌入堆叠成一个张量
        return cls_embeds  # 返回分类嵌入


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)  # 初始化嵌入层  #####################
        self.encoder = Encoder(config, vis)  # 初始化编码器     ##################

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)  # 通过嵌入层，获取嵌入输出

        encoded, attn_weights = self.encoder(embedding_output)  # 通过编码器，获取编码结果和注意力权重
        return encoded, attn_weights  # 返回编码结果和注意力权重

    def forward_cls_layerwise(self, input_ids):
        embedding_output = self.embeddings(input_ids)  # 通过嵌入层，获取嵌入输出

        cls_embeds = self.encoder.forward_cls_layerwise(embedding_output)  # 获取每层的分类嵌入
        return cls_embeds  # 返回分类嵌入


class VisionTransformer(nn.Module):
    def __init__(
            self, model_type,
            img_size=224, num_classes=21843, vis=False
    ):
        super(VisionTransformer, self).__init__()
        config = CONFIGS[model_type]  # 获取模型配置
        self.num_classes = num_classes  # 设置分类数
        self.classifier = config.classifier  # 获取分类器配置

        self.transformer = Transformer(config, img_size, vis)  # 初始化Transformer
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()  # 初始化分类头


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

    def forward_cls_layerwise(self, x):
        cls_embeds = self.transformer.forward_cls_layerwise(x)  # 获取每层的分类嵌入
        return cls_embeds  # 返回分类嵌入

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))  # 加载patch嵌入的权重
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))  # 加载patch嵌入的偏置
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))  # 加载分类token
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))  # 加载编码器归一化的权重
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))  # 加载编码器归一化的偏置

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])  # 加载位置嵌入
            posemb_new = self.transformer.embeddings.position_embeddings  # 获取新位置嵌入
            if posemb.size() == posemb_new.size():  # 检查位置嵌入尺寸是否匹配
                self.transformer.embeddings.position_embeddings.copy_(posemb)  # 如果匹配，直接复制
            else:
                logger.info(
                    "load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))  # 尺寸不匹配，调整尺寸
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":  # 如果分类器是token类型
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # 切分token和grid
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]  # 仅使用grid

                gs_old = int(np.sqrt(len(posemb_grid)))  # 计算旧grid尺寸
                gs_new = int(np.sqrt(ntok_new))  # 计算新grid尺寸
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)  # 重塑旧grid

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)  # 计算缩放比例
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # 缩放grid
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)  # 重塑为新尺寸
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)  # 合并token和新grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))  # 复制新位置嵌入

            for bname, block in self.transformer.encoder.named_children():  # 遍历编码器中的每个块
                for uname, unit in block.named_children():  # 遍历块中的每个单元
                    unit.load_from(weights, n_block=uname)  # 加载单元权重

            if self.transformer.embeddings.hybrid:  # 如果使用混合模型
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))  # 加载根卷积权重
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)  # 加载根归一化权重
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)  # 加载根归一化偏置
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)  # 复制根归一化权重
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)  # 复制根归一化偏置

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():  # 遍历混合模型中的每个块
                    for uname, unit in block.named_children():  # 遍历块中的每个单元
                        unit.load_from(weights, n_block=bname, n_unit=uname)  # 加载单元权重


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])  # 转换卷积权重的维度
    return torch.from_numpy(weights)  # 转换为 PyTorch 张量


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight  # 获取卷积核权重
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)  # 计算权重的方差和均值
        w = (w - m) / torch.sqrt(v + 1e-5)  # 对权重进行标准化处理
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)  # 调用F.conv2d进行卷积操作


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)  # 返回一个3x3的标准卷积


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)  # 返回一个1x1的标准卷积



class PreActBottleneck(nn.Module):
    """预激活（v2）瓶颈块。
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)  # 第一个 GroupNorm 层，用于归一化 cmid 维度
        self.conv1 = conv1x1(cin, cmid, bias=False)  # 第一个 1x1 卷积层，将输入通道数降至 cmid
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)  # 第二个 GroupNorm 层，用于归一化 cmid 维度
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # 第二个 3x3 卷积层，用于特征提取
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)  # 第三个 GroupNorm 层，用于归一化输出通道数
        self.conv3 = conv1x1(cmid, cout, bias=False)  # 第三个 1x1 卷积层，将通道数升至 cout
        self.relu = nn.ReLU(inplace=True)  # 激活函数 ReLU

        if (stride != 1 or cin != cout):
            # 如果步幅不为 1 或输入通道数不等于输出通道数，则需要进行投影
            # 投影也进行预激活，即在卷积操作之前进行归一化
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # 残差分支
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # 单元分支
        y = self.relu(self.gn1(self.conv1(x)))  # 第一个卷积层、归一化和激活函数
        y = self.relu(self.gn2(self.conv2(y)))  # 第二个卷积层、归一化和激活函数
        y = self.gn3(self.conv3(y))  # 第三个卷积层和归一化

        y = self.relu(residual + y)  # 残差连接并激活
        return y  # 返回结果

    def load_from(self, weights, n_block, n_unit):
        # 加载卷积层1的权重
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        # 加载卷积层2的权重
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        # 加载卷积层3的权重
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        # 加载GN层1的权重
        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        # 加载GN层2的权重
        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        # 加载GN层3的权重
        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)  # 复制卷积层1的权重到模型中
        self.conv2.weight.copy_(conv2_weight)  # 复制卷积层2的权重到模型中
        self.conv3.weight.copy_(conv3_weight)  # 复制卷积层3的权重到模型中

        self.gn1.weight.copy_(gn1_weight.view(-1))  # 复制GN层1的权重到模型中，并转换为一维张量
        self.gn1.bias.copy_(gn1_bias.view(-1))  # 复制GN层1的偏置到模型中，并转换为一维张量

        self.gn2.weight.copy_(gn2_weight.view(-1))  # 复制GN层2的权重到模型中，并转换为一维张量
        self.gn2.bias.copy_(gn2_bias.view(-1))  # 复制GN层2的偏置到模型中，并转换为一维张量

        self.gn3.weight.copy_(gn3_weight.view(-1))  # 复制GN层3的权重到模型中，并转换为一维张量
        self.gn3.bias.copy_(gn3_bias.view(-1))  # 复制GN层3的偏置到模型中，并转换为一维张量

        if hasattr(self, 'downsample'):  # 如果模型具有下采样属性
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)  # 加载投影卷积层的权重
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])  # 加载投影GN层的权重
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])  # 加载投影GN层的偏置

            self.downsample.weight.copy_(proj_conv_weight)  # 复制投影卷积层的权重到下采样层
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))  # 复制投影GN层的权重到GN投影层，并转换为一维张量
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))  # 复制投影GN层的偏置到GN投影层，并转换为一维张量


class ResNetV2(nn.Module):
    """Pre-activation (v2) ResNet 模型的实现。"""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        # 下面的代码如果拆成多行将难以阅读。
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),  # 根层卷积
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),  # 根层GN归一化
            ('relu', nn.ReLU(inplace=True)),  # 根层ReLU激活
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 根层最大池化
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +  # 第一个残差块
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],  # 后续残差块
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +  # 第一个残差块，步长为2
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],  # 后续残差块
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +  # 第一个残差块，步长为2
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],  # 后续残差块
            ))),
        ]))

    def forward(self, x):
        x = self.root(x)  # 根层处理
        x = self.body(x)  # 主体部分处理
        return x  # 返回结果


