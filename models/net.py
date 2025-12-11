import gc

import torch
from .xcos import Xcos
from methods.crop_img import Crop
from  methods.SupConLoss import SupConLoss
from methods.attention import AttentionSimilarity
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
import math

# import pynvml  # 用于获取 GPU 利用率信息

# 初始化 NVML
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用 GPU 0
# def print_gpu_utilization():
#     info = pynvml.nvmlDeviceGetUtilizationRates(handle)
#     print(f"GPU Utilization: {info.gpu}% | Memory Utilization: {info.memory}%")

class Trainer():
    def __init__(self):
        super().__init__()

        self.is_training = True
        self.contrastive = SupConLoss()
        self.attention_ir = AttentionSimilarity(hidden_size=768, inner_size=768)

    def trainer(self, model, xtrain, ytrain, xtest, ytest, label_train, label_test, args):  # 前向传播函数定义，接收训练集和测试集数据以及它们对应的标签
        label_train = torch.squeeze(label_train, dim=0)
        label_test = torch.squeeze(label_test, dim=0)
        num_prompt = args.num_prompt
        batch_size, num_train = xtrain.size(0), xtrain.size(1)  # 获取训练集批次大小和样本数量
        num_test = xtest.size(1)  # 获取测试集样本数量
        K = ytrain.size(2)  # 获取标签类别数量
        ytrain = ytrain.transpose(1, 2)  # 调整标签张量的维度顺序

#         print_gpu_utilization()

        crop_instance = Crop(args)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  # 重新调整训练集数据张量的形状
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))  # 重新调整测试集数据张量的形状
        images = torch.cat((xtrain, xtest), 0)  # 将训练集和测试集数据合并

        att_mats = model(images, 1, is_extract=False)  # 批量处理
        # 计算注意力矩阵
        with torch.no_grad():
        # 将所有图片批量输入模型

            att_mats = torch.stack(att_mats)  # 压缩注意力头维度
            att_mats = att_mats.permute(1, 0, 2, 3, 4)

        # 处理注意力矩阵
            attention_h = att_mats.shape[-1]
            att_mats_mean = torch.mean(att_mats, dim=2)  # 对所有注意力头取平均
            residual_att = torch.eye(att_mats_mean.size(-1)).unsqueeze(0).cuda()  # 添加残差
            aug_att_mats = att_mats_mean + residual_att
            aug_att_mats = aug_att_mats / aug_att_mats.sum(dim=-1).unsqueeze(-1)   # 归一化

        # 加权
        alphas = torch.tensor([0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda()

        num_layers = aug_att_mats.size(1)

        # alphas = torch.linspace(0.1, 1.0, steps=num_layers).cuda()

        # 计算联合注意力
        joint_attentions = torch.zeros_like(aug_att_mats).cuda()
        # joint_attentions[:, 1] = aug_att_mats[:, 1]  # 初始化第一个联合注意力矩阵
        joint_attentions[:, 0] = alphas[0] * aug_att_mats[:, 0]  # 初始化第一个 第几层 联合注意力矩阵

        # 加权递归计算联合注意力
        for n in range(1,
                       num_layers):  # (2, num_layers)表示从第1层开始，0层是初始层,11层是末层 ，这里改数字上面初始层也要改成 joint_attentions[:,（2-1）]
            # joint_attentions[:, n] =  torch.matmul(alphas[n] * aug_att_mats[:, n], joint_attentions[:, n - 1])
            joint_attentions[:, n] = alphas[n] * aug_att_mats[:, n] + joint_attentions[:, n - 1]

        # 从输出token到输入空间的注意力
        v = joint_attentions[:, -1]
        v = v[:, [0] + list(range(1 + num_prompt, attention_h)), :][:, :,
            [0] + list(range(1 + num_prompt, attention_h))]

#         print_gpu_utilization()

        # 批量裁剪图片
        crops = crop_instance.crop_img(images, v)  # 假设crop_img支持批量处理

        crop_train = crops[:xtrain.shape[0],...]
        crop_test = crops[xtrain.shape[0]:,...]

        glo1 = model(images)  # 全局特征分类器1
        glo2 = model(crops)  # 全局特征分类器2

        ftrain = model(xtrain, 0, 1)
        ftest = model(xtest, 0, 1)

        fcrop_train = model(crop_train, 0, 1)
        fcrop_test = model(crop_test, 0, 1)


        s2 = Xcos(ftrain, ftest, ytrain, batch_size, num_train, num_test, K)  # 原始特征的相似度
        s1 = Xcos(fcrop_train, fcrop_test, ytrain, batch_size, num_train, num_test, K)  # 裁剪特征的相似度

        ########            全局对比损失             #########

        pool = nn.AdaptiveAvgPool1d(1)  # 平均池化
        #
        # # 交换维度，使得 768 作为最后一个维度才能使用 AdaptiveAvgPool1d
        pooled_ftrain = ftrain.transpose(1, 2)  # 形状变为 [5, 768, 196]
        pooled_ftrain = pool(pooled_ftrain)  # 形状变为 [5, 768, 1]
        pooled_ftrain = pooled_ftrain.squeeze(-1)

        pooled_ftest = ftest.transpose(1, 2)  # 形状变为 [5, 768, 196]
        pooled_ftest = pool(pooled_ftest)  # 形状变为 [5, 768, 1]
        pooled_ftest = pooled_ftest.squeeze(-1)

        pooled_fcrop_train = fcrop_train.transpose(1, 2)  # 形状变为 [5, 768, 196]
        pooled_fcrop_train = pool(pooled_fcrop_train)  # 形状变为 [5, 768, 1]
        pooled_fcrop_train = pooled_fcrop_train.squeeze(-1)

        pooled_fcrop_test = fcrop_test.transpose(1, 2)  # 形状变为 [5, 768, 196]
        pooled_fcrop_test = pool(pooled_fcrop_test)  # 形状变为 [5, 768, 1]
        pooled_fcrop_test = pooled_fcrop_test.squeeze(-1)

        loss_1 = self.contrastive(torch.cat((pooled_ftrain, pooled_ftest), dim=0),
                                  labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                  attention=self.attention_ir)
        loss_2 = self.contrastive(torch.cat((pooled_fcrop_train, pooled_fcrop_test), dim=0),
                                  labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                  attention=self.attention_ir)

        ########            空间对比损失             #########

        loss_spatial_1 = self.contrastive(torch.cat((ftrain, ftest), dim=0),
                                          labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                          attention=self.attention_ir)
        loss_spatial_2 = self.contrastive(torch.cat((fcrop_train, fcrop_test), dim=0),
                                          labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                          attention=self.attention_ir)

        loss_con = loss_spatial_1 + loss_spatial_2 + loss_1 +loss_2
        # loss_con = loss_spatial_1 + loss_spatial_2

        return s1, s2, glo1, glo2, loss_con
        # return s1, s2, glo1, glo2

    def tester(self, model, xtrain, ytrain, xtest, ytest, label_train, label_test, args, batch_num=0):  # 前向传播函数定义，接收训练集和测试集数据以及它们对应的标签
        label_train = torch.squeeze(label_train, dim=0)
        label_test = torch.squeeze(label_test, dim=0)
        num_prompt = args.num_prompt
        batch_size, num_train = xtrain.size(0), xtrain.size(1)  # 获取训练集批次大小和样本数量
        batch_num = batch_size*batch_num
        num_test = xtest.size(1)  # 获取测试集样本数量
        K = ytrain.size(2)  # 获取标签类别数量
        ytrain = ytrain.transpose(1, 2)  # 调整标签张量的维度顺序

        crop_instance = Crop(args)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  # 重新调整训练集数据张量的形状
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))  # 重新调整测试集数据张量的形状
        images = torch.cat((xtrain, xtest), 0)  # 将训练集和测试集数据合并

        att_mats = model(images, 1, is_extract=False)  # 批量处理
        # 计算注意力矩阵
        with torch.no_grad():
            # 将所有图片批量输入模型

            att_mats = torch.stack(att_mats)  # 压缩注意力头维度
            att_mats = att_mats.permute(1, 0, 2, 3, 4)

            # 处理注意力矩阵
            attention_h = att_mats.shape[-1]
            att_mats_mean = torch.mean(att_mats, dim=2)  # 对所有注意力头取平均
            residual_att = torch.eye(att_mats_mean.size(-1)).unsqueeze(0).cuda()  # 添加残差
            aug_att_mats = att_mats_mean + residual_att
            aug_att_mats = aug_att_mats / aug_att_mats.sum(dim=-1).unsqueeze(-1)

        # # 计算联合注意力
        # joint_attentions = torch.zeros_like(aug_att_mats).cuda()
        # joint_attentions[:, 0] = aug_att_mats[:, 0]  # 初始化第一个联合注意力矩阵
        #
        #
        # for n in range(1, aug_att_mats.size(1)):
        #     joint_attentions[:, n] = torch.matmul(aug_att_mats[:, n], joint_attentions[:, n - 1])

        # 计算联合注意力
        joint_attentions = torch.zeros_like(aug_att_mats).cuda()
        joint_attentions[:, 0] = aug_att_mats[:, 0]  # 初始化第一个联合注意力矩阵

        for n in range(1, aug_att_mats.size(1)):
            joint_attentions[:, n] = torch.matmul(aug_att_mats[:, n], joint_attentions[:, n - 1])

        # 从输出token到输入空间的注意力
        v = joint_attentions[:, -1]
        v = v[:, [0] + list(range(1 + num_prompt, attention_h)), :][:, :,
            [0] + list(range(1 + num_prompt, attention_h))]

        # 批量裁剪图片

        crops = crop_instance.crop_img(images, v, batch_num=batch_num)

        # crops = crop_instance.crop_img(images, v)  # 假设crop_img支持批量处理

        crop_train = crops[:xtrain.shape[0], ...]
        crop_test = crops[xtrain.shape[0]:, ...]

        ftrain = model(xtrain, 0, 1)
        ftest = model(xtest, 0, 1)

        fcrop_train = model(crop_train, 0, 1)
        fcrop_test = model(crop_test, 0, 1)

        s2 = Xcos(ftrain, ftest, ytrain, batch_size, num_train, num_test, K)  # 原始特征的相似度
        s1 = Xcos(fcrop_train, fcrop_test, ytrain, batch_size, num_train, num_test, K)  # 裁剪特征的相似度

        return s1.sum(-1) * 0.5 + s2.sum(-1) * 0.5