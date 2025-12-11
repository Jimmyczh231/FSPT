#!/usr/bin/env python3
"""
Model construction functions.
"""
from tabnanny import verbose
import torch

from .resnet import ResNet
from .convnext import ConvNeXt
from .vit_models import ViT, Swin, SSLViT
from ..utils import logging
logger = logging.get_logger("visual_prompt")
# Supported model types
_MODEL_TYPES = {
    "resnet": ResNet,
    "convnext": ConvNeXt,
    "vit": ViT,
    "swin": Swin,
    "ssl-vit": SSLViT,
}


def build_model(cfg):
    """
    在这里构建模型
    """
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)  # 确保模型类型受支持
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"  # 确保使用的GPU设备数不超过可用的GPU数量

    # 构建模型
    train_type = cfg.MODEL.TYPE
    model = _MODEL_TYPES[train_type](cfg)              ###########################模型vit_models

    log_model_info(model, verbose=cfg.DBG)  # 记录模型信息
    model, device = load_model_to_device(model, cfg)  # 将模型加载到设备上
    logger.info(f"Device used for model: {device}")  # 记录模型所使用的设备

    return model, device  # 返回模型和设备



def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")  # 如果verbose为True，记录模型的信息
    model_total_params = sum(p.numel() for p in model.parameters())  # 计算模型的总参数数量
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)  # 计算需要梯度的参数数量
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))  # 记录总参数和需要梯度参数的数量
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))  # 记录需要梯度参数占总参数的百分比



def get_current_device():
    if torch.cuda.is_available():  # 如果GPU可用
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()  # 获取当前进程使用的GPU
    else:
        cur_device = torch.device('cpu')  # 否则使用CPU
    return cur_device  # 返回当前设备



def load_model_to_device(model, cfg):
    cur_device = get_current_device()  # 获取当前设备
    if torch.cuda.is_available():  # 如果GPU可用
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)  # 将模型转移到当前GPU设备
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:  # 如果使用多GPU
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(  # 使用分布式数据并行
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,
            )
    else:
        model = model.to(cur_device)  # 否则，将模型转移到当前设备（CPU）
    return model, cur_device  # 返回模型和当前设备

