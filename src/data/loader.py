#!/usr/bin/env python3

"""Data loader."""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from ..utils import logging
from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset, xrayDataset
)

logger = logging.get_logger("visual_prompt")  # 获取日志记录器对象
_DATASET_CATALOG = {  # 数据集目录映射字典
    "CUB_200_2011": CUB200Dataset,  # 鸟类数据集
    'OxfordFlowers': FlowersDataset,  # 鲜花数据集
    'StanfordCars': CarsDataset,  # 汽车数据集
    'StanfordDogs': DogsDataset,  # 狗类数据集
    "nabirds": NabirdsDataset,  # North American Birds数据集
    "xray": xrayDataset
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """为给定的数据集构建数据加载器。"""
    dataset_name = cfg.DATA.NAME  # 获取数据集名称

    # 构建数据集
    if dataset_name.startswith("vtab-"):  # 如果数据集名称以"vtab-"开头
        # 仅在需要时导入tensorflow
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(cfg, split)  # 使用TFDataset构建数据集
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)  # 断言数据集名称在数据集目录中
        dataset = _DATASET_CATALOG[dataset_name](cfg, split)  # 使用对应的数据集构建数据集对象

    # 为多进程训练创建一个Sampler
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # 创建数据加载器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(cfg):
    """训练数据加载器封装函数。"""
    if cfg.NUM_GPUS > 1:  # 如果使用多GPU
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )  # 返回构建好的数据加载器



def construct_trainval_loader(cfg):
    """训练+验证数据加载器封装函数。"""
    if cfg.NUM_GPUS > 1:  # 如果使用多GPU
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )  # 返回构建好的训练+验证数据加载器


def construct_test_loader(cfg):
    """测试数据加载器封装函数。"""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )  # 返回构建好的测试数据加载器


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """验证数据加载器封装函数。"""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )  # 返回构建好的验证数据加载器


def shuffle(loader, cur_epoch):
    """"数据洗牌函数。"""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))  # 断言Sampler类型为RandomSampler或DistributedSampler
    # RandomSampler会自动处理数据洗牌
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler基于epoch进行数据洗牌
        loader.sampler.set_epoch(cur_epoch)  # 设置Sampler的epoch

