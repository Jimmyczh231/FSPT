#!/usr/bin/env python3

"""JSON dataset: support CUB_200_2011, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter

from ..transforms import get_transforms
from ...utils import logging
from ...utils.io_utils import read_json
logger = logging.get_logger("visual_prompt")


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)  # 确认split参数是否在支持的分割类型中，如果不支持则抛出断言错误
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))  # 记录构建数据集的信息

        self.cfg = cfg  # 保存配置参数
        self._split = split  # 保存数据集分割类型
        self.name = cfg.DATA.NAME  # 保存数据集名称
        self.data_dir = cfg.DATA.DATAPATH  # 保存数据集路径
        self.data_percentage = cfg.DATA.PERCENTAGE  # 保存数据集使用百分比
        self._construct_imdb(cfg)  # 构建图像数据库
        self.transform = get_transforms(split, cfg.DATA.CROPSIZE)  # 获取数据转换方法

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))  # 构建分割数据集的JSON文件路径
        if "train" in self._split:  # 如果分割类型包含"train"
            if self.data_percentage < 1.0:  # 如果数据使用百分比小于1.0
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)  # 使用数据百分比构建对应的JSON文件路径
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)  # 确认JSON文件路径是否存在，如果不存在则抛出断言错误

        return read_json(anno_path)  # 读取并返回JSON文件内容

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """构建图像数据库。"""
        img_dir = self.get_imagedir()  # 获取图像目录路径
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)  # 确认图像目录是否存在，如果不存在则抛出断言错误

        anno = self.get_anno()  # 获取注释数据
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))  # 获取注释中所有类别的唯一ID，并按顺序排序
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}  # 将类别ID映射到连续的ID

        # Construct the image db
        self._imdb = []  # 初始化图像数据库列表
        for img_name, cls_id in anno.items():  # 遍历注释中的每张图像和对应的类别ID
            cont_id = self._class_id_cont_id[cls_id]  # 获取类别的连续ID
            im_path = os.path.join(img_dir, img_name)  # 构建图像文件的完整路径
            self._imdb.append({"im_path": im_path, "class": cont_id})  # 将图像路径和类别ID添加到图像数据库中

        logger.info("Number of images: {}".format(len(self._imdb)))  # 记录图像数量
        logger.info("Number of classes: {}".format(len(self._class_ids)))  # 记录类别数量

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class pretrained_weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        # im_path = self._imdb[index]["im_path"]  # 原始路径
        label = self._imdb[index]["class"]
        im = self.transform(im)
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label,
            # "path": self._imdb[index]["im_path"],
            # "id": index
        }
        return sample

    def __len__(self):
        return len(self._imdb)


class CUB200Dataset(JSONDataset):
    """CUB_200 dataset."""

    def __init__(self, cfg, split):
        super(CUB200Dataset, self).__init__(cfg, split)  # 调用父类的初始化方法

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")  # 返回图像目录路径，将数据目录和"images"子目录拼接在一起



class CarsDataset(JSONDataset):
    """stanford-cars dataset."""

    def __init__(self, cfg, split):
        super(CarsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class DogsDataset(JSONDataset):
    """stanford-dogs dataset."""

    def __init__(self, cfg, split):
        super(DogsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")


class FlowersDataset(JSONDataset):
    """flowers dataset."""

    def __init__(self, cfg, split):
        super(FlowersDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return self.data_dir


class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        super(NabirdsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

class xrayDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        super(xrayDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

