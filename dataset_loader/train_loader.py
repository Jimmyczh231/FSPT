from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
# import lmdb
import io
import random

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class FewShotDataset_train(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest, Ycls) to classify'
        Xtrain: [nKnovel*nExpemplars, c, h, w].
        Ytrain: [nKnovel*nExpemplars].
        Xtest:  [nTestNovel, c, h, w].
        Ytest:  [nTestNovel].
        Ycls: [nTestNovel].
    """

    def __init__(self,
                 dataset,  # dataset of [(img_path, cats), ...].
                 labels2inds,  # labels of index {(cats: index1, index2, ...)}.
                 labelIds,  # train labels [0, 1, 2, 3, ...,].
                 nKnovel=5,  # number of novel categories.
                 nExemplars=1,  # number of training examples per novel category.
                 nTestNovel=6 * 5,  # number of test examples for all the novel categories.
                 epoch_size=2000,  # number of tasks per eooch.
                 transform=None,
                 load=False,
                 **kwargs
                 ):

        self.dataset = dataset
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        self.nKnovel = nKnovel
        self.transform = transform

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size
        self.load = load

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """

        Knovel = random.sample(self.labelIds, self.nKnovel)  # 从self.labelIds中随机选择nKnovel个类别作为Knovel
        nKnovel = len(Knovel)  # 获取Knovel的类别数量
        assert ((self.nTestNovel % nKnovel) == 0)  # 确保self.nTestNovel能够被nKnovel整除
        nEvalExamplesPerClass = int(self.nTestNovel / nKnovel)  # 每个类别的评估样本数量

        Tnovel = []  # 存储新颖类别的评估样本
        Exemplars = []  # 存储样本原型
        for Knovel_idx in range(len(Knovel)):  # 遍历Knovel中的类别索引
            ids = (nEvalExamplesPerClass + self.nExemplars)  # 计算每个类别所需的总样本数量
            img_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], ids)  # 从对应类别中随机选择样本

            imgs_tnovel = img_ids[:nEvalExamplesPerClass]  # 从样本中选择用于评估的样本
            imgs_emeplars = img_ids[nEvalExamplesPerClass:]  # 从样本中选择用于样本原型的样本

            Tnovel += [(img_id, Knovel_idx) for img_id in imgs_tnovel]  # 将评估样本添加到Tnovel中
            Exemplars += [(img_id, Knovel_idx) for img_id in imgs_emeplars]  # 将样本原型添加到Exemplars中
        assert (len(Tnovel) == self.nTestNovel)  # 确保Tnovel中的样本数量为self.nTestNovel
        assert (len(Exemplars) == nKnovel * self.nExemplars)  # 确保Exemplars中的样本数量为nKnovel乘以self.nExemplars
        random.shuffle(Exemplars)  # 随机打乱Exemplars中的样本顺序
        random.shuffle(Tnovel)  # 随机打乱Tnovel中的样本顺序

        return Tnovel, Exemplars  # 返回Tnovel和Exemplars

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [nExemplars, c, h, w]
            labels: a tensor [nExemplars]
            cls: a tensor [nExemplars]
        """

        images = []
        labels = []
        cls = []
        for (img_idx, label) in examples:
            img, ids = self.dataset[img_idx]  # 获取图像和标签
            if self.load:
                img = Image.fromarray(img)  # 如果需要加载图像，则将img转换为PIL图像
            else:
                img = read_image(img)  # 否则，读取图像

            if self.transform is not None:
                img = self.transform(img)  # 对图像进行变换

            images.append(img)  # 将处理后的图像添加到images列表
            labels.append(label)  # 将标签添加到labels列表
            cls.append(ids)  # 将ids添加到cls列表

        images = torch.stack(images, dim=0)  # 将images列表中的图像堆叠成张量
        labels = torch.LongTensor(labels)  # 将labels列表转换为LongTensor张量
        cls = torch.LongTensor(cls)  # 将cls列表转换为LongTensor张量
        return images, labels, cls  # 返回处理后的图像、标签和ids

    def __getitem__(self, index):
        Tnovel, Exemplars = self._sample_episode()  # 从数据集中采样一个episode
        Xt, Yt, Ytc = self._creatExamplesTensorData(Exemplars)  # 创建样本原型数据的张量表示
        Xe, Ye, Yec = self._creatExamplesTensorData(Tnovel)  # 创建评估样本数据的张量表示
        return Xt, Yt, Xe, Ye, Ytc, Yec  # 返回样本原型和评估样本的张量表示