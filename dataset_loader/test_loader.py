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


class FewShotDataset_test(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest) to classify'
        Xtrain: [nKnovel*nExpemplars, c, h, w].
        Ytrain: [nKnovel*nExpemplars].
        Xtest:  [nTestNovel, c, h, w].
        Ytest:  [nTestNovel].
    """

    def __init__(self,
                 dataset,  # 数据集
                 labels2inds,  # 标签到索引的映射字典
                 labelIds,  # 标签id列表
                 nKnovel=5,  # 每个episode中新类的数量
                 nExemplars=1,  # 每个类别的样本数量
                 nTestNovel=2 * 5,  # 每个episode中用于测试的新类样本数量
                 epoch_size=2000,  # epoch的大小
                 transform=None,  # 数据变换
                 load=True,  # 是否加载数据
                 **kwargs
                 ):
        self.dataset = dataset  # 数据集
        self.labels2inds = labels2inds  # 标签到索引的映射字典
        self.labelIds = labelIds  # 标签id列表
        self.nKnovel = nKnovel  # 每个episode中新类的数量
        self.transform = transform  # 数据变换

        self.nExemplars = nExemplars  # 每个类别的样本数量
        self.nTestNovel = nTestNovel  # 每个episode中用于测试的新类样本数量
        self.epoch_size = epoch_size  # epoch的大小
        self.load = load  # 是否加载数据

        seed = 112
        random.seed(seed)
        np.random.seed(seed)

        self.Epoch_Exemplar = []  # 每个epoch的样本示例列表
        self.Epoch_Tnovel = []  # 每个epoch的新类别样本列表
        for i in range(epoch_size):
            Tnovel, Exemplar = self._sample_episode()  # 采样一个episode
            self.Epoch_Exemplar.append(Exemplar)  # 将样本示例添加到列表中
            self.Epoch_Tnovel.append(Tnovel)  # 将新类别样本添加到列表中

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """

        Knovel = random.sample(self.labelIds, self.nKnovel)  # 从所有标签中随机选择 nKnovel 个作为新类别的标签
        nKnovel = len(Knovel)  # 计算新类别的数量
        # print('nKnovel:', nKnovel)

        assert ((self.nTestNovel % nKnovel) == 0)  # 断言确保 nTestNovel 能够被 nKnovel 整除

        nEvalExamplesPerClass = int(self.nTestNovel / nKnovel)  # 每个新类别的评估样本数量
        # print(self.nTestNovel)
        # print(nEvalExamplesPerClass)

        Tnovel = []  # 存储用于测试的新类别样本
        Exemplars = []  # 存储用于训练的样本示例

        # 遍历每个新类别
        for Knovel_idx in range(len(Knovel)):
            # print('Knovel_idx:',Knovel_idx)
            ids = (nEvalExamplesPerClass + self.nExemplars)  # 每个新类别所需的样本总数
            # print('ids:', ids)
            # print('labels2inds[Knovel[Knovel_idx]]:',self.labels2inds[Knovel[Knovel_idx]])
            img_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], ids)  # 从新类别的样本中随机选择所需数量的样本
            # print('img_ids:', img_ids)

            imgs_tnovel = img_ids[:nEvalExamplesPerClass]  # 新类别用于测试的样本
            imgs_emeplars = img_ids[nEvalExamplesPerClass:]  # 新类别用于训练的样本示例

            # 将新类别用于测试的样本添加到 Tnovel 中
            Tnovel += [(img_id, Knovel_idx) for img_id in imgs_tnovel]
            # 将新类别用于训练的样本示例添加到 Exemplars 中
            Exemplars += [(img_id, Knovel_idx) for img_id in imgs_emeplars]

        # 确保生成的测试集样本数量与预期一致
        assert (len(Tnovel) == self.nTestNovel)

        # 确保生成的样本原型数量与预期一致
        assert (len(Exemplars) == nKnovel * self.nExemplars)

        # 随机打乱样本原型的顺序
        random.shuffle(Exemplars)

        # 随机打乱测试集样本的顺序
        random.shuffle(Tnovel)

        # 返回测试集样本和样本原型
        return Tnovel, Exemplars

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [nExemplars, c, h, w]
            labels: a tensor [nExemplars]
        """

        images = []
        labels = []
        for (img_idx, label) in examples:
            img = self.dataset[img_idx][0]
            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        return images, labels

    def __getitem__(self, index):
        Tnovel = self.Epoch_Tnovel[index]
        Exemplars = self.Epoch_Exemplar[index]
        Xt, Yt = self._creatExamplesTensorData(Exemplars)
        Xe, Ye = self._creatExamplesTensorData(Tnovel)
        return Xt, Yt, Xe, Ye