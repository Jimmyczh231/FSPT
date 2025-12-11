from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import os.path as osp


class CUB_200_2011(object):
    """
    Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
    """

    # dataset_dir = '/home/10701006/Datasets/Fine_grained/CUB_200_2011'
    dataset_dir = 'CUB_200_2011'
    # dataset_dir = 'D:\Datasets\Fine_grained\CUB_200_2011'
    def __init__(self):
        super(CUB_200_2011, self).__init__()
        # 定义数据集的目录路径
        self.dataset_dir = 'CUB_200_2011'
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir = os.path.join(self.dataset_dir, 'test')

        # 加载训练集、验证集和测试集数据
        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)

        # 统计数据集的信息并打印
        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> CUB_200_2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds), len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds), len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        """在进一步操作之前检查所有文件是否可用"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path):
        """处理给定目录下的数据"""
        # 获取目录下的子目录列表
        cat_container = sorted(os.listdir(dir_path))
        # 将类别映射为标签
        cats2label = {cat: label for label, cat in enumerate(cat_container)}

        dataset = []
        labels = []
        # 遍历每个类别目录下的图片
        for cat in cat_container:
            for img_path in sorted(os.listdir(os.path.join(dir_path, cat))):
                if '.jpg' not in img_path:
                    continue
                label = cats2label[cat]
                dataset.append((os.path.join(dir_path, cat, img_path), label))  # 图片路径和标签对应
                labels.append(label)

        # 构建标签到索引的映射
        labels2inds = {}
        for idx, label in enumerate(labels):
            if label not in labels2inds:
                labels2inds[label] = []
            labels2inds[label].append(idx)

        # 对标签进行排序
        labelIds = sorted(labels2inds.keys())
        return dataset, labels2inds, labelIds  ### 图片路径+label  每个类有哪些图片（id）  有哪些label（类）
