from __future__ import absolute_import

import os
import os.path as osp
import errno
import json
import shutil

import torch


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best=False, fpath='checkpoint.pth.tar'):
    if len(os.path.dirname(fpath)) != 0:
        mkdir_if_missing(os.path.dirname(fpath))  # 如果文件路径的目录部分长度不为0，则创建目录

    if is_best:
        torch.save(state, os.path.join(os.path.dirname(fpath), 'best_model.pth.tar'))  # 如果是最佳模型，则保存模型状态到指定路径