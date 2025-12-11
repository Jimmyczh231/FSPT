#!/usr/bin/env python3
"""
launch helper functions
"""
import argparse
import os
import sys
import pprint
import PIL
from collections import defaultdict
from tabulate import tabulate
from typing import Tuple

import torch
from src.utils.file_io import PathManager
from src.utils import logging
from src.utils.distributed import get_rank, get_world_size

os.environ["CUDA_VISIBLE_DEVICES"]='0'
def collect_torch_env() -> str:
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module() -> Tuple[str]:
    var_name = "ENV_MODULE"  # 定义环境变量名称
    return var_name, os.environ.get(var_name, "<not set>")  # 获取环境变量的值，如果没有设置则返回"<not set>"



def collect_env_info() -> str:
    data = []  # 初始化一个空列表来存储环境信息
    data.append(("Python", sys.version.replace("\n", "")))  # 获取Python版本信息并添加到列表
    data.append(get_env_module())  # 获取环境模块信息并添加到列表
    data.append(("PyTorch", torch.__version__))  # 获取PyTorch版本信息并添加到列表
    data.append(("PyTorch Debug Build", torch.version.debug))  # 获取PyTorch是否为调试版本的信息并添加到列表

    has_cuda = torch.cuda.is_available()  # 检查CUDA是否可用
    data.append(("CUDA available", has_cuda))  # 将CUDA可用性信息添加到列表
    if has_cuda:  # 如果CUDA可用
        data.append(("CUDA ID", os.environ["CUDA_VISIBLE_DEVICES"]))  # 获取CUDA可见设备的环境变量并添加到列表
        devices = defaultdict(list)  # 初始化一个默认字典来存储设备信息
        for k in range(torch.cuda.device_count()):  # 遍历所有CUDA设备
            devices[torch.cuda.get_device_name(k)].append(str(k))  # 获取每个设备的名称并添加到字典
        for name, devids in devices.items():  # 遍历字典中的设备信息
            data.append(("GPU " + ",".join(devids), name))  # 将设备信息添加到列表
    data.append(("Pillow", PIL.__version__))  # 获取Pillow版本信息并添加到列表

    try:
        import cv2  # 尝试导入OpenCV
        data.append(("cv2", cv2.__version__))  # 获取OpenCV版本信息并添加到列表
    except ImportError:
        pass  # 如果导入失败，跳过

    env_str = tabulate(data) + "\n"  # 将数据列表转换为表格字符串并添加换行符
    env_str += collect_torch_env()  # 收集PyTorch环境信息并添加到字符串
    return env_str  # 返回环境信息字符串



def default_argument_parser():
    """
    create a simple parser to wrap around config file
    """
    parser = argparse.ArgumentParser(description="visual-prompt")  # 创建带有描述的ArgumentParser对象
    parser.add_argument(
        "--config-file", default="configs/prompt/cub.yaml", metavar="FILE", help="path to config file")  # 添加--config-file参数，默认值为configs/prompt/cub.yaml
    parser.add_argument(
        "--train-type", default="prompt", help="training types")  # 添加--train-type参数
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",  # 命令行中用于修改配置选项
        default=None,  # 默认值为None
        nargs=argparse.REMAINDER,  # 捕获命令行中其余的参数
    )
    return parser  # 返回解析器对象



def logging_train_setup(args, cfg) -> None:
    output_dir = cfg.OUTPUT_DIR  # 获取输出目录
    if output_dir:
        PathManager.mkdirs(output_dir)  # 创建输出目录

    logger = logging.setup_logging(  # 设置日志记录器
        cfg.NUM_GPUS, get_world_size(), output_dir, name="visual_prompt")

    # 记录环境、命令行参数和配置的基本信息
    rank = get_rank()  # 获取当前进程的排名
    logger.info(
        f"当前进程的排名Rank of current process: {rank}. World: {get_world_size()}")
    logger.info("环境信息:\n" + collect_env_info())  # 收集环境信息

    logger.info("命令行参数: " + str(args))  # 记录命令行参数
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "args.config_file={} 的内容:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r").read()
            )
        )
    # 显示配置
    logger.info("使用以下配置进行训练:")
    logger.info(pprint.pformat(cfg))
    # cudnn benchmark 在小型验证集上存在较大的开销。
    # 考虑到这一点，不应该使用它。
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK  # 设置cudnn benchmark属性

