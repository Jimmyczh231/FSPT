#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode  # 从config_node模块导入CfgNode

# Global config object
_C = CfgNode()  # 创建全局配置对象
# Example usage:
#   from configs.config import cfg

_C.DBG = False  # 调试模式标志
_C.OUTPUT_DIR = "./output"  # 输出目录
_C.RUN_N_TIMES = 1  # 运行次数
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False  # 是否进行CUDNN基准测试

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1  # 使用的GPU数量
_C.NUM_SHARDS = 1  # 分片数量

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = 42  # 随机种子

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()  # 模型配置节点
_C.MODEL.TRANSFER_TYPE = "end2end"  # end2end, prompt
_C.MODEL.WEIGHT_PATH = ""
_C.MODEL.SAVE_CKPT = True

_C.MODEL.MODEL_ROOT = "pretrained_weight"  # 预训练模型权重的根目录

_C.MODEL.TYPE = "vit"  # 模型类型
_C.MODEL.MLP_NUM = 0  # MLP层数

_C.MODEL.LINEAR = CfgNode()  # 线性模型配置节点
_C.MODEL.LINEAR.MLP_SIZES = []  # MLP尺寸
_C.MODEL.LINEAR.DROPOUT = 0.1  # Dropout率

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()  # 提示配置节点
_C.MODEL.PROMPT.NUM_TOKENS = 0   # 提示令牌数量              ####################
_C.MODEL.PROMPT.LOCATION = "prepend"  # 提示位置
# prompt initialization:
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.PROMPT.INITIATION = "random"  # 提示初始化方法，可以是"final-cls", "cls-first12"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""  # 提示嵌入文件夹
_C.MODEL.PROMPT.CLSEMB_PATH = ""  # 提示嵌入路径
_C.MODEL.PROMPT.PROJECT = -1  # 投影MLP隐藏维度
_C.MODEL.PROMPT.DEEP = True  # 是否进行深层提示，仅适用于prepend位置

_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None  # 部分深层提示调优的层数，如果设置为整数
_C.MODEL.PROMPT.REVERSE_DEEP = False  # 是否仅更新最后n层，不包括输入层
_C.MODEL.PROMPT.DEEP_SHARED = False  # 是否所有深层将使用相同的提示嵌入
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False  # 是否不为没有提示的层扩展输入序列
# how to get the output emb for cls head:
    # original: follow the original backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"  # 获取cls头部输出嵌入的方法
_C.MODEL.PROMPT.DROPOUT = 0.1  # Dropout率
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False  # 是否为每个epoch保存


# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()  # 求解器配置节点
_C.SOLVER.LOSS = "softmax"  # 损失函数类型
_C.SOLVER.LOSS_ALPHA = 0.01  # 损失函数的alpha值

_C.SOLVER.OPTIMIZER = "sgd"  # 优化器类型，可以是"sgd"或"adamw"
_C.SOLVER.MOMENTUM = 0.9  # 动量
_C.SOLVER.WEIGHT_DECAY = 0.001  # 权重衰减
_C.SOLVER.WEIGHT_DECAY_BIAS = 0  # 偏置权重衰减

_C.SOLVER.PATIENCE = 15  # 提前停止的耐心值

_C.SOLVER.SCHEDULER = "cosine"  # 学习率调度器

_C.SOLVER.BASE_LR = 0.1  # 基础学习率
_C.SOLVER.BIAS_MULTIPLIER = 1.0  # 提示和偏置的倍率

# _C.SOLVER.WARMUP_EPOCH = 5  # 预热epoch数量
# _C.SOLVER.TOTAL_EPOCH = 30  # 总epoch数量      ##### EPOCH = 0的时候为测试模式 ########
# _C.SOLVER.LOG_EVERY_N = 1000  # 每N步记录日志

_C.SOLVER.DBG_TRAINABLE = False  # 如果为True，将打印可训练参数的名称

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()  # 数据集配置节点

_C.DATA.NAME = "CUB_200_2011"  # 数据集名称
_C.DATA.DATAPATH = "VPTDataRelease/data/cub"  # 数据集路径
_C.DATA.FEATURE = ""  # 特征，例如inat2021_supervised    base-prompt.yaml

_C.DATA.PERCENTAGE = 1.0  # 数据集使用百分比
_C.DATA.NUMBER_CLASSES = 10  # 类别数量
_C.DATA.MULTILABEL = False  # 是否为多标签分类
_C.DATA.CLASS_WEIGHTS_TYPE = "none"  # 类别权重类型

_C.DATA.CROPSIZE = 96  # 裁剪大小

_C.DATA.NO_TEST = False  # 是否没有测试数据
_C.DATA.BATCH_SIZE = 100  # 批处理大小
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4  # 每个训练过程的数据加载器工作线程数
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True  # 是否将数据加载到固定内存

_C.DIST_BACKEND = "nccl"  # 分布式后端
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""  # 分布式初始化文件

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()  # 返回默认配置的副本
