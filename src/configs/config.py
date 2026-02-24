#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from .config_node import CfgNode

# Global config object
_C = CfgNode()
# Example usage:
#   from configs.config import cfg

_C.DBG = False
_C.OUTPUT_DIR = ""
_C.RUN_N_TIMES = 1
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN_BENCHMARK = False

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.SEED = 42

# ----------------------------------------------------------------------
# Model options
# ----------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.TRANSFER_TYPE = ""
_C.MODEL.WEIGHT_PATH = ""
_C.MODEL.SAVE_CKPT = True

_C.MODEL.MODEL_ROOT = "pretrained_weight"

_C.MODEL.TYPE = "vit"
_C.MODEL.MLP_NUM = 0

_C.MODEL.LINEAR = CfgNode()
_C.MODEL.LINEAR.MLP_SIZES = []
_C.MODEL.LINEAR.DROPOUT = 0.1

# ----------------------------------------------------------------------
# Prompt options
# ----------------------------------------------------------------------
_C.MODEL.PROMPT = CfgNode()
_C.MODEL.PROMPT.NUM_TOKENS = 2
_C.MODEL.PROMPT.LOCATION = "prepend"
# prompt initialization:
    # (1) default "random"
    # (2) "final-cls" use aggregated final [cls] embeddings from training dataset
    # (3) "cls-nolastl": use first 12 cls embeddings (exclude the final output) for deep prompt
    # (4) "cls-nofirstl": use last 12 cls embeddings (exclude the input to first layer)
_C.MODEL.PROMPT.INITIATION = "random"
_C.MODEL.PROMPT.CLSEMB_FOLDER = ""
_C.MODEL.PROMPT.CLSEMB_PATH = ""
_C.MODEL.PROMPT.PROJECT = -1
_C.MODEL.PROMPT.DEEP = True

_C.MODEL.PROMPT.NUM_DEEP_LAYERS = None
_C.MODEL.PROMPT.REVERSE_DEEP = False
_C.MODEL.PROMPT.DEEP_SHARED = False
_C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND = False
# how to get the output emb for cls head:
    # original: follow the original backbone choice,
    # img_pool: image patch pool only
    # prompt_pool: prompt embd pool only
    # imgprompt_pool: pool everything but the cls token
_C.MODEL.PROMPT.VIT_POOL_TYPE = "original"
_C.MODEL.PROMPT.DROPOUT = 0.1
_C.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH = False


# ----------------------------------------------------------------------
# Solver options
# ----------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.LOSS = "softmax"
_C.SOLVER.LOSS_ALPHA = 0.01

_C.SOLVER.OPTIMIZER = "sgd"
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.PATIENCE = 15

_C.SOLVER.SCHEDULER = "cosine"

_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.BIAS_MULTIPLIER = 1.0

# _C.SOLVER.WARMUP_EPOCH = 5
# _C.SOLVER.TOTAL_EPOCH = 30
# _C.SOLVER.LOG_EVERY_N = 1000

_C.SOLVER.DBG_TRAINABLE = False

# ----------------------------------------------------------------------
# Dataset options
# ----------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NAME = ""  #
_C.DATA.DATAPATH = ""  #
_C.DATA.FEATURE = ""  #

_C.DATA.PERCENTAGE = 1.0
_C.DATA.NUMBER_CLASSES = 10
_C.DATA.MULTILABEL = False
_C.DATA.CLASS_WEIGHTS_TYPE = "none"

_C.DATA.CROPSIZE = 96

_C.DATA.NO_TEST = False
_C.DATA.BATCH_SIZE = 100
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA.PIN_MEMORY = True

_C.DIST_BACKEND = "nccl"
_C.DIST_INIT_PATH = "env://"
_C.DIST_INIT_FILE = ""

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
