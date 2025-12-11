import argparse
from src.configs.config import get_cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()  # 获取默认配置
    cfg.merge_from_file(args.config_file)  # 从配置文件合并配置
    cfg.merge_from_list(args.opts)  # 从命令行选项合并配置

    cfg.freeze()  # 冻结配置
    return cfg  # 返回配置

"""
create a simple parser to wrap around config file
"""
parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')

parser.add_argument(
    "--config-file", default="configs/prompt/dogs.yaml", metavar="FILE",
    help="path to config file")  # 添加--config-file参数
parser.add_argument(
    "--train-type", default="prompt", help="training types")  # 添加--train-type参数
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",  # 命令行中用于修改配置选项
    default=None,  # 默认值为None
    nargs=argparse.REMAINDER,  # 捕获命令行中其余的参数
)

args = parser.parse_args()  # 解析参数

cfg = setup(args)  # 设置配置和参数