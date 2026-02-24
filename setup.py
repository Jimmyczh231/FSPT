import argparse
from src.configs.config import get_cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg

"""
create a simple parser to wrap around config file
"""
parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')

parser.add_argument(
    "--config-file", default="configs/prompt/xray.yaml", metavar="FILE",
    help="path to config file")  # config
parser.add_argument(
    "--train-type", default="prompt", help="training types")
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

cfg = setup(args)