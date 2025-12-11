#!/usr/bin/env python3

"""Logging."""

import builtins
import decimal
import functools
import logging
import simplejson
import sys
import os
from termcolor import colored

from .distributed import is_master_process
from .file_io import PathManager

# Show filename and line number in logs
_FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"


def _suppress_print():
    """Suppresses printing from the current process."""

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers  # noqa
def setup_logging(
    num_gpu, num_shards, output="", name="visual_prompt", color=True):
    """Sets up the logging."""
    # Enable logging only for the master process
    if is_master_process(num_gpu):  # 仅为主进程启用日志记录
        # Clear the root logger to prevent any existing logging config
        # (e.g. set by another module) from messing with our setup
        logging.root.handlers = []  # 清除根记录器的所有处理器
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format=_FORMAT, stream=sys.stdout
        )  # 配置基础日志记录
    else:
        _suppress_print()  # 抑制打印输出

    if name is None:
        name = __name__  # 如果name为空，使用模块的名称
    logger = logging.getLogger(name)  # 获取指定名称的记录器
    # remove any lingering handler
    logger.handlers.clear()  # 清除记录器的所有处理器

    logger.setLevel(logging.INFO)  # 设置记录器的日志级别为INFO
    logger.propagate = False  # 禁止日志传播

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )  # 设置日志的格式和日期格式
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(name),
        )  # 设置彩色格式的日志格式
    else:
        formatter = plain_formatter  # 使用普通格式的日志格式

    if is_master_process(num_gpu):  # 如果当前进程是主进程
        ch = logging.StreamHandler(stream=sys.stdout)  # 创建一个流处理器，将日志输出到标准输出
        ch.setLevel(logging.DEBUG)  # 设置处理器的日志级别为DEBUG
        ch.setFormatter(formatter)  # 设置处理器的格式
        logger.addHandler(ch)  # 将处理器添加到记录器

    if is_master_process(num_gpu * num_shards):  # 如果当前进程是所有GPU和分片中的主进程
        if len(output) > 0:  # 如果输出目录不为空
            if output.endswith(".txt") or output.endswith(".log"):  # 如果输出文件以.txt或.log结尾
                filename = output  # 使用输出文件名
            else:
                filename = os.path.join(output, "logs.txt")  # 否则，在输出目录中创建logs.txt文件

            PathManager.mkdirs(os.path.dirname(filename))  # 创建输出文件的目录

            fh = logging.StreamHandler(_cached_log_stream(filename))  # 创建一个文件处理器，将日志输出到文件
            fh.setLevel(logging.DEBUG)  # 设置处理器的日志级别为DEBUG
            fh.setFormatter(plain_formatter)  # 设置处理器的格式
            logger.addHandler(fh)  # 将处理器添加到记录器
    return logger  # 返回记录器对象



def setup_single_logging(name, output=""):
    """Sets up the logging."""
    # Enable logging only for the master process
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )

    if len(name) == 0:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
        datefmt="%m/%d %H:%M:%S",
        root_name=name,
        abbrev_name=str(name),
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if len(output) > 0:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs.txt")

        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)


def log_json_stats(stats, sort_keys=True):
    """Logs json stats."""
    # It seems that in Python >= 3.6 json.encoder.FLOAT_REPR has no effect
    # Use decimal+string as a workaround for having fixed length values in logs
    logger = get_logger(__name__)
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    if stats["_type"] == "test_epoch" or stats["_type"] == "train_epoch":
        logger.info("json_stats: {:s}".format(json_stats))
    else:
        logger.info("{:s}".format(json_stats))


class _ColorfulFormatter(logging.Formatter):
    # from detectron2
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."  # 提取并设置根名称，加上"."作为后缀
        self._abbrev_name = kwargs.pop("abbrev_name", "")  # 提取并设置缩写名称，如果没有提供默认为空字符串
        if len(self._abbrev_name):  # 如果缩写名称不为空
            self._abbrev_name = self._abbrev_name + "."  # 在缩写名称后加上"."作为后缀
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)  # 调用父类的初始化方法


    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)  # 替换日志记录器名称中的根名称为缩写名称
        log = super(_ColorfulFormatter, self).formatMessage(record)  # 调用父类的方法格式化日志记录
        if record.levelno == logging.WARNING:  # 如果日志级别是WARNING
            prefix = colored("WARNING", "red", attrs=["blink"])  # 设置WARNING日志的前缀为红色闪烁文本
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:  # 如果日志级别是ERROR或CRITICAL
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])  # 设置ERROR日志的前缀为红色闪烁并带下划线的文本
        else:  # 如果日志级别不是WARNING、ERROR或CRITICAL
            return log  # 直接返回格式化后的日志记录
        return prefix + " " + log  # 返回带前缀的日志记录

