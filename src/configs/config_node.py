#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from fvcore.common.config import CfgNode as _CfgNode
from ..utils.file_io import PathManager


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:
    support manifold path
    """
    @classmethod
    def _open_cfg(cls, filename):  # 类方法，用于打开配置文件
        return PathManager.open(filename, "r")  # 使用PathManager以只读模式打开文件

    def dump(self, *args, **kwargs):  # 重写dump方法
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)  # 调用父类的dump方法并返回

