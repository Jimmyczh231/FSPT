#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage


logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """

    def __init__(
            self,
            cfg: CfgNode,  # 配置节点对象，用于存储和管理配置信息
            model: nn.Module,  # PyTorch 模型对象，用于训练和推断
            evaluator: Evaluator,  # 评估器对象，用于评估模型性能
            device: torch.device,  # 设备对象，指定模型的运行设备（如 CPU 或 GPU）
    ) -> None:
        self.cfg = cfg  # 将配置节点对象保存为实例变量
        self.model = model  # 将模型对象保存为实例变量
        self.device = device  # 将设备对象保存为实例变量

        # solver related
        logger.info("\tSetting up the optimizer...")  # 记录设置优化器的信息

        self.optimizer = make_optimizer([self.model], cfg.SOLVER)  # 创建优化器对象，用于模型训练
        # self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)  # 创建学习率调度器对象，管理学习率的更新策略
        self.cls_criterion = build_loss(self.cfg)  # 根据配置构建分类损失函数对象
        self.evaluator = evaluator
        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,  # 模型检查点保存目录
            save_to_disk=True  # 是否保存检查点到磁盘
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:  # 如果指定了预训练模型权重路径

            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if
                               key not in ["head.last_layer.bias", "head.last_layer.pretrained_weight"]]
            # 构建检查点可恢复项列表，排除特定的参数（如尾部层的偏置和预训练权重）
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)# 加载指定路径下的模型权重，仅恢复不在排除列表中的部分
            logger.info(f"Model pretrained_weight loaded from {cfg.MODEL.WEIGHT_PATH}")# 记录从指定路径加载预训练权重的信息
            self.evaluator = evaluator# 将评估器对象保存为实例变量
            self.cpu_device = torch.device("cpu")# 创建一个 CPU 设备对象

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)  # 将输入数据移动到指定设备上，并使用非阻塞方式
        # inputs 的形状为 (batchsize, 2048)

        targets = targets.to(self.device, non_blocking=True)  # 将目标数据移动到指定设备上，并使用非阻塞方式
        # targets 的形状为 (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")  # 若处于调试模式，则记录输入数据的形状信息
            logger.info(f"shape of targets: {targets.shape}")  # 若处于调试模式，则记录目标数据的形状信息

        # forward
        with torch.set_grad_enabled(is_train):  # 如果 is_train 为 True，代码将运行在训练模式下，计算梯度,如果 is_train 为 False，代码将运行在评估模式下，不计算梯度
            outputs = self.model(inputs)  # 对输入数据进行前向传播，得到模型的输出分类结果   ##########进入vit_models.py的forward进入冻结model
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))  # 若处于调试模式，则记录模型输出和目标的形状信息


        return outputs  # 返回本次批处理的模型输出

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)  # 将数据从 NumPy 数组转换为 PyTorch 张量
        inputs = data["image"].float()  # 将图像数据转换为浮点型张量
        labels = data["label"]  # 获取标签数据
        return inputs, labels  # 返回处理后的输入数据和标签

    def train_classifier(self, crop_xtrain, targets):
        """
        Train a classifier using epoch
        """

            # 启用训练模式
        self.model.train()
        # 执行前向传播并计算损失

        outputs = self.forward_one_batch(crop_xtrain, targets, True)          ##############

        return outputs


    @torch.no_grad()  # 禁用梯度计算以节省内存并提高速度
    def save_prompt(self, epoch):
        # 只有在满足以下条件时才保存 prompt 嵌入
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()  # 获取并转换浅层 prompt 嵌入为 NumPy 数组
                out = {"shallow_prompt": prompt_embds}  # 将浅层 prompt 嵌入存储在字典中
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()  # 获取并转换深层 prompt 嵌入为 NumPy 数组
                    out["deep_prompt"] = deep_embds  # 将深层 prompt 嵌入添加到字典中
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))  # 将字典保存为一个 .pth 文件，文件名包含 epoch

    @torch.no_grad()  # 禁用梯度计算以节省内存并提高速度
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')  # 初始化批处理时间的计时器
        data_time = AverageMeter('Data', ':6.3f')  # 初始化数据加载时间的计时器
        losses = AverageMeter('Loss', ':.4e')  # 初始化损失值的计时器

        log_interval = self.cfg.SOLVER.LOG_EVERY_N  # 获取日志记录间隔
        test_name = prefix + "_" + data_loader.dataset.name  # 构建测试集的名称
        total = len(data_loader)  # 获取数据加载器的总批次数

        # 初始化存储 logits 和目标值的列表
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()  # 记录当前时间
            X, targets = self.get_input(input_data)  # 获取输入数据和目标值
            # 计算数据加载时间
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))  # 如果启用了调试模式，记录输入数据的形状
            loss, outputs = self.forward_one_batch(X, targets, False)  # 前向传播计算损失和输出
            if loss == -1:
                return  # 如果损失为 -1，表示出错，直接返回
            losses.update(loss, X.shape[0])  # 更新损失值

            # 计算批处理时间
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:  # 每隔 log_interval 次批处理记录一次日志
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,  # 当前批次编号
                        total,  # 总批次数
                        losses.val,  # 当前损失值
                        batch_time.val,  # 当前批处理时间
                        data_time.val  # 当前数据加载时间
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())  # 当前 GPU 内存使用情况
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))  # 将当前批次的目标值转换为 NumPy 数组，并扩展到 total_targets 列表中
            total_logits.append(outputs)  # 将当前批次的输出 logits 添加到 total_logits 列表中

            logger.info(
                f"Inference ({prefix}):"
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)  # 记录平均数据加载时间和平均批处理时间
                + "average loss: {:.4f}".format(losses.avg))  # 记录平均损失值

            if self.model.side is not None:
                logger.info(
                    "--> side tuning alpha = {:.4f}".format(
                        self.model.side_alpha))  # 如果模型有 side 调整参数，记录 side 调整的 alpha 值

            # 将 total_logits 列表中的所有 logits 连接起来，形成一个大的张量，然后转换为 NumPy 数组
            joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()

            # 使用 evaluator 对分类结果进行评估
            self.evaluator.classify(
                joint_logits, total_targets,  # 将所有的 logits 和目标值传递给评估器
                test_name, self.cfg.DATA.MULTILABEL,  # 传递测试集的名称和多标签标志
            )

            # 保存概率和目标值
            if save and self.cfg.MODEL.SAVE_CKPT:
                out = {"targets": total_targets, "joint_logits": joint_logits}  # 创建包含目标值和 logits 的字典
                out_path = os.path.join(
                    self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")  # 构建保存路径
                torch.save(out, out_path)  # 将字典保存为 .pth 文件
                logger.info(
                    f"Saved logits and targets for {test_name} at {out_path}")  # 记录保存日志和目标值的路径

