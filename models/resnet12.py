import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, kernel=3):
        # 初始化函数，输入参数为block(残差块类型), layers(每个阶段的残差块数), kernel(卷积核大小，默认为3)

        self.inplanes = 64
        # inplanes是输入平面的数量，初始化为64

        self.kernel = kernel
        # 设置卷积核大小

        super(ResNet, self).__init__()
        # 调用父类（nn.Module）的初始化函数

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 定义第一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为3，步长为1，填充为1，不使用偏置项

        self.bn1 = nn.BatchNorm2d(64)
        # 定义第一个批量归一化层，输入通道数为64

        self.relu = nn.ReLU(inplace=True)
        # 定义ReLU激活函数，inplace=True表示将运算结果直接覆盖到输入数据中，节省内存

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        # 定义第一个残差层，输入通道数为64，残差块数为layers[0]，步长为2

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 定义第二个残差层，输入通道数为128，残差块数为layers[1]，步长为2

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 定义第三个残差层，输入通道数为256，残差块数为layers[2]，步长为2

        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # 定义第四个残差层，输入通道数为512，残差块数为layers[3]，步长为1

        self.nFeat = 512 * block.expansion
        # 计算最后一层的特征图数量，等于最后一层的通道数乘以残差块的扩张系数

        for m in self.modules():
            # 遍历所有模块
            if isinstance(m, nn.Conv2d):
                # 如果模块是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 使用kaiming正态分布初始化权重，模式为'fan_out'，非线性函数为'relu'
            elif isinstance(m, nn.BatchNorm2d):
                # 如果模块是批量归一化层
                nn.init.constant_(m.weight, 1)
                # 将权重初始化为1
                nn.init.constant_(m.bias, 0)
                # 将偏置项初始化为0

    def _make_layer(self, block, planes, blocks, stride=1):  # 构建ResNet的一个层
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 如果步幅不为1或输入通道数与输出通道数不一致
            downsample = nn.Sequential(  # 构建下采样序列
                nn.Conv2d(self.inplanes, planes * block.expansion,  # 卷积层，用于调整通道数和分辨率
                          kernel_size=1, stride=stride, bias=False),  # 1x1卷积，调整步幅
                nn.BatchNorm2d(planes * block.expansion),  # 批归一化层，规范化数据
            )

        layers = []  # 构建层列表
        layers.append(block(self.inplanes, planes, self.kernel, stride, downsample))  # 添加第一个块
        self.inplanes = planes * block.expansion  # 更新输入通道数
        for i in range(1, blocks):  # 构建剩余块
            layers.append(block(self.inplanes, planes, self.kernel))  # 添加块

        return nn.Sequential(*layers)  # 返回序列化的层

    def forward(self, x):  # 前向传播函数
        x = self.conv1(x)  # 第一个卷积层
        x = self.bn1(x)  # 第一个批归一化层
        x = self.relu(x)  # ReLU激活函数
        x = self.layer1(x)  # 第一个层
        x = self.layer2(x)  # 第二个层
        x = self.layer3(x)  # 第三个层
        x = self.layer4(x)  # 第四个层

        return x  # 返回特征张量

def resnet12():  # 构建ResNet-12模型
    model = ResNet(BasicBlock, [1, 1, 1, 1], kernel=3)  # 创建ResNet模型实例
    return model  # 返回ResNet-12模型

