from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from utils import transforms as T
import datasets
import dataset_loader

class DataManager(object):
    """
    数据管理器，用于处理few-shot数据
    """

    def __init__(self, args, use_gpu):
        super(DataManager, self).__init__()
        self.args = args
        self.use_gpu = use_gpu

        print("初始化数据集 {}".format(args.dataset))
        dataset = datasets.init_imgfewshot_dataset(name=args.dataset)

        transform_train = T.Compose([
            T.Resize((128, 128), interpolation=3),  # 将图像大小调整为 (96, 96)
            T.RandomCrop(args.height),  # 对图像进行随机裁剪
            T.RandomHorizontalFlip(),  # 随机水平翻转图像
            T.ToTensor(),  # 将图像转换为张量
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 对图像进行标准化
            # T.RandomErasing(0.5)  # 随机擦除
        ])

        transform_test = T.Compose([
            T.Resize((96, 96), interpolation=3),
            T.CenterCrop(args.height),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pin_memory = True if use_gpu else False

        self.trainloader = DataLoader(  # 创建训练数据加载器
            dataset_loader.init_loader(name='train_loader',  # 初始化训练数据集加载器
                                       dataset=dataset.train,  # 使用训练数据集
                                       labels2inds=dataset.train_labels2inds,  # 训练数据集标签对应的索引
                                       labelIds=dataset.train_labelIds,  # 训练数据集标签
                                       nKnovel=args.nKnovel,  # 每个迭代的Knovel数量
                                       nExemplars=args.nExemplars,  # 每个类别的示例数量
                                       nTestNovel=args.train_nTestNovel,  # 测试阶段的新类别数量
                                       epoch_size=args.train_epoch_size,  # 每个epoch的大小
                                       transform=transform_train,  # 训练数据集的转换
                                       load=False,  # 是否加载数据
                                       ),
            batch_size=args.train_batch, shuffle=True, num_workers=args.workers,  # 批量大小、是否打乱数据、工作进程数
            pin_memory=pin_memory, drop_last=True,  # 是否将数据存储在固定内存中、是否丢弃最后一批数据
        )


        self.valloader = DataLoader(  # 创建验证数据加载器
            dataset_loader.init_loader(name='test_loader',  # 初始化验证数据集加载器
                                       dataset=dataset.val,  # 使用验证数据集
                                       labels2inds=dataset.val_labels2inds,  # 验证数据集标签对应的索引
                                       labelIds=dataset.val_labelIds,  # 验证数据集标签
                                       nKnovel=args.nKnovel,  # 每个迭代的Knovel数量
                                       nExemplars=args.nExemplars,  # 每个类别的示例数量
                                       nTestNovel=args.nTestNovel,  # 测试阶段的新类别数量
                                       epoch_size=args.epoch_size,  # 每个epoch的大小
                                       transform=transform_test,  # 验证数据集的转换
                                       load=False,  # 是否加载数据
                                       ),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,  # 批量大小、是否打乱数据、工作进程数
            pin_memory=pin_memory, drop_last=False,  # 是否将数据存储在固定内存中、是否丢弃最后一批数据
        )

        self.testloader = DataLoader(  # 创建测试数据加载器
            dataset_loader.init_loader(name='test_loader',  # 初始化测试数据集加载器
                                       dataset=dataset.test,  # 使用测试数据集
                                       labels2inds=dataset.test_labels2inds,  # 测试数据集标签对应的索引
                                       labelIds=dataset.test_labelIds,  # 测试数据集标签
                                       nKnovel=args.nKnovel,  # 每个迭代的Knovel数量
                                       nExemplars=args.nExemplars,  # 每个类别的示例数量
                                       nTestNovel=args.nTestNovel,  # 测试阶段的新类别数量
                                       epoch_size=args.epoch_size,  # 每个epoch的大小
                                       transform=transform_test,  # 测试数据集的转换
                                       load=False,  # 是否加载数据
                                       ),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,  # 批量大小、是否打乱数据、工作进程数
            pin_memory=pin_memory, drop_last=False,  # 是否将数据存储在固定内存中、是否丢弃最后一批数据
        )

    def return_dataloaders(self):  # 返回数据加载器函数
        if self.args.phase == 'test':  # 如果是测试阶段
            return self.trainloader, self.testloader  # 返回训练数据加载器和测试数据加载器
        elif self.args.phase == 'val':  # 如果是验证阶段
            return self.trainloader, self.valloader  # 返回训练数据加载器和验证数据加载器