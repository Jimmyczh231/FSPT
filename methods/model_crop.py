import torch
import torch.nn as nn
from utils import transforms as T
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from PIL import Image
from new4.visualizations import visualize_predictions
from skimage import measure
import math
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
from pathlib import Path
def AOLM(feature_maps):
    width = feature_maps.size(-1)
    height = feature_maps.size(-2)
    A = torch.sum(feature_maps, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > 0.8 * a).float()


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(height, width)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        if len(areas)==0:
            bbox = [0,0,height, width]
        else:

            max_idx = areas.index(max(areas))

            bbox = properties[max_idx].bbox

        temp = 224/width
        temp = math.floor(temp)
        x_lefttop = bbox[0] * temp - 1
        y_lefttop = bbox[1] * temp - 1
        x_rightlow = bbox[2] * temp- 1
        y_rightlow = bbox[3] * temp - 1
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0

        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates


def _plot_and_save(img, attention, scores, score_cropped, output_image_path):
    # 确保保存路径的文件夹存在
    output_folder = Path(output_image_path).parent
    output_folder.mkdir(parents=True, exist_ok=True)

    # 创建画布
    fig = plt.figure(figsize=[25, 10])

    # 添加原始图像子图
    ax = fig.add_subplot(1, 5, 1)
    ax.imshow(img)
    ax.set_title("Original Image")

    # # 取消注释以添加其他子图
    # ax = fig.add_subplot(1, 5, 2)
    # ax.imshow(attention)
    # ax.set_title("Attention")

    # ax = fig.add_subplot(1, 5, 3)
    # ax.imshow(scores)
    # ax.set_title("Scores for Cropping")

    # ax = fig.add_subplot(1, 5, 4)
    # ax.imshow(center_cropped)
    # ax.set_title("Center Cropped")

    # ax = fig.add_subplot(1, 5, 5)
    # ax.imshow(score_cropped)
    # ax.set_title("Cropped using Attention")

    # 保存图片
    fig.savefig(output_image_path, facecolor='white', transparent=False)
    plt.close(fig)  # 关闭绘图，释放内存
    print(f"Image saved at {output_image_path}")

class Crop():
    def __init__(self,args=None):
        super().__init__()
        if(args != None):
            self.h = args.height
            self.w = args.width
        else:
            self.h = 224
            self.w = 224
        self.preprocessor = self._get_preprocessor()


    def _get_preprocessor(self):
        resize = T.Compose([
            T.Resize((self.h, self.w)),  # 将图像大小调整为 ()
            T.ToTensor(),  # 将图像转换为张量
        ])
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 定义标准化操作

        def _preprocess(args, image):
            resized = resize(image)  # 调整图像大小
            image = normalize(resized)  # 标准化图像

            w = image.shape[1] - image.shape[1] % args.patch_size  # 调整图像宽度使其能够被patch size整除
            h = image.shape[2] - image.shape[2] % args.patch_size  # 调整图像高度使其能够被patch size整除

            image = image[:, :w, :h].unsqueeze(0)  # 调整图像尺寸并增加batch维度
            resized = resized[:, :w, :h].permute(1, 2, 0)  # 调整已调整大小的图像

            w_featmap = image.shape[-2] // args.patch_size  # 计算特征图的宽度
            h_featmap = image.shape[-1] // args.patch_size  # 计算特征图的高度

            return ((image, resized), (w_featmap, h_featmap))  # 返回处理后的图像及其特征图大小

        return _preprocess  # 返回预处理器函数

    def crop_img(self, args, images, attentions, feat=None):
        output_image_path = r'E:\小样本细粒度\实验\BSFA-FSFG\修改配置前代码\BSFA-FSFG-main\1122445'
        batch_size = images.size(0)
        num_prompt = args.num_prompt
        images = images.cpu()

        to_pil = ToPILImage()
        # 遍历批次中的每张图像并转换为 PIL 图像
        pil_images = [to_pil(images[i]) for i in range(images.shape[0])]

        # 创建一个保存图片的文件夹，比如 'output_images'
        # output_folder = Path("output_images")
        # output_folder.mkdir(parents=True, exist_ok=True)

        # # 循环保存每张图片，命名为 "image_0.png", "image_1.png", 等等
        # for i, pil_image in enumerate(pil_images):
        #     save_path = output_folder / f"image_{i}.png"
        #     pil_image.save(save_path)
        #     print(f"Saved image {i} at {save_path}")

        # last_attentions = attentions[-1]  # 获取最后一层的自注意力图
        attention_h = attentions.shape[3]
        crops = []  # 初始化 crops
        # image_tensors = []
        vis_folder = args.vis_folder
        for i, image in enumerate(pil_images):
            # 处理图像
            with torch.no_grad():  # 禁用梯度计算
                (image_tensor, resized_image), (w_featmap, h_featmap) = self.preprocessor(args, image)  # 使用预处理器处理图像
                last_attentions = attentions[i:i + 1]  # 注意 [i:i+1] 使形状变为 [1, 12, 197, 197]
                last_attentions = last_attentions[:, :, [0] + list(range(1 + num_prompt, attention_h)), :][:, :, :, [0] + list(range(1 + num_prompt, attention_h))]
                # image_tensors.append(image_tensor)
            nh = last_attentions.shape[1]  # 获取注意力图中的头数

            # 处理注意力图
            last_attentions = last_attentions[0, :, 0, 1:].reshape(nh, -1)  # 只保留每个头的[CLS]到其他token的注意力值
            if args.threshold != 0:  # 如果设置了阈值
                val, idx = torch.sort(last_attentions)  # 对注意力值进行排序
                val /= torch.sum(val, dim=1, keepdim=True)  # 归一化
                cumval = torch.cumsum(val, dim=1)  # 计算累积和
                th_attn = cumval > (1 - args.threshold)  # 选择超过阈值的注意力
                idx2 = torch.argsort(idx)  # 根据索引进行排序
                for head in range(nh):  # 对每个注意力头处理
                    th_attn[head] = th_attn[head][idx2[head]]  # 根据阈值调整注意力        #########
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()  # 调整形状
                th_attn = \
                nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
                    0]  # 使用插值放大到原图大小
                last_attentions = th_attn.sum(0)  # 合并所有头的注意力
            else:
                last_attentions = last_attentions.reshape(nh, w_featmap, h_featmap)  # 调整形状
                last_attentions = \
                nn.functional.interpolate(last_attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
                    0]  # 使用插值放大
                last_attentions = last_attentions.sum(0)  # 合并所有头的注意力

            # 裁剪图像
            h, w, _ = resized_image.size()  # 获取图像的高宽
            conv_weight = torch.ones((1, 1, args.sum_span, args.sum_span), dtype=torch.float32)  # 定义卷积核
            pad_size = args.sum_span // 2  # 计算填充大小
            padded_attention = nn.functional.pad(last_attentions, (pad_size, pad_size, pad_size, pad_size),
                                                 value=0)  # 对注意力图进行填充
            padded_attention = padded_attention.cuda()
            conv_weight = conv_weight.cuda()
            scores = nn.functional.conv2d(padded_attention.unsqueeze(0).unsqueeze(0), conv_weight)[0, 0]  # 计算卷积得分

            max_index = (scores == torch.max(scores)).nonzero()[0]  # 获取得分最高的位置
            max_h_start = h - args.output_height  # 计算裁剪起始点的最大值（高度）
            max_w_start = w - args.output_width  # 计算裁剪起始点的最大值（宽度）

            # 根据得分计算裁剪起始位置
            h_start = min(max(max_index[0] + (args.sum_span // 2) - (args.output_height // 2), 0), max_h_start)
            w_start = min(max(max_index[1] + (args.sum_span // 2) - (args.output_width // 2), 0), max_w_start)

            x_min = w_start
            y_min = h_start
            x_max = x_min + args.output_width
            y_max = y_min + args.output_height
            im_name = i
            visualize_predictions(image, x_min, y_min, x_max, y_max, vis_folder, im_name)



            # 根据得分裁剪图像
            score_cropped = resized_image[h_start:h_start + args.output_height, w_start:w_start + args.output_width, :]

            score_cropped = score_cropped.permute(2, 0, 1)

            score_cropped = score_cropped.unsqueeze(0)  # 转换为 4D Tensor (1, 3, H, W)
            score_cropped_resized = F.interpolate(score_cropped, size=(self.w, self.w), mode="bilinear", align_corners=True)  # 调整大小

            crops.append(score_cropped_resized)  # 将调整大小后的裁剪图像存入列表

        # image_tensors = torch.cat(image_tensors, dim=0)

        # coordinates = AOLM(attentions)  # 使用AOLM函数获取特征图中的坐标
        # for i, image in enumerate(pil_images):  # 遍历每个批次
        #     [x0, y0, x1, y1] = coordinates[i]
        #     im_name = i
        #     visualize_predictions(image, x0, y0, x1, y1, vis_folder, im_name)
        crops = torch.cat(crops, dim=0)  # 拼接成形状 [15, 3, 224, 224]
        #     _plot_and_save(resized_image, attentions, scores, score_cropped, output_image_path)  # 保存图像
        return crops

