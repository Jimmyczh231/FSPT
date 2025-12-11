import torch
import torch.nn as nn
from utils import transforms as T
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from skimage import measure
import math
import matplotlib.pyplot as plt

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure


class Crop():
    def __init__(self, args=None):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def denormalize(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)

    def crop_img(self, images, v, batch_num=None, visualize=True, start_index=0):
        grid_size = int(np.sqrt(v.size(-1)))
        num_images = images.size(0)
        masks = v[:, 0, 1:].reshape(num_images, grid_size, grid_size).detach().cpu().numpy()  # [B, 14, 14]
        # masks = v[:, 0, 1:].reshape(num_images, grid_size + 1, grid_size + 1).detach().cpu().numpy()  # [B, 14, 14]  #全调的时候

        cropped_images = []

        for i in range(num_images):
            mask = masks[i]
            attention = masks[i]
            image = images[i].unsqueeze(0)

            mask = cv2.resize(mask / mask.max(), (image.shape[-2], image.shape[-1]))[..., np.newaxis]
            mask_np = mask.squeeze(2)
            height, width = mask_np.shape

            a = np.mean(mask_np, axis=(0, 1), keepdims=True)
            mask_np = (mask_np > a * 1).astype(np.float32)

            component_labels = measure.label(mask_np)
            properties = measure.regionprops(component_labels)
            areas = [prop.area for prop in properties]

            if len(areas) == 0:
                bbox = [0, 0, height, width]
            else:
                max_idx = areas.index(max(areas))
                bbox = properties[max_idx].bbox

            temp = 1
            temp = math.floor(temp)
            y_lefttop = max(0, bbox[0] * temp)
            x_lefttop = max(0, bbox[1] * temp)
            y_rightlow = min(image.shape[-2], bbox[2] * temp)
            x_rightlow = min(image.shape[-1], bbox[3] * temp)

            cropped_tensor = image[:, :, y_lefttop:y_rightlow, x_lefttop:x_rightlow]

            resized_tensor = F.interpolate(cropped_tensor, size=[image.shape[-2], image.shape[-1]], mode='bilinear',
                                           align_corners=False)

            cropped_images.append(resized_tensor.squeeze(0))  # 去掉批次维度

        cropped_images = torch.stack(cropped_images, dim=0)
        return cropped_images









