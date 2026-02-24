import torch
from .xcos import Xcos
from methods.crop_img import Crop
from  methods.SupConLoss import SupConLoss
from methods.attention import AttentionSimilarity
import torch.nn as nn


class Trainer():
    def __init__(self):
        super().__init__()

        self.is_training = True
        self.contrastive = SupConLoss()
        self.attention_ir = AttentionSimilarity(hidden_size=768, inner_size=768)

    def trainer(self, model, xtrain, ytrain, xtest, ytest, label_train, label_test, args):
        label_train = torch.squeeze(label_train, dim=0)
        label_test = torch.squeeze(label_test, dim=0)
        num_prompt = args.num_prompt
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        crop_instance = Crop(args)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        images = torch.cat((xtrain, xtest), 0)

        att_mats = model(images, 1, is_extract=False)
        with torch.no_grad():
            att_mats = torch.stack(att_mats)
            att_mats = att_mats.permute(1, 0, 2, 3, 4)
            attention_h = att_mats.shape[-1]
            att_mats_mean = torch.mean(att_mats, dim=2)
            residual_att = torch.eye(att_mats_mean.size(-1)).unsqueeze(0).cuda()
            aug_att_mats = att_mats_mean + residual_att
            aug_att_mats = aug_att_mats / aug_att_mats.sum(dim=-1).unsqueeze(-1)

        alphas = torch.tensor([0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda()
        num_layers = aug_att_mats.size(1)
        joint_attentions = torch.zeros_like(aug_att_mats).cuda()
        joint_attentions[:, 0] = alphas[0] * aug_att_mats[:, 0]

        for n in range(1, num_layers):
            joint_attentions[:, n] = alphas[n] * aug_att_mats[:, n] + joint_attentions[:, n - 1]

        v = joint_attentions[:, -1]
        v = v[:, [0] + list(range(1 + num_prompt, attention_h)), :][:, :,
            [0] + list(range(1 + num_prompt, attention_h))]

        crops = crop_instance.crop_img(images, v)

        crop_train = crops[:xtrain.shape[0],...]
        crop_test = crops[xtrain.shape[0]:,...]

        glo1 = model(images)  # global
        glo2 = model(crops)

        ftrain = model(xtrain, 0, 1)
        ftest = model(xtest, 0, 1)

        fcrop_train = model(crop_train, 0, 1)
        fcrop_test = model(crop_test, 0, 1)

        s2 = Xcos(ftrain, ftest, ytrain, batch_size, num_train, num_test, K)
        s1 = Xcos(fcrop_train, fcrop_test, ytrain, batch_size, num_train, num_test, K)

        pool = nn.AdaptiveAvgPool1d(1)

        pooled_ftrain = ftrain.transpose(1, 2)
        pooled_ftrain = pool(pooled_ftrain)
        pooled_ftrain = pooled_ftrain.squeeze(-1)

        pooled_ftest = ftest.transpose(1, 2)
        pooled_ftest = pool(pooled_ftest)
        pooled_ftest = pooled_ftest.squeeze(-1)

        pooled_fcrop_train = fcrop_train.transpose(1, 2)
        pooled_fcrop_train = pool(pooled_fcrop_train)
        pooled_fcrop_train = pooled_fcrop_train.squeeze(-1)

        pooled_fcrop_test = fcrop_test.transpose(1, 2)
        pooled_fcrop_test = pool(pooled_fcrop_test)
        pooled_fcrop_test = pooled_fcrop_test.squeeze(-1)

        loss_1 = self.contrastive(torch.cat((pooled_ftrain, pooled_ftest), dim=0),
                                  labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                  attention=self.attention_ir)
        loss_2 = self.contrastive(torch.cat((pooled_fcrop_train, pooled_fcrop_test), dim=0),
                                  labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                  attention=self.attention_ir)

        loss_spatial_1 = self.contrastive(torch.cat((ftrain, ftest), dim=0),
                                          labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                          attention=self.attention_ir)
        loss_spatial_2 = self.contrastive(torch.cat((fcrop_train, fcrop_test), dim=0),
                                          labels=torch.cat((label_train.flatten(), label_test.flatten()), dim=0),
                                          attention=self.attention_ir)

        loss_con = loss_spatial_1 + loss_spatial_2 + loss_1 +loss_2

        return s1, s2, glo1, glo2, loss_con

    def tester(self, model, xtrain, ytrain, xtest, ytest, label_train, label_test, args, batch_num=0):
        label_train = torch.squeeze(label_train, dim=0)
        label_test = torch.squeeze(label_test, dim=0)
        num_prompt = args.num_prompt
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        batch_num = batch_size*batch_num
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        crop_instance = Crop(args)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        images = torch.cat((xtrain, xtest), 0)

        att_mats = model(images, 1, is_extract=False)
        with torch.no_grad():
            att_mats = torch.stack(att_mats)
            att_mats = att_mats.permute(1, 0, 2, 3, 4)

            attention_h = att_mats.shape[-1]
            att_mats_mean = torch.mean(att_mats, dim=2)
            residual_att = torch.eye(att_mats_mean.size(-1)).unsqueeze(0).cuda()
            aug_att_mats = att_mats_mean + residual_att
            aug_att_mats = aug_att_mats / aug_att_mats.sum(dim=-1).unsqueeze(-1)

        joint_attentions = torch.zeros_like(aug_att_mats).cuda()
        joint_attentions[:, 0] = aug_att_mats[:, 0]

        for n in range(1, aug_att_mats.size(1)):
            joint_attentions[:, n] = torch.matmul(aug_att_mats[:, n], joint_attentions[:, n - 1])

        v = joint_attentions[:, -1]
        v = v[:, [0] + list(range(1 + num_prompt, attention_h)), :][:, :,
            [0] + list(range(1 + num_prompt, attention_h))]

        crops = crop_instance.crop_img(images, v, batch_num=batch_num)
        crop_train = crops[:xtrain.shape[0], ...]
        crop_test = crops[xtrain.shape[0]:, ...]
        ftrain = model(xtrain, 0, 1)
        ftest = model(xtest, 0, 1)
        fcrop_train = model(crop_train, 0, 1)
        fcrop_test = model(crop_test, 0, 1)

        s2 = Xcos(ftrain, ftest, ytrain, batch_size, num_train, num_test, K)
        s1 = Xcos(fcrop_train, fcrop_test, ytrain, batch_size, num_train, num_test, K)

        return s1.sum(-1) * 0.5 + s2.sum(-1) * 0.5