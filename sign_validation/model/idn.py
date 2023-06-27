import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from .config import Config

"""
/*
* idn class
* this class is for Deep Learning Module
* created by Su Linyu &&Song Yuanping
* copyright USTC
* 11.01.2021
*/
"""
np.set_printoptions(threshold=np.inf)


class Net3(nn.Module):

    def __init__(self, num_classes=Config.class_number):
        super(Net3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=256),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=1024),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      groups=1024),
            nn.LeakyReLU(inplace=True),
            Flattern(),
            nn.Linear(4096, num_classes),
            nn.Tanh(),
        )

    def forward(self, sign):
        out1 = self.conv1(sign)
        return out1


class Flattern(nn.Module):
    def __init__(self):
        super(Flattern, self).__init__()

    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.25, gamma=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, output1, output2, label):
        # 计算向量v1、v2之间的距离（成次或者成对，意思是可以计算多个，可以参看后面的参数）
        euclidean_distance = F.pairwise_distance(output1, output2)
        # print('euclidean_distance shape:',  euclidean_distance,euclidean_distance.shape)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, self.gamma) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0), self.gamma))
        loss = loss_contrastive
        # print('loss:', loss)
        return loss


class ForwardIDN(nn.Module):
    def __init__(self):
        super(ForwardIDN, self).__init__()
        self.contrastiveLoss = ContrastiveLoss()

    def forward(self, imgs1, imgs2, targets, idn, train=True):
        if train:
            white_imgs1, white_imgs2 = imgs1, imgs2
            label = targets
            # print('white_imgs1 shape:', white_imgs1.shape, 'label shape: ', label.shape,'label:',label)
            inv_ref_out1 = idn(white_imgs1)
            inv_ref_out2 = idn(white_imgs2)
            out2 = self.contrastiveLoss(inv_ref_out1, inv_ref_out2, label)
            return out2
        else:
            # 用黑底白字的图片进行判断两张图片是否为同一人所写
            white_imgs1, white_imgs2 = imgs1, imgs2
            inv_ref_out1 = idn(white_imgs1)
            inv_ref_out2 = idn( white_imgs2)
            out1 = F.pairwise_distance(inv_ref_out1, inv_ref_out2)
            out = out1.view(-1)
            return out
