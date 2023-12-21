import os

import numpy as np
from torch import nn
import torch as t

t.cuda.empty_cache()
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import random

from torchvision import datasets, models, transforms
import os
import re
import pandas as pd
from PIL import Image
import jieba
import os.path
import csv
from transformers import BertTokenizer
# from utils import Processer

# -*- coding: utf-8 -*-

import re
from math import log

from PIL import Image
import numpy as np

import random
import torchvision

from .InceptionV3 import GoogLeNet




class ResidualBlock(nn.Module):
    """实现子modual：residualblock"""

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):

        out = self.left(x)
        if self.right is None:
            residual = x
        else:
            residual = self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """实现主module：ResNet34"""

    def __init__(self, numclasses=512):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self.make_layer(64, 128,
                                      4)
        self.layer2 = self.make_layer(128, 256, 4, stride=2)
        self.layer3 = self.make_layer(256, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)

        self.fc = nn.Linear(512, numclasses)

    def make_layer(self, inchannel, outchannel, block_num,
                   stride=1):

        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(
            *layers)

    def forward(self, x):

        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
     
        x = F.avg_pool2d(x, 7)

        x = x.view(x.size(0), -1)
        return self.fc(x)




class vgg(nn.Module):
    """
    obtain visual feature
    """

    def __init__(self):
        super(vgg, self).__init__()


        # image
        vgg_19 = torchvision.models.vgg19(pretrained=True)

        self.feature = vgg_19.features
        self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[:-3])
        pretrained_dict = vgg_19.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # delect the last layer
        model_dict.update(pretrained_dict)  # update
        self.classifier.load_state_dict(model_dict)  # load the new parameter

    def forward(self, img):
        # image
        # image = self.vgg(img) #[batch, num_ftrs]
        img = self.feature(img)
        img = img.view(img.size(0), -1)
        image = self.classifier(img)

        return image


