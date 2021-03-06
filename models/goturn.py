#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-12-30 12:45 zq <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Origin goturn model.
"""

import torch
from torchvision import models
import torch.nn as nn
import sys
import os
import os.path as osp
import numpy as np
CURR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURR_DIR, ".."))


class GONET(nn.Module):
    def __init__(self, pretrained_dir=os.path.join(CURR_DIR, "models", "init")):
        super(GONET, self).__init__()
        #alexnet = models.alexnet()
        #alexnet.load_state_dict(torch.load(osp.join(CURR_DIR, "alexnet.pth")))
        #self.features = alexnet.features
        from models.alexnet import AlexNet
        self.features = AlexNet()
        self.regressor = nn.Sequential(
                    nn.Linear(256*6*6*2, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4)
        )
        self.weight_init(pretrained_dir)
        #self.load_regressor_weights()

    def forward(self, x, y):
        x1 = self.features(x)
        x1 = x1.view(x.size(0), 256*6*6)
        x2 = self.features(y)
        x2 = x2.view(x.size(0), 256*6*6)
        x  = torch.cat((x1, x2), 1)
        x  = self.regressor(x)
        return x
    
    def load_regressor_weights(self):
        prefix = self.pretrained_dir
        fc6_w = np.load(osp.join(prefix, 'fc6-new_0.npy'))
        fc6_b = np.load(osp.join(prefix, 'fc6-new_1.npy'))
        fc7_w = np.load(osp.join(prefix, 'fc7-new_0.npy'))
        fc7_b = np.load(osp.join(prefix, 'fc7-new_1.npy'))
        fc7d_w = np.load(osp.join(prefix, 'fc7-newb_0.npy'))
        fc7d_b = np.load(osp.join(prefix, 'fc7-newb_1.npy'))
        fc8_w = np.load(osp.join(prefix, 'fc8-shapes_0.npy'))
        fc8_b = np.load(osp.join(prefix, 'fc8-shapes_1.npy'))
        self.regressor[0].weight.data = torch.from_numpy(fc6_w)
        self.regressor[0].bias.data = torch.from_numpy(fc6_b)
        self.regressor[3].weight.data = torch.from_numpy(fc7_w)
        self.regressor[3].bias.data = torch.from_numpy(fc7_b)
        self.regressor[6].weight.data = torch.from_numpy(fc7d_w)
        self.regressor[6].bias.data = torch.from_numpy(fc7d_b)
        self.regressor[9].weight.data = torch.from_numpy(fc8_w)
        self.regressor[9].bias.data = torch.from_numpy(fc8_b)

    def weight_init(self, init_dir):
        prefix = os.path.join(CURR_DIR, "init")
        conv1_w = np.load(osp.join(prefix, 'conv1_0.npy'))
        conv1_b = np.load(osp.join(prefix, 'conv1_1.npy'))
        conv2_w = np.load(osp.join(prefix, 'conv2_0.npy'))
        conv2_b = np.load(osp.join(prefix, 'conv2_1.npy'))
        conv3_w = np.load(osp.join(prefix, 'conv3_0.npy'))
        conv3_b = np.load(osp.join(prefix, 'conv3_1.npy'))
        conv4_w = np.load(osp.join(prefix, 'conv4_0.npy'))
        conv4_b = np.load(osp.join(prefix, 'conv4_1.npy'))
        conv5_w = np.load(osp.join(prefix, 'conv5_0.npy'))
        conv5_b = np.load(osp.join(prefix, 'conv5_1.npy'))
        self.features.layer1[0].weight.data = torch.from_numpy(conv1_w)
        self.features.layer1[0].bias.data = torch.from_numpy(conv1_b)
        self.features.layer2[0].weight.data = torch.from_numpy(conv2_w)
        self.features.layer2[0].bias.data = torch.from_numpy(conv2_b)
        self.features.layer3[0].weight.data = torch.from_numpy(conv3_w)
        self.features.layer3[0].bias.data = torch.from_numpy(conv3_b)
        self.features.layer4[0].weight.data = torch.from_numpy(conv4_w)
        self.features.layer4[0].bias.data = torch.from_numpy(conv4_b)
        self.features.layer5[0].weight.data = torch.from_numpy(conv5_w)
        self.features.layer5[0].bias.data = torch.from_numpy(conv5_b)
        fc6_w = np.load(osp.join(prefix, 'fc6-new_0.npy'))
        fc6_b = np.load(osp.join(prefix, 'fc6-new_1.npy'))
        fc7_w = np.load(osp.join(prefix, 'fc7-new_0.npy'))
        fc7_b = np.load(osp.join(prefix, 'fc7-new_1.npy'))
        fc7d_w = np.load(osp.join(prefix, 'fc7-newb_0.npy'))
        fc7d_b = np.load(osp.join(prefix, 'fc7-newb_1.npy'))
        fc8_w = np.load(osp.join(prefix, 'fc8-shapes_0.npy'))
        fc8_b = np.load(osp.join(prefix, 'fc8-shapes_1.npy'))
        self.regressor[0].weight.data = torch.from_numpy(fc6_w)
        self.regressor[0].bias.data = torch.from_numpy(fc6_b)
        self.regressor[3].weight.data = torch.from_numpy(fc7_w)
        self.regressor[3].bias.data = torch.from_numpy(fc7_b)
        self.regressor[6].weight.data = torch.from_numpy(fc7d_w)
        self.regressor[6].bias.data = torch.from_numpy(fc7d_b)
        self.regressor[9].weight.data = torch.from_numpy(fc8_w)
        self.regressor[9].bias.data = torch.from_numpy(fc8_b)
        #self.regressor[0].weight.data.normal_(0, 0.005)
        #self.regressor[0].bias.data.fill_(1)
        #self.regressor[3].weight.data.normal_(0, 0.005)
        #self.regressor[3].bias.data.fill_(1)
        #self.regressor[6].weight.data.normal_(0, 0.005)
        #self.regressor[6].bias.data.fill_(1)
        #self.regressor[9].weight.data.normal_(0, 0.005)
        #self.regressor[9].bias.data.fill_(0)

if __name__ == "__main__":
    gonet = GONET()
    a = torch.ones((1, 32, 6, 6))
    b = torch.ones((1, 32, 6, 6))
    c = CORR()
    a = Variable(a)
    b = Variable(b)
    torch.set_printoptions(threshold=5000)
    c(a, b)




