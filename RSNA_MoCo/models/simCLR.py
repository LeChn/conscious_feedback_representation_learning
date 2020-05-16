from __future__ import print_function

import torch
import torch.nn as nn
from models.resnet import InsResNet50
# from resnet import InsResNet50
# from torchvision.models.resnet import resnet50
import torch.nn.functional as F
import pdb


class simCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(simCLR, self).__init__()

        # localInfoMax
        self.f = nn.Sequential(*list(InsResNet50().encoder.module.children())[:-3])
        # h from simCLR
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        self.f[0] = nn.Conv2d(x.size(1), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        M = self.f(x)
        x = self.pool(M)
        feature = torch.flatten(x, start_dim=1)
        # return F.normalize(feature, dim=-1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), M


# mod = simCLR()
# pdb.set_trace()
