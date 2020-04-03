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

        self.f = []
        for name, module in InsResNet50().encoder.module.named_children():
            # if name == 'conv1':
            #     module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not name == 'l2norm':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.f = nn.DataParallel(self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        self.g = nn.DataParallel(self.g)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# mod = simCLR()
# norm = Normalize()
# pdb.set_trace()
