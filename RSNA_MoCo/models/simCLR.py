from __future__ import print_function

import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class simCLR(nn.Module):
    def __init__(self, width=128):
        super(simCLR, self).__init__()
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            'LiniearClassifierW1', nn.Linear(width, width))
        self.classifier.add_module('Relu', nn.ReLU(inplace=True))
        self.classifier.add_module(
            'LiniearClassifierW2', nn.Linear(width, width))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)
