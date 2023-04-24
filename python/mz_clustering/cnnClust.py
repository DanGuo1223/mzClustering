import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class cnnClust(nn.Module):
    def __init__(self, num_clust = 7, image_height = 40, image_width = 40, seed = 1):
        super(cnnClust, self).__init__()
        self.num_clust = num_clust
        self.image_height = image_height
        self.image_width = image_width


        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU()
        )
        self.conv2 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(2, 2), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2), (2,2)),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU()

        )
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=(2, 2), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU()

        )
        self.conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU()

        )
        self.conv5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2), (2,2)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU()

        )
        self.conv6 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU()

        )

        rand_input = torch.rand(1, 1, self.image_height, self.image_width)

        rand_output = self.conv1(rand_input)

        rand_output = self.conv2(rand_output)

        rand_output = self.conv3(rand_output)

        rand_output = self.conv4(rand_output)

        rand_output = self.conv5(rand_output)

        rand_output = self.conv6(rand_output)

        _, self.c, self.h, self.w = rand_output.shape

        self.fc = nn.Sequential(nn.Linear(self.c*self.h*self.w, num_clust),
                                nn.BatchNorm1d(num_clust, momentum = 0.01),
                                nn.Softmax())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
