import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
class CAE(nn.Module):
    def __init__(self, embed_dim = 7, image_height = 40, image_width = 40, k1 = (3,3), k2 = (3,3), k3 = (5,5), k4 = (3,3), seed = 1):
        super(CAE, self).__init__()
        self.embed_dim = embed_dim
        self.image_height = image_height
        self.image_width = image_width
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4

        # encoder

        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size = self.k1, stride=(2, 2), padding=(0, 0), bias=False),
                                   nn.BatchNorm2d(8, momentum = 0.01),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size = self.k2, stride=(3, 3), padding=(0, 0), bias=False),
                                nn.BatchNorm2d(16, momentum = 0.01),
                                nn.ReLU())

        rand_input = torch.rand(1, 1, self.image_height, self.image_width)

        rand_output = self.conv1(rand_input)

        rand_output = self.conv2(rand_output)

        _, self.c, self.h, self.w = rand_output.shape

        self.encoder = nn.Linear(self.c*self.h*self.w, self.embed_dim)


        # decoder

        self.fc3 = nn.Linear(self.embed_dim, self.c*self.h*self.w)


        self.conv_trans1 = nn.Sequential(nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.k3, stride = (3, 3), padding = (0,0)),
                                         nn.BatchNorm2d(8, momentum = 0.01),
                                         nn.ReLU(inplace=True)
                                         )

        self.conv_trans2 = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=self.k4, stride = (2, 2), padding = (0,0)),
                                         nn.BatchNorm2d(1, momentum = 0.01),
                                         nn.Sigmoid()
                                         )



    def encode(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)

        return x



    def decode(self, z):

        z = self.fc3(z)
        z = z.view(-1, self.c, self.h, self.w)
        z = self.conv_trans1(z)
        z = self.conv_trans2(z)
        z = F.interpolate(z, size = (self.image_height, self.image_width))

        return z

    def forward(self, x):

        z = self.encode(x)
        xp = self.decode(z)

        return xp, z
