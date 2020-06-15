## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from networks.ConvLSTM import ConvLSTM

class FeatFlow(nn.Module):

    def __init__(self):
        super(FeatFlow, self).__init__()
        gx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        gy = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        self.conv1 = ConvLayer(1, 1, kernel_size=1, stride=1, bias=False, norm="None")
        self.conv1.weight = nn.Parameter(torch.from_numpy(gx).float().unsqueeze(0).unsqueeze(0))

        self.conv2 = ConvLayer(1, 1, kernel_size=1, stride=1, bias=False, norm="None")
        self.conv2.weight = nn.Parameter(torch.from_numpy(gy).float().unsqueeze(0).unsqueeze(0))

    def forward(self, F1, F2):
        [b, c, h, w] = F1.size()

        F1 = F1.contiguous().view(b*c, 1, h, w)
        F1_delta_rs_x = self.conv1(F1)
        F1_delta_x = F1_delta_rs_x.contiguous().view(b, c, h, w)

        F1_delta_rs_y = self.conv2(F1)
        F1_delta_y = F1_delta_rs_y.contiguous().view(b, c, h, w)


        F2 = F2.contiguous().view(b*c, 1, h, w)
        F2_delta_rs_x = self.conv1(F2)
        F2_delta_x = F2_delta_rs_x.contiguous().view(b, c, h, w)

        F2_delta_rs_y = self.conv2(F2)
        F2_delta_y = F2_delta_rs_y.contiguous().view(b, c, h, w)

        pho_c = F1 - F2

        u_x = torch.zeros(b, c, h, w)
        u_y = torch.zeros(b, c, h, w)

        for i in range(0, c):
            pho = pho_c + torch.bmm(F2_delta_x)


        ttt = 0

        return ttt

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=None, bias=True, last_bias=0):
        super(ConvLayer, self).__init__()

        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if last_bias!=0:
            init.constant(self.conv2d.weight, 0)
            init.constant(self.conv2d.bias, last_bias)

    def forward(self, x):
        out = self.conv2d(x)

        return out