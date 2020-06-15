## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init

from utils import *
from networks.ConvLSTM import ConvLSTM

class TDModel(nn.Module):

    def __init__(self, opts, nc_out, nc_ch):
        super(TDModel, self).__init__()

        self.epoch = 0

        use_bias = True
      
        self.conv1 = nn.Conv3d(3, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv5 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv8 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv11 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv14 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv16 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv17 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv19 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv20 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv22 = nn.Conv3d(nc_ch, nc_out, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):

        F1 = self.relu(self.conv1(X))
        F2 = self.relu(self.conv3(self.relu(self.conv2(F1)))+F1)

        F3 = self.relu(self.conv4(F2))

        F4 = self.relu(self.conv6(self.relu(self.conv5(F3)))+F3)
        F5 = self.relu(self.conv7(F4))

        F6 = self.relu(self.conv9(self.relu(self.conv8(F5)))+F5)
        F7 = self.relu(self.conv10(F6))
        
        F8 = self.relu(self.conv12(self.relu(self.conv11(F7)))+F7)
        F9 = self.relu(self.conv13(F8))

        F10 = self.relu(self.conv15(self.relu(self.conv14(F9)))+F9)
        F11 = self.relu(self.conv16(F10))

        F12 = self.relu(self.conv18(self.relu(self.conv17(F11)))+F11)
        F13 = self.relu(self.conv19(F12))

        F14 = self.relu(self.conv21(self.relu(self.conv20(F13)))+F13)
        Y = self.conv22(F14)

        return Y


