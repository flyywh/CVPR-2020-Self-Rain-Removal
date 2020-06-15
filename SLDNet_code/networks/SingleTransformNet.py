## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init

from networks.ConvLSTM import ConvLSTM
from utils import *

class SingleTransformNet(nn.Module):

    def dehaze_initialize(self):

        self.trans_output.conv2d.weight.data[:] = 0
        self.trans_output.conv2d.bias.data[:] = 0.5
        self.alpha_output.conv2d.weight.data[:] = 0
        self.alpha_output.conv2d.bias.data[:] = 1
    def derain_initialize(self):
        self.s_net.apply(weight_init_scale)        

    def __init__(self, opts, nc_in, nc_out, streak_tag, haze_tag, flow_tag):
        super(SingleTransformNet, self).__init__()

        self.blocks = opts.blocks
        self.epoch = 0
        nf = opts.nf
        self.nf = nf
        use_bias = True
        opts.norm = "None"
        
        self.conv1 = ConvLayer(3 , nf, kernel_size=3, stride=1, bias=use_bias, norm=opts.norm) ## input: P_t, O_t-1
        self.res1 = ResidualBlock(nf, bias=use_bias, norm=opts.norm)
        self.conv2 = ConvLayer(nf, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.res2 = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.conv3 = ConvLayer(nf * 2, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)

        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))

        self.deconv3 = UpsampleConvLayer(nf * 8, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.dres2 = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.deconv2 = UpsampleConvLayer(nf * 4, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.dres1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.deconv1 = ConvLayer(nf * 2, nf, kernel_size=3, stride=1)

        self.s_res1 = ConvLayer(3, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_res2 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res3 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res4 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res5 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_out1 = ConvLayer(nf*4, 3, kernel_size=3, stride=1, bias=False)


        self.s_res6 = ConvLayer(6, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_res7 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res8 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res9 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res10 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_out2 = ConvLayer(nf*4, 3, kernel_size=3, stride=1, bias=False)

        self.s_res11 = ConvLayer(6, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_res12 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res13 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res14 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res15 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_out3 = ConvLayer(nf*4, 3, kernel_size=3, stride=1, bias=False)
        
        self.s_net1 = nn.Sequential(
                  self.s_res2,
                  self.s_res3,
                  self.s_res4,
                  self.s_res5,
                )

        self.s_net2 = nn.Sequential(
                  self.s_res7,
                  self.s_res8,
                  self.s_res9,
                  self.s_res10,
                )

        self.s_net3 = nn.Sequential(
                  self.s_res12,
                  self.s_res13,
                  self.s_res14,
                  self.s_res15,
                )

        self.streak_tag = streak_tag
        self.haze_tag = haze_tag
        self.flow_tag = flow_tag

        if self.streak_tag == 1:
            self.streak_step1_decoder1 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step1_decoder2 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step1_output = ConvLayer(nf * 4, 3, kernel_size=1, stride=1)

        if self.streak_tag == 1:
            self.streak_step2_decoder1 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step2_decoder2 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step2_output = ConvLayer(nf * 4, 3, kernel_size=1, stride=1)

        if self.streak_tag == 1:
            self.streak_step3_decoder1 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step3_decoder2 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step3_output = ConvLayer(nf * 4, 3, kernel_size=1, stride=1)

        if self.haze_tag == 1:
            self.alpha_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
            self.alpha_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
            self.alpha_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

            self.trans_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
            self.trans_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
            self.trans_output = ConvLayer(nf * 1, 1, kernel_size=1, stride=1)

        if self.flow_tag == 1:
            self.flow_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
            self.flow_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
            self.flow_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

        self.residue_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.residue_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.residue_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, streak_tag, haze_tag, flow_tag):
        E1 = self.res1(self.relu(self.conv1(X)))
        E2 = self.res2(self.relu(self.conv2(E1)))
        E3= self.relu(self.conv3(E2))

        RB = E3
        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)

        D3 = torch.cat((RB, E3), 1)
        D2 = self.dres2(self.relu(self.deconv3(D3)))
        D2 = torch.cat((D2, E2), 1)
        D1 = self.dres1(self.relu(self.deconv2(D2)))
        D0 = torch.cat((D1, E1), 1)
        D0 = self.relu(self.deconv1(D0))

        result = []

        one_tensor = torch.ones_like(X).cuda()
        zero_tensor = torch.zeros_like(X).cuda() 

        if streak_tag == 1:
            D0_F1 = self.s_net1(self.s_res1(X))
            streak1 = self.streak_step1_output(self.streak_step1_decoder2(self.streak_step1_decoder1(D0_F1)))

            D0_F2 = self.s_net2(self.s_res6(torch.cat((X, X-streak1), 1))) + D0_F1
            streak2 = self.streak_step2_output(self.streak_step2_decoder2(self.streak_step2_decoder1(D0_F2))) + streak1.detach()

            D0_F3 = self.s_net3(self.s_res11(torch.cat((X, X-streak2), 1))) + D0_F2
            streak3 = self.streak_step3_output(self.streak_step3_decoder2(self.streak_step3_decoder1(D0_F3))) + streak2.detach()

            result.append(streak1)
            result.append(streak2)
            result.append(streak3)
        else:
            result.append(zero_tensor)
            result.append(zero_tensor)
            result.append(zero_tensor)

        if haze_tag == 1:
            D0_alpha = D0
            alpha = self.alpha_output(self.alpha_decoder2(self.alpha_decoder1(D0_alpha)))

            D0_trans = D0
            trans = self.trans_output(self.trans_decoder2(self.trans_decoder1(D0_trans)))
            trans = torch.cat((trans, trans, trans), 1)

            result.append(alpha)
            result.append(trans)
        else:
            result.append(one_tensor)
            result.append(one_tensor)

        if flow_tag == 1:
            D0_flow = D0
            flow  = self.flow_output(self.flow_decoder2(self.flow_decoder1(D0_flow)))

            result.append(flow)
        else:
            result.append(zero_tensor)

        D0_residue = D0
        residue  = self.residue_output(self.residue_decoder2(self.residue_decoder1(D0_residue)))

        result.append(residue)

        return result

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=None, bias=True):
        super(ConvLayer, self).__init__()

        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class UpsampleConvLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm=None, bias=True):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample

        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)

        return out

class ResidualBlock(nn.Module):
    
    def __init__(self, channels, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1  = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.conv2  = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        #self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):

        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input

        return out
