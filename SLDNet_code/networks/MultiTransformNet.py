## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init

from utils import *
from networks.ConvLSTM import ConvLSTM

class MultiTransformNet(nn.Module):

    def __init__(self, opts, nc_in, nc_out, streak_tag, haze_tag, flow_tag):
        super(MultiTransformNet, self).__init__()

        self.blocks = opts.blocks
        self.epoch = 0
        nf = opts.nf
        self.nf = nf

        use_bias = True
        opts.norm = "None"
        
        self.conv1a = ConvLayer(3, nf, kernel_size=3, stride=1, bias=use_bias, norm=opts.norm)  ## input: P_t, O_t-1
        self.res1a = ResidualBlock(nf, bias=use_bias, norm=opts.norm)
        self.conv2a = ConvLayer(nf, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.res2a = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.conv3a = ConvLayer(nf * 2, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)

        self.conv1c = ConvLayer(3, nf, kernel_size=3, stride=1, bias=use_bias, norm=opts.norm)  ## input: P_t, O_t-1
        self.res1c = ResidualBlock(nf, bias=use_bias, norm=opts.norm)
        self.conv2c = ConvLayer(nf, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.res2c = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.conv3c = ConvLayer(nf * 2, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)

        self.conv1b = ConvLayer(6, nf, kernel_size=3, stride=1, bias=use_bias, norm=opts.norm)  ## input: P_t, O_t-1
        self.res1b = ResidualBlock(nf, bias=use_bias, norm=opts.norm)
        self.conv2b = ConvLayer(nf, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.res2b = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.conv3b = ConvLayer(nf * 2, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)

        self.conv3 = ConvLayer(nf * 12, nf * 4, kernel_size=3, stride=1, bias=use_bias, norm=opts.norm)

        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))

        self.convlstm = ConvLSTM(input_size=nf * 4, hidden_size=nf * 4, kernel_size=3)

        self.deconv3 = UpsampleConvLayer(nf * 8, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)

        self.dres2 = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.convlstm_l2 = ConvLSTM(input_size=nf * 4, hidden_size=nf * 4, kernel_size=3)
        self.deconv2 = UpsampleConvLayer(nf * 4, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)

        self.dres1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.deconv1 = ConvLayer(nf * 2, nf, kernel_size=3, stride=1)
        self.convlstm_l1 = ConvLSTM(input_size=nf * 2, hidden_size=nf * 1, kernel_size=3)

        self.s_res1 = ConvLayer(3, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_df_res1 = ConvLayer(3, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_res2 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res3 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res4 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res5 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_out1 = ConvLayer(nf*4, 3, kernel_size=3, stride=1, bias=False)

        self.s_res6 = ConvLayer(6, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_df_res6 = ConvLayer(3, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_res7 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res8 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res9 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_res10 = ResidualBlock(nf*4, bias=False, norm=opts.norm)
        self.s_out2 = ConvLayer(nf*4, 3, kernel_size=3, stride=1, bias=False)

        self.s_res11 = ConvLayer(6, nf*4, kernel_size=3, stride=1, bias=False)
        self.s_df_res11 = ConvLayer(3, nf*4, kernel_size=3, stride=1, bias=False)
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

        self.s_convlstm_l1 = ConvLSTM(input_size=nf * 4, hidden_size=nf * 4, kernel_size=3)

        self.s_net2 = nn.Sequential(
                  self.s_res7,
                  self.s_res8,
                  self.s_res9,
                  self.s_res10,
                )

        self.s_convlstm_l2 = ConvLSTM(input_size=nf * 4, hidden_size=nf * 4, kernel_size=3)

        self.s_net3 = nn.Sequential(
                  self.s_res12,
                  self.s_res13,
                  self.s_res14,
                  self.s_res15,
                )

        self.s_convlstm_l3 = ConvLSTM(input_size=nf * 4, hidden_size=nf * 4, kernel_size=3)

        self.streak_tag = streak_tag
        self.haze_tag = haze_tag
        self.flow_tag = flow_tag

        if self.streak_tag == 1:
            self.streak_step1_decoder1 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step1_decoder2 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step1_output = ConvLayer(nf * 4, 3, kernel_size=1, stride=1)

            self.streak_step2_decoder1 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step2_decoder2 = ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm)
            self.streak_step2_output = ConvLayer(nf * 4, 3, kernel_size=1, stride=1)

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

        #self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def dehaze_initialize(self):
        #self.apply(weight_init)
        self.trans_output.conv2d.weight.data[:] = 0
        self.trans_output.conv2d.bias.data[:] = 0
        self.alpha_output.conv2d.weight.data[:] = 0
        self.alpha_output.conv2d.bias.data[:] = 0

    def forward(self, X, prev_state, prev_state_l1, prev_state_l2, prev_s_state, prev_s_state_l1, prev_s_state_l2, streak_tag, haze_tag, flow_tag):
        Xa_1 = X[:, 6:9,:,:]
        Xa_2 = X[:,9:,:,:]
        Xb = X[:,:6,:,:]

        E1a_1 = self.res1a(self.relu(self.conv1a(Xa_1)))
        E2a_1 = self.res2a(self.relu(self.conv2a(E1a_1)))
        E3a_1 = self.conv3a(self.relu(E2a_1))

        E1a_2 = self.res1c(self.relu(self.conv1c(Xa_2)))
        E2a_2 = self.res2c(self.relu(self.conv2c(E1a_2)))
        E3a_2 = self.conv3c(self.relu(E2a_2))

        E1b = self.res1b(self.relu(self.conv1b(Xb)))
        E2b = self.res2b(self.relu(self.conv2b(E1b)))
        E3b = self.conv3b(self.relu(E2b))

        E3 = self.relu(self.conv3(torch.cat((E3a_1, E3a_2, E3b), 1)))

        RB = E3
        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)

        state = self.convlstm(RB, prev_state)
        D3 =  torch.cat((state[0]+RB, E3a_1), 1)

        D2 = self.dres2(self.relu(self.deconv3(D3)))

        D2_in = torch.cat((D2, E2a_1), 1)
        state_l2 = self.convlstm_l2(D2_in, prev_state_l2)

        D2_in = D2_in + state_l2[0] 

        D1 = self.relu(self.deconv2(D2_in))

        D0_in = torch.cat((D1, E1a_1), 1)
        D0 = self.relu(self.deconv1(D0_in))

        state_l1 = self.convlstm_l1(D0_in, prev_state_l1)
        D0 = D0 + state_l1[0]

        
        one_tensor = torch.ones_like(Xa_1).float().cuda()
        zero_tensor = torch.zeros_like(Xa_1).float().cuda()

        result = []

        if streak_tag == 1:
            feat_s1 = self.s_net1(self.s_res1(Xa_1) + self.s_df_res1(Xa_1-Xa_2))
            cur_s_state = self.s_convlstm_l1(feat_s1, prev_s_state)
            feat_s1 = feat_s1 #+ 0.1*cur_s_state[0]
            streak1 = self.streak_step1_output(self.streak_step1_decoder2(self.streak_step1_decoder1(feat_s1)))

            feat_s2 = self.s_net2(self.s_res6(torch.cat((Xa_1, Xa_1-streak1), 1)) + self.s_df_res6(Xa_1-Xa_2)) + feat_s1
            cur_s_state_l1 = self.s_convlstm_l2(feat_s2, prev_s_state_l1)
            feat_s2 = feat_s2 #+ 0.1*cur_s_state_l1[0]
            streak2 = self.streak_step2_output(self.streak_step2_decoder2(self.streak_step2_decoder1(feat_s2))) + streak1.detach()

            feat_s3 = self.s_net3(self.s_res11(torch.cat((Xa_1, Xa_1-streak2), 1)) + self.s_df_res11(Xa_1-Xa_2)) + feat_s2
            cur_s_state_l2 = self.s_convlstm_l3(feat_s3, prev_s_state_l2)
            feat_s3 = feat_s3 #+ 0.1*cur_s_state_l2[0]
            streak3 = self.streak_step3_output(self.streak_step3_decoder2(self.streak_step3_decoder1(feat_s3))) + streak2.detach()

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
            flow  = self.flow_output( self.flow_decoder2(self.flow_decoder1(D0_flow)))

            result.append(flow)
        else:
            result.append(zero_tensor)

        D0_residue = D0
        residue  = self.residue_output( self.residue_decoder2(self.residue_decoder1(D0_residue)))

        result.append(residue)

        return result, state, state_l1, state_l2, cur_s_state, cur_s_state_l1, cur_s_state_l2

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=None, bias=True, last_bias=0):
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

        self.relu = nn.ReLU(inplace=True)
        #self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input

        return out


