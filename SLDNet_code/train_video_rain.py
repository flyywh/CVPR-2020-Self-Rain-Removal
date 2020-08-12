#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import datasets_multiple
import utils
from utils import *
import torch.nn.init as init
from torch.nn.init import *
import torchvision
from loss import *
from option import *
import torch.nn.functional as F

def mask_generate(frame1, frame2, b, c, h, w, th):
    img_min = torch.min(frame1, dim=1)
    img_min = img_min[0]

    img_min2 = torch.min(frame2, dim=1)
    img_min2 = img_min2[0]

    frame_diff = img_min - img_min2
    frame_diff[frame_diff<0] = 0

    frame_mask = 1 - torch.exp(-frame_diff*frame_diff/th)
    frame_neg_mask = 1- frame_mask

    frame_neg_mask = frame_neg_mask.view(b, 1, h, w).detach()
    frame_mask = frame_mask.view(b, 1, h, w).detach()

    frame_neg_mask = torch.cat((frame_neg_mask, frame_neg_mask, frame_neg_mask), 1)
    frame_mask = torch.cat((frame_mask, frame_mask, frame_mask), 1)

    return frame_mask, frame_neg_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fast Blind Video Temporal Consistency")
    parser.add_argument('-three_dim_model',   type=str,     default="TDModel",     help='Multi-frame models for hanlde videos')

    parser.add_argument('-nf',              type=int,     default=16,               help='#Channels in conv layer')
    parser.add_argument('-blocks',          type=int,     default=2,                help='#ResBlocks') 
    parser.add_argument('-norm',            type=str,     default='IN',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')

    parser.add_argument('-datasets_tasks',  type=str,     default='video_rain_removal',    help='dataset-task pairs list')
    parser.add_argument('-data_dir',        type=str,     default='data_heavy',     help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=32,               help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.4,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=7,                help='#frames for training')
        
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L2",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-VGGLayers',       type=str,     default="1",              help="VGG layers for perceptual loss, combinations of 1, 2, 3, 4")

    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')

    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.01,             help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=16,               help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')

    parser.add_argument('-list_filename',   type=str,      help='use cpu?')

    opts = parser.parse_args()

    opts.checkpoint_dir = opt.checkpoint_dir
    opts.data_dir = opt.data_dir
    opts.list_filename = opt.list_filename
    opts.model_name = opt.model_name
    opts.batch_size = opt.batch_size
    opts.crop_size = opt.crop_size
    opts.vgg_path = opt.vgg_path

    opts.train_epoch_size = opt.train_epoch_size
    opts.valid_epoch_size = opt.valid_epoch_size
    opts.epoch_max = opt.epoch_max
    opts.threads = opt.threads
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m   

    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix

    opts.size_multiplier = 2 ** 6
    print(opts)

    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)

    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    print('===> Initializing model from %s...' %opts.three_dim_model)
    three_dim_model = networks.__dict__[opts.three_dim_model](opts, 3, 64)
    fusion_model = networks.__dict__[opts.three_dim_model](opts, 3, 64)

    three_dim_model.apply(weight_init)
    fusion_model.apply(weight_init)

    ### Load pretrained FlowNet2
    opts.rgb_max = 1.0
    opts.fp16 = False

    FlowNet = networks.FlowNet2(opts, requires_grad=True)
    model_filename = os.path.join("./pretrained_models", "FlowNet2_checkpoint.pth.tar")
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    FlowNet.load_state_dict(checkpoint['state_dict'])

    if opts.solver == 'SGD':
        optimizer = optim.SGD(three_dim_model.parameters(), \
                              lr=opts.lr_init, momentum=opts.momentum, weight_decay= opts.weight_decay )
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam([ \
                                {'params': three_dim_model.parameters(), 'lr': opts.lr_init }, \
                                {'params': fusion_model.parameters(), 'lr': opts.lr_init }, \
                               ], lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))

        optimizer_flow = optim.Adam([ \
                                {'params': FlowNet.parameters(), 'lr': opts.lr_init*0.001 }, \
                               ], \
                               lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))

    else:
        raise Exception("Not supported solver (%s)" %opts.solver)

    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = 0

    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]

    if epoch_st > 0:
        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')
        three_dim_model, fusion_model, FlowNet, optimizer = utils.load_model(three_dim_model, fusion_model, FlowNet, optimizer, opts, epoch_st)

    print(three_dim_model)

    num_params = utils.count_network_parameters(three_dim_model)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')

    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir)

    VGG = networks.Vgg16(requires_grad=False)
    device = torch.device("cuda" if opts.cuda else "cpu")

    three_dim_model = three_dim_model.cuda()
    FlowNet = FlowNet.cuda()
    fusion_model = fusion_model.cuda()

    vgg_model = vgg_init(opt.vgg_path)
    vgg = vgg(vgg_model)
    vgg.eval()

    fusion_model.train()
    three_dim_model.train()
    FlowNet.train()

    train_dataset = datasets_multiple.MultiFramesDataset(opts, "train")

    loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

    while three_dim_model.epoch < opts.epoch_max:
        three_dim_model.epoch += 1

        data_loader = utils.create_data_loader(train_dataset, opts, "train")
        current_lr = utils.learning_rate_decay(opts, three_dim_model.epoch)

        for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        for param_group in optimizer_flow.param_groups:
                param_group['lr'] = current_lr*0.001
        
        error_last = 1e8
        ts = datetime.now()

        for iteration, batch in enumerate(data_loader, 1):
            total_iter = (three_dim_model.epoch - 1) * opts.train_epoch_size + iteration
            cross_num = 1

            frame_i = []

            for t in range(opts.sample_frames):
                frame_i.append(batch[t * cross_num].cuda())

            data_time = datetime.now() - ts
            ts = datetime.now()

            optimizer.zero_grad()
            optimizer_flow.zero_grad()

            [b, c, h, w] = frame_i[0].shape

            frame_i0 = frame_i[0]
            frame_i1 = frame_i[1]
            frame_i2 = frame_i[2]
            frame_i3 = frame_i[3]

            frame_i3_target = frame_i3

            frame_i4 = frame_i[4]
            frame_i5 = frame_i[5]
            frame_i6 = frame_i[6]

            flow_warping = Resample2d().cuda()

            flow_i30 = FlowNet(frame_i3, frame_i0)
            flow_i31 = FlowNet(frame_i3, frame_i1)
            flow_i32 = FlowNet(frame_i3, frame_i2)

            flow_i34 = FlowNet(frame_i3, frame_i4)
            flow_i35 = FlowNet(frame_i3, frame_i5)
            flow_i36 = FlowNet(frame_i3, frame_i6)

            warp_i0 = flow_warping(frame_i0, flow_i30) 
            warp_i1 = flow_warping(frame_i1, flow_i31)
            warp_i2 = flow_warping(frame_i2, flow_i32)

            warp_i4 = flow_warping(frame_i4, flow_i34)
            warp_i5 = flow_warping(frame_i5, flow_i35)
            warp_i6 = flow_warping(frame_i6, flow_i36)

            warp_i0 = warp_i0.view(b, c, 1, h, w)
            warp_i1 = warp_i1.view(b, c, 1, h, w)
            warp_i2 = warp_i2.view(b, c, 1, h, w)
            warp_i4 = warp_i4.view(b, c, 1, h, w)
            warp_i5 = warp_i5.view(b, c, 1, h, w)
            warp_i6 = warp_i6.view(b, c, 1, h, w)

            frame_input = torch.cat((warp_i0.detach(), warp_i1.detach(), warp_i2.detach(), warp_i2.detach(), warp_i4.detach(), warp_i5.detach(), warp_i6.detach()), 2)
            frame_pred = three_dim_model(frame_input)
            frame_i3_rs = frame_i3.view(b, c, 1, h, w)
            frame_target = frame_i3_rs

            frame_input2 = torch.cat((warp_i0.detach(), warp_i1.detach(), warp_i2.detach(), frame_pred.detach(), frame_i3_rs,  warp_i4.detach(), warp_i5.detach()), 2)
            frame_pred_rf = fusion_model(frame_input2) + frame_pred.detach()

            frame_pred_rf = frame_pred_rf.view(b, c, h, w)

            flow_i03 = FlowNet(frame_i0, frame_i3)
            back_warp_i0 = flow_warping(frame_pred_rf, flow_i03)

            flow_i13 = FlowNet(frame_i1, frame_i3)
            back_warp_i1 = flow_warping(frame_pred_rf, flow_i13)

            flow_i23 = FlowNet(frame_i2, frame_i3)
            back_warp_i2 = flow_warping(frame_pred_rf, flow_i23)

            flow_i43 = FlowNet(frame_i4, frame_i3)
            back_warp_i4 = flow_warping(frame_pred_rf, flow_i43)

            flow_i53 = FlowNet(frame_i5, frame_i3)
            back_warp_i5 = flow_warping(frame_pred_rf, flow_i53)

            flow_i63 = FlowNet(frame_i6, frame_i3)
            back_warp_i6 = flow_warping(frame_pred_rf, flow_i63)


            optical_loss = loss_fn(warp_i0, frame_i3_rs) + loss_fn(warp_i1, frame_i3_rs) + loss_fn(warp_i2, frame_i3_rs) + loss_fn(warp_i4, frame_i3_rs) + loss_fn(warp_i5, frame_i3_rs) + loss_fn(warp_i6, frame_i3_rs)
            optical_loss.backward()

            overall_loss = loss_fn(frame_pred, frame_target)
            overall_loss.backward()

            frame_diff = frame_i3_rs - frame_pred
            frame_mask = 1 - torch.exp(-frame_diff*frame_diff/0.00001)
            frame_neg_mask = 1- frame_mask

            frame_neg_mask = frame_neg_mask.view(b, c, h, w).detach()
            frame_mask = frame_mask.view(b, c, h, w).detach()

            frame_neg_m0, frame_m0 =  mask_generate(frame_i0, back_warp_i0, b, c, h, w, 0.01)
            frame_neg_m1, frame_m1 =  mask_generate(frame_i1, back_warp_i1, b, c, h, w, 0.01)
            frame_neg_m2, frame_m2 =  mask_generate(frame_i2, back_warp_i2, b, c, h, w, 0.01)
            frame_neg_m4, frame_m4 =  mask_generate(frame_i4, back_warp_i4, b, c, h, w, 0.01)
            frame_neg_m5, frame_m5 =  mask_generate(frame_i5, back_warp_i5, b, c, h, w, 0.01)
            frame_neg_m6, frame_m6 =  mask_generate(frame_i6, back_warp_i6, b, c, h, w, 0.01)

            refine_loss = (1/7)*(loss_fn(back_warp_i0.view(b, c, h, w)*frame_m0, frame_i0.view(b, c, h, w)*frame_m0) + \
                                 loss_fn(back_warp_i1.view(b, c, h, w)*frame_m1, frame_i1.view(b, c, h, w)*frame_m1) + \
                                 loss_fn(back_warp_i2.view(b, c, h, w)*frame_m2, frame_i2.view(b, c, h, w)*frame_m2) + \
                                 loss_fn(back_warp_i4.view(b, c, h, w)*frame_m4, frame_i4.view(b, c, h, w)*frame_m4) + \
                                 loss_fn(back_warp_i5.view(b, c, h, w)*frame_m5, frame_i5.view(b, c, h, w)*frame_m5) + \
                                 loss_fn(back_warp_i6.view(b, c, h, w)*frame_m6, frame_i6.view(b, c, h, w)*frame_m6)) + loss_fn(frame_pred_rf*frame_neg_mask, frame_i3*frame_neg_mask)

            refine_loss.backward()
            
            optimizer.step()
            optimizer_flow.step()

            error_last_inner_epoch = overall_loss.item()
            network_time = datetime.now() - ts

            info = "[GPU %d]: " %(opts.gpu)
            info += "Epoch %d; Batch %d / %d; " %(three_dim_model.epoch, iteration, len(data_loader))

            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            info += "\tmodel = %s\n" %opts.model_name

            loss_writer.add_scalar('Rect Loss', overall_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Rect Loss", overall_loss.item())

            loss_writer.add_scalar('Fusion Loss', refine_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Fusion Loss", refine_loss.item())

            loss_writer.add_scalar('Optical Loss', optical_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Optical Loss", optical_loss.item())

            print(info)
            error_last = error_last_inner_epoch

        utils.save_model(three_dim_model, fusion_model, FlowNet, optimizer, opts)        

