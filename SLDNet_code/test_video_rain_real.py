#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    parser.add_argument('-method', type=str, required=True, help='test model name')
    parser.add_argument('-model_name', type=str, required=True, help='test model name')
    parser.add_argument('-epoch', type=int, required=True, help='epoch')
    parser.add_argument('-streak_tag',          type=str,     default=1,               help='Whether the model handle rain streak')
    parser.add_argument('-haze_tag',            type=str,     default=1,               help='Whether the model handle haze')
    parser.add_argument('-flow_tag',            type=str,     default=1,               help='Whether the model handle haze')

    parser.add_argument('-dataset', type=str, required=True, help='dataset to test')
    parser.add_argument('-phase', type=str, default="test_real", choices=["train", "test", "test_real"])
    parser.add_argument('-data_dir', type=str, default='data', help='path to data folder')
    parser.add_argument('-list_dir', type=str, default='lists', help='path to list folder')

    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint folder')
    parser.add_argument('-task', type=str, required=True, help='evaluated task')
    parser.add_argument('-list_filename', type=str, required=True, help='evaluated task')
    parser.add_argument('-redo', action="store_true", help='Re-generate results')
    parser.add_argument('-iter', type=int)
    parser.add_argument('-gpu', type=int, default=1, help='gpu device id')

    opts = parser.parse_args()
    opts.cuda = True

    opts.size_multiplier = 2 ** 3  ## Inputs to TransformNet need to be divided by 4

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")

    opts_filename = os.path.join(opts.checkpoint_dir, opts.model_name, "opts.pth")
    print("Load %s" % opts_filename)
    with open(opts_filename, 'rb') as f:
        model_opts = pickle.load(f)

    model_filename = os.path.join(opts.checkpoint_dir, opts.model_name, "model_epoch_%d.pth" % opts.epoch)
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename)

    single_sharp_model = networks.__dict__[model_opts.single_sharp_model](model_opts, nc_in=12, nc_out=3, streak_tag=opts.streak_tag, haze_tag=opts.haze_tag, flow_tag=opts.flow_tag)
    multi_sharp_model = networks.__dict__[model_opts.multi_sharp_model](model_opts, nc_in=12, nc_out=3, streak_tag=opts.streak_tag, haze_tag=opts.haze_tag, flow_tag=opts.flow_tag)

    multi_sharp_model.load_state_dict(state_dict['multi_sharp_model'])
    single_sharp_model.load_state_dict(state_dict['single_sharp_model'])

    device = torch.device("cuda" if opts.cuda else "cpu")
    single_sharp_model = single_sharp_model.cuda()
    multi_sharp_model = multi_sharp_model.cuda()

    flow_warping = Resample2d().cuda()
    downsampler = nn.AvgPool2d((2, 2), stride=2).cuda()

    multi_sharp_model.eval()
    single_sharp_model.eval()

    list_filename = opts.list_filename

    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    times = []

    for v in range(len(video_list)):
        v = opts.iter
        video = video_list[v]

        print("Test %s on %s video %d/%d: %s" % (opts.dataset, opts.phase, v + 1, len(video_list), video))

        input_dir = os.path.join(opts.data_dir, opts.phase, "input", video)
        rain_dir = os.path.join(opts.data_dir, opts.phase, "input", video)
        haze_dir = os.path.join(opts.data_dir, opts.phase, "input", video)

        output_dir = os.path.join(opts.data_dir, opts.phase, "output", opts.model_name, opts.method, "epoch_%d" % opts.epoch, opts.task,
                                  opts.dataset, video)

        print(input_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        frame_list = glob.glob(os.path.join(input_dir, "*.png"))
        output_list = glob.glob(os.path.join(output_dir, "*.png"))

        if len(frame_list) == len(output_list) and not opts.redo:
            print("Output frames exist, skip...")
            continue

        frame_i1 = utils.read_img(os.path.join(input_dir, "001.png"))
        frame_rain1 = utils.read_img(os.path.join(rain_dir, "001.png"))
        frame_haze1 = utils.read_img(os.path.join(haze_dir, "001.png"))

        H_orig = frame_i1.shape[0]
        W_orig = frame_i1.shape[1]

        print(H_orig)
        print(W_orig)

        H_sc = int(math.ceil(float(H_orig) / opts.size_multiplier) * opts.size_multiplier)
        W_sc = int(math.ceil(float(W_orig) / opts.size_multiplier) * opts.size_multiplier)

        frame_i1 = cv2.resize(frame_i1, (W_sc, H_sc))
        frame_i1 = utils.img2tensor(frame_i1).cuda()

        frame_rain1 = cv2.resize(frame_rain1, (W_sc, H_sc))
        frame_rain1 = utils.img2tensor(frame_rain1).cuda()

        frame_haze1 = cv2.resize(frame_haze1, (W_sc, H_sc))
        frame_haze1 = utils.img2tensor(frame_haze1).cuda()

        frame_result1_s = single_sharp_model(frame_i1, opts.streak_tag, opts.haze_tag, opts.flow_tag)

        frame_streak1_pre_s = frame_result1_s[2]
        frame_alpha1_pre_s = frame_result1_s[3]
        frame_trans1_pre_s = frame_result1_s[4]
        frame_flow1_pre_s = frame_result1_s[5]
        frame_residue1_pre_s = frame_result1_s[6]

        frame_trans1_pre_s[frame_trans1_pre_s<0.05] = 0.05
        frame_trans1_pre_s[frame_trans1_pre_s>1] = 1
        
        frame_o1_s = (frame_i1 - frame_streak1_pre_s - frame_alpha1_pre_s  + frame_alpha1_pre_s*frame_trans1_pre_s)/frame_trans1_pre_s + frame_residue1_pre_s
#        frame_o1_s = (frame_i1 - frame_streak1_pre_s - 1 - frame_flow1_pre_s + frame_trans1_pre_s)/frame_trans1_pre_s + frame_residue1_pre_s

        frame_o = utils.tensor2img(frame_o1_s)
        frame_streak = utils.tensor2img(frame_streak1_pre_s)
        frame_alpha = utils.tensor2img(frame_alpha1_pre_s)
        frame_trans = utils.tensor2img(frame_trans1_pre_s)
        frame_flow = utils.tensor2img(frame_flow1_pre_s)
        frame_residue = utils.tensor2img(frame_residue1_pre_s)

        t=1
        output_filename = os.path.join(output_dir, "%03d.png"%t)
        utils.save_img(frame_o, output_filename)
        output_filename = os.path.join(output_dir, "%03d_streak.png"%t)
        utils.save_img(frame_streak, output_filename)
        output_filename = os.path.join(output_dir, "%03d_alpha.png"%t)
        utils.save_img(frame_alpha, output_filename)
        output_filename = os.path.join(output_dir, "%03d_trans.png"%t)
        utils.save_img(frame_trans, output_filename)
        output_filename = os.path.join(output_dir, "%03d_flow.png"%t)
        utils.save_img(frame_flow, output_filename)
        output_filename = os.path.join(output_dir, "%03d_residue.png"%t)
        utils.save_img(frame_residue, output_filename)
       
        #
        # hn
        #
#        frame_result1_s_hn = single_sharp_model(frame_haze1, opts.streak_tag, opts.haze_tag, opts.flow_tag)
#        frame_alpha1_pre_s_hn = frame_result1_s_hn[1]
#        frame_trans1_pre_s_hn = frame_result1_s_hn[2]
#        frame_flow1_pre_s_hn = frame_result1_s_hn[3]
#        frame_residue1_pre_s_hn = frame_result1_s_hn[4]
#
#
#        frame_trans1_pre_s_hn[frame_trans1_pre_s_hn<0.05] = 0.05
#        frame_trans1_pre_s_hn[frame_trans1_pre_s_hn<0.05] = 0.05
#        frame_o1_s_hn = (frame_haze1 - frame_alpha1_pre_s_hn - frame_flow1_pre_s_hn + frame_alpha1_pre_s_hn*frame_trans1_pre_s_hn)/frame_trans1_pre_s_hn + frame_residue1_pre_s_hn
#
#        frame_o_hn = utils.tensor2img(frame_o1_s_hn)
#        frame_alpha_hn = utils.tensor2img(frame_alpha1_pre_s_hn)
#        frame_trans_hn = utils.tensor2img(frame_trans1_pre_s_hn)
#        frame_flow_hn = utils.tensor2img(frame_flow1_pre_s_hn)
#        frame_residue_hn = utils.tensor2img(frame_residue1_pre_s_hn)
#
#        output_filename = os.path.join(output_dir, "%03d_hn.png"%t)
#        utils.save_img(frame_o_hn, output_filename)
#        output_filename = os.path.join(output_dir, "%03d_alpha_hn.png"%t)
#        utils.save_img(frame_alpha_hn, output_filename)
#        output_filename = os.path.join(output_dir, "%03d_trans_hn.png"%t)
#        utils.save_img(frame_trans_hn, output_filename)
#        output_filename = os.path.join(output_dir, "%03d_flow_hn.png"%t)
#        utils.save_img(frame_flow_hn, output_filename)
#        output_filename = os.path.join(output_dir, "%03d_residue_hn.png"%t)
#        utils.save_img(frame_residue_hn, output_filename)
        
        #
        # rn
        #
#        frame_result1_s_rn = single_sharp_model(frame_rain1, opts.streak_tag, opts.haze_tag, opts.flow_tag)
#        frame_streak1_pre_s_rn = frame_result1_s_rn[0]
#
#        frame_o1_s_rn = frame_rain1 - frame_streak1_pre_s_rn
#
#        frame_o_rn = utils.tensor2img(frame_o1_s_rn)
#        frame_streak_rn = utils.tensor2img(frame_streak1_pre_s_rn)
#
#        output_filename = os.path.join(output_dir, "%03d_rn.png"%t)
#        utils.save_img(frame_o_rn, output_filename)
#        output_filename = os.path.join(output_dir, "%03d_streak_rn.png"%t)
#        utils.save_img(frame_streak_rn, output_filename)

        lstm_state = None
        lstm_state_l1 = None
        lstm_state_l2 = None

        lstm_s1_state = None
        lstm_s2_state = None
        lstm_s3_state = None 

        for t in range(2, len(frame_list)+1): #len(frame_list)+1):
            frame_i1 = utils.read_img(os.path.join(input_dir, "%03d.png" % (t-1)))
            frame_i2 = utils.read_img(os.path.join(input_dir, "%03d.png" % (t)))
            frame_o1 = utils.read_img(os.path.join(output_dir, "%03d.png" % (t - 1)))
            #frame_o1_hn = utils.read_img(os.path.join(output_dir, "%03d_hn.png" % (t - 1)))
            #frame_o1_rn = utils.read_img(os.path.join(output_dir, "%03d_rn.png" % (t - 1)))

            frame_rain1 = utils.read_img(os.path.join(rain_dir, "%03d.png")%(t-1))
            frame_haze1 = utils.read_img(os.path.join(haze_dir, "%03d.png")%(t-1))

            frame_rain2 = utils.read_img(os.path.join(rain_dir, "%03d.png")%(t))
            frame_haze2 = utils.read_img(os.path.join(haze_dir, "%03d.png")%(t))

            H_orig = frame_i1.shape[0]
            W_orig = frame_i1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / opts.size_multiplier ) * opts.size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / opts.size_multiplier ) * opts.size_multiplier)

            frame_i1 = cv2.resize(frame_i1, (W_sc, H_sc))
            frame_i2 = cv2.resize(frame_i2, (W_sc, H_sc))
            frame_o1 = cv2.resize(frame_o1, (W_sc, H_sc))

            frame_rain1 = cv2.resize(frame_rain1, (W_sc, H_sc))
            frame_haze1 = cv2.resize(frame_haze1, (W_sc, H_sc))

            frame_rain2 = cv2.resize(frame_rain2, (W_sc, H_sc))
            frame_haze2 = cv2.resize(frame_haze2, (W_sc, H_sc))
            
            with torch.no_grad():

                frame_i1 = utils.img2tensor(frame_i1).cuda()
                frame_i2 = utils.img2tensor(frame_i2).cuda()

                frame_o1 = utils.img2tensor(frame_o1).cuda()
                #frame_o1_hn = utils.img2tensor(frame_o1_hn).cuda()
                #frame_o1_rn = utils.img2tensor(frame_o1_rn).cuda()

                frame_rain1 = utils.img2tensor(frame_rain1).cuda()
                frame_haze1 = utils.img2tensor(frame_haze1).cuda()

                frame_rain2 = utils.img2tensor(frame_rain2).cuda()
                frame_haze2 = utils.img2tensor(frame_haze2).cuda()

                frame_result2_s = single_sharp_model(frame_i2, opts.streak_tag, opts.haze_tag, opts.flow_tag)
                frame_streak2_pre_s = frame_result2_s[2]
                frame_alpha2_pre_s = frame_result2_s[3]
                frame_trans2_pre_s = frame_result2_s[4]
                frame_flow2_pre_s = frame_result2_s[5]
                frame_residue2_pre_s = frame_result2_s[6]

                frame_trans2_pre_s[frame_trans2_pre_s<0.05] = 0.05
#                frame_trans2_pre_s[frame_trans2_pre_s<0.5] = 0.5
                frame_trans2_pre_s[frame_trans2_pre_s>1] = 1

                frame_o2_s = (frame_i2 - frame_streak2_pre_s - frame_alpha2_pre_s + frame_alpha2_pre_s*frame_trans2_pre_s)/frame_trans2_pre_s + frame_residue2_pre_s
#                frame_o2_s = (frame_i2 - frame_streak2_pre_s -1 - frame_flow2_pre_s + frame_trans2_pre_s)/frame_trans2_pre_s + frame_residue2_pre_s
                #frame_o2_s = (frame_i2 - frame_alpha2 - frame_flow2 + frame_alpha2*frame_trans2)/frame_trans2

                print(frame_o2_s.shape)
                print(frame_o1.shape)

                inputs = torch.cat((frame_o2_s, frame_o1, frame_i2, frame_i1), dim=1)

                ts = time.time()

                frame_result2_pre, lstm_state, lstm_state_l1, lstm_state_l2, lstm_s1_state, lstm_s2_state, lstm_s3_state = multi_sharp_model(inputs, lstm_state, lstm_state_l1, lstm_state_l2, lstm_s1_state, lstm_s2_state, lstm_s3_state, opts.streak_tag, opts.haze_tag, opts.flow_tag)

                frame_streak2_pre = frame_result2_pre[2]
                frame_alpha2_pre = frame_result2_pre[3] + frame_alpha2_pre_s
                frame_trans2_pre = frame_result2_pre[4] + frame_trans2_pre_s
                frame_flow2_pre = frame_result2_pre[5] + frame_flow2_pre_s
                frame_residue2_pre = frame_result2_pre[6] + frame_residue2_pre_s

                frame_trans2_pre[frame_trans2_pre<0.05] = 0.05
                #frame_trans2_pre[frame_trans2_pre<0.5] = 0.5
                frame_trans2_pre[frame_trans2_pre>1] = 1

                frame_o2 = (frame_i2 - frame_streak2_pre - frame_alpha2_pre + frame_alpha2_pre*frame_trans2_pre)/frame_trans2_pre + frame_residue2_pre
#                frame_o2 = (frame_i2 - frame_streak2_pre - 1 - frame_flow2_pre + frame_trans2_pre)/frame_trans2_pre + frame_residue2_pre

                #
                # hn
                #
#                frame_result2_s_hn = single_sharp_model(frame_haze2, opts.streak_tag, opts.haze_tag, opts.flow_tag)
#                #frame_streak2_pre_s_hn = frame_result2_s_hn[0]
#                frame_alpha2_pre_s_hn = frame_result2_s_hn[1]
#                frame_trans2_pre_s_hn = frame_result2_s_hn[2]
#                frame_flow2_pre_s_hn = frame_result2_s_hn[3]
#                frame_residue2_pre_s_hn = frame_result2_s_hn[4]
#
#                frame_trans2_pre_s_hn[frame_trans2_pre_s_hn<0.05] = 0.05
#                frame_trans2_pre_s_hn[frame_trans2_pre_s_hn>1] = 1
#
#                frame_o2_s_hn = (frame_haze2 - frame_alpha2_pre_s_hn - frame_flow2_pre_s_hn + frame_alpha2_pre_s_hn*frame_trans2_pre_s_hn)/frame_trans2_pre_s_hn + frame_residue2_pre_s_hn
#                #frame_o2_s = (frame_i2 - frame_alpha2 - frame_flow2 + frame_alpha2*frame_trans2)/frame_trans2
#
#                print(frame_o2_s_hn.shape)
#                print(frame_o1_hn.shape)
#
#                inputs = torch.cat((frame_o2_s_hn, frame_o1_hn, frame_haze2, frame_haze1), dim=1)
#
#                ts = time.time()
#
#                frame_result2_pre_hn, lstm_state_hn = multi_sharp_model(inputs, lstm_state_hn, opts.streak_tag, opts.haze_tag, opts.flow_tag)
#
#                #frame_streak2_pre_hn = frame_result2_pre_hn[0] + frame_streak2_pre_s_hn
#                frame_alpha2_pre_hn = frame_result2_pre_hn[1] + frame_alpha2_pre_s_hn
#                frame_trans2_pre_hn = frame_result2_pre_hn[2] + frame_trans2_pre_s_hn
#                frame_flow2_pre_hn = frame_result2_pre_hn[3] + frame_flow2_pre_s_hn
#                frame_residue2_pre_hn = frame_result2_pre_hn[4] + frame_residue2_pre_s_hn
#
#                frame_trans2_pre_hn[frame_trans2_pre_hn<0.05] = 0.05
#                frame_trans2_pre_hn[frame_trans2_pre_hn>1] = 1
#
#                frame_o2_hn = (frame_haze2 - frame_alpha2_pre_hn - frame_flow2_pre_hn + frame_alpha2_pre_hn*frame_trans2_pre_hn)/frame_trans2_pre_hn + frame_residue2_pre_hn

                #
                # rn
                #

#                frame_result2_s_rn = single_sharp_model(frame_rain2, opts.streak_tag, opts.haze_tag, opts.flow_tag)
#                frame_streak2_pre_s_rn = frame_result2_s_rn[0]
#
#                frame_o2_s_rn = frame_rain2 - frame_streak2_pre_s_rn
#                #frame_o2_s = (frame_i2 - frame_alpha2 - frame_flow2 + frame_alpha2*frame_trans2)/frame_trans2
#
#                print(frame_o2_s_rn.shape)
#                print(frame_o1_rn.shape)
#
#                inputs = torch.cat((frame_o2_s_rn, frame_o1_rn, frame_rain2, frame_rain1), dim=1)
#
#                ts = time.time()
#
#                frame_result2_pre_rn, lstm_state_rn = multi_sharp_model(inputs, lstm_state_rn, opts.streak_tag, opts.haze_tag, opts.flow_tag)
#                frame_streak2_pre_rn = frame_result2_pre_rn[0] + frame_streak2_pre_s_rn
#
#                frame_o2_rn = frame_rain2 - frame_streak2_pre_rn
#
#                te = time.time()
#                times.append(te - ts)

                lstm_state = utils.repackage_hidden(lstm_state)
                lstm_state_l1 = utils.repackage_hidden(lstm_state_l1)
                lstm_state_l2 = utils.repackage_hidden(lstm_state_l2)

                lstm_s1_state = utils.repackage_hidden(lstm_s1_state)
                lstm_s2_state = utils.repackage_hidden(lstm_s2_state)
                lstm_s3_state = utils.repackage_hidden(lstm_s3_state)

#                lstm_state_hn = utils.repackage_hidden(lstm_state_hn)
#                lstm_state_rn = utils.repackage_hidden(lstm_state_rn)

                #lstm_state = utils.repackage_hidden(lstm_state)

            frame_o = utils.tensor2img(frame_o2)
            frame_streak = utils.tensor2img(frame_streak2_pre)
            frame_alpha = utils.tensor2img(frame_alpha2_pre)
            frame_trans = utils.tensor2img(frame_trans2_pre)
            frame_flow = utils.tensor2img(frame_flow2_pre)
            frame_residue = utils.tensor2img(frame_residue2_pre)

            output_filename = os.path.join(output_dir, "%03d.png"%t)
            utils.save_img(frame_o, output_filename)
            output_filename = os.path.join(output_dir, "%03d_streak.png"%t)
            utils.save_img(frame_streak, output_filename)
            output_filename = os.path.join(output_dir, "%03d_alpha.png"%t)
            utils.save_img(frame_alpha, output_filename)
            output_filename = os.path.join(output_dir, "%03d_trans.png"%t)
            utils.save_img(frame_trans, output_filename)
            output_filename = os.path.join(output_dir, "%03d_flow.png"%t)
            utils.save_img(frame_flow, output_filename)
            output_filename = os.path.join(output_dir, "%03d_residue.png"%t)
            utils.save_img(frame_residue, output_filename)

            frame_o_s = utils.tensor2img(frame_o2_s)
            frame_streak_s = utils.tensor2img(frame_streak2_pre_s)
            frame_alpha_s = utils.tensor2img(frame_alpha2_pre_s)
            frame_trans_s = utils.tensor2img(frame_trans2_pre_s)
            frame_flow_s = utils.tensor2img(frame_flow2_pre_s)
            frame_residue_s = utils.tensor2img(frame_residue2_pre_s)

            output_filename = os.path.join(output_dir, "%03d_s.png"%t)
            utils.save_img(frame_o_s, output_filename)
            output_filename = os.path.join(output_dir, "%03d_streak_s.png"%t)
            utils.save_img(frame_streak_s, output_filename)
            output_filename = os.path.join(output_dir, "%03d_alpha_s.png"%t)
            utils.save_img(frame_alpha_s, output_filename)
            output_filename = os.path.join(output_dir, "%03d_trans_s.png"%t)
            utils.save_img(frame_trans_s, output_filename)
            output_filename = os.path.join(output_dir, "%03d_flow_s.png"%t)
            utils.save_img(frame_flow_s, output_filename)
            output_filename = os.path.join(output_dir, "%03d_residue_s.png"%t)
            utils.save_img(frame_residue_s, output_filename)

            del frame_o2
            del frame_streak2_pre
            del frame_alpha2_pre
            del frame_trans2_pre
            del frame_flow2_pre
            del frame_residue2_pre

            del frame_o2_s
            del frame_streak2_pre_s
            del frame_alpha2_pre_s
            del frame_trans2_pre_s
            del frame_flow2_pre_s
            del frame_residue2_pre_s
            del frame_result2_pre
            del frame_result2_s
#            #
#            # rn
#            #
#
#            frame_o_rn = utils.tensor2img(frame_o2_rn)
#            frame_streak_rn = utils.tensor2img(frame_streak2_pre_rn)
#
#            output_filename = os.path.join(output_dir, "%03d_rn.png"%t)
#            utils.save_img(frame_o_rn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_streak_rn.png"%t)
#            utils.save_img(frame_streak_rn, output_filename)
#
#            frame_o_s_rn = utils.tensor2img(frame_o2_s_rn)
#            frame_streak_s_rn = utils.tensor2img(frame_streak2_pre_s_rn)
#
#            output_filename = os.path.join(output_dir, "%03d_s_rn.png"%t)
#            utils.save_img(frame_o_s_rn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_streak_s_rn.png"%t)
#            utils.save_img(frame_streak_s_rn, output_filename)
#
#            #
#            # hn
#            #
#
#            frame_o_hn = utils.tensor2img(frame_o2_hn)
#            frame_alpha_hn = utils.tensor2img(frame_alpha2_pre_hn)
#            frame_trans_hn = utils.tensor2img(frame_trans2_pre_hn)
#            frame_flow_hn = utils.tensor2img(frame_flow2_pre_hn)
#            frame_residue_hn = utils.tensor2img(frame_residue2_pre_hn)
#
#            output_filename = os.path.join(output_dir, "%03d_hn.png"%t)
#            utils.save_img(frame_o_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_alpha_hn.png"%t)
#            utils.save_img(frame_alpha_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_trans_hn.png"%t)
#            utils.save_img(frame_trans_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_flow_hn.png"%t)
#            utils.save_img(frame_flow_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_residue_hn.png"%t)
#            utils.save_img(frame_residue_hn, output_filename)
#
#            frame_o_s_hn = utils.tensor2img(frame_o2_s_hn)
#            frame_alpha_s_hn = utils.tensor2img(frame_alpha2_pre_s_hn)
#            frame_trans_s_hn = utils.tensor2img(frame_trans2_pre_s_hn)
#            frame_flow_s_hn = utils.tensor2img(frame_flow2_pre_s_hn)
#            frame_residue_s_hn = utils.tensor2img(frame_residue2_pre_s_hn)
#
#            output_filename = os.path.join(output_dir, "%03d_s_hn.png"%t)
#            utils.save_img(frame_o_s_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_alpha_s_hn.png"%t)
#            utils.save_img(frame_alpha_s_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_trans_s_hn.png"%t)
#            utils.save_img(frame_trans_s_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_flow_s_hn.png"%t)
#            utils.save_img(frame_flow_s_hn, output_filename)
#            output_filename = os.path.join(output_dir, "%03d_residue_s_hn.png"%t)
#            utils.save_img(frame_residue_s_hn, output_filename)
        break
    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" % (time_avg, len(times)))
