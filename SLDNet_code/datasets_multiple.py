### python lib
import os, sys, math, random, glob, cv2
import numpy as np

### torch lib
import torch
import torch.utils.data as data

### custom lib
import utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]

class MultiFramesDataset(data.Dataset):

    def __init__(self, opts, mode):
        super(MultiFramesDataset, self).__init__()
        self.opts = opts
        self.mode = mode
        self.task_videos = []
        self.num_frames = []

        list_filename = opts.list_filename

        with open(list_filename) as f:
            videos = [line.rstrip() for line in f.readlines()]

        for video in videos:
            self.task_videos.append([os.path.join(video)])

            input_dir = os.path.join(self.opts.data_dir, self.mode, video)
            frame_list = glob.glob(os.path.join(input_dir, '*.jpg'))

            if len(frame_list) == 0:
                raise Exception("No frames in %s" %input_dir)

            self.num_frames.append(len(frame_list))

        print("[%s] Total %d videos (%d frames)" %(self.__class__.__name__, len(self.task_videos), sum(self.num_frames)))

    def __len__(self):
        return len(self.task_videos)


    def __getitem__(self, index):

        N = self.num_frames[index]
        T = random.randint(0, N - self.opts.sample_frames)

        video = self.task_videos[index][0]

        input_dir = os.path.join(self.opts.data_dir, self.mode, video)

        frame_i = []

        for t in range(T+1, T + self.opts.sample_frames+1):
            frame_i.append(utils.read_img(os.path.join(input_dir, "%05d.jpg" % t)))

        if self.mode == 'train':
            if self.opts.geometry_aug:

                H_in = frame_i[0].shape[0]
                W_in = frame_i[0].shape[1]

                sc = np.random.uniform(self.opts.scale_min, self.opts.scale_max)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                ## scaled size should be greater than opts.crop_size
                if H_out < W_out:
                    if H_out < self.opts.crop_size:
                        H_out = self.opts.crop_size
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else: ## W_out < H_out
                    if W_out < self.opts.crop_size:
                        W_out = self.opts.crop_size
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.opts.sample_frames):
                    frame_i[t] = cv2.resize(frame_i[t], (W_out, H_out))

            cropper = RandomCrop(frame_i[0].shape[:2], (self.opts.crop_size, self.opts.crop_size))
            
            for t in range(self.opts.sample_frames):
                frame_i[t] = cropper(frame_i[t])

            if self.opts.geometry_aug:
                ## horizontal flip
                if np.random.random() >= 0.5:
                    for t in range(self.opts.sample_frames):
                        frame_i[t] = cv2.flip(frame_i[t], flipCode=0)

            if self.opts.order_aug:
                ## reverse temporal order
                if np.random.random() >= 0.5:
                    frame_i.reverse()
        
        elif self.mode == "test":
            ## resize image to avoid size mismatch after downsampline and upsampling
            H_i = frame_i[0].shape[0]
            W_i = frame_i[0].shape[1]

            H_o = int(math.ceil(float(H_i) / self.opts.size_multiplier) * self.opts.size_multiplier)
            W_o = int(math.ceil(float(W_i) / self.opts.size_multiplier) * self.opts.size_multiplier)

            for t in range(self.opts.sample_frames):
                frame_i[t] = cv2.resize(frame_i[t], (W_o, H_o))
        else:
            raise Exception("Unknown mode (%s)" %self.mode)

        ### convert (H, W, C) array to (C, H, W) tensor
        data = []
        for t in range(self.opts.sample_frames):
            data.append(torch.from_numpy(frame_i[t].transpose(2, 0, 1).astype(np.float32)).contiguous())

        return data
