
class item:
    def __init__(self):
        self.name = ''

opt = item()

opt.checkpoint_dir = './checkpoints/P401_video_rain_self/'
opt.data_dir = 'data_NTU/'
opt.list_filename = './lists/video_rain_removal_train.txt'
opt.test_list_filename = './lists/video_rain_removal_test.txt'
opt.self_tag = 'P401_video_rain_self'

opt.model_name = 'derain_self'
opt.batch_size = 8
opt.crop_size = 64
opt.vgg_path = '/home/yangwenhan/.torch/models/vgg16-397923af.pth'

opt.threads = 8
opt.input_show = False

opt.train_epoch_size = 500
opt.valid_epoch_size = 100
opt.epoch_max = 100
