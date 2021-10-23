import argparse
import sys
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import datasets
from multiscaleloss import compute_photometric_loss, estimate_corresponding_gt_flow, flow_error_dense, smooth_loss
import datetime
from tensorboardX import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint
import cv2
import torch
import os, os.path
import numpy as np
import h5py
import random
from vis_utils import *
from torch.utils.data import Dataset, DataLoader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='Spike-FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, metavar='DIR', default='./datasets/celex_datasets',
                    help='path to dataset')
parser.add_argument('--savedir', type=str, metavar='DATASET', default='spikeflownet',
                    help='results save dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='spike_flownets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                         ' | '.join(model_names))
parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=800, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[1, 1, 1, 1], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--evaluate-interval', default=5, type=int, metavar='N',
                    help='Evaluate every \'evaluate interval\' epochs ')
parser.add_argument('--print-freq', '-p', default=8000, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder')
parser.add_argument('--div-flow', default=1,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[5, 10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 170], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')
parser.add_argument('--render', dest='render', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--celexencode', action='store_true')
args = parser.parse_args()

# Initializations
best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_resize = 800
event_interval = 0
spiking_ts = 1
sp_threshold = 0

testenv = 'celex_l2r'
# testenv = 'celex_r2l'
# testenv = 'celex_t2b'
# testenv = 'celex_b2t'

testdir = os.path.join(args.data, testenv)

testfile = testdir + '/' + testenv + '_data.txt'

test_flow_map_dir = os.path.join(testdir, 'celex_image_flow')


class Test_loading(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 1
        self.xoff = 240
        self.yoff = 0
        self.split = 10
        self.half_split = int(self.split / 2)

        d_set = h5py.File(testfile, 'r')

        # Training input data, label parse
        # ts size is 623
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        self.length = len(os.listdir(testdir))
        d_set = None

    def __getitem__(self, index):
        # 这四个对应
        # former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off
        # 这个20又是什么道理呢？
        if (index + 20 < self.length) and (index > 20):
            # 256,256,5
            aa = np.zeros((800, 800, self.half_split), dtype=np.uint8)
            bb = np.zeros((800, 800, self.half_split), dtype=np.uint8)
            cc = np.zeros((800, 800, self.half_split), dtype=np.uint8)
            dd = np.zeros((800, 800, self.half_split), dtype=np.uint8)

            im_et = np.load(testdir + '/count_data/' + str(int(index + 1)) + '.npy')

            # 800-0-0=800 1280-240-240=800,,,,256?
            # 也就是说他不是resize的，而是直接剪裁的
            aa[:, :, :] = im_et[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:5].astype(float)
            bb[:, :, :] = im_et[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:5].astype(float)
            cc[:, :, :] = im_et[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 5:10].astype(float)
            dd[:, :, :] = im_et[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 5:10].astype(float)

            # 这里似乎应该给他resize

            # return aa, bb, cc, dd, self.image_raw_ts[index], self.image_raw_ts[index + self.dt]
            return aa, bb, cc, dd, 0, 0
        else:
            pp = np.zeros((image_resize, image_resize, self.half_split))
            # return pp, pp, pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros(
            #     (self.image_raw_ts[index].shape))
            return pp, pp, pp, pp, 0, 0

    def __len__(self):
        return self.length


def drawOptFlowMap(flow, img, step, color, name_idx):
    for x in range(0, 800, step):
        for y in range(0, 800, step):
            start_point = (x, y)
            end_point = (round(x + flow[x, y, 0]), round(y + flow[x, y, 1]))
            cv2.line(img, start_point, end_point, color)
            cv2.arrowedLine(img, start_point, end_point, color)
            cv2.circle(img, start_point, 2, color)
    cv2.imshow('celex', img)
    cv2.imwrite(test_flow_map_dir + '/' + name_idx + '.jpg', img)


def validate(test_loader, model, epoch, output_writers):
    global args, image_resize, sp_threshold

    # batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    batch_size_v = 4
    sp_threshold = 0.75

    iters = 0.
    scale = 1

    # 0 is start location
    for i, data in enumerate(test_loader, 0):
        # 这一帧灰度的开始时间和结束时间
        # etet格式
        former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off, st_time, ed_time = data

        if torch.sum(former_inputs_on + former_inputs_off) > 0:
            # 1*4*256*256*5
            input_representation = torch.zeros(former_inputs_on.size(0), batch_size_v, image_resize, image_resize,
                                               former_inputs_on.size(3)).float()
            # 所谓4通道
            for b in range(batch_size_v):
                if b == 0:
                    input_representation[:, 0, :, :, :] = former_inputs_on
                elif b == 1:
                    input_representation[:, 1, :, :, :] = former_inputs_off
                elif b == 2:
                    input_representation[:, 2, :, :, :] = latter_inputs_on
                elif b == 3:
                    input_representation[:, 3, :, :, :] = latter_inputs_off

            # compute output
            input_representation = input_representation.to(device)
            output = model(input_representation.type(torch.cuda.FloatTensor), image_resize, sp_threshold)
            # output is torch.Size([1, 2, 256, 256])
            # pred_flow = output
            pred_flow = np.zeros((image_resize, image_resize, 2))
            output_temp = output.cpu()
            pred_flow[:, :, 0] = cv2.resize(np.array(output_temp[0, 0, :, :]), (image_resize, image_resize),
                                            interpolation=cv2.INTER_LINEAR)
            pred_flow[:, :, 1] = cv2.resize(np.array(output_temp[0, 1, :, :]), (image_resize, image_resize),
                                            interpolation=cv2.INTER_LINEAR)
            # 0 x direction; 1 y direction
            # 在这里直接save或者可视化

            #   ----------- Visualization
            if epoch < 0:
                pass
            #     # 1 256 256 5 mask_temp
                mask_temp = former_inputs_on + latter_inputs_on
                mask_temp = torch.sum(torch.sum(mask_temp, 0), 2)
                mask_temp_np = np.squeeze(np.array(mask_temp)) > 0

                # 所谓spike_image，就是指在两张灰度图之间，出现事件，那就认为他是spike,就是255
                spike_image = mask_temp
                spike_image[spike_image > 0] = 255
                # spike_image.shape 256*256
                if args.render:
                    cv2.imshow('Spike Image', np.array(spike_image, dtype=np.uint8))

                # gray = cv2.resize(gray_image[i], (scale * image_resize, scale * image_resize),
                #                   interpolation=cv2.INTER_LINEAR)
                # if args.render:
                #     cv2.imshow('Gray Image', cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

                out_temp = np.array(output_temp.cpu().detach())
                x_flow = cv2.resize(np.array(out_temp[0, 0, :, :]), (scale * image_resize, scale * image_resize),
                                    interpolation=cv2.INTER_LINEAR)
                y_flow = cv2.resize(np.array(out_temp[0, 1, :, :]), (scale * image_resize, scale * image_resize),
                                    interpolation=cv2.INTER_LINEAR)
            #     # 炫彩光流图
            #     flow_rgb = flow_viz_np(x_flow, y_flow)
            #     if args.render:
            #         cv2.imshow('Predicted Flow Output', cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))
            #
            #     gt_flow_x = cv2.resize(gt_flow[:, :, 0], (scale * image_resize, scale * image_resize),
            #                            interpolation=cv2.INTER_LINEAR)
            #     gt_flow_y = cv2.resize(gt_flow[:, :, 1], (scale * image_resize, scale * image_resize),
            #                            interpolation=cv2.INTER_LINEAR)

            #     # mask_tmp_np就是把所有产生事件的位置置为1了
            #     masked_x_flow = cv2.resize(np.array(out_temp[0, 0, :, :] * mask_temp_np),
            #                                (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
            #     masked_y_flow = cv2.resize(np.array(out_temp[0, 1, :, :] * mask_temp_np),
            #                                (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
            #     # 炫彩mask_pred_flow
            #     flow_rgb_masked = flow_viz_np(masked_x_flow, masked_y_flow)
            #     if args.render:
            #         cv2.imshow('Masked Predicted Flow', cv2.cvtColor(flow_rgb_masked, cv2.COLOR_BGR2RGB))
            #
            #
                cv2.waitKey(1)

            if i < len(output_writers):  # log first output of first batches
                output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

            iters += 1
    # 这个n_points好像有点问题啊，这样一来不是只计算最后的了吗
    print('-------------------------------------------------------')
    print('-------------------------------------------------------')

    return


def celexEncode():
    print('----> encoding celex data <----')

    count_dir = os.path.join(testdir, 'count_data')
    if not os.path.exists(count_dir):
        os.makedirs(count_dir)
    gray_dir = os.path.join(testdir, 'gray_data')
    if not os.path.exists(gray_dir):
        os.makedirs(gray_dir)

    with open(testfile) as f:
        lines = f.readlines()

    data_split = 10

    # split in 30 ms, 整体做分割
    # frame_dates[frame10[frame_single,],...]
    frame_cnt = 1
    frame_dates = []
    cur_frame10 = []
    cur_frame_single = []
    for line in lines:
        if line[2] < 3000 * frame_cnt:
            # if line[2]<30000*(frame-1)+
            cur_frame_single.append(line)
        else:
            cur_frame10.append(cur_frame_single)
            # frame_dates.append(cur_frame10)
            cur_frame_single.clear()
            cur_frame_single.append(line)
            frame_cnt += 1
        if len(cur_frame10) == 10:
            frame_dates.append(cur_frame10)
            cur_frame10.clear()

    data_cnt = 0
    # frame_data include n frame
    # frame include 10 data,
    # every data is 3 ms
    for frame in frame_dates:
        cur_mat_10 = np.zeros((2, 800, 1280, data_split))
        min_t = frame[0][2]
        max_t = frame[-1][2]
        # update event cnt and latest time
        # cur_mat=np.zeros(2,800,1280)
        for idx, data in enumerate(frame):
            for p in data:
                cur_mat_10[0][p[0]][p[1]][idx] += 1
                cur_mat_10[1][p[0]][p[1]][idx] = (p[2] - min_t) / (max_t - min_t)
        np.save(os.path.join(count_dir, str(data_cnt)), cur_mat_10)

        bin_img = np.sum(cur_mat_10[1], axis=2)
        np.save(os.path.join(gray_dir, str(data_cnt)), bin_img)
        data_cnt += 1


def main():
    global args, best_EPE, image_resize, event_interval, spiking_ts, device, sp_threshold
    if args.celexencode:
        celexEncode()
        return
    if not os.path.exists(test_flow_map_dir):
        os.makedirs(test_flow_map_dir)

    # spikeflownet, adam, 100epochs,epochSize800,8,5e-5
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    # no_data: False for default
    # 没啥特别的，就是用时间戳区分一下
    #############################################z这里应该可以删掉
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.savedir, save_path)
    print('=> Everything will be saved to {}'.format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
#######################################################################
    # 这一步本质就是创建文件夹，后续会有更多操作
    # train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    # test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    ##################################s####################################3
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    # Data loading code
    # torchvison.transforms
    # co_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomVerticalFlip(0.5),
    #     transforms.RandomRotation(30),
    #     transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    #     transforms.ToTensor(),
    # ])
    Test_dataset = Test_loading()
    # workers default is 8 make -j8的那个8，wnmd
    test_loader = DataLoader(dataset=Test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.workers)

    # create model
    # args.ptrtrained='./pretrain/checkpoint_dt1.pth.tar'
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        # args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    # 这个操作没太看懂啊
    # 就是spike_flownets,他调用了models中的spike_flownet函数，参数是network_data
    # 改成下边哪行似乎也没差啊，他可能就是为了强行使用args.arch
    model = models.__dict__[args.arch](network_data).cuda()
    # model = models.spike_flownets(network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # sys.exit()

    assert (args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    # 这两个参数也不对啊，真特么神奇啊，自己就变成下划线了
    # print([k for k in model.named_parameters()])
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)
    # sys.exit()
    args.evaluate = True
    if args.evaluate:
        # 强制之后的内容不进行计算图构建，不追踪梯度
        with torch.no_grad():
            # best_EPE = validate(test_loader, model, -1, output_writers)
            validate(test_loader, model, -1, output_writers)
        return
    # ----------------------- line between train and test --------------------------------------------


if __name__ == '__main__':
    main()
