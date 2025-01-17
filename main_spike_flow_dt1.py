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
parser.add_argument('--data', type=str, metavar='DIR', default='./datasets',
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
args = parser.parse_args()

# Initializations
best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('--device is:',device)
image_resize = 256
event_interval = 0
spiking_ts = 1
sp_threshold = 0

trainenv = 'outdoor_day2'
# trainenv = 'indoor_flying4'
# trainenv = 'indoor_flying1'

# testenv = 'indoor_flying1'
# testenv = 'indoor_flying2'
testenv = 'indoor_flying3'

traindir = os.path.join(args.data, trainenv)
testdir = os.path.join(args.data, testenv)

trainfile = traindir + '/' + trainenv + '_data.hdf5'
testfile = testdir + '/' + testenv + '_data.hdf5'

gt_file = testdir + '/' + testenv + '_gt.hdf5'


class Train_loading(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, transform=None):
        self.transform = transform
        # Training input data, label parse
        self.dt = 1
        self.split = 10
        self.half_split = int(self.split / 2)
        self.x = 260
        self.y = 346

        d_set = h5py.File(trainfile, 'r')
        self.image_raw_event_inds = np.float64(d_set['davis']['left']['image_raw_event_inds'])
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        # gray image re-size
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        if index + 100 < self.length and index > 100:
            aa = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)
            bb = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)
            cc = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)
            dd = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)

            im_onoff = np.load(traindir + '/count_data/' + str(int(index + 1)) + '.npy')

            aa[:, :, :] = im_onoff[0, :, :, 0:5]
            bb[:, :, :] = im_onoff[1, :, :, 0:5]
            cc[:, :, :] = im_onoff[0, :, :, 5:10]
            dd[:, :, :] = im_onoff[1, :, :, 5:10]

            ee = np.uint8(np.load(traindir + '/gray_data/' + str(int(index)) + '.npy'))
            ff = np.uint8(np.load(traindir + '/gray_data/' + str(int(index + self.dt)) + '.npy'))

            if self.transform:
                seed = np.random.randint(2147483647)

                aaa = torch.zeros(256, 256, int(aa.shape[2]))
                bbb = torch.zeros(256, 256, int(bb.shape[2]))
                ccc = torch.zeros(256, 256, int(cc.shape[2]))
                ddd = torch.zeros(256, 256, int(dd.shape[2]))

                # range(5)
                for p in range(int(self.split / 2 * self.dt)):
                    # 保证每个批次结果一样
                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_a = aa[:, :, p].max()
                    aaa[:, :, p] = self.transform(aa[:, :, p])
                    # 这个是干嘛呢
                    if torch.max(aaa[:, :, p]) > 0:
                        aaa[:, :, p] = scale_a * aaa[:, :, p] / torch.max(aaa[:, :, p])

                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_b = bb[:, :, p].max()
                    bbb[:, :, p] = self.transform(bb[:, :, p])
                    if torch.max(bbb[:, :, p]) > 0:
                        bbb[:, :, p] = scale_b * bbb[:, :, p] / torch.max(bbb[:, :, p])

                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_c = cc[:, :, p].max()
                    ccc[:, :, p] = self.transform(cc[:, :, p])
                    if torch.max(ccc[:, :, p]) > 0:
                        ccc[:, :, p] = scale_c * ccc[:, :, p] / torch.max(ccc[:, :, p])

                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_d = dd[:, :, p].max()
                    ddd[:, :, p] = self.transform(dd[:, :, p])
                    if torch.max(ddd[:, :, p]) > 0:
                        ddd[:, :, p] = scale_d * ddd[:, :, p] / torch.max(ddd[:, :, p])

                    # print('abcd',scale_a,scale_b,scale_c,scale_d)

                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                ee = self.transform(ee)

                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                ff = self.transform(ff)

            # 这都是啥
            if torch.max(aaa) > 0 and torch.max(bbb) > 0 and torch.max(ccc) > 0 and torch.max(ddd) > 0 and torch.max(
                    ee) > 0 and torch.max(ff) > 0:

                return aaa, bbb, ccc, ddd, ee / torch.max(ee), ff / torch.max(ff)
            else:
                pp = torch.zeros(image_resize, image_resize, self.half_split)
                return pp, pp, pp, pp, torch.zeros(1, image_resize, image_resize), torch.zeros(1, image_resize,
                                                                                               image_resize)
        else:
            pp = torch.zeros(image_resize, image_resize, self.half_split)
            return pp, pp, pp, pp, torch.zeros(1, image_resize, image_resize), torch.zeros(1, image_resize,
                                                                                           image_resize)

    def __len__(self):
        return self.length


class Test_loading(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 1
        self.xoff = 45
        self.yoff = 2
        self.split = 10
        self.half_split = int(self.split / 2)

        # testfile:datasets/indoor_flying4/indoor_flying4_datahdf5
        d_set = h5py.File(testfile, 'r')

        # Training input data, label parse
        # ts size is 623
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        # 这四个对应
        # former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off
        # 这个20又是什么道理呢？
        if (index + 20 < self.length) and (index > 20):
            # 256,256,5
            aa = np.zeros((256, 256, self.half_split), dtype=np.uint8)
            bb = np.zeros((256, 256, self.half_split), dtype=np.uint8)
            cc = np.zeros((256, 256, self.half_split), dtype=np.uint8)
            dd = np.zeros((256, 256, self.half_split), dtype=np.uint8)

            im_onoff = np.load(testdir + '/count_data/' + str(int(index + 1)) + '.npy')
            # print("im_onoff",im_onoff.shape)

            # 2, 260 346 10
            # 260-2-2=256 346-45-45=256
            # 也就是说他不是resize的，而是直接剪裁的
            aa[:, :, :] = im_onoff[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:5].astype(float)
            bb[:, :, :] = im_onoff[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:5].astype(float)
            cc[:, :, :] = im_onoff[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 5:10].astype(float)
            dd[:, :, :] = im_onoff[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 5:10].astype(float)

            return aa, bb, cc, dd, self.image_raw_ts[index], self.image_raw_ts[index + self.dt]
        else:
            pp = np.zeros((image_resize, image_resize, self.half_split))
            return pp, pp, pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros(
                (self.image_raw_ts[index].shape))

    def __len__(self):
        return self.length


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args, event_interval, image_resize, sp_threshold
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    # batch_size is 8
    mini_batch_size_v = args.batch_size
    batch_size_v = 4
    sp_threshold = 0.75

    for ww, data in enumerate(train_loader, 0):
        # get the inputs
        # 前四个8*256*256*5  后两个8*1*256*256
        # batch是8，在load时期设置的
        former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off, former_gray, latter_gray = data

        # 两张灰度图之间有事件，才进行操作
        if torch.sum(former_inputs_on + former_inputs_off) > 0:
            # print('aaaaaaaaaaaaaaaaaaaaa')
            # input torch.Size([8, 4, 256, 256, 5])
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

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            input_representation = input_representation.to(device)
            # print('hhh')
            # 执行20次NNforward，
            # 输出是flow1 ~ flow4
            # output tuple(8*2*256*256,8*2*128*128,8*2*64*64,8*2*32*32)
            output = model(input_representation.type(torch.cuda.FloatTensor), image_resize, sp_threshold)
            # print('jjj')

            # Photometric loss.
            photometric_loss = compute_photometric_loss(former_gray[:, 0, :, :], latter_gray[:, 0, :, :],
                                                        torch.sum(input_representation, 4), output,
                                                        weights=args.multiscale_weights)

            # Smoothness loss.
            smoothness_loss = smooth_loss(output)

            # total_loss
            loss = photometric_loss + 10 * smoothness_loss

            # 这块已经看不懂了
            # 先将梯度归零（optimizer.zero_grad()），
            # 然后反向传播计算得到每个参数的梯度值（loss.backward()），
            # 最后通过梯度下降执行一步参数更新（optimizer.step()）
            # compute gradient and do optimization step
            optimizer.zero_grad()
            # 15次backward
            loss.backward()
            optimizer.step()

            # record loss and EPE
            train_writer.add_scalar('train_loss', loss.item(), n_iter)
            losses.update(loss.item(), input_representation.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if mini_batch_size_v * ww % args.print_freq < mini_batch_size_v:
                print('Epoch: [{0}][{1}/{2}]    \t Time {3}\t Data {4}\t Loss {5}'
                      .format(epoch, mini_batch_size_v * ww, mini_batch_size_v * len(train_loader), batch_time,
                              data_time, losses))
            n_iter += 1

    return losses.avg


def validate(test_loader, model, epoch, output_writers):
    global args, image_resize, sp_threshold
    d_label = h5py.File(gt_file, 'r')
    # 390*2*260*346
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    d_label = None

    d_set = h5py.File(testfile, 'r')
    gray_image = d_set['davis']['left']['image_raw']

    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    batch_size_v = 4
    sp_threshold = 0.75

    AEE_sum = 0.
    AEE_sum_sum = 0.
    AEE_sum_gt = 0.
    AEE_sum_sum_gt = 0.
    percent_AEE_sum = 0.
    iters = 0.
    scale = 1

    # 0 is start location
    for i, data in enumerate(test_loader, 0):
        # 这一帧灰度的开始时间和结束时间
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
            # u v is x y direction
            U_gt_all = np.array(gt_temp[:, 0, :, :])
            V_gt_all = np.array(gt_temp[:, 1, :, :])

            # 得到在两个灰度图之间的光流信息
            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all, gt_ts_temp, np.array(st_time),
                                                        np.array(ed_time))
            gt_flow = np.stack((U_gt, V_gt), axis=2)

            #   ----------- Visualization
            if epoch < 0:
                # 1 256 256 5 mask_temp
                mask_temp = former_inputs_on + former_inputs_off + latter_inputs_on + latter_inputs_off
                mask_temp = torch.sum(torch.sum(mask_temp, 0), 2)
                mask_temp_np = np.squeeze(np.array(mask_temp)) > 0

                # 所谓spike_image，就是指在两张灰度图之间，出现事件，那就认为他是spike,就是255
                spike_image = mask_temp
                spike_image[spike_image > 0] = 255
                # spike_image.shape 256*256
                if args.render:
                    cv2.imshow('Spike Image', np.array(spike_image, dtype=np.uint8))

                gray = cv2.resize(gray_image[i], (scale * image_resize, scale * image_resize),
                                  interpolation=cv2.INTER_LINEAR)
                if args.render:
                    cv2.imshow('Gray Image', cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

                out_temp = np.array(output_temp.cpu().detach())
                x_flow = cv2.resize(np.array(out_temp[0, 0, :, :]), (scale * image_resize, scale * image_resize),
                                    interpolation=cv2.INTER_LINEAR)
                y_flow = cv2.resize(np.array(out_temp[0, 1, :, :]), (scale * image_resize, scale * image_resize),
                                    interpolation=cv2.INTER_LINEAR)
                # 炫彩光流图
                flow_rgb = flow_viz_np(x_flow, y_flow)
                if args.render:
                    cv2.imshow('Predicted Flow Output', cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))

                gt_flow_x = cv2.resize(gt_flow[:, :, 0], (scale * image_resize, scale * image_resize),
                                       interpolation=cv2.INTER_LINEAR)
                gt_flow_y = cv2.resize(gt_flow[:, :, 1], (scale * image_resize, scale * image_resize),
                                       interpolation=cv2.INTER_LINEAR)
                # 炫彩gt光流
                gt_flow_large = flow_viz_np(gt_flow_x, gt_flow_y)
                if args.render:
                    cv2.imshow('GT Flow', cv2.cvtColor(gt_flow_large, cv2.COLOR_BGR2RGB))
                # mask_tmp_np就是把所有产生事件的位置置为1了
                masked_x_flow = cv2.resize(np.array(out_temp[0, 0, :, :] * mask_temp_np),
                                           (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                masked_y_flow = cv2.resize(np.array(out_temp[0, 1, :, :] * mask_temp_np),
                                           (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                # 炫彩mask_pred_flow
                flow_rgb_masked = flow_viz_np(masked_x_flow, masked_y_flow)
                if args.render:
                    cv2.imshow('Masked Predicted Flow', cv2.cvtColor(flow_rgb_masked, cv2.COLOR_BGR2RGB))

                gt_flow_cropped = gt_flow[2:-2, 45:-45]
                gt_flow_masked_x = cv2.resize(gt_flow_cropped[:, :, 0] * mask_temp_np,
                                              (scale * image_resize, scale * image_resize),
                                              interpolation=cv2.INTER_LINEAR)
                gt_flow_masked_y = cv2.resize(gt_flow_cropped[:, :, 1] * mask_temp_np,
                                              (scale * image_resize, scale * image_resize),
                                              interpolation=cv2.INTER_LINEAR)
                # 炫彩mask_gt_flow
                gt_masked_flow = flow_viz_np(gt_flow_masked_x, gt_flow_masked_y)
                if args.render:
                    cv2.imshow('GT Masked Flow', cv2.cvtColor(gt_masked_flow, cv2.COLOR_BGR2RGB))

                cv2.waitKey(1)

            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xsize = full_size[1]
            ysize = full_size[0]
            xcrop = image_size[1]
            ycrop = image_size[0]
            xoff = (xsize - xcrop) // 2
            yoff = (ysize - ycrop) // 2

            gt_flow = gt_flow[yoff:-yoff, xoff:-xoff, :]

            AEE, percent_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = flow_error_dense(gt_flow, pred_flow, (
                torch.sum(torch.sum(torch.sum(input_representation, dim=0), dim=0), dim=2)).cpu(), is_car=False)

            AEE_sum = AEE_sum + args.div_flow * AEE
            AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

            AEE_sum_gt = AEE_sum_gt + args.div_flow * AEE_gt
            AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

            percent_AEE_sum += percent_AEE

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i < len(output_writers):  # log first output of first batches
                output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

            iters += 1
    # 这个n_points好像有点问题啊，这样一来不是只计算最后的了吗
    print('-------------------------------------------------------')
    print('Mean AEE: {:.2f}, sum AEE: {:.2f}, Mean AEE_gt: {:.2f}, sum AEE_gt: {:.2f}, mean %AEE: {:.2f}, # pts: {:.2f}'
          .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters,
                  percent_AEE_sum / iters, n_points))
    print('-------------------------------------------------------')
    gt_temp = None

    return AEE_sum / iters


def main():
    global args, best_EPE, image_resize, event_interval, spiking_ts, device, sp_threshold
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
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.savedir, save_path)
    print('=> Everything will be saved to {}'.format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 这一步本质就是创建文件夹，后续会有更多操作
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    # Data loading code
    # torchvison.transforms
    co_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.ToTensor(),
    ])
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
    # args.evaluate = True
    if args.evaluate:
        # 强制之后的内容不进行计算图构建，不追踪梯度
        with torch.no_grad():
            best_EPE = validate(test_loader, model, -1, output_writers)
        return
    # ----------------------- line between train and test --------------------------------------------
    Train_dataset = Train_loading(transform=co_transform)
    train_loader = DataLoader(dataset=Train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)
    # 学习率调整
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.7)

    # range(0,100)
    # for epoch in range(5):
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        # mean_loss update in every epoch
        train_loss = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean loss', train_loss, epoch)

        # Test at every 5 epoch during training
        if (epoch + 1) % args.evaluate_interval == 0:
            # evaluate on validation set
            with torch.no_grad():
                EPE = validate(test_loader, model, epoch, output_writers)
            test_writer.add_scalar('mean EPE', EPE, epoch)

            if best_EPE < 0:
                best_EPE = EPE

            is_best = EPE < best_EPE
            best_EPE = min(EPE, best_EPE)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_EPE': best_EPE,
                'div_flow': args.div_flow
            }, is_best, save_path)


if __name__ == '__main__':
    main()
