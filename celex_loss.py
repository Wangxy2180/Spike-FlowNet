import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F

from multiscaleloss import charbonnier_loss, warp


def celex_compute_photometric_loss(prev_images_temp, next_images_temp, event_images, output, weights=None,
                                   loss_area=-1):
    # size is
    # prev_images_tmp: 8*256*256
    # next_images_tmp: 8*256*256
    # event_images:    8*4*256*256
    # output:          tuple(8*2*256*256,8*2*128*128,8*2*64*64,8*2*32*32)

    prev_images = np.array(prev_images_temp)
    next_images = np.array(next_images_temp)

    total_photometric_loss = 0.
    loss_weight_sum = 0.

    # 256 128 64 32 四个尺寸都计算一遍
    for i in range(len(output)):
        flow = output[i]

        m_batch = flow.size(0)  # 8
        height = flow.size(2)  # 256
        width = flow.size(3)  # 256

        prev_images_resize = torch.zeros(m_batch, 1, height, width)
        next_images_resize = torch.zeros(m_batch, 1, height, width)

        for p in range(m_batch):
            # 这里有必要resize吗，两个应该是相同大小的啊。不同！！！，这里的宽高是flow的宽高，不一定是256*256
            prev_images_resize[p, 0, :, :] = torch.from_numpy(
                cv2.resize(prev_images[p, :, :], (height, width), interpolation=cv2.INTER_LINEAR))
            next_images_resize[p, 0, :, :] = torch.from_numpy(
                cv2.resize(next_images[p, :, :], (height, width), interpolation=cv2.INTER_LINEAR))

        # 根据光流，将第二张图反推，得到第一张图，然后和原始的第一张图做比较
        next_images_warped = warp(next_images_resize.cuda(), flow.cuda())
        # 得到灰度值差，如果为了适应celex这里的差值似乎应该做一些修改
        if loss_area == -1:
            error_temp = next_images_warped - prev_images_resize.cuda()
        else:
            # 这里考虑边界部分不进行loss计算
            for row in range(loss_area // 2, height - loss_area // 2):
                for col in range(loss_area // 2, width - loss_area // 2):
                    error_temp[row][col] = np.mean(next_images_warped[row - loss_area // 2:row + loss_area // 2,
                                                  col - loss_area // 2:col + loss_area // 2] -
                                                  prev_images_resize[row - loss_area // 2:row + loss_area // 2,
                                                  col - loss_area // 2:col + loss_area // 2])

        # charbonnier损失
        photometric_loss = charbonnier_loss(error_temp)

        # 对不同的尺寸赋予不同的权重，但是这里都是的weights都是1
        total_photometric_loss += weights[len(weights) - i - 1] * photometric_loss
        loss_weight_sum += 1.
    # 最后再求个平均
    total_photometric_loss = total_photometric_loss / loss_weight_sum

    return total_photometric_loss
