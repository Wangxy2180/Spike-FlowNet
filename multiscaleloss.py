import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn

"""
Robust Charbonnier loss.
"""


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    # alpha is r in paper; epsilon is eta
    loss = torch.sum(torch.pow(torch.mul(delta, delta) + torch.mul(epsilon, epsilon), alpha))
    return loss


"""
warp an image/tensor (im2) back to im1, according to the optical flow
x: [B, C, H, W] (im2), flo: [B, 2, H, W] flow
"""


# next_img flow
# 按论文里说的，他是用预测光流，将第二张图恢复到第一张图
def warp(x, flo):
    B, C, H, W = x.size()
    # mesh grid
    # xx yy 256*256
    # torch.arange(),左闭右开
    # view:将连续的一维数据拿来做数据填充;repeat 重扩张N次
    # xx Size:(256,)->(1,256)->(256,256)
    # xx是每一行是256(行向量)， yy是每一列是256(列向量)
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    # xx yy 8*1*256*256
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # grid(8*2*256*256)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    # flo中存储的是x，y方向分别走了多远
    # 这样相加，就把他运动后的位置求出来了
    vgrid = grid + flo

    # scale grid to [-1,1]
    # 这又是在干嘛呢，怎么个化思路？见cnblog.com/zi-wang/9950917.html
    # grid_sample要求的，将坐标归一化到[-1,1]其中横坐标对应(0:-1; 1:width-1)
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    # 把灰度值做了双线性采样,用vgrid中的值去x中采样
    # output:(8*1*256*256)原始图片根据光流运动后对应的位置的值;x(8*1*256*256)原始图片;vgrid(8*256*256*2)256*256个坐标对(x,y)
    # 对应的是这个根据(x,y)走完后，在当前图上的取值
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # 理论上来说，mask的最大值只能是1，也就是说:
    # mask中，[0.9999,1]的为1，[0,0.9999)为0
    # 应该是为了去除超出的边界点，确实都是边界点
    # for i in range(256):
    #     for j in range(256):
    #         if(mask[0,0,i,j]<0.9999):
    #             print('ij:',vgrid[0,i,j,0],vgrid[0,i,j,1],'val:',mask[0,0,i,j])
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    # 距离像素点过远的就直接置0
    return output * mask


"""
Multi-scale photometric loss, as defined in equation (3) of the paper.
"""


def compute_photometric_loss(prev_images_temp, next_images_temp, event_images, output, weights=None):
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
        error_temp = next_images_warped - prev_images_resize.cuda()
        # charbonnier损失
        photometric_loss = charbonnier_loss(error_temp)

        # 对不同的尺寸赋予不同的权重，但是这里都是的weights都是1
        total_photometric_loss += weights[len(weights) - i - 1] * photometric_loss
        loss_weight_sum += 1.
    # 最后再求个平均
    total_photometric_loss = total_photometric_loss / loss_weight_sum

    return total_photometric_loss

# BCHW
# pred_map tuple(8*2*256*256,8*2*128*128,8*2*64*64,8*2*32*32)
def smooth_loss(pred_map):
    # pred 8*2*256*256 etc
    def gradient(pred):
        # 1: 从第一个开始，直到最后； :-1，从第0个开始，直到最后一个(不含)
        # D_dy(行少1)是行相减，D_dx(列少1)是列之间相减
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    # 这里的权重是变化的，尺寸从大到小，每次权重减少一半
    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
        weight /= 2.0
    return loss


"""
Calculates per pixel flow error between flow_pred and flow_gt. event_img is used to mask out any pixels without events
"""


def flow_error_dense(flow_gt, flow_pred, event_img, is_car=False):
    max_row = flow_gt.shape[1]
    if is_car == True:
        max_row = 190

    # flow_pred:(256,256,2);event_img(256,256)
    flow_pred = np.array(flow_pred)
    event_img = np.array(event_img)

    # 还是原来的尺寸没变
    event_img_cropped = np.squeeze(event_img)[:max_row, :]
    flow_gt_cropped = flow_gt[:max_row, :]
    flow_pred_cropped = flow_pred[:max_row, :]

    # 出现事件才为1的mask(256,256)
    event_mask = event_img_cropped > 0

    # Only compute error over points that are valid in the GT (not inf or 0).
    # 某点xy均为有效数值,且范数大于0(应该就是x^2+y^2>0的意思，至少有一个不为0)，flow_mask才为1
    # 范数的特殊含义是啥呢
    # flow_mask 256*256,里边的3个也全是256*256
    flow_mask = np.logical_and(np.logical_and(~np.isinf(flow_gt_cropped[:, :, 0]), ~np.isinf(flow_gt_cropped[:, :, 1])),
                               np.linalg.norm(flow_gt_cropped, axis=2) > 0)
    # 光流值存在，且该点也触发了事件 total_mask(256*256)
    total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))
    # 这两个mask，15*2；这个方法真神奇啊,好像是把他x，y各拉成一维的了
    # 当total_mask对应位置为true时这个数据才会被拉成一维
    gt_masked = flow_gt_cropped[total_mask, :]
    pred_masked = flow_pred_cropped[total_mask, :]

    # EE_gt是要干啥呢
    # 这俩尺寸都是(1425*1)
    EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
    EE_gt = np.linalg.norm(gt_masked, axis=-1)
    # gt和pred同事都存在光流的点的数量
    n_points = EE.shape[0]

    # Percentage of points with EE < 3 pixels.
    thresh = 3.
    percent_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

    EE = torch.from_numpy(EE)
    EE_gt = torch.from_numpy(EE_gt)

    if torch.sum(EE) == 0:
        AEE = 0
        AEE_sum_temp = 0

        AEE_gt = 0
        AEE_sum_temp_gt = 0
    else:
        AEE = torch.mean(EE)
        AEE_sum_temp = torch.sum(EE)

        AEE_gt = torch.mean(EE_gt)
        AEE_sum_temp_gt = torch.sum(EE_gt)

    return AEE, percent_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt


"""Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow. x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement."""


def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor
    return


"""The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we need to propagate the ground truth flow over the time between two images.
This function assumes that the ground truth flow is in terms of pixel displacement, not velocity. Pseudo code for this process is as follows:
x_orig = range(cols)      y_orig = range(rows)
x_prop = x_orig           y_prop = y_orig
Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
for all of these flows:
  x_prop = x_prop + gt_flow_x(x_prop, y_prop)
  y_prop = y_prop + gt_flow_y(x_prop, y_prop)
The final flow, then, is x_prop - x-orig, y_prop - y_orig.
Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.
Inputs:
  x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at each timestamp.
  gt_timestamps - timestamp for each flow array.  start_time, end_time - gt flow will be estimated between start_time and end time."""


def estimate_corresponding_gt_flow(x_flow_in, y_flow_in, gt_timestamps, start_time, end_time):
    # 光流gt，390*260*346
    x_flow_in = np.array(x_flow_in, dtype=np.float64)
    y_flow_in = np.array(y_flow_in, dtype=np.float64)
    gt_timestamps = np.array(gt_timestamps, dtype=np.float64)
    start_time = np.array(start_time, dtype=np.float64)
    end_time = np.array(end_time, dtype=np.float64)

    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between gt_iter and gt_iter+1.
    gt_iter = np.searchsorted(gt_timestamps, start_time, side='right') - 1
    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    # 当前时间段内的光流数据
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    # 意思就是，当前时段的光流能够完美覆盖所需的
    if gt_dt > dt:
        # 计算时间上的比例从而计算除光流的比例
        return x_flow * dt / gt_dt, y_flow * dt / gt_dt
    # 这下边几乎不会被用到，indoor_flying4是没有的
    print('tttttt',start_time)

    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0]))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
    total_dt = gt_timestamps[gt_iter + 1] - start_time

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor)
    gt_iter += 1

    while gt_timestamps[gt_iter + 1] < end_time:
        x_flow = np.squeeze(x_flow_in[gt_iter, ...])
        y_flow = np.squeeze(y_flow_in[gt_iter, ...])

        prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask)
        total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

        gt_iter += 1

    final_dt = end_time - gt_timestamps[gt_iter]
    total_dt += final_dt

    final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    scale_factor = final_dt / final_gt_dt

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor)

    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0

    return x_shift, y_shift
