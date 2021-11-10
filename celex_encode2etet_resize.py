import sys
import cv2
import numpy as np
import os
import argparse

count_dir = 0
gray_dir = 0


def config():
    parser = argparse.ArgumentParser(description='Spike Encoding')
    parser.add_argument('--save-dir', type=str, default='./datasets/celex_datasets/encoded_data/', metavar='PARAMS',
                        help='Main Directory to save all encoding results')
    parser.add_argument('--data-env', type=str, default='celex_b2t', metavar='PARAMS',
                        help='Sub-Directory name to save Environment specific encoding results')
    parser.add_argument('--data-path', type=str, default='./datasets/celex_datasets/txt_data/',
                        metavar='PARAMS', help='txt datafile path to load raw data from')
    args_ = parser.parse_args()
    return args_


def get_inc_cnt(input_event, time_interval):
    # 事件增量数据统计
    # 用来做事件的分割，每个间隔time_interval(us)记录一次当前的事件量
    # 如[0,100,200,300]这样，其中每个数字代表从起始0时刻算起，一共产生了多少事件
    # 第i帧事件再序列中的下标是从event_inc_cnt[i]到event_inc_cnt[i+1]
    # 一定要注意，下边的起始是有点问题的，idx的值有问题，他是25000us之后的一个的下标，不过也不重要
    event_inc_cnt = [0]
    st_time = input_event[0][2]
    for idx, k in enumerate(input_event):
        if k[2] >= st_time + time_interval:
            event_inc_cnt.append(idx)
            st_time = k[2]
    return event_inc_cnt


class Events(object):
    def __init__(self, events_cnt, width=1280, height=800):
        self.width = width
        self.height = height
        # 目标宽高
        self.dst_height = 260
        self.dst_width = 346
        # 裁剪的边缘宽度 (1280-1038)/2; (800-780)/2
        self.x_off = 121
        self.y_off = 10

    def generate_fimage(self, input_event, event_cnt_idx, dt_time_temp=1):
        # 切割除的图片的个数，这里要减1，因为他多了一个0的下标
        split_cnt = len(event_cnt_idx) - 1
        # N is 5 *  group is 2
        data_split = 10  # N * (number of event frames from each groups)

        # 对25ms内的事件进行编码
        # 2 260 346 10 (polar, height, weight,),2代表的是事件量和时间各一个维度
        td_img_1280 = np.zeros((2, self.height, self.width, data_split), dtype=np.uint8)
        # td_img_346 = np.zeros((2, self.dst_height, self.dst_width, data_split), dtype=np.uint8)

        for i in range(split_cnt - (dt_time_temp - 1)):
            if i % 50 == 0:
                print(args.data_env, ": {}/{}".format(i, split_cnt))
            # 这个if基本不会有人走，现在似乎更没有存在的的必要了
            # if image_raw_event_inds_temp[i - 1] < 0:
            if event_cnt_idx[i - 1] < 0:
                frame_data = input_event[0:event_cnt_idx[i + (dt_time_temp - 1)], :]
            else:
                # 把当前帧对应的事件间隔内的事件收集起来
                frame_data = input_event[event_cnt_idx[i]:event_cnt_idx[i + dt_time_temp], :]
            if frame_data.size > 0:
                td_img_1280.fill(0)
                # td_img_346.fill(0)
                # data_split is 10 遍历并存储十个图像的数据
                for m in range(data_split):
                    # 把这个时间段内的 split in 10 part; vv is size of every part
                    for vv in range(int(frame_data.shape[0] / data_split)):
                        v = int(frame_data.shape[0] / data_split) * m + vv
                        # #############################################注意这里要做上下翻转
                        # e 事件累加 原始数据是xy，目标数据下标height width
                        td_img_1280[0, 800 - frame_data[v, 1].astype(int) - 1, frame_data[v, 0].astype(int), m] += 1
                        # t 找最大最小时间，归一化
                        min_t = input_event[event_cnt_idx[i]][2]
                        max_t = input_event[event_cnt_idx[i + dt_time_temp]][2]
                        t = (frame_data[v, 2].item() - min_t) / (max_t - min_t)
                        td_img_1280[1, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] = round(t * 255)
                        # ###########################################1038*780 end
                        # ##########下边是一次性的完成编码工作。***不保证代码没问题***
                        # if self.y_off < frame_data[v, 1] < self.height - self.y_off and \
                        #         self.x_off < frame_data[v, 0] < self.width - self.x_off:
                        #     td_img_c_346[0, (frame_data[v, 1].astype(int) - 10) // 3, (
                        #                 frame_data[v, 0].astype(int) - 121) // 3, m] += 1
                        #     # t 找最大最小时间，归一化
                        #     min_t = input_event[event_cnt_idx[i]][2]
                        #     max_t = input_event[event_cnt_idx[i + dt_time_temp]][2]
                        #     t = (frame_data[v, 2].item() - min_t) / (max_t - min_t)
                        #     td_img_c_346[1, (frame_data[v, 1].astype(int) - 10) // 3, (
                        #                 frame_data[v, 0].astype(int) - 121) // 3, m] = round(t * 255)
                        #     # 下边是灰度图的构造
                        #     gray_346[
                        #         (frame_data[v, 1].astype(int) - 10) // 3, (frame_data[v, 0].astype(int) - 121) // 3] = 1
                        # ##########一次性编码 end
            # 切割 三倍的尺寸 先弄成1038*780，再缩小,
            td_img_1038 = td_img_1280[:, self.y_off:-self.y_off, self.x_off:-self.x_off, :]
            # 这里做一个resize，这直接resize会不会更快一些,现在这样实在是太慢了太慢了
            td_img_346 = np.zeros((2, self.dst_height, self.dst_width, data_split), dtype=np.uint8)
            for idx in range(data_split):
                for row in range(self.dst_height):
                    for col in range(self.dst_width):
                        # 做一个sum去噪，否则几乎所的地方都会有1的事件量(噪声),!!这个数调不好啊，只能设小一点1了
                        # 可调这个值，大于等于几，事件数量大于这个值才进行计算，如果不符合那就是默认的0了
                        if np.sum(td_img_1038[0, row * 3: row * 3 + 3, col * 3:col * 3 + 3, :]) >= 2:
                            # if True:
                            # 可修改为max sum mean，似乎max是比较好的选择，平均至少每个点出现一个事件，才是有效的
                            td_img_346[0, row, col, idx] = np.max(
                                td_img_1038[0, row * 3: row * 3 + 3, col * 3:col * 3 + 3, idx])
                            td_img_346[1, row, col, idx] = np.max(
                                td_img_1038[1, row * 3: row * 3 + 3, col * 3:col * 3 + 3, idx])
            np.save(os.path.join(count_dir, str(i)), td_img_346)

            # #############灰度图制作 start####################
            # 4是试出来的(3,25ms,max);(0.2,25ms,mean),不需要经常更新，需要的时候uncomment就行
            gray_346 = np.zeros((self.dst_height, self.dst_width), dtype=np.uint8)
            for row in range(self.dst_height):
                for col in range(self.dst_width):
                    gray_346[row, col] = 255 if np.sum(td_img_346[0, row, col, :]) > 2 else 0
            np.save(os.path.join(gray_dir, str(i)), gray_346)
            cv2.imwrite(os.path.join(gray_dir, str(i) + '.jpg'), gray_346)
            # #############灰度图制作 end####################

            cv2.imshow(args.data_env, gray_346)
            cv2.waitKey(1)


def main(args__):
    # 首先在这里读取文件，包括像素点和时间(微秒)
    # lines = []
    events = []
    data_path = os.path.join(args__.data_path, args__.data_env + '.txt')
    with open(data_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        x = line.split()
        # (x,y,t)(col,row,t)
        events.append([eval(x[0]), eval(x[1]), eval(x[2])])

    # 对应论文中的步长 这个应该还能取值4吧
    dt_time = 1
    # 根据时间做分割，单位是微秒
    event_cnt_idx = get_inc_cnt(events, 25000)

    td = Events(len(events))
    events = np.array(events)
    # 开始编码
    td.generate_fimage(input_event=events, event_cnt_idx=event_cnt_idx, dt_time_temp=dt_time)

    print('Encoding complete!')


if __name__ == '__main__':
    args = config()

    save_path = os.path.join(args.save_dir, args.data_env)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count_dir = os.path.join(save_path, 'count_data')
    if not os.path.exists(count_dir):
        os.makedirs(count_dir)

    gray_dir = os.path.join(save_path, 'gray_data')
    if not os.path.exists(gray_dir):
        os.makedirs(gray_dir)

    main(args)
