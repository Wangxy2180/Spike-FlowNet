import sys
import cv2
import numpy as np
import os
import h5py
import argparse

parser = argparse.ArgumentParser(description='Spike Encoding')
parser.add_argument('--save-dir', type=str, default='../datasets', metavar='PARAMS',
                    help='Main Directory to save all encoding results')
parser.add_argument('--save-env', type=str, default='indoor_flying4', metavar='PARAMS',
                    help='Sub-Directory name to save Environment specific encoding results')
parser.add_argument('--data-path', type=str, default='../datasets/indoor_flying1/indoor_flying1_data.hdf5',
                    metavar='PARAMS', help='HDF5 datafile path to load raw data from')
args = parser.parse_args()

save_path = os.path.join(args.save_dir, args.save_env)
if not os.path.exists(save_path):
    os.makedirs(save_path)

count_dir = os.path.join(save_path, 'count_data')
if not os.path.exists(count_dir):
    os.makedirs(count_dir)


# gray_dir = os.path.join(save_path, 'gray_data')
# if not os.path.exists(gray_dir):
#     os.makedirs(gray_dir)


class Events(object):
    def __init__(self, num_events, width=346, height=260):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)],
                                 shape=(num_events))
        self.width = width
        self.height = height

    def generate_fimage(self, input_event=0,  image_raw_event_inds_temp=0, image_raw_ts_temp=0, dt_time_temp=0):
        # print(image_raw_event_inds_temp.shape, image_raw_ts_temp.shape)
        # 623
        split_interval = image_raw_ts_temp.shape[0]
        # N is 5 *  group is 2
        data_split = 10  # N * (number of event frames from each groups)

        # 2 260 346 10 (polar height weight )
        td_img_c = np.zeros((2, self.height, self.width, data_split), dtype=np.uint8)
        # td_img_c = np.zeros((2, self.height, self.width, data_split), dtype=float)
        #
        # td_img_c_cnt = np.zeros((self.height, self.width, data_split), dtype=np.uint8)
        # td_img_c_time = np.zeros((self.height, self.width, data_split), dtype=np.float)
        # td_img_c = np.stack((td_img_c_cnt, td_img_c_time))

        t_index = 0
        # dt_time_temp is 1; split_interval is 623
        # i:0~622
        for i in range(split_interval - (dt_time_temp - 1)):
            # print(i)
            # 这个if基本不会有人走
            if image_raw_event_inds_temp[i - 1] < 0:
                # print(11111)
                frame_data = input_event[0:image_raw_event_inds_temp[i + (dt_time_temp - 1)], :]
            else:
                # image_raw_event_inds_temp 623张图片有623个时间间隔，存放当前时间间隔累加的事件数量
                # 似乎下边都是从0开始的
                # 第一次循环，这里绝对是错的，i-1是-1啊
                # 简单的说，就是从上一帧的最后一个事件到现在这帧
                # 但是这里时间并不是从第0个开始的，而是从第0个ind对应的事件开始的
                # 这里是有问题的，第二帧事件因该是108-267，但这里是从107开始的
                # 那两个+1是我后来加的，修复了这个bug
                frame_data = input_event[
                             image_raw_event_inds_temp[i - 1] + 1:image_raw_event_inds_temp[i + (dt_time_temp - 1)] + 1,
                             :]

            if frame_data.size > 0:
                td_img_c.fill(0)
                # data_split is 10
                # 遍历十个图像的数据
                for m in range(data_split):
                    # 把这个时间段内的 split in 10 part
                    # vv is size of every part
                    # 第m个图像的事件遍历
                    for vv in range(int(frame_data.shape[0] / data_split)):
                        v = int(frame_data.shape[0] / data_split) * m + vv
                        # 这里是在做polar的判定 做的是累加
                        # if frame_data[v, 3].item() == -1:
                        #     td_img_c[1, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1
                        # elif frame_data[v, 3].item() == 1:
                        #     td_img_c[0, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1
                        if frame_data[v, 3].item() in [1, -1]:
                            td_img_c[0, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] += 1
                            min_t = image_raw_ts_temp[i - 1]
                            max_t = image_raw_ts_temp[i + (dt_time_temp - 1)]
                            t = (frame_data[v, 2].item() - min_t) / (max_t - min_t)
                            td_img_c[1, frame_data[v, 1].astype(int), frame_data[v, 0].astype(int), m] = round(t * 255)

            t_index = t_index + 1

            np.save(os.path.join(count_dir, str(i)), td_img_c)
            # 这个gray就是单纯的弄了个save



# indoor_flying1_data.hdf5
d_set = h5py.File(args.data_path, 'r')

# size is
# raw data(8024748, 4)
# image raw ind <HDF5 dataset "image_raw_event_inds": shape (623,), type "<i8">
# image raw  ts (623,)

# xytp
raw_data = d_set['davis']['left']['events']
# 这个里边是每个灰度图对应的事件序号吗？，最大值8024747 事件数量是8024748
image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
d_set = None

# 第一个ts对应事件107附近，他是怎么做的这个呢

# 这个应该还能取值4吧
dt_time = 1

print('raw data shape', raw_data.shape)
print('image raw ind', image_raw_event_inds)
print('image raw ts', image_raw_ts.shape)
# sys.exit()
# shape[0] is num_events
td = Events(raw_data.shape[0])
# Events
td.generate_fimage(input_event=raw_data, image_raw_event_inds_temp=image_raw_event_inds,
                   image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)
raw_data = None

print('Encoding complete!')
#

