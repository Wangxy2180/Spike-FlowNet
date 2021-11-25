import argparse
import numpy as np
import os
import cv2
from vis_utils import flow_viz_np


# 根据灰度图创建gt信息

def drawOptFlowMap(flow, img, step, color, save_path):
    # 灰度转rgb，应该有便捷的方法吧
    # 这里的图也有问题，尺寸不对,img转
    # row_off = 2
    # col_off = 45
    # img = img[row_off:-row_off, col_off:-col_off]
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = img
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img
    for row in range(0, img.shape[0], 8):
        for col in range(0, img.shape[1], 8):
            start_point = (col, row)
            end_point = (int(round(col + flow[row, col, 0])), int(round(row + flow[row, col, 1])))
            # 为了可视化好看。。。
            dis_threshold = 2000
            if abs(flow[row, col, 0]) < dis_threshold and abs(flow[row, col, 1]) < dis_threshold:
                cv2.line(rgb, start_point, end_point, color)
                cv2.arrowedLine(rgb, start_point, end_point, color)
            cv2.circle(rgb, start_point, 1, color)
    cv2.imshow('celex', rgb)
    cv2.waitKey(30)
    cv2.imwrite(str(save_path), rgb)


def config():
    parser = argparse.ArgumentParser(description='celex in pixel csv data to (x,y,t) txt data')
    parser.add_argument('--data-path', type=str, default='./datasets/celex_datasets/encoded_data')
    parser.add_argument('--data-env', type=str, default='walk')
    args = parser.parse_args()
    return args


def main():
    args = config()
    data_folder = os.path.join(args.data_path, args.data_env, 'gray_data')
    arrow_save_folder = os.path.join(args.data_path, args.data_env, 'gt_flow_data')
    color_save_folder = os.path.join(args.data_path, args.data_env, 'gt_flow_color_data')
    if not os.path.exists(arrow_save_folder):
        os.makedirs(arrow_save_folder)

    if not os.path.exists(color_save_folder):
        os.makedirs(color_save_folder)

    curr_img = 0
    prev_img = 0
    cnt = len(os.listdir(data_folder)) // 2
    for i in range(0, cnt):
        curr_img = np.load(data_folder + '/' + str(i) + '.npy')
        if i > 1:
            flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # 保存的序号是当前帧的序号，存的是从上一帧到当前帧的光流
            drawOptFlowMap(flow, prev_img, 8, (0, 255, 0), os.path.join(arrow_save_folder, str(i) + '.jpg'))
            np.save(os.path.join(arrow_save_folder, str(i)), flow)
            rgb_flow_img = flow_viz_np(flow[:, :, 0], flow[:, :, 1])
            cv2.imwrite(os.path.join(color_save_folder, str(i) + '.jpg'),
                        rgb_flow_img)
        prev_img = curr_img


if __name__ == '__main__':
    main()
