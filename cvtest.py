import numpy as np
import cv2
import os


def onpass():
    data_path = './datasets/celex_datasets/encoded_data/'
    data_name = 'b2t'
    gray_path = os.path.join(data_path, data_name, 'gray_data')

    name_list = [os.path.join(gray_path, str(i) + '.npy') for i in range(0, 27)]

    for name in name_list:
        a = np.load(name)
        a = np.array(a, dtype='uint8')
        cv2.imshow("123", a)
        cv2.waitKey(30)


def drawOptFlowMap(flow, img, step, color, name_idx):
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
            dis_threshold = 20
            if abs(flow[row, col, 0]) < dis_threshold and abs(flow[row, col, 1]) < dis_threshold:
                cv2.line(rgb, start_point, end_point, color)
                cv2.arrowedLine(rgb, start_point, end_point, color)
            cv2.circle(rgb, start_point, 1, color)
    cv2.imshow('celex', rgb)
    cv2.waitKey(3000)
    cv2.imwrite(str(name_idx) + '.jpg', rgb)


def op_test():
    data_name = 'walk'
    idx = 101
    prev = os.path.join('./datasets/celex_datasets/encoded_data/', data_name, 'gray_data', str(idx) + '.jpg')
    curr = os.path.join('./datasets/celex_datasets/encoded_data/', data_name, 'gray_data', str(idx + 1) + '.jpg')
    # print(prev)
    # prev = r'./LKimg/LK1.png'
    # curr = r'./LKimg/LK2.png'

    prev_img = cv2.imread(prev, cv2.IMREAD_GRAYSCALE)
    curr_img = cv2.imread(curr, cv2.IMREAD_GRAYSCALE)
    flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    drawOptFlowMap(flow, prev_img, 8, (0, 255, 0), 111)


if __name__ == '__main__':
    op_test()
