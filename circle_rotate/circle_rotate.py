from PIL import Image
import numpy as np
import cv2

im = Image.open('circle_img.jpg')
#################可修改参数########################
# 刷新率，每秒转几圈，持续时间
rate = 60
cnt_per_sec = 3
duration_s = 10
########################################
# 每次多少度
degree = 360 / (rate / cnt_per_sec)


def main():
    img_list = []
    for i in range(0, int(rate / cnt_per_sec)):
        # 负数，逆时针转
        img_list.append(np.array(im.rotate(-1 * i * degree)))
    img_list = remove_margin(img_list)
    show(img_list)


def remove_margin(img_list_):
    # 01分别对应宽高
    cen_x = int(im.size[0] / 2)
    cen_y = int(im.size[1] / 2)
    radius = 300
    for img in img_list_:
        for h in range(im.height):
            for w in range(im.width):
                if pow(w - cen_x, 2) + pow(h - cen_y, 2) > pow(radius, 2):
                    img[h, w] = 255
    return img_list_


def show(img_list):
    for i in range(duration_s * cnt_per_sec):
        for img in img_list:
            cv2.imshow('circle', img)
            cv2.waitKey(int(round(1 / rate, 4) * 1000))


if __name__ == '__main__':
    main()
