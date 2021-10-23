import numpy as np


def norm_test():
    # 所有数的平方相加再开根号
    a = np.array([[2, 2]])
    print('norm_test:', np.linalg.norm(a))

    a = np.array([[2, 2], [1, 1]])
    print('norm_test:', np.linalg.norm(a))

    a = np.array([[[2, 2], [1, 1]]])
    print('norm_test:', np.linalg.norm(a, axis=-1))

def load_npy():
    im_onoff = np.load('./datasets/indoor_flying1' + '/count_data/' + str(99) + '.npy')
    print(np.sum(im_onoff))

    im_onoff = np.load('./datasets/outdoor_day2' + '/count_data/' + str(99) + '.npy')
    print(np.sum(im_onoff))

def main():
    # norm_test()
    load_npy()


if __name__ == '__main__':
    main()
