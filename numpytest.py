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
    for i in range(51):
        im1 = np.load('./datasets/celex_datasets/encoded_data/' + 'celex_b2t' + '/count_data/' + str(i) + '.npy')
        print(np.sum(im1))

        # im2 = np.load('./datasets/celex_datasets/encoded_data/' + 'b2t' + '/count_data/' + str(i) + '.npy')
        # print(np.sum(im2))


def compare_test():
    a = np.zeros(10)
    b = np.linspace(0, 9, 10)
    for i in range(10):
        print(np.sum(b[0:i]))
        a[i] = np.sum(b[0:i]) > 3
    print(a)


def maohao_test():
    a = np.array([0, 1, 2, 3, 4, 5])
    print(a[2:4])


def main():
    # norm_test()
    load_npy()
    # compare_test()
    # maohao_test()


if __name__ == '__main__':
    main()
