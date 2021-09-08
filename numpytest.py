import numpy as np


def norm_test():
    # 所有数的平方相加再开根号
    a = np.array([[2, 2]])
    print('norm_test:', np.linalg.norm(a))

    a = np.array([[2, 2], [1, 1]])
    print('norm_test:', np.linalg.norm(a))

    a = np.array([[[2, 2], [1, 1]]])
    print('norm_test:', np.linalg.norm(a, axis=-1))


def main():
    norm_test()


if __name__ == '__main__':
    main()
