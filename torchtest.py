import numpy as np
import torch.nn as nn
import torch


def nn_functional_threshold():
    thre = nn.functional.threshold(torch.FloatTensor([1, 2, 3, 4, 5]), 2, 0)
    print(thre)


def torchrand():
    torch.manual_seed(2)
    print(torch.rand(2))

    torch.manual_seed(1)
    print(torch.rand(2))

    torch.manual_seed(2)
    print(torch.rand(2))

    torch.manual_seed(1)
    print(torch.rand(2))


def fourDtest():
    input_rep = torch.zeros(4, 2, 2, 5)
    for i in range(4):
        v = i
        if i == 0:
            tmp = np.array([[[1, 1], [1, 1]],
                            [[2, 2], [2, 2]],
                            [[3, 3], [3, 3]],
                            [[4, 4], [4, 4]],
                            [[5, 5], [5, 5]]])
            input_rep[i, :, :, :] = torch.transpose(torch.from_numpy(tmp), 0, 2)
        if i == 1:
            tmp = np.array([[[-1, -1], [-1, -1]],
                            [[-2, -2], [-2, -2]],
                            [[-3, -3], [-3, -3]],
                            [[-4, -4], [-4, -4]],
                            [[-5, -5], [-5, -5]],
                            ])
            # input_rep[i, :, :, :] = torch.from_numpy(tmp)
            input_rep[i, :, :, :] = torch.transpose(torch.from_numpy(tmp), 0, 2)

        if i == 2:
            tmp = np.array([[[6, 6], [6, 6]],
                            [[7, 7], [7, 7]],
                            [[8, 8], [8, 8]],
                            [[9, 9], [9, 9]],
                            [[10, 10], [10, 10]]
                            ])
            # input_rep[i, :, :, :] = torch.from_numpy(tmp)
            input_rep[i, :, :, :] = torch.transpose(torch.from_numpy(tmp), 0, 2)

        if i == 3:
            tmp = np.array([[[-6, -6], [-6, -6]],
                            [[-7, -7], [-7, -7]],
                            [[-8, -8], [-8, -8]],
                            [[-9, -9], [-9, -9]],
                            [[-10, -10], [-10, -10]]
                            ])
            # input_rep[i, :, :, :] = torch.from_numpy(tmp)
            input_rep[i, :, :, :] = torch.transpose(torch.from_numpy(tmp), 0, 2)
    print(input_rep.shape)


def grid_sample():
    # 1*1*3*3
    img_in = np.array([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    img_in = torch.from_numpy(img_in)
    # 1*1*3*2
    grid_np = np.array([[[[-1, -1], [0.5, 0.5], [0.75, 0.75]],
                         [[-1, -1], [0, 0], [1, 1]],
                        [[-0.5, -0.5], [-0.75, -0.75], [1, 1]]]])
    # print(grid_np.shape)
    grid = torch.from_numpy(grid_np)
    # 本质上就是双线性采样，用grid中的值去img_in中采样
    sampled = torch.nn.functional.grid_sample(img_in.to(torch.float), grid.to(torch.float))
    print(sampled)
    print('img  shape:', img_in.shape)
    print('grid shape:', grid.shape)
    print('samp shape:', sampled.shape)


def torch_arange():
    print('range :', torch.range(1, 5))
    print('arange:', torch.arange(1, 5))


def main():
    # nn_functional_threshold()
    # torchrand()
    # fourDtest()
    # torch_arange()
    grid_sample()


if __name__ == '__main__':
    main()
