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


def main():
    # nn_functional_threshold()
    # torchrand()
    fourDtest()


if __name__ == '__main__':
    main()
