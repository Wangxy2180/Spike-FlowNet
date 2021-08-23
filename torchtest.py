import torch.nn as nn
import torch


def nn_functional_threshold():
    thre = nn.functional.threshold(torch.FloatTensor([1, 2, 3, 4, 5]), 2, 0)
    print(thre)


def main():
    nn_functional_threshold()


if __name__ == '__main__':
    main()
