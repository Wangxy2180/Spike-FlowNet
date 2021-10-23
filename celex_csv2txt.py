import numpy as np
import argparse


def config():
    parser = argparse.ArgumentParser(description='celex in pixel csv data to (x,y,t) txt data')
    parser.add_argument('--file_path', type=str, default='celex_data.csv')
    args = parser.parse_args()
    return args


def main():
    args = config()
    file_path = args.file_path
    # 不使用科学计数法
    np.set_printoptions(suppress=True)
    csv = np.loadtxt(file_path, delimiter=',', encoding='UTF-8')

    # csv = np.array([[1, 3, 3], [7, 2, 0], [8, 5, 4], [2, 1, 7], [7, 6, 2], [1, 2, 9], [1, 4, 3]])
    # csv = [[1, 2, 3], [1, 2, 0], [1, 2, 4], [1, 2, 7], [1, 2, 2], [1, 2, 9], [1, 4, 3]]
    # print(csv)
    csv = csv.tolist()
    csv.sort(key=lambda x: x[2])
    # 转来转去，我简直有病
    csv = np.array(csv)
    csv = csv[:, :3]
    csv = csv[:, [1, 0, 2]]
    # print(csv)
    # 同时替换后缀和文件夹名
    save_path = file_path.replace('csv', 'txt')
    np.savetxt(save_path, csv, encoding='UTF-8', fmt='%d')


if __name__ == '__main__':
    main()
