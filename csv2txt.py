import numpy as np

file_path = r'./datasets/celex_1.csv'
# with open(file_path) as f:
csv = np.loadtxt(file_path, delimiter=',', encoding='UTF-8')

# csv = np.array([[1, 3, 3], [7, 2, 0], [8, 5, 4], [2, 1, 7], [7, 6, 2], [1, 2, 9], [1, 4, 3]])
# csv = [[1, 2, 3], [1, 2, 0], [1, 2, 4], [1, 2, 7], [1, 2, 2], [1, 2, 9], [1, 4, 3]]
# print(csv)
csv = csv.tolist()
csv.sort(key=lambda x: x[2])
# print(csv)

np.savetxt(r'./datasets/celex_1/celex_1.txt', csv, encoding='UTF-8')
