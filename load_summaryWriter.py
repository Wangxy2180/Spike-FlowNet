from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['uming']
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器

# ea = event_accumulator.EventAccumulator("D:\FishRecognazation\pythonProject\events.out.tfevents.1630837823.server-Precision")
ea = event_accumulator.EventAccumulator(
    "./spikeflownet/11-10-09:42/spike_flownets,adam,10000epochs,epochSize800,b8,lr5e-05/test/events.out.tfevents.1636508570.server-Precision")
# ea = event_accumulator.EventAccumulator('./data/events.out.tfevents.1630837823.server-Precision')
# D:\FishRecognazation\pythonProject\data\test.server-Precision# 初始化EventAccumulator对象
ea.Reload()  # 这一步是必须的，将事件的内容都导进去
print(ea.scalars.Keys())  # 我们知道tensorboard可以保存Image scalars等对象，我们主要关注scalars
# train画mean_loss, test画mean_EPE
# mean_EPE = ea.scalars.Items("mean_loss")
mean_EPE = ea.scalars.Items("mean_EPE")
# print(mean_EPE[0])
# train_loss = ea.scalars.Items("mean_loss")
# print("123")
step = []
value = []
for i in range(len(mean_EPE)):
    data = mean_EPE[i]
    step.append(data.step)
    value.append(data.value)
plt.xlabel("步长", fontsize=15)  # x轴上的名字
plt.ylabel("mean_EPE(像素)", fontsize=15)  # y轴上的名字
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(step, value)
plt.show()
