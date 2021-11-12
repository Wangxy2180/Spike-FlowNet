from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器


ea = event_accumulator.EventAccumulator("./spikeflownet/eeee_test-indoor4/spike_flownets,adam,100epochs,epochSize800,b8,lr5e-05/test/events.out.tfevents.1630837823.server-Precision")  # 初始化EventAccumulator对象
ea.Reload()  # 这一步是必须的，将事件的内容都导进去
print(ea.scalars.Keys())  # 我们知道tensorboard可以保存Image scalars等对象，我们主要关注scalars
# train画mean_loss, test画mean_EPE
mean_EPE = ea.scalars.Items("mean_EPE")
train_loss = ea.scalars.Items("train_loss")
print("123")
