1. 为啥平均末端误差一直没有大的变动呢？
2. etet训练100epoch后似乎还有余力



对于indoor_flying4,在原始状态下，最好的是3.492



batch还可以修改





出现了一些问题，时间归一化后是float，但是网络是uint8，于是给他乘255并向上取整

全0训练结果：Mean AEE1.51

![](https://s3.bmp.ovh/imgs/2021/09/bdb4974d2cf92b98.png)



# 验证对比

indoor2 用预训练模型，结果如下,和；和论文中差的远呢![Screenshot from 2021-09-09 16-27-53.png](https://i.loli.net/2021/09/09/VbChLex5f28nvHW.png)

indoor2用使用etet 100epoch训练出来的，好像是好了一些些哈


![Screenshot from 2021-09-13 12-09-32.png](https://i.loli.net/2021/09/13/GpcIQz81s4YlNgy.png)












还有一个很神奇的地方，事件数量一定是10的倍数，这样应该是有问题的啊，应该时间均分好一点吧



