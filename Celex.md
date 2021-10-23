这里记录为了celex所做的一些修改
# Usage
1. 再使用`/encoding/split_coding_etet.py`对数据集进行etet的编码的工作(celex可以使用的个数)。
2. 再使用正常的训练方法进行训练，即可训练出一个etet格式的模型。
3. 先使用`csv2txt.py`将celex数据进行转换成按时间排序的事件流数据
4. 再使用`celex_encode2etet_resize.py`对celex数据进行etet编码，基本和普通的etet一样，就是把输入改成了celex的输入并resize到346\*260(因为这个尺寸的数据与原始代码中的处理是一致的，所以后续就可以较少改动)(先从1280\*800->1038\*780，再从1038\*780->346\*260)
5. 最后使用`validCelexData.py`进行celex数据的可视化工作


`/encoding/split_coding_etet.py`，对事件进行etet编码，其中是按照1+6组合的方式编码的，并将时间统一缩放到0~255

`csv2txt.py` : 将celex输出的csv文件转换为txt文件，并按内部时间戳(微秒)排序

`validCelexData.py` : 对celex文件的运行评估，该文件不包含任何训练的部分，训练使用他原始的etet进行训练
该文件需要完成以下内容：

- [ ] 对于像素进行缩放，从1280\*800缩放为256\*256，暂时还没有缩放思路
  - 暂定是进行步长为3的缩放，先剪切为，可能在这9的范围中出现事件的数量作为他缩放后的值，如果大于阈值设置为 1，小于阈值设为0
- [ ] 画出光流的可视化，最好有箭头的那种，能体现出物体运动的轨迹