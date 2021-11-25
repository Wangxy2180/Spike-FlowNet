这里记录为了celex所做的一些修改
!!!注意一下td_img_c的xy问题

# Usage

1. 首先使用CeleX-V事件相机的可视化Demo进行数据采集，使用内部时间戳模式。并将bin文件转换为csv文件。

2. 使用`celex_csv2txt.py`，将csv数据进行内部时间戳排序，并转换为txt格式。

   ```
   python celex_csv2txt.py --file_path='path/to/celex_data.csv'
   ```

   > 注意，对于一些较大的csv数据，可能会爆内存，应该是因为python内部排序算法的空间复杂度所导致，可以自己写一个O(1)的排序（我没验证是否可行），或者再windows上用“Data.ollo_version_3.1”进行排序处理，然后用vim删掉多余的行，再使用`celex_csv2txt_nosort.py`即可。

3. 使用`celex_encode2etet_resize.py`对事件数据进行etet编码，并将其resize为346*260，输出count数据和gray数据。

   ```
   python celex_encode2etet_ersize.py --data-env=your_data_name
   ```

4. 使用`celex_generate_gt_flow.py`对测试数据集进行gt光流生成。

   ```
   python celex_generate_gt_flow.py --data-env=your_data_name
   ```

5. 使用`celex_validCelexData.py`进行训练或测试。

   训练：`python celex_validCelexData.py`

   测试：`python celex_validCelexData.py --evaluate --pretrained='checkpoint_path'`





# Usage

`/encoding/celex_encode2etet_resize.py`，对事件进行etet编码，其中是按照1+6组合的方式编码的，并将时间统一缩放到0~255,然后resize到346\*260

> 该部分存在可修改的内容，即对t_img_resize中事件数量的处理策略，是sum？mean？还是max

`csv2txt.py` : 将celex输出的csv文件转换为txt文件，并按内部时间戳(微秒)排序

`validCelexData.py` : 对celex文件的运行评估，该文件不包含任何训练的部分，训练使用他原始的etet进行训练
该文件需要完成以下内容：画出光流的可视化，最好有箭头的那种，能体现出物体运动的轨迹



`celex_loss.py`这里修改了一个loss的计算方法，即`patch_loss`，普通loss是单个相减，我们这个是一个范围的相减计算，因为他毕竟是二值图吗，所以有时单个像素看不出啥

