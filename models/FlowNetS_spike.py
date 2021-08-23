import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_
from .util import predict_flow, crop_like, conv_s, conv, deconv

__all__ = ['spike_flownets']


# __all__,当被其他包导入时，只能使用这其中的成员,这个对应的就是那个函数

# pytorch可以自动求导，但是有时不能求导，就需要来这么自定义一下求导方式
class SpikingNN(torch.autograd.Function):
    # 所有的变量在forward中被换为tensor
    def forward(self, input):
        # print('nnfor')
        self.save_for_backward(input)
        # 大于为1，小于为0, 可以认为大于0的为1，否则为0
        # ReLU求导
        return input.gt(1e-5).type(torch.cuda.FloatTensor)

    def backward(self, grad_output):
        print('nnbak')
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 1e-5] = 0
        return grad_input


# 这个就是IF模型
# mem_1: 8 * x * image_resize/2 * image_resize/2
def IF_Neuron(membrane_potential, threshold):
    global threshold_k
    threshold_k = threshold
    # check exceed membrane potential and reset
    # 超出阈值的保持不变，未超出的置0
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)
    # print("2 sum",torch.sum(membrane_potential),torch.sum(ex_membrane))

    # 这是干嘛呢，这样一来不就翻转了吗，就是超出阈值的置0，不超出的则不变
    membrane_potential = membrane_potential - ex_membrane  # hard reset
    # 他甚至会有很多负的
    # generate spike
    # 超出0的置为1，反之置为0，看上去像是ReLU求导
    out = SpikingNN()(ex_membrane)
    # detach(),清除梯度，不进行反向传播
    # 这又是干嘛呢
    out = out.detach() + (1 / threshold) * out - (1 / threshold) * out.detach()
    # membrane_potential: 膜电位超出阈值0.75的置0，其余不变；
    # out: 膜电位超出阈值0.75部分的部分中，超出0的置1，其余为0？这不就是该部分全部置1吗
    return membrane_potential, out


class FlowNetS_spike(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetS_spike, self).__init__()
        self.batchNorm = batchNorm
        self.conv1 = conv_s(self.batchNorm, 4, 64, kernel_size=3, stride=2)
        self.conv2 = conv_s(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3 = conv_s(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4 = conv_s(self.batchNorm, 256, 512, kernel_size=3, stride=2)

        self.conv_r11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        self.deconv3 = deconv(self.batchNorm, 512, 128)
        self.deconv2 = deconv(self.batchNorm, 384 + 2, 64)
        self.deconv1 = deconv(self.batchNorm, 192 + 2, 4)

        self.predict_flow4 = predict_flow(self.batchNorm, 32)
        self.predict_flow3 = predict_flow(self.batchNorm, 32)
        self.predict_flow2 = predict_flow(self.batchNorm, 32)
        self.predict_flow1 = predict_flow(self.batchNorm, 32)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=512, out_channels=32, kernel_size=4, stride=2,
                                                       padding=1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(in_channels=384 + 2, out_channels=32, kernel_size=4, stride=2,
                                                       padding=1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(in_channels=192 + 2, out_channels=32, kernel_size=4, stride=2,
                                                       padding=1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(in_channels=68 + 2, out_channels=32, kernel_size=4, stride=2,
                                                       padding=1, bias=False)

        # 这是干嘛呢,而且这俩看上去一样啊
        for m in self.modules():
            # print('modules',m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, input, image_resize, sp_threshold):
        # print('fff')
        # 这里的主干网络应该来自于EV-flownet和firenet
        # o.75
        threshold = sp_threshold
        # input ([8, 4, 256, 256, 5])
        # image_resize is 256
        # mem_1 8 * x * image_resize/2 * image_resize/2
        mem_1 = torch.zeros(input.size(0), 64, int(image_resize / 2), int(image_resize / 2)).cuda()
        mem_2 = torch.zeros(input.size(0), 128, int(image_resize / 4), int(image_resize / 4)).cuda()
        mem_3 = torch.zeros(input.size(0), 256, int(image_resize / 8), int(image_resize / 8)).cuda()
        mem_4 = torch.zeros(input.size(0), 512, int(image_resize / 16), int(image_resize / 16)).cuda()

        mem_1_total = torch.zeros(input.size(0), 64, int(image_resize / 2), int(image_resize / 2)).cuda()
        mem_2_total = torch.zeros(input.size(0), 128, int(image_resize / 4), int(image_resize / 4)).cuda()
        mem_3_total = torch.zeros(input.size(0), 256, int(image_resize / 8), int(image_resize / 8)).cuda()
        mem_4_total = torch.zeros(input.size(0), 512, int(image_resize / 16), int(image_resize / 16)).cuda()

        # size(4) is 5
        # nnforward20次
        # 就是在这4*5=20
        # 因为一共有五张图啊,
        # 这里应该对应SNN-Block中的四层卷积吧,也就是这个循环对应的是蓝色的编码部分
        for i in range(input.size(4)):
            input11 = input[:, :, :, :, i].cuda()

            # mem都是累加的，只有out_conv不是
            current_1 = self.conv1(input11)
            mem_1 = mem_1 + current_1
            mem_1_total = mem_1_total + current_1
            # 输入分别是膜电位和阈值
            mem_1, out_conv1 = IF_Neuron(mem_1, threshold)
            # print(1)

            current_2 = self.conv2(out_conv1)
            mem_2 = mem_2 + current_2
            mem_2_total = mem_2_total + current_2
            mem_2, out_conv2 = IF_Neuron(mem_2, threshold)
            # print(2)

            current_3 = self.conv3(out_conv2)
            mem_3 = mem_3 + current_3
            mem_3_total = mem_3_total + current_3
            mem_3, out_conv3 = IF_Neuron(mem_3, threshold)
            # print(3)

            current_4 = self.conv4(out_conv3)
            mem_4 = mem_4 + current_4
            mem_4_total = mem_4_total + current_4
            mem_4, out_conv4 = IF_Neuron(mem_4, threshold)
            # print(4)

        # 这三个应该是SNN部分那三个绿线残差
        mem_4_residual = 0
        mem_3_residual = 0
        mem_2_residual = 0

        # 这里的四个就是红色的四个累计输出
        out_conv4 = mem_4_total + mem_4_residual
        out_conv3 = mem_3_total + mem_3_residual
        out_conv2 = mem_2_total + mem_2_residual
        out_conv1 = mem_1_total

        # 这里应该是res层
        out_rconv11 = self.conv_r11(out_conv4)
        out_rconv12 = self.conv_r12(out_rconv11) + out_conv4
        out_rconv21 = self.conv_r21(out_rconv12)
        out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12

        # 这里的concat应该是EV-FlowNet
        # flow1 is 256*256; flow4 is 32*32
        # 所有的flow_up都没有修改尺寸,所有的flow输出前两维都是8*2

        # 第四层,out_rconv22是残差结构的输出
        # upsampled 8*512*16*16->8*32*32*32; perdict:8*32*32*32->8*2*32*32
        flow4 = self.predict_flow4(self.upsampled_flow4_to_3(out_rconv22))
        flow4_up = crop_like(flow4, out_conv3)
        # out_deconv3:8*128*32*32
        out_deconv3 = crop_like(self.deconv3(out_rconv22), out_conv3)

        # 第三层的绿线concat 256+128+2=386
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        # upsampled 32*32->64*64
        flow3 = self.predict_flow3(self.upsampled_flow3_to_2(concat3))
        flow3_up = crop_like(flow3, out_conv2)
        # out_deconv2:8*64*64*64
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        # 第二层的绿线concat 128+64+2=194
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        # upsampled 64*64->128*128
        flow2 = self.predict_flow2(self.upsampled_flow2_to_1(concat2))
        flow2_up = crop_like(flow2, out_conv1)
        # out_deconv1:8*4*128*128
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1)

        # 第一层的绿线concat 64+4+2=70
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        # upsampled 128*128->256*256
        flow1 = self.predict_flow1(self.upsampled_flow1_to_0(concat1))

        if self.training:
            return flow1, flow2, flow3, flow4
        else:
            return flow1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def spike_flownets(data=None):
    # data is network_data
    model = FlowNetS_spike(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
