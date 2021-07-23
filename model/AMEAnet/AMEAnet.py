import torch
import torch.nn as nn
from torch import autograd
import os
from PIL import Image
import cv2
from torch.nn import functional as F
import sys
sys.path.append("/home/lpy/paper/experiment/")
from model.Res2block import Res2block
from model.SELayer import SElayer
import numpy as np

class downDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)
 
class upDouble3dConv(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1):
        super(upDouble3dConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding = padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding = padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

class AMEAnet(nn.Module):
    def __init__(self, in_ch, out_ch, deepvision = False):
        super(AMEAnet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.deepvision = deepvision
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        self.conv_other_input1 = nn.Conv3d(1, 16, kernel_size = 3, padding = 1)
        self.conv_other_input2 = nn.Conv3d(1, 16, kernel_size = 3, padding = 1)
        self.conv_other_input3 = nn.Conv3d(1, 16, kernel_size = 3, padding = 1)
        self.conv1 = downDouble3dConv(80, 128)
        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(144, 256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.spp_convc1 = nn.Conv3d(128, 16, kernel_size = 3, dilation = 1, padding=1)
        self.spp_convc2 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 3, padding= 3)
        self.spp_up_c1 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        self.spp_up_c2 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 4, 4), stride=(1, 4, 4))
        self.weight_cat_all3 = SElayer(96)
        self.spp_convc22 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 1, padding=1)
        self.spp_up_c22 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        self.weight_cat_all2 = SElayer(144)

        self.res_out = Res2NetBlock(256)
        self.refine = nn.Conv3d(272, 256, kernel_size = 1) # 只为改变通道数
        self.bridge = downDouble3dConv(256, 512)
        self.output_l3 = nn.Conv3d(768, 1, 3, padding = 1)
        self.BN3d_l3 = nn.BatchNorm3d(out_ch)

        self.up1 = nn.ConvTranspose3d(768, 768, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(1024, 256)
        self.output_l2 = nn.Conv3d(256, 1, 3, padding=1)
        self.BN3d_l2 = nn.BatchNorm3d(out_ch)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2))
        
        self.conv5 = upDouble3dConv(400, 128)
        self.output_l1 = nn.Conv3d(128, 1, 3, padding=1)
        self.BN3d_l1 = nn.BatchNorm3d(out_ch)
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(224, 64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
        
    def forward(self, input):
        # 之前是参数共享 现在3个input用了三个卷积核
        input1 = self.conv_other_input1(F.interpolate(input, size=(input.shape[2], input.shape[3] // 2, input.shape[4] // 2)))
        # print(input1.shape)
        input2 = self.conv_other_input2(F.interpolate(input, size=(input.shape[2], input.shape[3] // 4, input.shape[4] // 4)))
        input3 = self.conv_other_input3(F.interpolate(input, size=(input.shape[2], input.shape[3] // 8, input.shape[4] // 8)))
        # print(input1.shape, input2.shape, input3.shape)
        # assert 1>3
        c0 = self.conv0(input) 
        p1 = self.pool1(c0)
        cat_input1 = torch.cat((p1, input1), dim=1)
        c1 = self.conv1(cat_input1) 

        p2 = self.pool2(c1)# 64 64 
        cat_input2 = torch.cat((p2, input2), dim=1)
        c2 = self.conv2(cat_input2)

        spp_convc1 = self.spp_convc1(c1)
        spp_convc2 = self.spp_convc2(c2)

        spp_up_c1 = self.spp_up_c1(spp_convc1)
        spp_up_c2 = self.spp_up_c2(spp_convc2)

        cat_all3 = torch.cat((spp_up_c1, spp_up_c2, c0), dim = 1)
        weight_cat_all3 = self.weight_cat_all3(cat_all3)
 
        spp_convc22 = self.spp_convc22(c2)
        spp_up_c22 = self.spp_up_c22(spp_convc22) 
        cat_all2 = torch.cat((spp_up_c22, c1), dim = 1)
        weight_cat_all2 = self.weight_cat_all2(cat_all2)
        p3 = self.pool3(c2)
        cat_input3 = self.refine(torch.cat((p3, input3), dim=1))
        res_out = self.res_out(cat_input3)
        c3 = self.bridge(cat_input3)
        c3 = torch.cat([c3, res_out], dim = 1) # 768
        output3 = self.BN3d_l3(self.output_l3(c3)) 
    
        up_1 = self.up1(c3)
        merge5 = torch.cat([up_1, c2], dim = 1)
        # print(merge5.shape)
        # assert 1>3
        c4 = self.conv4(merge5)
        output2 = self.BN3d_l2(self.output_l2(c4))
        up_2 = self.up2(c4) 
        merge6 = torch.cat([up_2, weight_cat_all2], dim = 1) #32
        c5 = self.conv5(merge6)
        output1 = self.BN3d_l1(self.output_l1(c5))
        up_3 = self.up3(c5)
        merge7 = torch.cat([up_3, weight_cat_all3], dim = 1) #64
        c6 = self.conv6(merge7)
        # assert 1>3
        c7 = self.conv7(c6)
        out = self.BN3d(c7)
        if self.deepvision:
            return out, output1, output2, output3
        else:
            return out

class MyselfUnet3d_res2block(nn.Module):
    def __init__(self, in_ch, out_ch, deepvision = False):
        super(MyselfUnet3d_res2block, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.deepvision = deepvision
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        self.conv_other_input = nn.Conv3d(1, 16, kernel_size = 3, padding = 1)
        self.conv1 = downDouble3dConv(80, 128)
        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(144, 256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.res2block = Res2NetBlock(272)
        self.res_se = SElayer(272)

        self.spp_convc1 = nn.Conv3d(128, 16, kernel_size = 3, dilation = 1, padding=1)
        self.spp_convc2 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 3, padding= 3)

        self.spp_up_c1 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        self.spp_up_c2 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 4, 4), stride=(1, 4, 4))

        self.weight_cat_all3 = SElayer(96)

        self.spp_convc22 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 1, padding=1)
        self.spp_up_c22 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        self.weight_cat_all2 = SElayer(144)

        self.bridge = downDouble3dConv(272, 512)
        self.output_l3 = nn.Conv3d(784, 1, 3, padding = 1)
        self.up1 = nn.ConvTranspose3d(784, 784, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(784, 256)
        self.output_l2 = nn.Conv3d(256, 1, 3, padding=1)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2))
        
        self.conv5 = upDouble3dConv(400, 128)
        self.output_l1 = nn.Conv3d(128, 1, 3, padding=1)
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(224, 64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
        
    def forward(self, input):
        input1 = self.conv_other_input(F.interpolate(input, size=(input.shape[2], input.shape[3] // 2, input.shape[4] // 2)))
        # print(input1.shape)
        input2 = self.conv_other_input(F.interpolate(input, size=(input.shape[2], input.shape[3] // 4, input.shape[4] // 4)))
        input3 = self.conv_other_input(F.interpolate(input, size=(input.shape[2], input.shape[3] // 8, input.shape[4] // 8)))
        # print(input1.shape, input2.shape, input3.shape)
        # assert 1>3
        c0 = self.conv0(input) 
        p1 = self.pool1(c0)
        cat_input1 = torch.cat((p1, input1), dim=1)
        c1 = self.conv1(cat_input1) 
        p2 = self.pool2(c1)# 64 64 
        cat_input2 = torch.cat((p2, input2), dim=1)
        c2 = self.conv2(cat_input2)
        p3 = self.pool3(c2)
        cat_input3 = torch.cat((p3, input3), dim=1)
        
        res_out = self.res2block(cat_input3)
        res_out = self.res_se(res_out)
        c3 = self.bridge(cat_input3)
        c3 = torch.cat([c3, res_out], dim = 1)
        output3 = nn.Sigmoid()(self.output_l3(c3)) # 
    
        spp_convc1 = self.spp_convc1(c1)
        spp_convc2 = self.spp_convc2(c2)
        
        spp_up_c1 = self.spp_up_c1(spp_convc1)
        spp_up_c2 = self.spp_up_c2(spp_convc2)
        
        cat_all3 = torch.cat((spp_up_c1, spp_up_c2, c0), dim = 1)
        weight_cat_all3 = self.weight_cat_all3(cat_all3)

        spp_convc22 = self.spp_convc22(c2)
        spp_up_c22 = self.spp_up_c22(spp_convc22)
        
        cat_all2 = torch.cat((spp_up_c22, c1), dim = 1)
        weight_cat_all2 = self.weight_cat_all2(cat_all2)
        
        up_1 = self.up1(c3)
        c4 = self.conv4(up_1)
        output2 = nn.Sigmoid()(self.output_l2(c4))
   
        up_2 = self.up2(c4) 
        merge6 = torch.cat([up_2, weight_cat_all2], dim = 1) #32
        c5 = self.conv5(merge6)
        output1 = nn.Sigmoid()(self.output_l1(c5))
        up_3 = self.up3(c5)
        merge7 = torch.cat([up_3, weight_cat_all3], dim = 1) #64
        c6 = self.conv6(merge7)
        # assert 1>3
        c7 = self.conv7(c6)
        out = self.BN3d(c7)
        if self.deepvision:
            return out, output1, output2, output3
        else:
            return out

class MyselfUnet3d(nn.Module):
    def __init__(self, in_ch, out_ch, attention = False):
        super(MyselfUnet3d, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.attention = attention
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        
        self.conv1 = downDouble3dConv(64, 128)
        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(128, 256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.spp_convc1 = nn.Conv3d(128, 16, kernel_size = 3, dilation = 1, padding=1)
        self.spp_convc2 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 3, padding= 3)

        self.spp_up_c1 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        self.spp_up_c2 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 4, 4), stride=(1, 4, 4))

        self.weight_cat_all3 = SElayer(96)
        self.refine3 = nn.Conv3d(96, 96, 3, padding = 1)

        self.spp_convc22 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 1, padding=1)
        
        self.spp_up_c22 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        
        self.weight_cat_all2 = SElayer(144)
        self.refine2 = nn.Conv3d(144, 144, 3, padding = 1)

        self.spp_convp33 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 1, padding=1)
        self.spp_up_p33 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))

        self.bridge = downDouble3dConv(256, 512)
        self.output_l3 = nn.Conv3d(512, 1, 3, padding = 1)
        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2)) 
        
        self.conv5 = upDouble3dConv(400, 128)
        
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(224, 64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
        
    def forward(self, input):
        c0 = self.conv0(input) 
        p1 = self.pool1(c0)
        c1 = self.conv1(p1) 
        p2 = self.pool2(c1)# 64 64 
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.bridge(p3) 

        spp_convc1 = self.spp_convc1(c1)
        spp_convc2 = self.spp_convc2(c2)
        spp_up_c1 = self.spp_up_c1(spp_convc1)
        spp_up_c2 = self.spp_up_c2(spp_convc2)
        
        cat_all3 = torch.cat((spp_up_c1, spp_up_c2, c0), dim = 1)
        if self.attention:
            weight_cat_all3 = self.weight_cat_all3(cat_all3)
        else:
            weight_cat_all3 = self.refine3(cat_all3)
        spp_convc22 = self.spp_convc22(c2)
        
        spp_up_c22 = self.spp_up_c22(spp_convc22)
        cat_all2 = torch.cat((spp_up_c22, c1), dim = 1)
        if self.attention:
            weight_cat_all2 = self.weight_cat_all(cat_all2)
        else:
            weight_cat_all2 = self.refine2(cat_all2)
        
        up_1 = self.up1(c3)
        merge5 = torch.cat((up_1, c2), dim = 1)
        c4 = self.conv4(merge5)
        up_2 = self.up2(c4) 
        
        merge6 = torch.cat([up_2, weight_cat_all2], dim = 1) #32
    
        c5 = self.conv5(merge6)
        up_3 = self.up3(c5)
        merge7 = torch.cat([up_3, weight_cat_all3], dim = 1) #64
        c6 = self.conv6(merge7)
        
        c7 = self.conv7(c6)
        out = self.BN3d(c7)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMEAnet(1, 1, deepvision=True)
    model = nn.DataParallel(model,device_ids=[0])
    input = torch.randn(1, 1, 19, 256, 256) # BCDHW 
    input = input.to(device)
    out, out1, out2, out3 = model(input) 
    print("output.shape:", out.shape, out1.shape, out2.shape, out3.shape) # 4, 1, 8, 256, 256
