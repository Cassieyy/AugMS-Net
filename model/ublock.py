import torch
import torch.nn as nn
from torch import autograd
import os
from PIL import Image
import cv2
from torch.nn import functional as F
import sys
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

class UBlockFirst(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UBlockFirst, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, padding = 1), 
            nn.ReLU(inplace = True),
            nn.Conv3d(8, 16, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )

        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.pool4 = nn.MaxPool3d((1,2,2), (1,2,2))
 
        self.conv1 = downDouble3dConv(16, 32)
        self.conv2 = downDouble3dConv(32, 64)
        self.conv3 = downDouble3dConv(64, 128)
        self.bridge = downDouble3dConv(128, 256)

        self.up1 = nn.ConvTranspose3d(256, 128, (1,2,2), stride = (1,2,2))
        self.conv4 = upDouble3dConv(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, (1,2,2), stride=(1,2,2))
        self.conv5 = upDouble3dConv(128, 64)

        self.up3 = nn.ConvTranspose3d(64, 32, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(64, 32)

        self.up4 = nn.ConvTranspose3d(32, 16, (1,2,2), stride=(1,2,2)) ##
        self.conv7 = upDouble3dConv(32, in_ch)

    def forward(self, x):
        # encoder
        c0 = self.conv0(x) 
        p1 = self.pool1(c0)
        c1 = self.conv1(p1) 
        p2 = self.pool2(c1) 
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.conv3(p3)
        p4 = self.pool4(c3)
        c4 = self.bridge(p4)
        
        # decoder
        up_1 = self.up1(c4)
        merge5 = torch.cat((up_1, c3), dim = 1)
        c5 = self.conv4(merge5)
        up_2 = self.up2(c5) 
        merge6 = torch.cat([up_2, c2], dim = 1) #32
    
        c6 = self.conv5(merge6)
        up_3 = self.up3(c6)
        merge7 = torch.cat([up_3, c1], dim = 1) #64
        c7 = self.conv6(merge7)
        up_4 = self.up4(c7)
        merge8 = torch.cat([up_4, c0], dim = 1) #64
        c8 = self.conv7(merge8)

        return c0, c1, c2, c3, c4, c5, c6, c7, c8


class UBlocks(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UBlocks, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, padding = 1), 
            nn.ReLU(inplace = True),
            nn.Conv3d(8, 16, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )

        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))        
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.pool4 = nn.MaxPool3d((1,2,2), (1,2,2))

        self.conv1 = downDouble3dConv(48, 32)
        self.conv2 = downDouble3dConv(96, 64)
        self.conv3 = downDouble3dConv(192, 128)
        self.bridge = downDouble3dConv(384, 256)

        self.up1 = nn.ConvTranspose3d(256, 128, (1,2,2), stride = (1,2,2))
        self.conv4 = upDouble3dConv(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, (1,2,2), stride=(1,2,2))
        self.conv5 = upDouble3dConv(128, 64)

        self.up3 = nn.ConvTranspose3d(64, 32, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(64, 32)

        self.up4 = nn.ConvTranspose3d(32, 16, (1,2,2), stride=(1,2,2)) ##
        self.conv7 = upDouble3dConv(32, out_ch)

    def forward(self, ic0, ic1, ic2, ic3, ic4, ic5, ic6, ic7, ic8):
        # encoder
        c0 = self.conv0(ic8)
        c0 = c0 + ic0
        p1 = self.pool1(c0)
        l1 = torch.cat((ic7, p1), dim = 1)
        c1 = self.conv1(l1)
        c1 = c1 + ic1 
        p2 = self.pool2(c1)
        l2 = torch.cat((ic6, p2), dim = 1)
        c2 = self.conv2(l2)
        c2 = c2 + ic2
        p3 = self.pool3(c2)
        l3 = torch.cat((ic5, p3), dim = 1)
        c3 = self.conv3(l3)
        c3 = c3 + ic3
        p4 = self.pool4(c3)
        l4 = torch.cat((ic4, p4), dim = 1)

        c4 = self.bridge(l4)
        
        # decoder
        up_1 = self.up1(c4)
        merge5 = torch.cat((up_1, c3), dim = 1)
        c5 = self.conv4(merge5)
        up_2 = self.up2(c5) 
        merge6 = torch.cat([up_2, c2], dim = 1)
    
        c6 = self.conv5(merge6)
        up_3 = self.up3(c6)
        merge7 = torch.cat([up_3, c1], dim = 1)

        c7 = self.conv6(merge7)
        up_4 = self.up4(c7)
        merge8 = torch.cat([up_4, c0], dim = 1)

        c8 = self.conv7(merge8)

        return c0, c1, c2, c3, c4, c5, c6, c7, c8

class CascadeUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CascadeUnet, self).__init__()
        self.u1 = UBlockFirst(in_ch, in_ch)
        self.u2 = UBlocks(in_ch*2, in_ch)
        self.u3 = UBlocks(in_ch*2, in_ch)
        self.u4 = UBlocks(in_ch*2, in_ch)
        self.u5 = UBlocks(in_ch*2, in_ch)
        # self.u6 = UBlocks(in_ch*2, in_ch)
        # self.u7 = UBlocks(in_ch*2, in_ch)

        self.output_l2 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.BNl2 = nn.BatchNorm3d(out_ch)
        self.output_l1 = nn.Conv3d(32, out_ch, 3, padding=1)
        self.BNl1 = nn.BatchNorm3d(out_ch)
        self.output = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.BN = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        c0, c1, c2, c3, c4, c5, c6, c7, c8 = self.u1(x)

        c8 = torch.cat((c8, x), dim = 1)
        c0, c1, c2, c3, c4, c5, c6, c7, c8 = self.u2(c0, c1, c2, c3, c4, c5, c6, c7, c8)

        c8 = torch.cat((c8, x), dim = 1)
        c0, c1, c2, c3, c4, c5, c6, c7, c8 = self.u3(c0, c1, c2, c3, c4, c5, c6, c7, c8)

        c8 = torch.cat((c8, x), dim = 1)
        c0, c1, c2, c3, c4, c5, c6, c7, c8 = self.u4(c0, c1, c2, c3, c4, c5, c6, c7, c8)

        c8 = torch.cat((c8, x), dim = 1)
        c0, c1, c2, c3, c4, c5, c6, c7, c8 = self.u5(c0, c1, c2, c3, c4, c5, c6, c7, c8)

        # c8 = torch.cat((c8, x), dim = 1)
        # c0, c1, c2, c3, c4, c5, c6, c7, c8 = self.u6(c0, c1, c2, c3, c4, c5, c6, c7, c8)

        # c8 = torch.cat((c8, x), dim = 1)
        # c0, c1, c2, c3, c4, c5, c6, c7, c8 = self.u7(c0, c1, c2, c3, c4, c5, c6, c7, c8)

        out2 = self.BNl2(self.output_l2(c6))
        out1 = self.BNl1(self.output_l1(c7))
        out = self.BN(self.output(c8))
        return out, out1, out2

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CascadeUnet(1, 1).cuda()
    model = nn.DataParallel(model,device_ids=[0])
    params = sum(param.numel() for param in model.parameters()) / 1e6
    print(params)
    input = torch.randn(1, 1, 19, 256, 256) # BCDHW 
    input = input.to(device)
    out, outl1, outl2 = model(input) 
    print("output.shape:", out.shape, outl1.shape, outl2.shape) # 4, 1, 8, 256, 256
