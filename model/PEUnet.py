import torch
import torch.nn as nn
from torch import autograd
import os
from PIL import Image
import cv2
from torch.nn import functional as F
import sys
from SELayer import ProjectExciteLayer
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

class PEUnet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(PEUnet3D, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.pe1 = ProjectExciteLayer(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        
        
        self.conv1 = downDouble3dConv(64, 128)
        self.pe2 = ProjectExciteLayer(128)

        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(128, 256)
        self.pe3 = ProjectExciteLayer(256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.bridge = downDouble3dConv(256, 512)
        self.pe4 = ProjectExciteLayer(512)

        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        self.conv4 = upDouble3dConv(768, 256)
        self.pe5 = ProjectExciteLayer(256)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2))
        
        self.conv5 = upDouble3dConv(384, 128)
        self.pe6 = ProjectExciteLayer(128)
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) 
        self.conv6 = upDouble3dConv(192, 64)
        self.pe7 = ProjectExciteLayer(64)

        self.conv7 = nn.Conv3d(64, out_ch, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
    def forward(self, x): # input 1, 1, 27, 256, 256
        
        c0 = self.conv0(x) 
        pe1 = self.pe1(c0)
        p1 = self.pool1(pe1)
        c1 = self.conv1(p1)
        pe2 = self.pe2(c1)
        p2 = self.pool2(pe2)# 64 64 
    
        c2 = self.conv2(p2)
        pe3 = self.pe3(c2)

        p3 = self.pool3(pe3)
        c3 = self.bridge(p3)  
        pe4 = self.pe4(c3)

        up_1 = self.up1(pe4)
        merge5 = torch.cat((up_1, pe3), dim = 1)
        c4 = self.conv4(merge5)
        pe5 = self.pe5(c4)

        up_2 = self.up2(pe5) 
        merge6 = torch.cat([up_2, pe2], dim=1) #32
        c5 = self.conv5(merge6)
        pe6 = self.pe6(c5)
        up_3 = self.up3(pe6)
        
        merge7 = torch.cat([up_3, pe1], dim=1) #64
        c6 = self.conv6(merge7)
        pe7 = self.pe7(c6)
        c7 = self.conv7(pe7)
        out = self.BN3d(c7)
        return out

if __name__ == "__main__":
    from thop import profile
    from thop import clever_format
    from ptflops import get_model_complexity_info
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PEUnet3D(1, 2).cuda()
    params = sum(param.numel() for param in model.parameters()) / 1e6
    print(params)

    macs, params = get_model_complexity_info(model, (1, 19, 256, 256), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    input = torch.randn(1, 1, 19, 256, 256).to(device)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)