import torch
import torch.nn as nn
import os
import sys
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info

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

class Unet3D_DM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet3D_DM, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = False),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = False)
        )
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        self.conv1 = downDouble3dConv(64, 128)
        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(128, 256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.bridge = downDouble3dConv(256, 512)
        
        self.output_l2 = nn.Conv3d(256, out_ch, 3, padding=1)
        self.output_l1 = nn.Conv3d(128, out_ch, 3, padding=1)
        self.BNl1 = nn.BatchNorm3d(out_ch)
        self.BNl2 = nn.BatchNorm3d(out_ch)

        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        self.conv4 = upDouble3dConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2)) 
        self.conv5 = upDouble3dConv(384, 128)
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(192, 64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding=1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
    def forward(self, input):
        c0 = self.conv0(input) 
        p1 = self.pool1(c0)
        c1 = self.conv1(p1) 
        p2 = self.pool2(c1)# 64 64 
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.bridge(p3)
        
        up_1 = self.up1(c3)
        merge5 = torch.cat((up_1, c2), dim = 1)
        c4 = self.conv4(merge5)
        output_l2 = self.BNl1(self.output_l2(c4))
        # output_l2 = self.output_l2(c4)
        up_2 = self.up2(c4) 
        merge6 = torch.cat([up_2, c1], dim = 1) #32
        c5 = self.conv5(merge6)
        output_l1 = self.BNl2(self.output_l1(c5))
        # output_l1 = self.output_l1(c5)
        up_3 = self.up3(c5)
        merge7 = torch.cat([up_3, c0], dim = 1) #64
        c6 = self.conv6(merge7)
        
        c7 = self.conv7(c6)
        out = self.BN3d(c7)
        out = c7
        return out, output_l1, output_l2

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet3D_DM(1, 1, scale=4).cuda()
    params = sum(param.numel() for param in model.parameters()) / 1e6
    print(params)
    macs, params = get_model_complexity_info(model, (1, 19, 256, 256), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # assert 1>3
    input = torch.randn(1, 1, 19, 256, 256).to(device)
    print(model(input)[0].shape)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
