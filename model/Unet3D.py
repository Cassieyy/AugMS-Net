import torch
import torch.nn as nn
import sys
sys.path.append("/home/lpy/experiment/model")
from SELayer import SpatialSELayer3D, ChannelSELayer3D, ChannelSpatialSELayer3D, ProjectExciteLayer, SElayer
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

class Unet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet3D, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        self.conv1 = downDouble3dConv(64, 128)
        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(128, 256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.bridge = downDouble3dConv(256, 512)
        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2))
        
        self.conv5 = upDouble3dConv(384, 128)
        
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(192, 64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
    def forward(self, x): # input 1, 1, 27, 256, 256
        # assert 1>3
        c0 = self.conv0(x) 
        # print("c0.shape:", c0.shape) # 1, 64, 27, 256, 256
        # assert 1>3
        p1 = self.pool1(c0)
        # print("p1.shape:", p1.shape) # 1, 64, 27, 128, 128
        # assert 1>3
        c1 = self.conv1(p1)
        # print("c1.shape:", c1.shape) # 1, 128, 27, 128, 128
        # assert 1>3    
        p2 = self.pool2(c1)# 64 64
        # print("p2.shape:", p2.shape) # 1, 128, 27, 64, 64
        # assert 1>3   
        c2 = self.conv2(p2)
        # print("c2.shape:", c2.shape) # 1, 256, 27, 64, 64
        # assert 1>3  
        p3 = self.pool3(c2)
        # print("p3.shape", p3.shape) # 1, 256, 27, 32, 32
        # assert 1>3

        c3 = self.bridge(p3)  
        # print("c3.shape:", c3.shape) # 1, 512, 27, 32, 32
        # assert 1>3 

        up_1 = self.up1(c3)
        # print("up_1.shape:", up_1.shape) # 1, 512, 27, 64, 64
        # assert 1>3
        merge5 = torch.cat((up_1, c2), dim = 1)
        # print("merge5.shape:", merge5.shape) # 1, 768,27,64,64
        # assert 1>3
        c4 = self.conv4(merge5)
        # print("c4.shape", c4.shape) # 8 256 6 64 64
        # assert 1>3
        up_2 = self.up2(c4) #######注意啊！！！上采样不是pool实现！！！
        # print("up_2.shape", up_2.shape) # 8 256 1 32 32
        # assert 1>3
        merge6 = torch.cat([up_2, c1], dim=1) #32
        # print("merge6.shape:", merge6.shape) #12 128 128
    
        c5 = self.conv5(merge6)
        # print("c5.shape:", c5.shape) # 2, 128, 12, 128, 128
        # assert 1>3
        up_3 = self.up3(c5)
        # print("up_3.shape:", up_3.shape, c0.shape)

        merge7 = torch.cat([up_3, c0], dim=1) #64
        # print("merge7.shape:", merge7.shape)
        # assert 1>3

        c6 = self.conv6(merge7)
        # print("c6.shape:", c6.shape)
        # assert 1>3

        c7 = self.conv7(c6)
        # print("c7.shape:", c7.shape)
        out = self.BN3d(c7)
        return out

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
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(192, 64)
        self.pe7 = ProjectExciteLayer(64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
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
        # print("up_1.shape:", up_1.shape) # 1, 512, 27, 64, 64
        # assert 1>3
        merge5 = torch.cat((up_1, pe3), dim = 1)
        c4 = self.conv4(merge5)
        pe5 = self.pe5(c4)

        up_2 = self.up2(pe5) #######注意啊！！！上采样不是pool实现！！！
        merge6 = torch.cat([up_2, pe2], dim=1) #32
        c5 = self.conv5(merge6)
        pe6 = self.pe6(c5)
        up_3 = self.up3(pe6)
        
        merge7 = torch.cat([up_3, pe1], dim=1) #64
        c6 = self.conv6(merge7)
        pe7 = self.pe7(c6)
        c7 = self.conv7(pe7)
        # print("c7.shape:", c7.shape)
        out = self.BN3d(c7)
        return out


class scSEUnet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(scSEUnet3D, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.sc1 = ChannelSpatialSELayer3D(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        
        
        self.conv1 = downDouble3dConv(64, 128)
        self.sc2 = ChannelSpatialSELayer3D(128)

        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(128, 256)
        self.sc3 = ChannelSpatialSELayer3D(256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.bridge = downDouble3dConv(256, 512)
        self.sc4 = ChannelSpatialSELayer3D(512)
        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(768, 256)
        self.sc5 = ChannelSpatialSELayer3D(256)

        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2))
        self.conv5 = upDouble3dConv(384, 128)
        self.sc6 = ChannelSpatialSELayer3D(128)

        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(192, 64)
        self.sc7 = ChannelSpatialSELayer3D(64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
    def forward(self, x): # input 1, 1, 27, 256, 256
        
        c0 = self.conv0(x) 
        sc1 = self.sc1(c0)
        
        p1 = self.pool1(sc1)
        c1 = self.conv1(p1)
        sc2 = self.sc2(c1)  
        
        p2 = self.pool2(sc2)# 64 64   
        c2 = self.conv2(p2)
        sc3 = self.sc3(c2)
        p3 = self.pool3(sc3)
        c3 = self.bridge(p3)  
        sc4 = self.sc4(c3) 
       

        up_1 = self.up1(sc4)
        merge5 = torch.cat((up_1, sc3), dim = 1)
        c4 = self.conv4(merge5)
        sc5 = self.sc5(c4)
        up_2 = self.up2(sc5) 
        merge6 = torch.cat([up_2, sc2], dim=1) #32
        c5 = self.conv5(merge6)
        sc6 = self.sc6(c5)
        up_3 = self.up3(sc6)

        merge7 = torch.cat([up_3, sc1], dim=1) #64
        c6 = self.conv6(merge7)
        sc7 = self.sc7(c6)
        # print("c6.shape:", c6.shape)
        # assert 1>3

        sc7 = self.conv7(sc7)
        # print("c7.shape:", c7.shape)
        out = self.BN3d(sc7)
        return out

class SeUnet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SeUnet3D, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )

        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2)) # (kernel_size, stride)
        self.se1 = SElayer(64)
        
        self.conv1 = downDouble3dConv(64, 128)
        self.se2 = SElayer(128)

        self.pool2 = nn.MaxPool3d((1,2,2), (1,2,2))
        self.conv2 = downDouble3dConv(128, 256)
        self.se3 = SElayer(256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.bridge = downDouble3dConv(256, 512)
        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2))
        
        self.conv5 = upDouble3dConv(384, 128 )
        
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(192, 64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 
    def forward(self, x): # input 1, 1, 27, 256, 256
        
        c0 = self.conv0(x) 
        # print("c0.shape:", c0.shape) # 1, 64, 27, 256, 256
        # assert 1>3
        p1 = self.pool1(c0)
        # print("p1.shape:", p1.shape) # 1, 64, 27, 128, 128
        # assert 1>3
        se1 = self.se1(p1)
        # print(se1.shape)
        # assert 1>3
        c1 = self.conv1(se1)
        # print("c1.shape:", c1.shape) # 1, 128, 27, 128, 128
        # assert 1>3    
        p2 = self.pool2(c1)# 64 64
        # print("p2.shape:", p2.shape) # 1, 128, 27, 64, 64
        # assert 1>3   
        se2 = self.se2(p2)
        # print(se2.shape)
        # assert 1>3
        c2 = self.conv2(se2)
        # print("c2.shape:", c2.shape) # 1, 256, 27, 64, 64
        # assert 1>3  
        p3 = self.pool3(c2)
        # print("p3.shape", p3.shape) # 1, 256, 27, 32, 32
        # assert 1>3
        se3 = self.se3(p3)
        # print(se3.shape)
        # assert 1>3
        c3 = self.bridge(se3)  
        # print("c3.shape:", c3.shape) # 1, 512, 27, 32, 32
        # assert 1>3 

        up_1 = self.up1(c3)
        # print("up_1.shape:", up_1.shape) # 1, 512, 27, 64, 64
        # assert 1>3
        merge5 = torch.cat((up_1, c2), dim = 1)
        # print("merge5.shape:", merge5.shape) # 1, 768,27,64,64
        # assert 1>3
        c4 = self.conv4(merge5)
        # print("c4.shape", c4.shape) # 8 256 6 64 64
        # assert 1>3
        up_2 = self.up2(c4) #######注意啊！！！上采样不是pool实现！！！
        # print("up_2.shape", up_2.shape) # 8 256 1 32 32
        # assert 1>3
        merge6 = torch.cat([up_2, c1], dim=1) #32
        # print("merge6.shape:", merge6.shape) #12 128 128
    
        c5 = self.conv5(merge6)
        # print("c5.shape:", c5.shape) # 2, 128, 12, 128, 128
        # assert 1>3
        up_3 = self.up3(c5)
        # print("up_3.shape:", up_3.shape, c0.shape)

        merge7 = torch.cat([up_3, c0], dim=1) #64
        # print("merge7.shape:", merge7.shape)
        # assert 1>3

        c6 = self.conv6(merge7)
        # print("c6.shape:", c6.shape)
        # assert 1>3

        c7 = self.conv7(c6)
        # print("c7.shape:", c7.shape)
        out = self.BN3d(c7)
        return out

class Munet3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Munet3d, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding = 1),    
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, padding = 1),  
            nn.ReLU(inplace = True)
        )
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.detach1 = nn.ConvTranspose3d(64, 32, (1, 2, 2), stride = (1, 2, 2))
        self.detachconv1 = downDouble3dConv(32, 32)
        self.predict1 = nn.Conv3d(32, out_ch, 3, padding = 1)

        self.conv1 = downDouble3dConv(64, 128)
        
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.detach2 = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride = (1, 2, 2))
        self.detachconv2 = downDouble3dConv(64, 64)
        self.detach22 = nn.ConvTranspose3d(64, 32, (1, 2, 2), stride = (1, 2, 2))
        self.detachconv22 = downDouble3dConv(32, 32)
        self.predict2 = nn.Conv3d(32, out_ch, 3, padding = 1)


        self.conv2 = downDouble3dConv(128, 256)
        self.pool3 = nn.MaxPool3d((1,2,2), (1,2,2))
        
        self.bridge = downDouble3dConv(256, 512)
        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(768, 256)
        self.up2 = nn.ConvTranspose3d(256, 256, (1,2,2), stride=(1,2,2))
        
        self.conv5 = upDouble3dConv(384, 128 )
        
        self.up3 = nn.ConvTranspose3d(128, 128, (1,2,2), stride=(1,2,2)) ##
        self.conv6 = upDouble3dConv(192, 64)

        self.conv7 = nn.Conv3d(64, 1, 3, padding = 1)
        self.BN3d = nn.BatchNorm3d(out_ch)
 

    def forward(self, input):
        c0 = self.conv0(input)
        p1 = self.pool1(c0)
        detach1 = self.detach1(p1.detach())
        detachconv1 = self.detachconv1(detach1)
        predict1 = self.predict1(detachconv1)
        # print(predict1.shape)
        # assert 1>3

        c1 = self.conv1(p1)
        p2 = self.pool2(c1)
        detach2 = self.detach2(p2.detach())
        detachconv2 = self.detachconv2(detach2)
        detach22 = self.detach22(detachconv2)
        detachconv22 = self.detachconv22(detach22)
        predict2 = self.predict2(detachconv22)
        # print(predict2.shape)
        # assert 1>3

        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.bridge(p3)
        up_1 = self.up1(c3)

        merge5 = torch.cat((up_1, c2), dim = 1)
        c4 = self.conv4(merge5)
        up_2 = self.up2(c4) #######注意啊！！！上采样不是pool实现！！！
        merge6 = torch.cat([up_2, c1], dim=1) #32
        c5 = self.conv5(merge6)
        up_3 = self.up3(c5)

        merge7 = torch.cat([up_3, c0], dim=1) #64
        
        c6 = self.conv6(merge7)
        c7 = self.conv7(c6)
        out = self.BN3d(c7 + predict1 + predict2)
        return out
'''
    without res2block but have been fixed
'''
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
        # self.spp_convp1 = nn.Conv3d(64, 16, 1)
        self.spp_convc2 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 3, padding= 3)

        self.spp_up_c1 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        self.spp_up_c2 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 4, 4), stride=(1, 4, 4))
        # self.spp_up_p3 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 8, 8), stride=(1, 8, 8))

        self.weight_cat_all3 = SElayer(96)
        self.refine3 = nn.Conv3d(96, 96, 3, padding = 1)

        self.spp_convc22 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 1, padding=1)
        # self.spp_convc32 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 3, padding=3)
        
        self.spp_up_c22 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))
        # self.spp_up_p32 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 4, 4), stride=(1, 4, 4))
        
        self.weight_cat_all2 = SElayer(144)
        self.refine2 = nn.Conv3d(144, 144, 3, padding = 1)

        self.spp_convp33 = nn.Conv3d(256, 16, kernel_size = 3, dilation = 1, padding=1)
        self.spp_up_p33 = nn.ConvTranspose3d(16, 16, kernel_size = (1, 2, 2), stride=(1, 2, 2))

        self.bridge = downDouble3dConv(256, 512)
        # 输出level3的预测结果
        self.output_l3 = nn.Conv3d(512, 1, 3, padding = 1)
        self.up1 = nn.ConvTranspose3d(512, 512, (1,2,2), stride = (1,2,2))
        
        self.conv4 = upDouble3dConv(768, 256)
        # self.output_l2 = nn.Conv3d(256)
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

        ##  将p1 p2 p3 按spp结构处理
        # 1. 先分别卷积 得到所有的outch = 16
        spp_convc1 = self.spp_convc1(c1)
        spp_convc2 = self.spp_convc2(c2)
        # print(spp_convc1.shape, spp_convc2.shape)
        # assert 1>3
        # 2. 上采样到HW相同 全部为256*256
        spp_up_c1 = self.spp_up_c1(spp_convc1)
        # print(spp_up_c1.shape)
        spp_up_c2 = self.spp_up_c2(spp_convc2)
        # print(spp_up_c2.shape)
        # assert 1>3
        
        # 3.把他们cat起来
        cat_all3 = torch.cat((spp_up_c1, spp_up_c2, c0), dim = 1)
        # print(cat_all3.shape) # 96
        # assert 1>3
        if self.attention:
            weight_cat_all3 = self.weight_cat_all3(cat_all3)
        else:
            weight_cat_all3 = self.refine3(cat_all3)
        # print(weight_cat_all3.shape)

        spp_convc22 = self.spp_convc22(c2)
        # spp_convp32 = self.spp_convp32(p3)
        
        spp_up_c22 = self.spp_up_c22(spp_convc22)
        cat_all2 = torch.cat((spp_up_c22, c1), dim = 1)
        # print(cat_all2.shape)
        # assert 1>3
        if self.attention:
            weight_cat_all2 = self.weight_cat_all(cat_all2)
        else:
            weight_cat_all2 = self.refine2(cat_all2)
        # print(weight_cat_all2.shape)
        # assert 1>3
        
        up_1 = self.up1(c3)
        # assert 1>3
        merge5 = torch.cat((up_1, c2), dim = 1)
        # print(merge5.shape) # 768
        c4 = self.conv4(merge5)
        # print(c4.shape)
        # assert 1>3
        up_2 = self.up2(c4) 
        # print(up_2.shape)
        
        merge6 = torch.cat([up_2, weight_cat_all2], dim = 1) #32
        # print(merge6.shape) # 400
    
        c5 = self.conv5(merge6)
        up_3 = self.up3(c5)
        # assert 1>3
        merge7 = torch.cat([up_3, weight_cat_all3], dim = 1) #64
        c6 = self.conv6(merge7)
        
        c7 = self.conv7(c6)
        out = self.BN3d(c7)
        return out


# spp 考虑将所有的上采样都变成inception结构
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet3D(1, 1).cuda()
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