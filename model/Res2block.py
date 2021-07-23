import torch
import torch.nn as nn
import sys

class Res2block(nn.Module):
    def __init__(self, in_ch, scale = 4):
        super(Res2block, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_ch)

        self.nums = scale -1 if scale != 1 else 1
        convs = []
        bns = []
        for _ in range(self.nums):
          convs.append(nn.Conv3d(in_ch // scale, in_ch // scale, kernel_size=3, stride = 1, padding=1, bias=False))
          bns.append(nn.BatchNorm3d(in_ch // scale))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns) 

        self.conv3 = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(in_ch)
        self.in_ch = in_ch
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  
        spx = torch.split(out, self.in_ch // self.scale, 1) # len(spx) = scale

        for i in range(self.nums): # 0
          if i==0:
            sp = spx[i] 
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]),1)
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu(out)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Res2block(16, scale=16).to(device)
    params = sum(param.numel() for param in model.parameters()) / 1e6
    print(params)
    input = torch.randn(1, 16, 16, 256, 256).to(device) # BCDHW 
    out = model(input) 
    print("input.shape:", input.shape, "output.shape:", out.shape) # 4, 1, 8, 256, 256