from model.modules.wtconv import WTConv2d
from model.modules.refconv import RefConv
from FTN.model.modules.Frep_Mapping import Frep_Mapping
import torch 
import torch.nn as nn
import torch.nn.functional as F

class FCM(nn.Module):
    def __init__(self, c):
        super(FCM, self).__init__()
        self.conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.refconv = RefConv(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.wtconv = WTConv2d(c, c, kernel_size=3, stride=1, bias=False)
        self.ffc = Frep_Mapping(c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False, ratio_gin=0, ratio_gout=0.5)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.refconv(x)
        x3 = self.wtconv(x)
        x4 = self.ffc(x)
        
        return 


