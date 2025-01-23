import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 编码器部分
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 512)
        
        # 解码器部分
        self.decoder1 = self.deconv_block(512, 512)
        self.decoder2 = self.deconv_block(512, 256)
        self.decoder3 = self.deconv_block(256, 128)
        self.decoder4 = self.deconv_block(128, 64)
        self.decoder5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码过程
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # 解码过程
        d1 = self.decoder1(e5)
        d2 = self.decoder2(d1 + e4)  # 跳跃连接
        d3 = self.decoder3(d2 + e3)  # 跳跃连接
        d4 = self.decoder4(d3 + e2)  # 跳跃连接
        out = self.decoder5(d4 + e1)  # 跳跃连接

        return torch.tanh(out)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = self.conv_block(6, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.fc = nn.Linear(512 * 8 * 8, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, y):
        # 拼接光斑图像和生成图像
        x = torch.cat((x, y), 1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)  # 展平
        out = torch.sigmoid(self.fc(x))  # 判别图像是否真实
        return out
