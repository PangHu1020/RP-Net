import torch.nn as nn
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from model.head.kan import KAN
from model.network.vae  import VAE


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Load the pre-trained ResNet50 model
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
        # self.mlp = mlp 
        # self.kan = KAN(layers_hidden = [3*256*256, 3*256*256])
        # self.unet = unet
        # self.vae = VAE()

        # Extract only the backbone (remove the last fully connected layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # self.backbone.conv1 = FFC(3, 64, kernel_size=7, stride=2, padding=3, bias=False, ratio_gin=0, ratio_gout=0.3)

    def forward(self, x):
        # x = self.mlp(x)
        # x = self.unet(x)
        # x = self.kan(x)    
        # x, m, l = self.vae(x)
        #     
        x = self.backbone(x)  # Extract features using ResNet50's backbone
        return x



class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.network(x).view(x.size(0), 3, 256, 256)

mlp = MLP(196608, 196608)  


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        
        # Contracting path (Encoder)
        self.encoder1 = self.contracting_block(input_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.contracting_block(512, 1024)
        
        # Expanding path (Decoder)
        self.upconv4 = self.expanding_block(1024, 512)
        self.decoder4 = self.contracting_block(1024, 512)
        
        self.upconv3 = self.expanding_block(512, 256)
        self.decoder3 = self.contracting_block(512, 256)
        
        self.upconv2 = self.expanding_block(256, 128)
        self.decoder2 = self.contracting_block(256, 128)
        
        self.upconv1 = self.expanding_block(128, 64)
        self.decoder1 = self.contracting_block(128, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        # Final Convolution
        out = self.final_conv(d1)
        
        return out

    def pool(self, x):
        return nn.MaxPool2d(2, 2)(x)

unet = UNet(3, 3)


