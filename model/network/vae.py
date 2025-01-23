import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 改进的编码器（适应256x256图像输入）
        self.encoder = nn.Sequential(
            # 第一层卷积，输入3通道，输出32个通道，尺寸减半
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 第二层卷积，输出64个通道，尺寸减半
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 第三层卷积，输出128个通道，尺寸减半
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 第四层卷积，输出256个通道，尺寸减半
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 第五层卷积，输出512个通道，尺寸减半
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 展平层，计算卷积后输出的尺寸，256x256 -> 8x8
            nn.Flatten(),
            # 全连接层，512x8x8=32768 -> 1024
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU()
        )
        
        # 输出潜在空间的均值和对数方差
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # 改进的解码器（恢复图像尺寸为 256x256x3）
        self.decoder_fc = nn.Linear(latent_dim, 1024)
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            # 解码阶段的全连接层，256x8x8恢复为512x8x8
            nn.Linear(1024, 512 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            # 反卷积，尺寸恢复，逐步增大图像尺寸
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出图像像素值范围在[0, 1]之间
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        x_reconstructed = self.decoder_conv(h)
        return x_reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
