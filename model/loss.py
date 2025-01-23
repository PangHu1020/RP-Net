import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        """
        初始化余弦相似度损失类，只计算均值损失。
        """
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x, y):
        """
        计算输入x和y之间的余弦相似度损失,并返回均值。
        :param x: 预训练模型的特征，形状为 (batch_size, feature_dim)
        :param y: 未训练模型的特征，形状为 (batch_size, feature_dim)
        :return: 余弦相似度损失的均值
        """
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(x, y, dim=1)
        
        # 计算损失 (1 - cosine_sim)
        loss = 1 - cosine_sim

        # 返回损失的均值
        return loss.mean()
    

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        # 定义可训练参数 Lambda 加权损失
        self.lambda_param = nn.Parameter(torch.tensor(1.0))  # 初始值设为1.0

    def forward(self, loss_cls, loss_alig):
        """
        计算综合损失：loss = loss_cls + Lambda * loss_alig
        :param loss_cls: 分类损失
        :param loss_alig: 对齐损失
        :return: 综合损失
        """
        # 综合损失计算，Lambda 是可训练的参数
        total_loss = loss_cls + self.lambda_param * loss_alig
        return total_loss    
    
    
class FFTLoss(nn.Module):
    def __init__(self, high_freq_weight=1.0):
        super(FFTLoss, self).__init__()
        self.high_freq_weight = high_freq_weight

    def forward(self, pred, target):
        """
        计算频域损失。基于每个通道的频率成分来计算损失。
        Args:
            pred (Tensor): 预测的超分辨率图像，形状为 (B, 3, H, W)
            target (Tensor): 真实的高分辨率图像，形状为 (B, 3, H, W)
        Returns:
            loss (Tensor): 频域损失值
        """
        # 确保输入是RGB图像（三通道）
        assert pred.shape[1] == 3 and target.shape[1] == 3, "Input must be RGB images."

        # 计算每个通道的频域损失
        loss = 0.0
        for i in range(3):  # 对每个通道进行傅里叶变换
            pred_channel = pred[:, i, :, :].unsqueeze(1)  # 获取预测图像的当前通道
            target_channel = target[:, i, :, :].unsqueeze(1)  # 获取目标图像的当前通道

            # 转换到频域
            pred_freq = torch.fft.fft2(pred_channel)
            target_freq = torch.fft.fft2(target_channel)

            # 计算幅度
            pred_magnitude = torch.abs(pred_freq)
            target_magnitude = torch.abs(target_freq)

            # 计算该通道的频域MSE损失
            channel_loss = F.mse_loss(pred_magnitude, target_magnitude)

            # 如果需要加权高频部分，可以在此应用
            if self.high_freq_weight > 0:
                # 这里我们简单地乘上权重，如果需要更复杂的加权策略，可以在这里调整
                channel_loss *= self.high_freq_weight

            # 累加各通道的损失
            loss += channel_loss

        return loss

class PSFLoss(torch.nn.Module):
    def __init__(self, psf_kernel, weight=1.0):
        """
        Optical Loss based on PSF simulation.

        Args:
            psf_kernel (Tensor): Point Spread Function (PSF) kernel, shape (C, H, W)
            weight (float): 权重，用于控制损失的强度
        """
        super(PSFLoss, self).__init__()
        self.psf_kernel = psf_kernel
        self.weight = weight

    def forward(self, output, target):
        """
        计算基于光学机理的损失。
        
        Args:
            output (Tensor): 生成的超分辨率图像，形状为 (N, C, H, W)
            target (Tensor): 真实的高分辨率图像，形状为 (N, C, H, W)
        
        Returns:
            loss (Tensor): 光学损失
        """
        # 使用 PSF 对目标图像进行模糊处理
        blurred_target = F.conv2d(target, self.psf_kernel, padding=self.psf_kernel.shape[-1] // 2)

        # 计算生成图像和经过 PSF 模糊后的真实图像之间的差异
        loss = F.mse_loss(output, blurred_target)

        return self.weight * loss
    


class OTFLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        """
        OTF Loss based on Optical Transfer Function (OTF).
        Supports multi-channel (e.g., RGB) images.

        Args:
            weight (float): 权重，用于控制损失的强度。
        """
        super(OTFLoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        """
        计算基于OTF的损失，支持3通道（RGB）图像。
        
        Args:
            output (Tensor): 生成的超分辨率图像，形状为 (N, C, H, W)，C为通道数（3通道RGB图像）
            target (Tensor): 真实的高分辨率图像，形状为 (N, C, H, W)
        
        Returns:
            loss (Tensor): OTF 损失
        """
        # 初始化损失值
        total_loss = 0.0

        # 对于每个通道分别计算损失
        for c in range(output.shape[1]):  # output.shape[1] 是通道数 C
            output_channel = output[:, c, :, :]
            target_channel = target[:, c, :, :]

            # 计算每个通道的频域表示
            output_fft = torch.fft.fft2(output_channel)
            target_fft = torch.fft.fft2(target_channel)

            # 计算频域表示的幅度谱
            output_mag = torch.abs(output_fft)
            target_mag = torch.abs(target_fft)

            # 计算频域的相位谱
            output_phase = torch.angle(output_fft)
            target_phase = torch.angle(target_fft)

            # 计算幅度谱损失
            mag_loss = F.mse_loss(output_mag, target_mag)

            # 计算相位谱损失
            phase_loss = F.mse_loss(output_phase, target_phase)

            # 总损失：幅度损失 + 相位损失
            channel_loss = mag_loss + phase_loss

            # 汇总到总损失
            total_loss += channel_loss

        # 返回加权的总损失
        return self.weight * total_loss
    

class DiffractionLoss(torch.nn.Module):
    def __init__(self, wavelength=550e-9, distance=1.0, pixel_size=1e-6, weight=1.0, max_size=256):
        """
        Diffraction Loss based on Fresnel diffraction model.
        
        Args:
            wavelength (float): 光的波长（默认为550nm，即绿色光）。
            distance (float): 衍射的传播距离（单位：米）。
            pixel_size (float): 图像像素大小（单位：米）。
            weight (float): 损失的权重。
            max_size (int): 图像的最大尺寸，用于计算衍射传递函数（DTF）时使用的傅里叶变换大小。
        """
        super(DiffractionLoss, self).__init__()
        self.wavelength = wavelength
        self.distance = distance
        self.pixel_size = pixel_size
        self.weight = weight
        self.max_size = max_size

    def create_diffraction_transfer_function(self, h, w):
        """
        创建衍射传递函数 (DTF)，基于 Fresnel 衍射模型。

        Args:
            h (int): 图像的高度。
            w (int): 图像的宽度。
        
        Returns:
            diffraction_transfer_function (Tensor): 衍射传递函数，形状为 (h, w)。
        """
        # 计算频率坐标
        fx = torch.fft.fftfreq(w, d=self.pixel_size)
        fy = torch.fft.fftfreq(h, d=self.pixel_size)
        fx, fy = torch.meshgrid(fx, fy)
        
        # 计算频域的波数
        k = 2 * np.pi / self.wavelength  # 波数

        # Fresnel衍射公式，计算衍射传递函数
        # 这里是简化的 Fresnel 衍射公式
        DTF = torch.exp(1j * np.pi * self.wavelength * self.distance * (fx**2 + fy**2))
        
        return DTF

    def forward(self, output, target):
        """
        计算衍射损失。

        Args:
            output (Tensor): 生成的超分辨率图像，形状为 (N, C, H, W)
            target (Tensor): 真实的高分辨率图像，形状为 (N, C, H, W)

        Returns:
            loss (Tensor): 衍射损失
        """
        # 获取图像的大小
        N, C, H, W = output.shape
        
        # 创建衍射传递函数（DTF）
        DTF = self.create_diffraction_transfer_function(H, W).to(output.device)
        
        # 初始化损失值
        total_loss = 0.0

        for c in range(C):  # 对于每个通道（RGB）
            output_channel = output[:, c, :, :]
            target_channel = target[:, c, :, :]

            # 计算频域表示（傅里叶变换）
            output_fft = torch.fft.fft2(output_channel)
            target_fft = torch.fft.fft2(target_channel)

            # 应用衍射传递函数（DTF）
            output_diffraction = output_fft * DTF
            target_diffraction = target_fft * DTF

            # 计算频域差异
            output_mag = torch.abs(output_diffraction)
            target_mag = torch.abs(target_diffraction)

            # 计算幅度损失
            mag_loss = F.mse_loss(output_mag, target_mag)

            # 计算相位损失
            output_phase = torch.angle(output_diffraction)
            target_phase = torch.angle(target_diffraction)
            phase_loss = F.mse_loss(output_phase, target_phase)

            # 总损失：幅度损失 + 相位损失
            channel_loss = mag_loss + phase_loss

            # 汇总到总损失
            total_loss += channel_loss

        # 返回加权的总损失
        return self.weight * total_loss