import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal
import cv2
from torchvision import transforms
import torch.nn as nn

# class DenoiseModule(nn.Module):  # 继承 nn.Module
#     def __init__(self, diffraction_radius=200, diffraction_noise_scale=0.1):
#         super(DenoiseModule, self).__init__()  # 初始化父类
#         self.diffraction_radius = diffraction_radius
#         self.diffraction_noise_scale = diffraction_noise_scale
    
#     def remove_diffraction_noise(self, image):
#         """
#         去除衍射噪声
#         """
#         # 假设输入的 image 是一个 Tensor，形状为 [batch_size, channels, height, width]
#         batch_size, channels, height, width = image.shape
        
#         # 获取输入数据的设备（CPU 或 GPU）
#         device = image.device
        
#         # 将图像从 [0, 1] 范围转换到 [0, 255] 范围
#         image_np = (image * 255).cpu().numpy().astype(np.float32)

#         # 对每个图像进行处理
#         denoised_image_np = np.zeros_like(image_np)

#         for i in range(batch_size):
#             # 对单个图像应用傅里叶变换
#             f = np.fft.fft2(image_np[i].transpose(1, 2, 0))  # [height, width, channels]
#             fshift = np.fft.fftshift(f)

#             # 创建低通滤波器（圆形掩膜）
#             crow, ccol = height // 2, width // 2
#             mask = np.zeros((height, width), np.uint8)
#             cv2.circle(mask, (ccol, crow), self.diffraction_radius, 1, -1)

#             # 如果是彩色图像，将掩膜应用到每个通道
#             if channels == 3:
#                 mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

#             # 在频域上应用低通滤波器
#             # fshift_filtered = fshift * (1 - mask)
#             fshift_filtered = fshift * mask
#             f_ishift = np.fft.ifftshift(fshift_filtered)
#             filtered_image = np.fft.ifft2(f_ishift)
#             filtered_image = np.abs(filtered_image)

#             # 保存处理后的图像
#             denoised_image_np[i] = filtered_image.transpose(2, 0, 1)  # 转换回 [channels, height, width]
        
#         # 将去噪图像转换回 tensor 并归一化到 [0, 1] 范围
#         denoised_image = torch.tensor(denoised_image_np).float() / 255.0
        
#         # 将结果移到与输入数据相同的设备
#         return denoised_image.to(device)
class DenoiseModule(torch.nn.Module):
    def __init__(self, diffraction_radius=40, diffraction_noise_scale=0.9, kernel_size=25):
        super(DenoiseModule, self).__init__()
        self.diffraction_radius = diffraction_radius
        self.diffraction_noise_scale = diffraction_noise_scale
        self.kernel_size = kernel_size

    def remove_diffraction_noise(self, image):
        """
        去除衍射噪声
        """
        # 假设输入的 image 是一个 Tensor，形状为 [batch_size, channels, height, width]
        batch_size, channels, height, width = image.shape
        
        # 获取输入数据的设备（CPU 或 GPU）
        device = image.device
        
        # 将图像从 [0, 1] 范围转换到 [0, 255] 范围
        image_np = (image * 255).cpu().numpy().astype(np.float32)

        # 对每个图像进行处理
        denoised_image_np = np.zeros_like(image_np)

        for i in range(batch_size):
            # 对单个图像应用傅里叶变换
            f = np.fft.fft2(image_np[i].transpose(1, 2, 0))  # [height, width, channels]
            fshift = np.fft.fftshift(f)

            # 创建高斯低通滤波器
            crow, ccol = height // 2, width // 2
            mask = np.zeros((height, width), np.float32)
            x = np.linspace(-ccol, ccol, width)
            y = np.linspace(-crow, crow, height)
            X, Y = np.meshgrid(x, y)
            d = np.sqrt(X**2 + Y**2)
            mask = np.exp(-(d**2) / (2 * (self.diffraction_radius**2)))

            # 如果是彩色图像，将掩膜应用到每个通道
            if channels == 3:
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # 在频域上应用低通滤波器
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            filtered_image = np.fft.ifft2(f_ishift)
            filtered_image = np.abs(filtered_image)

            # 保存处理后的图像
            denoised_image_np[i] = filtered_image.transpose(2, 0, 1)  # 转换回 [channels, height, width]
        
        # 将去噪图像转换回 tensor 并归一化到 [0, 1] 范围
        denoised_image = torch.tensor(denoised_image_np).float() / 255.0
        
        # 将结果移到与输入数据相同的设备
        return denoised_image.to(device)
    # def remove_diffuse_noise(self, image):
    #     """
    #     使用双边滤波去噪
    #     :param image: 输入图像 Tensor，形状为 [batch_size, channels, height, width]
    #     """
    #     batch_size, channels, height, width = image.shape
    #     device = image.device
    #     denoised_image = torch.zeros_like(image)

    #     for i in range(batch_size):
    #         img_np = image[i].cpu().numpy().transpose(1, 2, 0)  # 转换为 [height, width, channels]

    #         # 使用双边滤波去噪
    #         img_denoised = cv2.bilateralFilter(img_np, d=9, sigmaColor=25, sigmaSpace=5)

    #         # 转回 float32 并恢复到 [0, 1] 范围
    #         denoised_image[i] = torch.tensor(img_denoised.transpose(2, 0, 1)).float() / 255.0

    #     return denoised_image.to(device)
    def remove_diffuse_noise(self, image):
        """
        使用 Wiener 滤波去噪
        :param image: 输入图像 Tensor，形状为 [batch_size, channels, height, width]
        """
        batch_size, channels, height, width = image.shape
        device = image.device
        denoised_image = torch.zeros_like(image)

        for i in range(batch_size):
            img_np = image[i].cpu().numpy().transpose(1, 2, 0)  # 转换为 [height, width, channels]

            # 对每个通道使用 Wiener 滤波去噪
            for c in range(channels):
                img_np[:, :, c] = scipy.signal.wiener(img_np[:, :, c], (2, 2))  # 可以调整窗口大小

            # 转回 [channels, height, width] 格式并归一化
            denoised_image[i] = torch.tensor(img_np.transpose(2, 0, 1)).float()

        return denoised_image.to(device)

    def forward(self, image):
        """
        调用方法去除衍射噪声和漫反射噪声
        """
        # 去除衍射噪声
        image = self.remove_diffraction_noise(image)
        
        # 去除漫反射噪声
        image = self.remove_diffuse_noise(image)

        return image
