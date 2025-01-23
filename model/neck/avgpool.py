import torch
import torch.nn as nn

class Avgpool(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Args:
            in_features (int): 输入特征的维度（来自前面的特征提取部分）。
            num_classes (int): 类别的数量。
        """
        super(Avgpool, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征，形状为 (batch_size, in_features, H, W)。

        Returns:
            torch.Tensor: 输出类别的概率，形状为 (batch_size, num_classes)。
        """
        x = self.avgpool(x)  # 输出形状为 (batch_size, in_features, 1, 1)
        x = torch.flatten(x, 1)       # 将张量展平为 (batch_size, in_features)
        return x