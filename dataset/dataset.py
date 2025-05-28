import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
from utils.transform import  DiffractionNoise_priori, DiffuseNoise_priori
from args_parser import parse_args

args = parse_args()

torch.manual_seed(666)

class LightspotDataset(Dataset):
    def __init__(self, ls_dir, gt_dir, ls_transform=None, gt_transform=None):
        """
        Args:
            ls_dir (string): 光斑图片的主目录。
            gt_dir (string): 原图图片的主目录。
            transform (callable, optional): 可选的变换，应用到图像上。
        """
        self.ls_dir = ls_dir
        self.gt_dir = gt_dir
        self.ls_transform = ls_transform
        self.gt_transform = gt_transform

        # 获取所有类别
        self.classes = os.listdir(self.ls_dir)

        # 读取所有光斑和原图的文件路径
        self.ls_images = []
        self.gt_images = []
        self.labels = []

        for label, cls in enumerate(self.classes):
            ls_cls_dir = os.path.join(self.ls_dir, cls)
            gt_cls_dir = os.path.join(self.gt_dir, cls)

            ls_images_in_class = os.listdir(ls_cls_dir)
            for img_name in ls_images_in_class:
                # 添加光斑图片路径和标签
                ls_img_path = os.path.join(ls_cls_dir, img_name)
                self.ls_images.append(ls_img_path)

                # 将 .png 扩展名转换为 .jpg 以匹配原图
                gt_img_name = img_name.replace('.png', '.jpg')
                gt_img_path = os.path.join(gt_cls_dir, gt_img_name)
                self.gt_images.append(gt_img_path)

                # 标签是类别索引
                self.labels.append(label)

    def __len__(self):
        # 数据集的总长度
        return len(self.ls_images)

    def __getitem__(self, idx):
        # 加载光斑图片和原图图片
        ls_img_path = self.ls_images[idx]
        gt_img_path = self.gt_images[idx]

        ls_image = Image.open(ls_img_path).convert("RGB")
        gt_image = Image.open(gt_img_path).convert("RGB")

        try:
            ls_image = Image.open(ls_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading lightspot image: {ls_img_path}, error: {e}")
            return None

        try:
            gt_image = Image.open(gt_img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading ground truth image: {gt_img_path}, error: {e}")
            return None

        # 标签
        label = self.labels[idx]

        # 如果有变换，应用变换
        if self.ls_transform:
            ls_image = self.ls_transform(ls_image)
        if self.gt_transform:
            gt_image = self.gt_transform(gt_image)

        return ls_image, gt_image, label
    

if args.is_recons:
     ls_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),  # Resize 应该放在 ToTensor 之前
        DiffractionNoise_priori(radius=200, noise_scale=0.1), 
        DiffuseNoise_priori(noise_scale=0.8, kernel_size=25),
        transforms.ToTensor(),                     # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
else:
    ls_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),  # Resize 应该放在 ToTensor 之前
        transforms.ToTensor(),                     # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# ls_transform = transforms.Compose([
#         transforms.Resize((args.size, args.size)),  # Resize 应该放在 ToTensor 之前
#         transforms.ToTensor(),                     # 转换为 Tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
#     ])

gt_transform = transforms.Compose([
    transforms.Resize((args.size, args.size)),  # Resize 应该放在 ToTensor 之前
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

MyDataset = LightspotDataset(ls_dir=args.ls_path, gt_dir=args.gt_path, ls_transform=ls_transform, gt_transform=gt_transform)


# 计算训练、验证、测试集的大小
total_size = len(MyDataset)
train_size = int(0.7 * total_size)  # 训练集占 70%
val_size = total_size - train_size
# val_size = int(0.15 * total_size)   # 验证集占 15%
# test_size = total_size - train_size - val_size  # 剩下的作为测试集，占 15%

# 使用 random_split 划分数据集
# train_dataset, val_dataset, test_dataset = random_split(MyDataset, [train_size, val_size, test_size])
train_dataset, val_dataset = random_split(MyDataset, [train_size, val_size]) 




# labels = MyDataset.labels
# train_indices, val_indices = train_test_split(
#     range(len(labels)),
#     test_size=0.3,  # 你的 val_size 是 30%
#     stratify=labels,  # 保证类别平衡
#     random_state=666  # 保证结果可复现
# )

# 根据索引创建子集
# train_dataset = Subset(MyDataset, train_indices)
# val_dataset = Subset(MyDataset, val_indices)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size,          # 批量大小
    shuffle=True,                       # 是否在每个epoch开始时随机打乱数据
    num_workers=args.num_workers         # 加载数据的子进程数量
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size,          # 批量大小
    shuffle=False,                       # 验证集不需要打乱
    num_workers=args.num_workers         # 子进程数量
)

test_dataloader = val_dataloader
# test_dataloader = DataLoader(
#     test_dataset, 
#     batch_size=args.batch_size,          # 批量大小
#     shuffle=False,                       # 测试集不需要打乱
#     num_workers=args.num_workers         # 子进程数量
# )
# test_dataset = MyDataset
# test_dataloader = DataLoader(
#     test_dataset, 
#     batch_size=args.batch_size,          # 批量大小
#     shuffle=False,                       # 验证集不需要打乱
#     num_workers=args.num_workers         # 子进程数量
# )

