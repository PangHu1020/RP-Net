
import torch
import torch.nn as nn
import timm
from transformers import AutoModelForImageClassification
from timm import create_model

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载预训练模型并修改最后一层以适应40类分类

def load_model(model_name, num_classes=40):
    # 使用timm加载预训练模型
    model = timm.create_model(model_name, pretrained=True)
    
    # 修改最后的全连接层 (FC Layer)，让其输出 40 个类别
    if 'convnext' in model_name:
        model.head.fc = nn.Linear(model.head.in_features, num_classes)
    elif 'vit' in model_name:
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif 'xception' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'mobilenetv3' in model_name:
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'resnext' in model_name:  # 处理 ResNeXt
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'densenet' in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'efficientnet' in model_name:
        if isinstance(model.classifier, nn.Sequential):
        # 如果 classifier 是一个 Sequential，则修改最后一层
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            # 如果 classifier 是一个 Linear，则直接替换
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'convnextv2' in model_name:
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif 'beit' in model_name:
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif 'inception' in model_name:  # 处理 Inception
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'regnet' in model_name:
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    
    # 将模型迁移到设备
    model = model.to(device)
    return model

# 扩展后的模型列表
model_names = [
    'convnext_tiny', 'vit_base_patch16_224', 'xception', 'mobilenetv3_large_100',
    'resnext50_32x4d', 'densenet121', 'efficientnet_b0', 
    'convnextv2_base', 'beit_base_patch16_224', 'inception_v3', 'regnety_016'
]
