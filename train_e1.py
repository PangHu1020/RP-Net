import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os

# from model import Linearhead, CosineSimilarityLoss, CenterLoss
from model.head.mlp import MLP
from model.neck.avgpool import Avgpool
from model.backbone.resnet import ResNet
from FTN.model.modules.Frep_Mapping import Frep_Mapping
from dataset.dataset import train_dataloader, val_dataloader
from utils.optimizer import get_optimizer
from utils.metric import Metrics
from utils.scheduler import Schedule
from args_parser import parse_args
from utils.log import setup_logging

args = parse_args()

logger = setup_logging(args.exp, args.name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# denoise = DenoiseModule(diffraction_radius=200, diffraction_noise_scale=0.6, kernel_size=25).to(device)
# denoise = FFC(3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, ratio_gin=0, ratio_gout=0.5).to(device)
E1 = ResNet().to(device)
neck = Avgpool().to(device)
head = MLP(in_features=args.num_him, num_classes=args.num_classes).to(device)

criterion = nn.CrossEntropyLoss().to(device)


optimizer = get_optimizer(
    models=[E1, head],  # Pass a list of the models
    optimizer_name=args.optimizer, 
    learning_rate=args.lr, 
    weight_decay=args.weight_decay, 
    momentum=args.momentum
)

scheduler = Schedule(optimizer=optimizer, warmup_epochs=args.warmup, min_lr=1e-6, factor=args.decay_factor, patience=args.patience, verbose=False)

metric = Metrics(num_classes=args.num_classes)

best_acc = float('-inf')
best_f1 = float('-inf')
best_auc = float('-inf')
best_metric = float('-inf') 


checkpoint_dir = os.path.join(args.checkpoints, args.name)

# 确保保存模型的目录存在
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 如果指定了恢复路径，加载模型和优化器状态
if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    E1.load_state_dict(checkpoint['model_state_dict']['E1'])
    neck.load_state_dict(checkpoint['model_state_dict']['neck'])
    head.load_state_dict(checkpoint['model_state_dict']['head'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
else:
    start_epoch = 0

for epoch in range(start_epoch, args.epochs):
    E1.train()
    head.train()
    running_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for ls_images, gt_images, labels in progress_bar:
        ls_images = ls_images.to(device)
        gt_images = gt_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # feature_cls = neck(E1(gt_images))
        feature_cls = neck(E1((ls_images)))
        outputs = head(feature_cls)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 更新进度条显示的信息
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

    avg_loss = running_loss / len(train_dataloader)

    # 计算指标
    E1.eval()
    head.eval()
    progress_val = tqdm(val_dataloader, desc=f'val')
    metric.reset()
    with torch.no_grad():
        for ls_images, gt_images, labels in progress_val:
            ls_images = ls_images.to(device)
            gt_images = gt_images.to(device)
            labels = labels.to(device)
            # outputs = head(neck(E1(gt_images)))
            outputs = head(neck(E1((ls_images))))
            outputs = F.softmax(outputs, dim=1)
            metric.update(outputs, labels)
        
        acc = metric.accuracy()
        f1 = metric.f1_score()
        auc = metric.auc_score()
    scheduler.step(acc)

    best_metric = acc + f1 + auc  # 计算所有指标之和

    # 保存最佳模型权重
    if best_metric > (best_acc + best_f1 + best_auc):
        best_acc = acc
        best_f1 = f1
        best_auc = auc
        torch.save({
            'model_state_dict': {
                'E1': E1.state_dict(),
                'head': head.state_dict(),
                # 'denoise': denoise.state_dict(),
            },
        }, os.path.join(checkpoint_dir, 'best.pth'))

    # 记录当前学习率和指标
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f'Epoch {epoch+1}/{args.epochs}, Current lr: {current_lr:.2e}, Loss: {avg_loss:.4f}')
    logger.info(f'Best Acc: {best_acc:.2f}, Current Acc: {acc:.2f}, '
                f'Best F1: {best_f1:.2f}, Current F1: {f1:.2f}, '
                f'Best AUC: {best_auc:.2f}, Current AUC: {auc:.2f}')

    # 保存最新模型权重
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': {
        'E1': E1.state_dict(),
        'head': head.state_dict(),
        # 'denoise': denoise.state_dict(),
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_acc,
    }, os.path.join(checkpoint_dir, 'last.pth'))

    
