import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model Parameters")
    
    # 数据路径参数
    parser.add_argument('--ls_path', type=str, default='./data/lightspot', help='LightSpot data path')
    parser.add_argument('--gt_path', type=str, default='./data/original', help='Original image data path')

    # 网络设置参数
    parser.add_argument('--num_classes', type=int, default=40, help='Number of classification categories')
    parser.add_argument('--num_him', type=int, default=2048, help='Number of hidden layer channels')
    parser.add_argument('--Lambda', type=float, default=0.5, help='Center loss weight')
    parser.add_argument('--size', type=int, default=128, help='Image resize size')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading threads')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Name of the optimizer to use,Supported optimizers: "sgd", ''adam'', ''adamw''')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum values')
    parser.add_argument('--warmup', type=int, default=3, help='Learning rate warm-up number')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='Learning rate decay coefficient')
    parser.add_argument('--patience', type=int, default=3, help='Loss tolerance times')

    # 文件保存路径
    parser.add_argument('--name', type=str, default='experiment', help='Save file name')
    parser.add_argument('--simA', type=str, default='./simA', help='Optical transmission matrix maintains the path')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='Weight saving path')
    parser.add_argument('--exp', type=str, default='./exp', help='Experimental result storage path')

    # 运行模式设置
    parser.add_argument('--is_recons', type=bool, default=False, help='Training the model')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return args