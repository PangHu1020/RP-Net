import torch.optim as optim
import torch

def get_optimizer(models, optimizer_name='SGD', learning_rate=0.01, weight_decay=0.0, momentum=0.0):
    """
    获取指定的优化器

    Args:
        model (torch.nn.Module): 要优化的模型
        optimizer_name (str): 优化器名称，支持 'SGD', 'Adam', 'AdamW'
        learning_rate (float): 学习率
        weight_decay (float): (权重衰减(L2 正则化）)

    Returns:
        optimizer: 初始化的优化器
    """

     # If a single model is passed, convert it to a list
    if isinstance(models, torch.nn.Module):
        models = [models]
    
    # Collect parameters from all models
    parameters = []
    for model in models:
        parameters += list(model.parameters())

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers: 'SGD', 'Adam', 'AdamW'.")

    return optimizer