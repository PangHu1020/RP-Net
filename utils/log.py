import logging
import os

def setup_logging(save_dir, save_name, filename='training.log'):
    """配置日志系统，包括控制台和文件输出"""
    logger = logging.getLogger()
    logger.handlers = []  # 清除现有的处理器，防止重复
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 控制台日志处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件日志处理器
    if save_dir:
        # 创建目录路径，包括子目录 save_name
        full_path = os.path.join(save_dir, save_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        
        # 检查文件是否存在，并为日志文件名生成唯一名称
        base_filename = os.path.join(full_path, filename)
        file_suffix = 1
        new_filename = base_filename
        while os.path.exists(new_filename):
            # 如果文件已存在，增加后缀
            name, ext = os.path.splitext(filename)
            new_filename = os.path.join(full_path, f"{name}{file_suffix}{ext}")
            file_suffix += 1

        # 创建并配置文件处理器
        fh = logging.FileHandler(new_filename, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



# import logging
# import os
# import torch
# import json

# def setup_logging(save_dir, save_name, filename='training.log', model=None, optimizer=None, loss_fn=None, additional_info=None):
#     """配置日志系统，包括控制台和文件输出"""
#     logger = logging.getLogger()
#     logger.handlers = []  # 清除现有的处理器，防止重复
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

#     # 控制台日志处理器
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     # 文件日志处理器
#     if save_dir:
#         # 创建目录路径，包括子目录 save_name
#         full_path = os.path.join(save_dir, save_name)
#         if not os.path.exists(full_path):
#             os.makedirs(full_path)
        
#         # 检查文件是否存在，并为日志文件名生成唯一名称
#         base_filename = os.path.join(full_path, filename)
#         file_suffix = 1
#         new_filename = base_filename
#         while os.path.exists(new_filename):
#             # 如果文件已存在，增加后缀
#             name, ext = os.path.splitext(filename)
#             new_filename = os.path.join(full_path, f"{name}{file_suffix}{ext}")
#             file_suffix += 1

#         # 创建并配置文件处理器
#         fh = logging.FileHandler(new_filename, mode='w')
#         fh.setLevel(logging.INFO)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     # 准备要保存的信息
#     config_info = {}

#     # 模型结构
#     if model:
#         config_info['Model Structure'] = str(model)
    
#     # 优化器信息
#     if optimizer:
#         optimizer_info = {
#             "Optimizer Type": type(optimizer).__name__,
#             "Learning Rate": optimizer.defaults.get("lr", "Unknown"),
#             "Weight Decay": optimizer.defaults.get("weight_decay", "Unknown"),
#             "Momentum": optimizer.defaults.get("momentum", "Unknown")
#         }
#         config_info['Optimizer Information'] = optimizer_info
    
#     # 损失函数信息
#     if loss_fn:
#         config_info['Loss Function'] = type(loss_fn).__name__

#     # 额外的信息（例如训练参数等）
#     if additional_info:
#         config_info['Additional Information'] = additional_info

#     # 保存配置信息到 JSON 文件
#     if save_dir:
#         json_filename = os.path.join(full_path, 'config.json')
#         with open(json_filename, 'w') as json_file:
#             json.dump(config_info, json_file, indent=4)

#     return logger

# # # 示例调用
# # if __name__ == "__main__":
# #     # 示例模型和优化器
# #     model = torch.nn.Linear(10, 2)
# #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
# #     loss_fn = torch.nn.CrossEntropyLoss()
    
# #     additional_info = {
# #         'Training Epochs': 50,
# #         'Batch Size': 32,
# #         'Device': 'cuda'
# #     }
    
# #     logger = setup_logging(save_dir='logs', save_name='experiment_1', model=model, optimizer=optimizer, loss_fn=loss_fn, additional_info=additional_info)
# #     logger.info("Logging setup complete.")