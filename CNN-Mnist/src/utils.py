import os
import random
import logging
import numpy as np
import torch
import yaml
from datetime import datetime


def setup_logging(log_dir='logs'):
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载YAML配置文件并确保数值类型正确"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 递归转换所有数值字段
    def convert_numerics(data):
        if isinstance(data, dict):
            return {k: convert_numerics(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_numerics(item) for item in data]
        elif isinstance(data, str) and data.replace('.', '', 1).replace('e-', '', 1).isdigit():
            try:
                return float(data)
            except ValueError:
                return data
        return data

    return convert_numerics(config)

def save_config(config, save_dir):
    """保存配置到文件"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'config.yaml')
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
