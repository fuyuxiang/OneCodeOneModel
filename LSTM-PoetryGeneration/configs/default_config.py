from pathlib import Path
import torch


class TrainingConfig:
    def __init__(self):
        # 数据参数
        self.data_path = Path('data/poetry.txt')
        self.seq_length = 30

        # 模型参数
        self.input_size = 0  # 将在初始化后设置
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout_rate = 0.5

        # 训练参数
        self.batch_size = 512
        self.epochs = 100
        self.learning_rate = 0.002
        self.clip_norm = 5.0

        # 优化器参数
        self.step_size = 20
        self.gamma = 0.5

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 保存路径
        self.model_dir = Path('saved_models')
        self.model_dir.mkdir(exist_ok=True)
        self.best_model_path = self.model_dir / 'best_poetry_model.pth'
        self.final_model_path = self.model_dir / 'final_poetry_model.pth'
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_every = 10

        # 日志配置
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / 'training.log'