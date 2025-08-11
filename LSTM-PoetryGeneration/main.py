import torch
import logging
from pathlib import Path
from data_loader import create_data_loader
from utils.data_utils  import load_and_preprocess_data
from trainer import PoetryTrainer
from models.poetry_lstm import PoetryLSTM
from configs.default_config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config = TrainingConfig()

    # 1. 加载数据
    poetry_text, char_to_idx, idx_to_char = load_and_preprocess_data(Path(config.data_path))

    # 2. 创建 DataLoader
    data_loader = create_data_loader(poetry_text, char_to_idx, config)

    # 3. 初始化模型
    model = PoetryLSTM(
        input_size=len(char_to_idx),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate
    ).to(config.device)

    # 4. 训练器
    trainer = PoetryTrainer(config, model, data_loader, char_to_idx, idx_to_char)

    # 5. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
