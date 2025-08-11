import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict
from models.poetry_lstm import PoetryLSTM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PoetryTrainer:
    def __init__(self, config, model, data_loader,
                 char_to_idx, idx_to_char):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.device = config.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        logger.info("开始训练...")
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0
            for batch_idx, (x, y) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                output, _ = self.model(x)
                loss = self.criterion(output.view(-1, output.size(-1)), y.view(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Step {batch_idx+1}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.data_loader)
            logger.info(f"Epoch {epoch+1} 完成，平均Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch):
        checkpoint_path = Path(self.config.checkpoint_dir) / f"poetry_model_epoch{epoch}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'vocab_size': len(self.char_to_idx),
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'seq_length': self.config.seq_length
        }, checkpoint_path)

        logger.info(f"模型已保存到 {checkpoint_path}")
