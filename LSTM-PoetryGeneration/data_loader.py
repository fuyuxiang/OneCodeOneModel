import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.StreamHandler())


class PoetryDataset(Dataset):
    def __init__(self, poetry_data, char_to_idx,
                 seq_length, device):
        self.device = device
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.data = self._prepare_data(poetry_data)

    def _text_to_indices(self, text):
        return [self.char_to_idx.get(char, 0) for char in text]

    def _prepare_data(self, poetry_data: str):
        """准备训练数据"""
        indices = self._text_to_indices(poetry_data)
        sequences = []
        targets = []

        # 使用滑动窗口创建序列
        for i in range(0, len(indices) - self.seq_length):
            sequences.append(indices[i:i + self.seq_length])
            targets.append(indices[i + 1:i + self.seq_length + 1])

        logger.info(f"创建 {len(sequences)} 个训练样本")
        return list(zip(sequences, targets))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq, dtype=torch.long).to(self.device), \
            torch.tensor(target, dtype=torch.long).to(self.device)


def create_data_loader(poetry_data, char_to_idx, config):
    """创建数据加载器"""
    dataset = PoetryDataset(
        poetry_data=poetry_data,
        char_to_idx=char_to_idx,
        seq_length=config.seq_length,
        device=config.device
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
