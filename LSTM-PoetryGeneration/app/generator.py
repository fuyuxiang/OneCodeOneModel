import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from models.poetry_lstm import PoetryLSTM

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.StreamHandler())


class PoetryGenerator:
    def __init__(self, checkpoint_path, device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.model, self.char_to_idx, self.idx_to_char, self.config = self._load_checkpoint()
        self.model.eval()
        logger.info(f"诗歌生成器初始化完成，使用设备: {self.device}")
        logger.info(f"词汇表大小: {len(self.idx_to_char)}")

    def _load_checkpoint(self) :
        """加载检查点，返回模型、字符映射和配置信息"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件 {self.checkpoint_path} 不存在")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        vocab_size = checkpoint['vocab_size']

        model = PoetryLSTM(
            input_size=vocab_size,
            hidden_size=checkpoint.get('hidden_size', 256),
            num_layers=checkpoint.get('num_layers', 2),
            dropout_rate=0
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        config = {
            'seq_length': checkpoint.get('seq_length', 30),
            'hidden_size': checkpoint.get('hidden_size', 256),
            'num_layers': checkpoint.get('num_layers', 2)
        }

        return model, char_to_idx, idx_to_char, config

    def _preprocess_input(self, text):
        """将字符串转换为张量输入"""
        indices = [self.char_to_idx.get(char, 0) for char in text]
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        hidden = self._init_hidden(1)
        return input_tensor, hidden

    def _init_hidden(self, batch_size):
        """初始化隐藏状态"""
        weight = next(self.model.parameters())
        return (weight.new_zeros(self.config['num_layers'], batch_size, self.config['hidden_size']),
                weight.new_zeros(self.config['num_layers'], batch_size, self.config['hidden_size']))

    def generate(self, start_string = '', max_length = 100, temperature = 1.0):
        """生成诗歌，遇到 <END> 停止"""
        START_TOKEN = '<START>'
        END_TOKEN = '<END>'

        # 用 START_TOKEN 开始
        input_seq, hidden = self._preprocess_input(START_TOKEN + start_string)
        generated = [START_TOKEN] + list(start_string)

        for _ in range(max_length):
            with torch.no_grad():
                output, hidden = self.model(input_seq, hidden)

            last_output = output[:, -1, :]
            probs = torch.softmax(last_output / temperature, dim=-1).squeeze()
            next_idx = torch.multinomial(probs, 1).item()
            next_char = self.idx_to_char.get(next_idx, '')

            if next_char == END_TOKEN:
                break

            generated.append(next_char)
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(self.device)

        # 去掉特殊标记
        poem = ''.join(generated).replace(START_TOKEN, '').replace(END_TOKEN, '')
        return poem

    def format_poem(self, raw_poem: str) -> str:
        """格式化诗歌，仅去除首尾空格，不强制加标点"""
        return raw_poem.strip()
