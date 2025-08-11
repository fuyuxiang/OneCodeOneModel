from typing import Tuple, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_and_preprocess_data(data_path: Path):
    """加载并预处理诗歌数据，保留结构和特殊标记"""
    try:
        START_TOKEN = '<START>'
        END_TOKEN = '<END>'
        SEP_TOKEN = '\n'  # 保留换行符

        poems = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' not in line:
                    continue
                content = line.strip().split(':', 1)[1]
                if not content:
                    continue
                # 保留原始标点和结构
                poems.append(f"{START_TOKEN}{content}{END_TOKEN}")

        logger.info(f"成功加载 {len(poems)} 首诗")

        # 拼成一个大字符串（保留诗与诗之间的换行）
        poetry_text = SEP_TOKEN.join(poems)

        # 创建字符映射
        all_chars = sorted(list(set(poetry_text)))
        char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        idx_to_char = {idx: char for idx, char in enumerate(all_chars)}

        logger.info(f"总字符数（去重）：{len(all_chars)}")

        return poetry_text, char_to_idx, idx_to_char

    except Exception as e:
        logger.error(f"数据处理错误: {e}")
        raise


def create_char_mappings(text: str):
    """创建字符到索引的映射"""
    all_chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
    return char_to_idx, idx_to_char
