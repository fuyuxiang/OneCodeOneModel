import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_checkpoint(epoch, model, optimizer, loss, vocab_size,
                    char_to_idx, path, is_best = False):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab_size': vocab_size,
        'char_to_idx': char_to_idx,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers
    }

    torch.save(checkpoint, path)

    if is_best:
        logger.info(f"保存最佳模型，损失: {loss:.4f} 在 {path}")