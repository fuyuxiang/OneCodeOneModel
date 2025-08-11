import logging
from pathlib import Path
import os


def setup_logger(log_file: Path, logger_name: str = __name__):
    """配置并返回日志记录器"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 确保日志目录存在
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
