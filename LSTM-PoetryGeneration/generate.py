import argparse
import torch
from pathlib import Path
import logging
from app.generator import PoetryGenerator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='AI诗歌生成器')
    parser.add_argument('--model', type=str, default='saved_models/best_poetry_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--start', type=str, default='夏',
                        help='诗歌起始词')
    parser.add_argument('--length', type=int, default=60,
                        help='生成诗歌长度')
    parser.add_argument('--temp', type=float, default=0.8,
                        help='温度参数（0.1-2.0）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备（cuda 或 cpu）')

    args = parser.parse_args()

    try:
        # 创建生成器
        generator = PoetryGenerator(
            checkpoint_path=Path(args.model),
            device=torch.device(args.device))

        # 生成诗歌
        raw_poem = generator.generate(
            start_string=args.start,
            max_length=args.length,
            temperature=args.temp)

        # 格式化并打印结果
        formatted_poem = generator.format_poem(raw_poem)

        print("\n生成的诗歌：")
        print("-" * 40)
        print(formatted_poem)
        print("-" * 40)
        print(f"起始词: {args.start}")
        print(f"长度: {args.length} 字符")
        print(f"温度参数: {args.temp}")

    except Exception as e:
        logger.error(f"生成诗歌时出错: {e}")


if __name__ == "__main__":
    main()
