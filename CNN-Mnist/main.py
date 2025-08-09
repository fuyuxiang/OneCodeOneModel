import argparse
from src.utils import setup_logging, set_seed, load_config, save_config
from src.data_loader import get_mnist_loaders
from src.model import CNN
from src.train import Trainer
import torch

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MNIST Training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()

    # 初始化
    logger = setup_logging()
    config = load_config(args.config)
    set_seed(config['training']['seed'])
    save_config(config, config['checkpoint']['dir'])

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 数据加载
    train_loader, val_loader, test_loader = get_mnist_loaders(config)
    logger.info(f"Dataset loaded: Train={len(train_loader.dataset)} "
                f"Val={len(val_loader.dataset)} Test={len(test_loader.dataset)}")

    # 模型初始化
    model = CNN().to(device)
    logger.info(f"Model created: \n{model}")

    # 训练器
    trainer = Trainer(model, config, device, logger)

    # 恢复训练（如果指定）
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_acc = checkpoint['best_val_acc']
        trainer.epoch = checkpoint['epoch']
        trainer.history = checkpoint['history']
        logger.info(f"Resuming training from epoch {trainer.epoch + 1}")

    # 训练模型
    trainer.train(train_loader, val_loader)

    # 最终测试
    test_loss, test_acc = trainer.run_epoch(test_loader, training=False)
    logger.info(f"Final test results - Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # 导出模型
    trainer.export_model()


if __name__ == "__main__":
    main()