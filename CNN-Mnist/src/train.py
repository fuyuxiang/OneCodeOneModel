import time
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import os

class Trainer:
    def __init__(self, model, config, device, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 确保数值类型正确
        lr = float(config['training']['lr'])
        weight_decay = float(config['training']['weight_decay'])
        grad_clip = float(config['training']['grad_clip'])

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        step_size = int(config['scheduler']['step_size'])
        gamma = float(config['scheduler']['gamma'])

        self.scheduler = StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )

        # 训练状态
        self.best_val_acc = 0.0
        self.epoch = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def run_epoch(self, loader, training=True):
        """运行一个epoch"""
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 反向传播和优化
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()

                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                    self.optimizer.step()

                # 统计指标
                epoch_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.size(0)

        return epoch_loss / total, correct / total

    def train(self, train_loader, val_loader):
        """主训练循环"""
        start_time = time.time()

        try:
            for epoch in range(self.config['training']['epochs']):
                self.epoch = epoch
                epoch_start = time.time()

                # 训练阶段
                train_loss, train_acc = self.run_epoch(train_loader, training=True)

                # 验证阶段
                val_loss, val_acc = self.run_epoch(val_loader, training=False)

                # 更新学习率
                self.scheduler.step()

                # 记录历史
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # 记录日志
                epoch_time = time.time() - epoch_start
                lr = self.optimizer.param_groups[0]['lr']

                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config['training']['epochs']} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"LR: {lr:.1e} | Time: {epoch_time:.1f}s"
                )

                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint(is_best=True)

                # 定期保存检查点
                if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user. Saving checkpoint...")
            self.save_checkpoint()
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")

    def save_checkpoint(self, is_best=False):
        """保存训练检查点"""
        checkpoint_dir = self.config['checkpoint']['dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }

        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(state, path)
            self.logger.info(f"Saved best model to {path}")

        # 定期保存
        path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{self.epoch + 1}.pth"
        )
        torch.save(state, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def export_model(self, export_dir='export'):
        """导出模型为ONNX格式"""
        os.makedirs(export_dir, exist_ok=True)
        dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
        onnx_path = os.path.join(export_dir, 'mnist_model.onnx')

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        self.logger.info(f"Exported model to {onnx_path}")