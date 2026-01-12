"""
DeepSpeed Multi-GPU Training for OSFormer
使用ZeRO-2优化的4卡训练
基于单卡版本 train_rgbb_tiny.py 改造
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
import json
import deepspeed
from deepspeed.utils import RepeatingLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.rgbt_tiny_coco import RGBTTinyCOCODataset, collate_fn
from models.osformer import build_osformer
from utils.loss import compute_loss
from utils.target_utils import convert_targets_for_loss
from utils.visualize import visualize_detection_results, images_to_tensorboard_grid


def setup_logging(log_dir, rank):
    """设置日志（只在主进程记录）"""
    os.makedirs(log_dir, exist_ok=True)
    
    if rank == 0:
        log_file = os.path.join(log_dir, f'train_{datetime.now():%Y%m%d_%H%M%S}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        # 非主进程只输出到控制台，级别为WARNING
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - [Rank {}] - %(levelname)s - %(message)s'.format(rank),
            handlers=[logging.StreamHandler()]
        )
    
    return logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_datasets(config):
    """构建数据集"""
    train_dataset = RGBTTinyCOCODataset(
        root_dir=config['data']['root_dir'],
        split='train',
        num_frames=config['model']['num_frames'],
        img_size=config['model']['img_size'],
        modality=config['data'].get('modality', 'both')
    )
    
    val_dataset = RGBTTinyCOCODataset(
        root_dir=config['data']['root_dir'],
        split='test',
        num_frames=config['model']['num_frames'],
        img_size=config['model']['img_size'],
        modality=config['data'].get('modality', 'both')
    )
    
    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, config, rank, world_size):
    """构建数据加载器（支持分布式）"""
    # 分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def build_model(config):
    """构建模型"""
    model = build_osformer(
        num_classes=config['model']['num_classes'],
        num_frames=config['model']['num_frames'],
        img_size=config['model']['img_size'],
        use_doppler=config['model']['use_doppler']
    )
    return model


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = (score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - score) > self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_one_epoch(model_engine, train_loader, device, config, epoch, writer, rank, world_size):
    """训练一个epoch（DeepSpeed版本）"""
    model_engine.train()
    
    total_loss = 0.0
    loss_components = {
        'cls_loss': 0.0,
        'bbox_loss': 0.0,
        'centerness_loss': 0.0,
        'offset_loss': 0.0
    }
    
    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        # 数据移动到设备
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)
        targets_batch = batch['targets']
        
        # 前向传播
        predictions = model_engine(rgb, thermal)
        
        # 转换目标格式
        feature_size = config['model']['img_size'] // 16
        targets = convert_targets_for_loss(
            targets_batch,
            num_frames=config['model']['num_frames'],
            img_size=config['model']['img_size'],
            feature_size=feature_size,
            device=device
        )
        
        # 计算损失
        loss, loss_dict = compute_loss(
            predictions, targets, config['train']['loss_weights']
        )
        
        # DeepSpeed backward
        model_engine.backward(loss)
        
        # DeepSpeed step
        model_engine.step()
        
        # 累积损失
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        
        # 记录到 TensorBoard (只在主进程)
        if rank == 0 and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', model_engine.get_lr()[0], global_step)
            
            # 每100个batch记录损失分量
            if batch_idx % 100 == 0:
                writer.add_scalar('train/batch_cls_loss', loss_dict['cls_loss'], global_step)
                writer.add_scalar('train/batch_bbox_loss', loss_dict['bbox_loss'], global_step)
                writer.add_scalar('train/batch_centerness_loss', loss_dict['centerness_loss'], global_step)
                writer.add_scalar('train/batch_offset_loss', loss_dict['offset_loss'], global_step)
        
        # 更新进度条（只在主进程）
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{model_engine.get_lr()[0]:.6f}'
            })
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def save_checkpoint(model_engine, epoch, loss, config, save_path, rank):
    """保存检查点（DeepSpeed）"""
    if rank == 0:
        # DeepSpeed保存
        model_engine.save_checkpoint(
            save_dir=os.path.dirname(save_path),
            tag=f'epoch_{epoch}',
            client_state={'loss': loss, 'config': config}
        )
        print(f'Checkpoint saved: epoch_{epoch}')


def find_best_checkpoint(checkpoint_dir):
    """自动查找最佳模型"""
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    return None


def find_latest_checkpoint(checkpoint_dir):
    """自动查找最新checkpoint"""
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    def extract_epoch(path):
        basename = os.path.basename(path)
        epoch_str = basename.replace('epoch_', '').replace('.pth', '')
        try:
            return int(epoch_str)
        except:
            return -1
    
    checkpoint_files.sort(key=extract_epoch, reverse=True)
    return checkpoint_files[0] if checkpoint_files else None


def main():
    parser = argparse.ArgumentParser(description='Train OSFormer with DeepSpeed (Multi-GPU)')
    
    # ========== 与单卡训练完全兼容的参数 ==========
    parser.add_argument('--config', type=str,
                       default='./configs/rgbt_tiny_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from. Special values: "best", "latest", "auto"')
    parser.add_argument('--resume_from_best', action='store_true',
                       help='Automatically resume from best_model.pth')
    parser.add_argument('--resume_from_latest', action='store_true',
                       help='Automatically resume from latest epoch checkpoint')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer when resuming (for fine-tuning)')
    parser.add_argument('--reset_epochs', action='store_true',
                       help='Reset epoch counter to 0 when resuming')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (override config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size per GPU (override config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (override config)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (ignored in DeepSpeed, kept for compatibility)')
    parser.add_argument('--amp', action='store_true',
                       help='Use mixed precision training (automatically enabled in DeepSpeed)')
    
    # ========== DeepSpeed特定参数 ==========
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (auto-set by DeepSpeed)')
    
    # 添加DeepSpeed参数（DeepSpeed会自动添加--deepspeed_config）
    parser = deepspeed.add_config_arguments(parser)
    
    # 设置deepspeed_config的默认值（如果没有通过命令行指定）
    # 注意：这个参数由deepspeed.add_config_arguments()添加
    
    args = parser.parse_args()
    
    # 设置deepspeed_config默认值（如果没有指定）
    if args.deepspeed_config is None:
        args.deepspeed_config = './configs/deepspeed_config.json'
    
    # 加载配置
    config = load_config(args.config)
    
    # ========== 命令行参数覆盖配置（与单卡训练完全一致） ==========
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.lr:
        # 更新DeepSpeed配置中的学习率
        if os.path.exists(args.deepspeed_config):
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
            ds_config['optimizer']['params']['lr'] = args.lr
            ds_config['scheduler']['params']['warmup_max_lr'] = args.lr
            # 临时保存修改后的配置
            temp_ds_config = args.deepspeed_config.replace('.json', '_temp.json')
            with open(temp_ds_config, 'w') as f:
                json.dump(ds_config, f, indent=2)
            args.deepspeed_config = temp_ds_config
    
    # 初始化DeepSpeed
    deepspeed.init_distributed()
    
    # 获取rank和world_size
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 设置设备
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # 设置日志（只在主进程详细记录）
    logger = setup_logging(config['save']['log_dir'], rank)
    
    if rank == 0:
        logger.info(f'Starting DeepSpeed training with {world_size} GPUs')
        logger.info(f'Config: {args.config}')
        logger.info(f'DeepSpeed config: {args.deepspeed_config}')
    
    # 构建数据集
    if rank == 0:
        logger.info('Building datasets...')
    train_dataset, val_dataset = build_datasets(config)
    
    if rank == 0:
        logger.info(f'Train dataset: {len(train_dataset)} samples')
        logger.info(f'Val dataset: {len(val_dataset)} samples')
        logger.info(f'Per GPU batch size: {config["train"]["batch_size"]}')
        logger.info(f'Global batch size: {config["train"]["batch_size"] * world_size}')
    
    # 构建数据加载器
    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        train_dataset, val_dataset, config, rank, world_size
    )
    
    # 构建模型
    if rank == 0:
        logger.info('Building model...')
    model = build_model(config)
    
    # 计算参数量（只在主进程）
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')
    
    # ==================== 初始化训练状态和Resume逻辑 ====================
    start_epoch = 0
    best_metric_name = config['save'].get('best_metric', 'map50')
    best_epoch = 0
    
    # 初始化 best_metric
    if best_metric_name in ['loss', 'val_loss']:
        best_metric = float('inf')
        best_mode = 'min'
    else:
        best_metric = 0.0
        best_mode = 'max'
    
    if rank == 0:
        logger.info(f'Model selection metric: {best_metric_name} (mode: {best_mode})')
    
    # ==================== 智能恢复训练（与单卡完全一致） ====================
    resume_path = None
    load_info = None  # 初始化load_info
    
    # 确定要恢复的checkpoint路径
    if args.resume_from_best:
        resume_path = find_best_checkpoint(config['save']['checkpoint_dir'])
        if resume_path and rank == 0:
            logger.info(f'🔍 Found best model: {resume_path}')
        elif rank == 0:
            logger.warning('❌ No best_model.pth found')
    
    elif args.resume_from_latest:
        resume_path = find_latest_checkpoint(config['save']['checkpoint_dir'])
        if resume_path and rank == 0:
            logger.info(f'🔍 Found latest checkpoint: {resume_path}')
        elif rank == 0:
            logger.warning('❌ No epoch checkpoints found')
    
    elif args.resume:
        # 处理特殊值
        if args.resume.lower() == 'best':
            resume_path = find_best_checkpoint(config['save']['checkpoint_dir'])
        elif args.resume.lower() == 'latest':
            resume_path = find_latest_checkpoint(config['save']['checkpoint_dir'])
        elif args.resume.lower() == 'auto':
            resume_path = find_best_checkpoint(config['save']['checkpoint_dir'])
            if not resume_path:
                resume_path = find_latest_checkpoint(config['save']['checkpoint_dir'])
            if resume_path and rank == 0:
                logger.info(f'🔍 Auto-selected checkpoint: {resume_path}')
        else:
            resume_path = args.resume
    
    # 加载checkpoint（在DeepSpeed初始化前）
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            logger.info('=' * 60)
            logger.info('📥 LOADING CHECKPOINT')
            logger.info('=' * 60)
        
        try:
            checkpoint = torch.load(resume_path, map_location='cpu')
            
            # 提取信息但不立即加载（DeepSpeed会处理模型权重）
            if not args.reset_epochs:
                start_epoch = checkpoint.get('epoch', 0) + 1
                if rank == 0:
                    logger.info(f'✅ Will resume from epoch {start_epoch}')
            else:
                start_epoch = 0
                if rank == 0:
                    logger.info('⚠️  Epoch counter reset to 0 (fine-tuning mode)')
            
            # 加载最佳指标信息
            if 'loss' in checkpoint and isinstance(checkpoint['loss'], dict):
                loss_info = checkpoint['loss']
                best_metric = loss_info.get('best_metric', best_metric)
                best_epoch = loss_info.get('best_epoch', 0)
                
                if rank == 0:
                    logger.info(f'✅ Best metric: {best_metric_name}={best_metric:.4f} (epoch {best_epoch})')
            
            load_info = checkpoint
            
            if rank == 0:
                logger.info('=' * 60)
        
        except Exception as e:
            if rank == 0:
                logger.error(f'❌ Failed to load checkpoint: {e}')
                import traceback
                traceback.print_exc()
    
    # 初始化DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    
    # 加载模型权重（如果有checkpoint）
    if load_info is not None:
        if 'model_state_dict' in load_info:
            # 加载模型权重
            model_engine.load_state_dict(load_info['model_state_dict'], strict=True)
            if rank == 0:
                logger.info('✅ Model weights loaded')
            
            # 加载优化器（如果不reset）
            if not args.reset_optimizer and 'optimizer_state_dict' in load_info:
                try:
                    optimizer.load_state_dict(load_info['optimizer_state_dict'])
                    if rank == 0:
                        logger.info('✅ Optimizer state loaded')
                except Exception as e:
                    if rank == 0:
                        logger.warning(f'⚠️  Failed to load optimizer state: {e}')
            elif args.reset_optimizer and rank == 0:
                logger.info('⚠️  Optimizer reset (fine-tuning mode)')
    
    # 创建 TensorBoard writer (只在主进程)
    writer = None
    if rank == 0:
        tensorboard_dir = config['save'].get('tensorboard_dir', './runs/rgbt_tiny_deepspeed')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f'TensorBoard logs: {tensorboard_dir}')
        
        # 记录配置
        writer.add_text('config/training', f'DeepSpeed ZeRO-2, {world_size} GPUs', 0)
        writer.add_text('config/loss_function', 'CIoU Loss', 0)
        writer.add_text('config/loss_weights', str(config['train']['loss_weights']), 0)
    
    # 初始化早停机制（只在主进程）
    early_stopping = None
    if rank == 0 and config.get('early_stopping', {}).get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['early_stopping'].get('patience', 15),
            min_delta=config['early_stopping'].get('min_delta', 0.001),
            mode=config['early_stopping'].get('mode', 'max')
        )
        logger.info(f'Early stopping enabled with patience={early_stopping.patience}')
    
    # 创建保存目录
    if rank == 0:
        os.makedirs(config['save']['checkpoint_dir'], exist_ok=True)
    
    # 训练循环
    if rank == 0:
        logger.info('Starting training...')
    
    for epoch in range(start_epoch, config['train']['num_epochs']):
        # 设置epoch（用于分布式采样器）
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f'Epoch {epoch}/{config["train"]["num_epochs"]}')
        
        # 训练一个epoch
        train_loss, loss_components = train_one_epoch(
            model_engine, train_loader, device, config, epoch, writer, rank, world_size
        )
        
        if rank == 0:
            logger.info(f'Train loss: {train_loss:.4f}')
            logger.info(f'Loss components: {loss_components}')
            
            # 记录训练损失到 TensorBoard
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            for key, value in loss_components.items():
                writer.add_scalar(f'epoch/train_{key}', value, epoch)
        
        # 定期保存检查点（只在主进程）
        if rank == 0 and (epoch + 1) % config['save']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['save']['checkpoint_dir'], f'epoch_{epoch}.pth'
            )
            save_checkpoint(
                model_engine, epoch,
                {'train_loss': train_loss, 'loss_components': loss_components},
                config, checkpoint_path, rank
            )
            logger.info(f'💾 Checkpoint saved: epoch_{epoch}')
        
        # 同步所有进程
        torch.distributed.barrier()
    
    # 训练结束
    if rank == 0:
        writer.close()
        logger.info('=' * 60)
        logger.info('🎉 TRAINING COMPLETED!')
        logger.info('=' * 60)
        logger.info(f'Total Epochs: {config["train"]["num_epochs"]}')
        logger.info(f'Checkpoints: {config["save"]["checkpoint_dir"]}')
        logger.info('=' * 60)
    
    # 清理
    deepspeed.sys.exit()


if __name__ == '__main__':
    main()

