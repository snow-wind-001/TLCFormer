"""
OSFormer 训练脚本
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
from tqdm import tqdm
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import OSFormer, OSFormerConfig
from datasets import RGBTTinyDataset, collate_fn
from utils.loss import compute_loss
from utils.metrics import DetectionMetrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train OSFormer')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, default='./data/RGBT-Tiny',
                        help='数据集根目录')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作线程数')
    
    # 模型参数
    parser.add_argument('--num_frames', type=int, default=5,
                        help='输入帧数')
    parser.add_argument('--sample_frames', type=int, default=3,
                        help='采样帧数')
    parser.add_argument('--img_size', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--embed_dim', type=int, default=96,
                        help='嵌入维度')
    parser.add_argument('--use_doppler', action='store_true', default=True,
                        help='是否使用 Doppler 滤波')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='warmup 轮数')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='评估间隔（epoch）')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存间隔（epoch）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='是否使用混合精度训练')
    
    args = parser.parse_args()
    return args


def build_optimizer(model, args):
    """构建优化器"""
    # 分离权重衰减参数
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=args.lr, betas=(0.9, 0.999))
    
    return optimizer


def build_scheduler(optimizer, args, num_steps_per_epoch):
    """构建学习率调度器"""
    # Warmup + Cosine Annealing
    warmup_steps = args.warmup_epochs * num_steps_per_epoch
    total_steps = args.num_epochs * num_steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler


def train_one_epoch(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    scaler,
    epoch, 
    args, 
    writer
):
    """训练一个 epoch"""
    model.train()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs}')
    
    total_loss = 0.0
    loss_stats = {
        'cls_loss': 0.0,
        'bbox_loss': 0.0,
        'centerness_loss': 0.0,
        'offset_loss': 0.0
    }
    
    for batch_idx, (rgb_frames, thermal_frames, targets) in enumerate(pbar):
        # 转移到设备
        rgb_frames = rgb_frames.to(args.device)
        thermal_frames = thermal_frames.to(args.device)
        
        # 前向传播（混合精度）
        optimizer.zero_grad()
        
        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(rgb_frames, thermal_frames)
                loss, loss_dict = compute_loss(outputs, targets)
        else:
            outputs = model(rgb_frames, thermal_frames)
            loss, loss_dict = compute_loss(outputs, targets)
        
        # 反向传播
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        for key in loss_stats:
            loss_stats[key] += loss_dict.get(key, 0.0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}"
        })
        
        # 记录到 TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
    
    # 平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_stats:
        loss_stats[key] /= num_batches
    
    return avg_loss, loss_stats


@torch.no_grad()
def evaluate(model, dataloader, args):
    """评估模型"""
    model.eval()
    
    metrics = DetectionMetrics(num_classes=1)
    
    pbar = tqdm(dataloader, desc='Evaluating')
    
    for rgb_frames, thermal_frames, targets_list in pbar:
        rgb_frames = rgb_frames.to(args.device)
        thermal_frames = thermal_frames.to(args.device)
        
        # 前向传播
        outputs = model(rgb_frames, thermal_frames)
        
        # 解码预测结果
        # 这里需要实现解码逻辑，将网络输出转换为边界框
        # 简化处理：使用中间帧的预测
        from models.seq_head import AnchorFreeDecoder
        decoder = AnchorFreeDecoder(score_thresh=0.3, nms_thresh=0.5)
        
        B = rgb_frames.shape[0]
        mid_frame = args.num_frames // 2
        
        for b in range(B):
            output = outputs[mid_frame]
            
            cls_pred = output['cls'][b]
            bbox_pred = output['bbox'][b]
            centerness_pred = output.get('centerness', None)
            if centerness_pred is not None:
                centerness_pred = centerness_pred[b]
            
            # 解码
            boxes, scores, labels = decoder.decode_single_frame(
                cls_pred, bbox_pred, centerness_pred
            )
            
            # GT
            targets = targets_list[b][mid_frame]
            gt_boxes = targets['boxes'].cpu().numpy()
            gt_labels = targets['labels'].cpu().numpy()
            
            # 更新指标
            if len(boxes) > 0:
                metrics.update(
                    boxes.cpu().numpy(),
                    scores.cpu().numpy(),
                    labels.cpu().numpy(),
                    gt_boxes,
                    gt_labels
                )
            else:
                # 没有预测
                metrics.update(
                    np.zeros((0, 4)),
                    np.zeros(0),
                    np.zeros(0, dtype=np.int64),
                    gt_boxes,
                    gt_labels
                )
    
    # 计算指标
    results = metrics.compute()
    
    return results


def main():
    """主函数"""
    args = parse_args()
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"使用设备: {device}")
    
    # 创建模型
    config = OSFormerConfig(
        num_frames=args.num_frames,
        sample_frames=args.sample_frames,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        use_doppler=args.use_doppler,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
    
    model = OSFormer(
        num_frames=config.num_frames,
        sample_frames=config.sample_frames,
        img_size=config.img_size,
        embed_dim=config.embed_dim,
        use_doppler=config.use_doppler
    )
    model = model.to(device)
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {num_params:.2f}M")
    
    # 创建数据集
    train_dataset = RGBTTinyDataset(
        root_dir=args.data_root,
        split='train',
        num_frames=args.num_frames,
        img_size=args.img_size
    )
    
    val_dataset = RGBTTinyDataset(
        root_dir=args.data_root,
        split='val',
        num_frames=args.num_frames,
        img_size=args.img_size
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 创建优化器和调度器
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 恢复训练
    start_epoch = 1
    best_map50 = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_map50 = checkpoint.get('best_map50', 0.0)
        print(f"从 epoch {checkpoint['epoch']} 恢复训练")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        # 训练
        avg_loss, loss_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, args, writer
        )
        
        # 记录
        writer.add_scalar('epoch/loss', avg_loss, epoch)
        for key, val in loss_stats.items():
            writer.add_scalar(f'epoch/{key}', val, epoch)
        
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f}")
        
        # 评估
        if epoch % args.eval_interval == 0:
            print("评估中...")
            results = evaluate(model, val_loader, args)
            
            # 记录指标
            for key, val in results.items():
                writer.add_scalar(f'val/{key}', val, epoch)
                print(f"  {key}: {val:.4f}")
            
            # 保存最佳模型
            map50 = results.get('mAP50', 0.0)
            if map50 > best_map50:
                best_map50 = map50
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_map50': best_map50,
                    'config': config.to_dict()
                }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
                print(f"  保存最佳模型 (mAP50={best_map50:.4f})")
        
        # 定期保存
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_map50': best_map50,
                'config': config.to_dict()
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print("\n训练完成！")
    writer.close()


if __name__ == '__main__':
    main()

