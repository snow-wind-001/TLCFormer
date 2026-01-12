"""
使用RGBT-Tiny数据集训练OSFormer
适配7类别多目标检测
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.osformer import OSFormer, build_osformer
from datasets.rgbt_tiny_coco import RGBTTinyCOCODataset, collate_fn
from utils.loss import compute_loss
from utils.metrics import DetectionMetrics
from utils.target_utils import convert_targets_for_loss
from utils.visualize import visualize_detection_results, images_to_tensorboard_grid


class EarlyStopping:
    """
    早退机制
    
    参数：
        patience (int): 容忍的 epoch 数
        min_delta (float): 最小改进量
        mode (str): 'min' 或 'max'
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, current_score):
        """
        检查是否应该早退
        
        参数：
            current_score: 当前指标值
            
        返回：
            should_stop: 是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.is_better(current_score, self.best_score):
            # 有改进
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            # 没有改进
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def reset(self):
        """重置计数器"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """构建学习率调度器"""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
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


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_datasets(config):
    """构建数据集"""
    train_config = config['train']
    data_config = config['data']
    model_config = config['model']
    
    # 获取 tracking 和 frame_interval 配置
    frame_interval = model_config.get('frame_interval', 1)
    tracking_config = data_config.get('tracking', {})

    # 训练数据集
    train_dataset = RGBTTinyCOCODataset(
        root_dir=data_config['root_dir'],
        split='train',
        num_frames=model_config['num_frames'],
        frame_interval=frame_interval,  # 新增
        img_size=model_config['img_size'],
        modality=data_config.get('modality', 'both'),
        tracking_config=tracking_config  # 新增
    )

    # 验证数据集
    val_dataset = RGBTTinyCOCODataset(
        root_dir=data_config['root_dir'],
        split='test',
        num_frames=model_config['num_frames'],
        frame_interval=frame_interval,  # 新增
        img_size=model_config['img_size'],
        modality=data_config.get('modality', 'both'),
        tracking_config=tracking_config  # 新增
    )

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, config):
    """构建数据加载器"""
    train_config = config['train']
    data_config = config['data']

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval']['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def build_model(config):
    """构建模型"""
    model_config = config['model']

    model = build_osformer(
        num_frames=model_config['num_frames'],
        sample_frames=model_config['sample_frames'],
        img_size=model_config['img_size'],
        num_classes=model_config['num_classes'],
        embed_dim=model_config['embed_dim'],
        depths=model_config['depths'],
        use_doppler=model_config['use_doppler'],
        anchor_free=model_config['anchor_free'],
        dropout=model_config['dropout']
    )

    return model


def build_optimizer(model, config):
    """构建优化器"""
    train_config = config['train']

    # 分层设置权重衰减
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': train_config['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=train_config['lr'],
        betas=train_config['betas']
    )

    return optimizer


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, config, epoch, writer=None):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    loss_components = {
        'loss': 0,
        'cls_loss': 0,
        'bbox_loss': 0,
        'centerness_loss': 0,
        'offset_loss': 0
    }

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # 数据移动到设备
        rgb = batch['rgb'].to(device)  # (B, T, 3, H, W)
        thermal = batch['thermal'].to(device)  # (B, T, 1, H, W)
        targets_batch = batch['targets']
        
        # 前向传播（先推理获取feature_size）
        with autocast(enabled=config['train']['amp']):
            predictions = model(rgb, thermal)  # List of dict, 长度为 T
            
            # 🔥 从实际模型输出获取特征图尺寸（stride=8，即640/8=80）
            _, _, feature_size, _ = predictions[0]['cls'].shape  # 使用实际输出尺寸
            
            # 转换目标格式（从数据集格式转换为损失函数格式）
            use_tracking = config['data'].get('tracking', {}).get('enabled', False)
            targets = convert_targets_for_loss(
                targets_batch,
                num_frames=config['model']['num_frames'],
                img_size=config['model']['img_size'],
                feature_size=feature_size,
                device=device,
                use_tracking_offset=use_tracking  # 新增
            )
            
            # 计算损失
            total_loss, loss_dict = compute_loss(predictions, targets, config['train']['loss_weights'])
            loss = total_loss
        
        # 检测 NaN/Inf，如果发现则跳过此 batch
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf detected in loss at batch {batch_idx}, skipping this batch...")
            # 跳过此batch，不更新梯度
            pbar.update(1)
            continue

        # 反向传播
        optimizer.zero_grad()
        
        if config['train']['amp']:
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if config['train']['clip_grad'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['train']['clip_grad']
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # 梯度裁剪
            if config['train']['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['train']['clip_grad']
                )
            
            optimizer.step()

        # 更新学习率
        scheduler.step()

        # 统计损失
        # 🔥 修复：累加当前batch的loss值
        total_loss += loss.item()
        # 🔥 修复：累加loss_dict中的各分量值（注意loss_dict已经是item值）
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]

        # 记录到 TensorBoard (每个 batch)
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            # 总损失和学习率
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            # 各个损失分量（每100个batch记录一次，避免日志过多）
            if batch_idx % 100 == 0:
                writer.add_scalar('train/batch_cls_loss', loss_dict['cls_loss'], global_step)
                writer.add_scalar('train/batch_bbox_loss', loss_dict['bbox_loss'], global_step)
                writer.add_scalar('train/batch_centerness_loss', loss_dict['centerness_loss'], global_step)
                writer.add_scalar('train/batch_offset_loss', loss_dict['offset_loss'], global_step)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}

    return avg_loss, avg_components


def visualize_epoch_results(model, dataloader, device, config, epoch, writer, split='train'):
    """
    在每个epoch结束后可视化一些样本
    
    参数：
        model: 模型
        dataloader: 数据加载器
        device: 设备
        config: 配置
        epoch: 当前epoch
        writer: TensorBoard writer
        split: 'train' 或 'val'
    """
    model.eval()
    
    # 随机选择一个batch进行可视化
    import random
    batch_idx = random.randint(0, min(len(dataloader) - 1, 10))
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx != batch_idx:
                continue
            
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            targets_batch = batch['targets']
            
            # 前向传播
            predictions = model(rgb, thermal)
            
            # 可视化
            vis_images = visualize_detection_results(
                rgb, thermal, predictions, targets_batch,
                config['classes']['names'],
                score_thresh=config['eval'].get('score_thresh', 0.3),
                max_samples=4,
                mid_frame_only=True
            )
            
            # 转换为网格并写入TensorBoard
            if len(vis_images) > 0:
                grid = images_to_tensorboard_grid(vis_images, nrow=2)
                writer.add_image(f'{split}/predictions', grid, epoch)
            
            break
    
    model.train()


def compute_iou(boxes1, boxes2):
    """
    计算两组框的IoU
    boxes1: (N, 4) [x1, y1, x2, y2]
    boxes2: (M, 4) [x1, y1, x2, y2]
    返回: (N, M) IoU矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def decode_predictions(predictions, img_size, feature_size, score_thresh=0.3, num_classes=7):
    """
    解码预测结果为边界框
    
    Args:
        predictions: dict with 'cls', 'bbox', 'centerness'
        img_size: 原图尺寸 (640)
        feature_size: 特征图尺寸 (40)
        score_thresh: 分数阈值
        num_classes: 类别数
    
    Returns:
        boxes: (N, 4) [x1, y1, x2, y2] 原图坐标
        scores: (N,) 置信度
        labels: (N,) 类别ID
    """
    cls_pred = predictions['cls']  # (B, num_classes, H, W)
    bbox_pred = predictions['bbox']  # (B, 4, H, W)
    centerness_pred = predictions.get('centerness', None)  # (B, 1, H, W)
    
    B, C, H, W = cls_pred.shape
    device = cls_pred.device
    
    # 计算stride
    stride = img_size / feature_size
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for b in range(B):
        # 对每个样本解码
        cls_b = cls_pred[b]  # (num_classes, H, W)
        bbox_b = bbox_pred[b]  # (4, H, W)
        
        # 应用sigmoid到分类分数
        scores_b = torch.sigmoid(cls_b)  # (num_classes+1, H, W)
        
        # 🔥 改进：排除背景类（最后一个通道）
        # 只考虑前num_classes个前景类
        scores_b_fg = scores_b[:num_classes]  # (num_classes, H, W) 前景类
        
        # 如果有centerness，应用到分数上
        if centerness_pred is not None:
            centerness_b = torch.sigmoid(centerness_pred[b, 0])  # (H, W)
            scores_b_fg = scores_b_fg * centerness_b.unsqueeze(0)  # (num_classes, H, W)
        
        # 找到每个类别的最大分数和位置
        max_scores, _ = scores_b_fg.max(dim=0)  # (H, W)
        max_labels = scores_b_fg.argmax(dim=0)  # (H, W)
        
        # 筛选高于阈值的位置
        mask = max_scores > score_thresh
        
        if mask.sum() == 0:
            # 没有检测到任何目标
            all_boxes.append(torch.zeros(0, 4, device=device))
            all_scores.append(torch.zeros(0, device=device))
            all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
            continue
        
        # 获取有效位置的索引
        valid_indices = mask.nonzero(as_tuple=False)  # (N, 2) [h, w]
        valid_scores = max_scores[mask]  # (N,)
        valid_labels = max_labels[mask]  # (N,)
        
        # 获取对应的bbox预测 (l, t, r, b)
        valid_bbox = bbox_pred[b, :, mask]  # (4, N)
        valid_bbox = valid_bbox.t()  # (N, 4)
        
        # 转换FCOS格式 (l,t,r,b) 到 (x1,y1,x2,y2)
        h_idx = valid_indices[:, 0].float()
        w_idx = valid_indices[:, 1].float()
        
        # 网格中心坐标（原图坐标系）
        center_x = (w_idx + 0.5) * stride
        center_y = (h_idx + 0.5) * stride
        
        # 转换为 x1, y1, x2, y2
        l, t, r, b = valid_bbox[:, 0], valid_bbox[:, 1], valid_bbox[:, 2], valid_bbox[:, 3]
        x1 = (center_x - l * stride).clamp(min=0, max=img_size)
        y1 = (center_y - t * stride).clamp(min=0, max=img_size)
        x2 = (center_x + r * stride).clamp(min=0, max=img_size)
        y2 = (center_y + b * stride).clamp(min=0, max=img_size)
        
        boxes_b = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)
        
        # ✅ 添加 NMS
        if len(boxes_b) > 0:
            try:
                from torchvision.ops import nms
                keep_indices = nms(boxes_b, valid_scores, iou_threshold=0.5)
                boxes_b = boxes_b[keep_indices]
                valid_scores = valid_scores[keep_indices]
                valid_labels = valid_labels[keep_indices]
            except Exception as e:
                # NMS失败，使用原始结果
                pass
        
        all_boxes.append(boxes_b)
        all_scores.append(valid_scores)
        all_labels.append(valid_labels)
    
    return all_boxes, all_scores, all_labels


def evaluate(model, val_loader, device, config, class_names):
    """
    评估模型 - 计算真实的mAP, Precision, Recall等指标
    使用IoU匹配和AP计算
    """
    model.eval()
    
    # 存储所有预测和GT，用于计算mAP
    all_predictions = []  # List of dicts: {'boxes', 'scores', 'labels'}
    all_ground_truths = []  # List of dicts: {'boxes', 'labels'}
    
    total_val_loss = 0.0
    loss_components_val = {'cls_loss': 0.0, 'bbox_loss': 0.0, 'centerness_loss': 0.0, 'offset_loss': 0.0}
    
    img_size = config['model']['img_size']
    # 🔥 修改：stride=8（Neck已上采样）
    feature_size = img_size // 8  # stride=8 (640/8=80)
    score_thresh = config['eval'].get('score_thresh', 0.3)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            # 数据移动到设备
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            targets_batch = batch['targets']
            
            # 前向传播
            predictions = model(rgb, thermal)  # List of dict, length T
            
            # 计算验证损失
            use_tracking = config['data'].get('tracking', {}).get('enabled', False)
            targets = convert_targets_for_loss(
                targets_batch,
                num_frames=config['model']['num_frames'],
                img_size=img_size,
                feature_size=feature_size,
                device=device,
                use_tracking_offset=use_tracking  # 新增
            )
            
            total_loss, loss_dict = compute_loss(
                predictions, targets, config['train']['loss_weights']
            )
            total_val_loss += total_loss.item()
            for key in loss_components_val:
                if key in loss_dict:
                    loss_components_val[key] += loss_dict[key]
            
            # 解码预测结果（使用中间帧）
            mid_frame = len(predictions) // 2
            pred_frame = predictions[mid_frame]
            
            # 解码为边界框
            boxes_list, scores_list, labels_list = decode_predictions(
                pred_frame, img_size, feature_size, score_thresh, config['model']['num_classes']
            )
            
            # 🔥 关键修复：归一化预测坐标到 [0, 1] 以匹配GT
            # GT是归一化的[x1, y1, x2, y2]，预测是像素坐标，需要归一化
            for b_idx in range(len(boxes_list)):
                if len(boxes_list[b_idx]) > 0:
                    boxes_list[b_idx] = boxes_list[b_idx] / img_size
            
            # 收集预测和GT
            B = len(boxes_list)
            for b in range(B):
                # 预测（已归一化）
                all_predictions.append({
                    'boxes': boxes_list[b].cpu(),  # (N, 4) [x1, y1, x2, y2] 归一化
                    'scores': scores_list[b].cpu(),  # (N,)
                    'labels': labels_list[b].cpu()  # (N,)
                })
                
                # Ground Truth
                # 🔥 关键修复：GT已经是[x1, y1, x2, y2]归一化格式，无需转换！
                gt_boxes = targets_batch[b]['boxes']  # (M, 4) [x1, y1, x2, y2] 归一化 ✅
                gt_labels = targets_batch[b]['labels']  # (M,)
                
                # 直接使用GT，无需任何转换
                all_ground_truths.append({
                    'boxes': gt_boxes,  # (M, 4) [x1, y1, x2, y2] 归一化
                    'labels': gt_labels  # (M,)
                })
    
    # 计算平均损失
    avg_val_loss = total_val_loss / len(val_loader)
    for key in loss_components_val:
        loss_components_val[key] /= len(val_loader)
    
    # 计算mAP
    map50, map75, precision, recall = compute_map(
        all_predictions, all_ground_truths, 
        iou_thresholds=[0.5, 0.75],
        num_classes=config['model']['num_classes']
    )
    
    # 计算统计信息
    total_pred = sum(len(p['boxes']) for p in all_predictions)
    total_gt = sum(len(g['boxes']) for g in all_ground_truths)
    avg_pred_per_image = total_pred / len(all_predictions) if len(all_predictions) > 0 else 0
    avg_gt_per_image = total_gt / len(all_ground_truths) if len(all_ground_truths) > 0 else 0
    
    results = {
        # 真实mAP指标
        'map50': map50,
        'map75': map75,
        'map50_95': (map50 + map75) / 2,  # 简化的mAP50-95
        
        # 精度和召回率
        'precision': precision,
        'recall': recall,
        
        # 验证损失
        'val_loss': avg_val_loss,
        'val_cls_loss': loss_components_val['cls_loss'],
        'val_bbox_loss': loss_components_val['bbox_loss'],
        'val_centerness_loss': loss_components_val['centerness_loss'],
        'val_offset_loss': loss_components_val['offset_loss'],
        
        # 统计信息
        'avg_pred_per_image': avg_pred_per_image,
        'avg_gt_per_image': avg_gt_per_image,
        'total_samples': len(all_predictions)
    }
    
    return results


def compute_map(predictions, ground_truths, iou_thresholds=[0.5], num_classes=7):
    """
    计算mAP
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of dicts with 'boxes', 'labels'
        iou_thresholds: List of IoU thresholds
        num_classes: 类别数
    
    Returns:
        map50, map75, precision, recall
    """
    aps = []
    all_precisions = []
    all_recalls = []
    
    for iou_thresh in iou_thresholds:
        # 对每个类别计算AP
        class_aps = []
        for cls_id in range(num_classes):
            # 收集该类别的所有预测和GT
            cls_preds = []
            cls_gts = []
            
            for pred, gt in zip(predictions, ground_truths):
                # 筛选该类别的预测
                cls_mask = pred['labels'] == cls_id
                if cls_mask.sum() > 0:
                    cls_preds.append({
                        'boxes': pred['boxes'][cls_mask],
                        'scores': pred['scores'][cls_mask]
                    })
                else:
                    cls_preds.append({'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0)})
                
                # 筛选该类别的GT
                gt_cls_mask = gt['labels'] == cls_id
                if gt_cls_mask.sum() > 0:
                    cls_gts.append({'boxes': gt['boxes'][gt_cls_mask]})
                else:
                    cls_gts.append({'boxes': torch.zeros(0, 4)})
            
            # 计算该类别的AP
            ap, prec, rec = compute_class_ap(cls_preds, cls_gts, iou_thresh)
            class_aps.append(ap)
            all_precisions.append(prec)
            all_recalls.append(rec)
        
        # 平均所有类别的AP
        aps.append(np.mean(class_aps) if len(class_aps) > 0 else 0.0)
    
    # 返回mAP@50 and mAP@75
    map50 = aps[0] if len(aps) > 0 else 0.0
    map75 = aps[1] if len(aps) > 1 else 0.0
    
    # 平均precision和recall
    avg_precision = np.mean(all_precisions) if len(all_precisions) > 0 else 0.0
    avg_recall = np.mean(all_recalls) if len(all_recalls) > 0 else 0.0
    
    return map50, map75, avg_precision, avg_recall


def compute_class_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    计算单个类别的AP
    
    Returns:
        ap, precision, recall
    """
    # 收集所有预测框和分数
    all_boxes = []
    all_scores = []
    all_image_ids = []
    
    for img_id, pred in enumerate(predictions):
        if len(pred['boxes']) > 0:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
            all_image_ids.extend([img_id] * len(pred['boxes']))
    
    if len(all_boxes) == 0:
        return 0.0, 0.0, 0.0
    
    all_boxes = torch.cat(all_boxes, dim=0)  # (N, 4)
    all_scores = torch.cat(all_scores, dim=0)  # (N,)
    all_image_ids = torch.tensor(all_image_ids)
    
    # 按分数排序
    sorted_indices = torch.argsort(all_scores, descending=True)
    all_boxes = all_boxes[sorted_indices]
    all_scores = all_scores[sorted_indices]
    all_image_ids = all_image_ids[sorted_indices]
    
    # 统计GT数量
    num_gts = sum(len(gt['boxes']) for gt in ground_truths)
    
    if num_gts == 0:
        return 0.0, 0.0, 0.0
    
    # 匹配预测和GT
    tp = torch.zeros(len(all_boxes))
    fp = torch.zeros(len(all_boxes))
    
    # 记录每个GT是否已被匹配
    gt_matched = [torch.zeros(len(gt['boxes']), dtype=torch.bool) for gt in ground_truths]
    
    for i in range(len(all_boxes)):
        img_id = all_image_ids[i].item()
        pred_box = all_boxes[i:i+1]  # (1, 4)
        
        gt_boxes = ground_truths[img_id]['boxes']
        
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        
        # 计算IoU
        ious = compute_iou(pred_box, gt_boxes)  # (1, M)
        max_iou, max_idx = ious.max(dim=1)
        max_iou = max_iou.item()
        max_idx = max_idx.item()
        
        if max_iou >= iou_threshold:
            if not gt_matched[img_id][max_idx]:
                tp[i] = 1
                gt_matched[img_id][max_idx] = True
            else:
                fp[i] = 1  # 该GT已被匹配
        else:
            fp[i] = 1
    
    # 计算累积TP和FP
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    # 计算precision和recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / num_gts
    
    # 计算AP (使用11点插值)
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.sum() > 0:
            ap += precisions[mask].max().item()
    ap /= 11
    
    # 返回最终的precision和recall
    final_precision = precisions[-1].item() if len(precisions) > 0 else 0.0
    final_recall = recalls[-1].item() if len(recalls) > 0 else 0.0
    
    return ap, final_precision, final_recall


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }

    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved to {save_path}')


def find_best_checkpoint(checkpoint_dir):
    """
    自动查找检查点目录中的最佳模型
    
    Returns:
        best_checkpoint_path: str or None
    """
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path
    return None


def find_latest_checkpoint(checkpoint_dir):
    """
    自动查找检查点目录中的最新epoch模型
    
    Returns:
        latest_checkpoint_path: str or None
    """
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # 按epoch数字排序
    def extract_epoch(path):
        basename = os.path.basename(path)
        # epoch_50.pth -> 50
        epoch_str = basename.replace('epoch_', '').replace('.pth', '')
        try:
            return int(epoch_str)
        except:
            return -1
    
    checkpoint_files.sort(key=extract_epoch, reverse=True)
    return checkpoint_files[0] if checkpoint_files else None


def main():
    parser = argparse.ArgumentParser(description='Train OSFormer on RGBT-Tiny dataset')
    parser.add_argument('--config', type=str,
                       default='./configs/rgbt_tiny_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from. Special values: "best", "latest", "auto"')
    parser.add_argument('--resume_from_best', action='store_true',
                       help='Automatically resume from best_model.pth in checkpoint dir')
    parser.add_argument('--resume_from_latest', action='store_true',
                       help='Automatically resume from latest epoch checkpoint')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer and scheduler when resuming (useful for fine-tuning)')
    parser.add_argument('--reset_epochs', action='store_true',
                       help='Reset epoch counter to 0 when resuming (for fine-tuning)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (override config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (override config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (override config)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--amp', action='store_true',
                       help='Use mixed precision training')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.lr:
        config['train']['lr'] = args.lr
    if args.amp:
        config['train']['amp'] = True

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 设置日志
    logger = setup_logging(config['save']['log_dir'])
    logger.info(f'Starting training with config: {args.config}')

    # 构建数据集
    logger.info('Building datasets...')
    train_dataset, val_dataset = build_datasets(config)
    logger.info(f'Train dataset: {len(train_dataset)} samples')
    logger.info(f'Val dataset: {len(val_dataset)} samples')

    # 构建数据加载器
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, config)

    # 构建模型
    logger.info('Building model...')
    model = build_model(config)
    model = model.to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')

    # 构建优化器和调度器
    optimizer = build_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer, 
        warmup_epochs=config['train'].get('warmup_epochs', 5),
        total_epochs=config['train']['num_epochs'],
        steps_per_epoch=steps_per_epoch
    )

    # 混合精度训练
    if config['train']['amp']:
        try:
            from torch.amp import GradScaler as NewGradScaler
            scaler = NewGradScaler('cuda')
        except (ImportError, AttributeError):
            # Fallback for older PyTorch versions
            scaler = GradScaler()
    else:
        scaler = None

    # 创建 TensorBoard writer
    tensorboard_dir = config['save'].get('tensorboard_dir', './runs/rgbt_tiny')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f'TensorBoard logs will be saved to: {tensorboard_dir}')
    
    # 记录配置信息到TensorBoard
    writer.add_text('config/loss_function', '✅ CIoU Loss (upgraded from L1)', 0)
    writer.add_text('config/loss_weights', str(config['train']['loss_weights']), 0)
    writer.add_text('config/model', f"Doppler={config['model']['use_doppler']}, Classes={config['model']['num_classes']}", 0)

    # 初始化早退机制
    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['early_stopping'].get('patience', 15),
            min_delta=config['early_stopping'].get('min_delta', 0.001),
            mode=config['early_stopping'].get('mode', 'max')
        )
        logger.info(f'Early stopping enabled with patience={early_stopping.patience}')

    # 恢复训练
    start_epoch = 0
    best_metric_name = config['save'].get('best_metric', 'map50')
    best_epoch = 0
    
    # 初始化 best_metric (根据指标类型决定初始值)
    if best_metric_name in ['loss', 'val_loss']:
        best_metric = float('inf')  # Loss类指标越小越好
        best_mode = 'min'
    else:
        best_metric = 0.0  # mAP类指标越大越好
        best_mode = 'max'
    
    logger.info(f'Model selection metric: {best_metric_name} (mode: {best_mode})')

    # ==================== 智能恢复训练 ====================
    resume_path = None
    
    # 确定要恢复的checkpoint路径
    if args.resume_from_best:
        resume_path = find_best_checkpoint(config['save']['checkpoint_dir'])
        if resume_path:
            logger.info(f'🔍 Found best model: {resume_path}')
        else:
            logger.warning('❌ No best_model.pth found in checkpoint directory')
    
    elif args.resume_from_latest:
        resume_path = find_latest_checkpoint(config['save']['checkpoint_dir'])
        if resume_path:
            logger.info(f'🔍 Found latest checkpoint: {resume_path}')
        else:
            logger.warning('❌ No epoch checkpoints found in checkpoint directory')
    
    elif args.resume:
        # 处理特殊值
        if args.resume.lower() == 'best':
            resume_path = find_best_checkpoint(config['save']['checkpoint_dir'])
            if not resume_path:
                logger.warning('❌ No best_model.pth found, will train from scratch')
        elif args.resume.lower() == 'latest':
            resume_path = find_latest_checkpoint(config['save']['checkpoint_dir'])
            if not resume_path:
                logger.warning('❌ No epoch checkpoints found, will train from scratch')
        elif args.resume.lower() == 'auto':
            # 自动选择：优先best，其次latest
            resume_path = find_best_checkpoint(config['save']['checkpoint_dir'])
            if not resume_path:
                resume_path = find_latest_checkpoint(config['save']['checkpoint_dir'])
            if resume_path:
                logger.info(f'🔍 Auto-selected checkpoint: {resume_path}')
            else:
                logger.warning('❌ No checkpoints found, will train from scratch')
        else:
            # 直接指定的路径
            resume_path = args.resume
    
    # 加载checkpoint
    if resume_path and os.path.exists(resume_path):
        logger.info('=' * 60)
        logger.info('📥 LOADING CHECKPOINT')
        logger.info('=' * 60)
        
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info('✅ Model weights loaded')
            
            # 加载优化器和调度器（除非指定reset）
            if not args.reset_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info('✅ Optimizer and scheduler loaded')
            else:
                logger.info('⚠️  Optimizer and scheduler reset (fine-tuning mode)')
            
            # 加载epoch信息（除非指定reset）
            if not args.reset_epochs:
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f'✅ Will resume from epoch {start_epoch}')
            else:
                start_epoch = 0
                logger.info('⚠️  Epoch counter reset to 0 (fine-tuning mode)')
            
            # 加载最佳指标信息
            if 'loss' in checkpoint and isinstance(checkpoint['loss'], dict):
                loss_info = checkpoint['loss']
                best_metric = loss_info.get('best_metric', best_metric)
                best_epoch = loss_info.get('best_epoch', 0)
                saved_metric_name = loss_info.get('best_metric_name', best_metric_name)
                
                if saved_metric_name == best_metric_name:
                    logger.info(f'✅ Best metric loaded: {best_metric_name}={best_metric:.4f} (epoch {best_epoch})')
                else:
                    logger.warning(f'⚠️  Metric name mismatch: saved={saved_metric_name}, current={best_metric_name}')
                    logger.warning(f'   Will use saved best_metric value but may not be comparable')
            
            # 显示checkpoint详细信息
            logger.info('📊 Checkpoint Info:')
            logger.info(f'  Checkpoint file:  {resume_path}')
            logger.info(f'  Saved epoch:      {checkpoint.get("epoch", "N/A")}')
            logger.info(f'  Resume from:      Epoch {start_epoch}')
            logger.info(f'  Best metric:      {best_metric_name}={best_metric:.4f} (Epoch {best_epoch})')
            
            if 'loss' in checkpoint and isinstance(checkpoint['loss'], dict):
                if 'val_results' in checkpoint['loss']:
                    val_res = checkpoint['loss']['val_results']
                    logger.info(f'  Last val mAP50:   {val_res.get("map50", 0):.4f}')
                    logger.info(f'  Last val loss:    {val_res.get("val_loss", 0):.4f}')
            
            logger.info('=' * 60)
            
        except Exception as e:
            logger.error(f'❌ Failed to load checkpoint: {e}')
            logger.warning('Will train from scratch')
            import traceback
            traceback.print_exc()
    
    elif resume_path:
        logger.warning(f'❌ Checkpoint not found: {resume_path}')
        logger.warning('Will train from scratch')

    # 创建保存目录
    os.makedirs(config['save']['checkpoint_dir'], exist_ok=True)

    # 训练循环
    logger.info('Starting training...')

    for epoch in range(start_epoch, config['train']['num_epochs']):
        logger.info(f'Epoch {epoch}/{config["train"]["num_epochs"]}')

        # 训练
        train_loss, loss_components = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, device, config, epoch, writer
        )

        # 🔥 修复：更清晰的损失打印格式
        logger.info(f'📊 Train Loss: {train_loss:.4f}')
        logger.info(f'   ├─ Total Loss:      {loss_components["loss"]:.4f}')
        logger.info(f'   ├─ Cls Loss:        {loss_components["cls_loss"]:.6f}')
        logger.info(f'   ├─ BBox Loss:       {loss_components["bbox_loss"]:.4f}')
        logger.info(f'   ├─ Centerness Loss: {loss_components["centerness_loss"]:.4f}')
        logger.info(f'   └─ Offset Loss:     {loss_components["offset_loss"]:.6f}')

        # 记录训练损失到 TensorBoard
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        for key, value in loss_components.items():
            writer.add_scalar(f'epoch/train_{key}', value, epoch)

        # 可视化训练样本（每个epoch）
        if config.get('visualize', {}).get('enabled', True):
            try:
                logger.info('生成可视化结果...')
                visualize_epoch_results(
                    model, train_loader, device, config, epoch, writer, 'train'
                )
            except Exception as e:
                logger.warning(f'可视化失败: {e}')

        # ==================== 评估阶段 ====================
        val_results = None
        if (epoch + 1) % config['eval']['eval_interval'] == 0:
            logger.info('=' * 60)
            logger.info('🔍 Starting Evaluation...')
            
            val_results = evaluate(
                model, val_loader, device, config,
                config['classes']['names']
            )

            # 🔥 改进：更清晰突出的验证结果打印
            logger.info('=' * 60)
            logger.info('📊 VALIDATION RESULTS 📊')
            logger.info('=' * 60)
            logger.info('🎯 Detection Metrics:')
            logger.info(f'   ├─ mAP@50:     {val_results["map50"]:.4f} ⭐⭐⭐')
            logger.info(f'   ├─ mAP@75:     {val_results["map75"]:.4f}')
            logger.info(f'   ├─ mAP@50-95:  {val_results["map50_95"]:.4f}')
            logger.info(f'   ├─ Precision:  {val_results["precision"]:.4f}')
            logger.info(f'   └─ Recall:     {val_results["recall"]:.4f}')
            logger.info('')
            logger.info('📉 Validation Loss:')
            logger.info(f'   ├─ Total:      {val_results["val_loss"]:.4f}')
            logger.info(f'   ├─ Cls:        {val_results["val_cls_loss"]:.6f}')
            logger.info(f'   ├─ BBox:       {val_results["val_bbox_loss"]:.4f}')
            logger.info(f'   ├─ Centerness: {val_results["val_centerness_loss"]:.4f}')
            logger.info(f'   └─ Offset:     {val_results["val_offset_loss"]:.6f}')
            logger.info('')
            logger.info('📈 Statistics:')
            logger.info(f'   ├─ Avg Pred/Img: {val_results["avg_pred_per_image"]:.2f}')
            logger.info(f'   └─ Avg GT/Img:   {val_results["avg_gt_per_image"]:.2f}')
            logger.info('=' * 60)

            # 🔥 改进：记录所有评估指标到 TensorBoard，按类别分组
            # 1. mAP指标（重点突出）
            writer.add_scalar('mAP/mAP@50', val_results['map50'], epoch)
            writer.add_scalar('mAP/mAP@75', val_results['map75'], epoch)
            writer.add_scalar('mAP/mAP@50-95', val_results['map50_95'], epoch)
            
            # 2. Precision & Recall
            writer.add_scalar('metrics/Precision', val_results['precision'], epoch)
            writer.add_scalar('metrics/Recall', val_results['recall'], epoch)
            
            # 3. Validation Loss
            writer.add_scalar('val_loss/total', val_results['val_loss'], epoch)
            writer.add_scalar('val_loss/cls', val_results['val_cls_loss'], epoch)
            writer.add_scalar('val_loss/bbox', val_results['val_bbox_loss'], epoch)
            writer.add_scalar('val_loss/centerness', val_results['val_centerness_loss'], epoch)
            writer.add_scalar('val_loss/offset', val_results['val_offset_loss'], epoch)
            
            # 4. Statistics
            writer.add_scalar('stats/avg_pred_per_image', val_results['avg_pred_per_image'], epoch)
            writer.add_scalar('stats/avg_gt_per_image', val_results['avg_gt_per_image'], epoch)
            
            # 5. 保持原有的通用记录（兼容性）
            for key, value in val_results.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # 可视化验证样本
            if config.get('visualize', {}).get('enabled', True):
                try:
                    visualize_epoch_results(
                        model, val_loader, device, config, epoch, writer, 'val'
                    )
                except Exception as e:
                    logger.warning(f'验证集可视化失败: {e}')

            # ==================== 最佳模型保存 ====================
            current_metric = val_results.get(best_metric_name, 0.0)
            
            # 判断是否为最佳模型
            is_best = False
            if best_mode == 'min':
                is_best = current_metric < best_metric
            else:  # 'max'
                is_best = current_metric > best_metric
            
            if is_best and config['save'].get('save_best', True):
                old_best = best_metric
                best_metric = current_metric
                best_epoch = epoch
                
                best_checkpoint_path = os.path.join(
                    config['save']['checkpoint_dir'], 'best_model.pth'
                )
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {
                        'val_results': val_results,
                        'best_metric': best_metric,
                        'best_epoch': best_epoch,
                        'best_metric_name': best_metric_name
                    },
                    config, best_checkpoint_path
                )
                
                logger.info('=' * 60)
                logger.info('✨ NEW BEST MODEL SAVED!')
                logger.info(f'  Metric:       {best_metric_name}')
                logger.info(f'  Previous:     {old_best:.4f}')
                logger.info(f'  Current:      {best_metric:.4f} ⬆️')
                logger.info(f'  Improvement:  +{(best_metric - old_best):.4f}')
                logger.info(f'  Saved to:     {best_checkpoint_path}')
                logger.info('=' * 60)
                
                writer.add_text(
                    'best_model', 
                    f'Epoch {epoch}: {best_metric_name}={best_metric:.4f} (improved by {(best_metric - old_best):.4f})', 
                    epoch
                )
                writer.add_scalar('epoch/best_metric', best_metric, epoch)
            else:
                logger.info(f'  Current {best_metric_name}: {current_metric:.4f}')
                logger.info(f'  Best {best_metric_name}:    {best_metric:.4f} (Epoch {best_epoch})')
                logger.info(f'  No improvement.')
            
            # ==================== 早停检查 ====================
            if early_stopping is not None:
                # 使用与模型保存相同的指标进行早停判断
                should_stop = early_stopping(current_metric)
                
                if should_stop:
                    logger.info('=' * 60)
                    logger.info('🛑 EARLY STOPPING TRIGGERED')
                    logger.info(f'  Reason:       No improvement for {early_stopping.patience} epochs')
                    logger.info(f'  Stopped at:   Epoch {epoch}')
                    logger.info(f'  Best metric:  {best_metric_name}={best_metric:.4f} (Epoch {best_epoch})')
                    logger.info(f'  Total epochs: {epoch - start_epoch + 1}/{config["train"]["num_epochs"]}')
                    logger.info('=' * 60)
                    
                    writer.add_text(
                        'training', 
                        f'Early stopped at epoch {epoch}. Best {best_metric_name}: {best_metric:.4f} at epoch {best_epoch}', 
                        epoch
                    )
                    break
                else:
                    patience_left = early_stopping.patience - early_stopping.counter
                    logger.info(f'  Early stopping: {patience_left} epochs left before stopping')
            
            logger.info('=' * 60)

        # 定期保存检查点
        if (epoch + 1) % config['save']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['save']['checkpoint_dir'], f'epoch_{epoch}.pth'
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train_loss': train_loss, 'loss_components': loss_components},
                config, checkpoint_path
            )
            logger.info(f'💾 Checkpoint saved: epoch_{epoch}.pth')

    # ==================== 训练结束 ====================
    writer.close()
    
    logger.info('=' * 60)
    logger.info('🎉 TRAINING COMPLETED!')
    logger.info('=' * 60)
    logger.info('📊 Final Statistics:')
    logger.info(f'  Total Epochs:     {epoch - start_epoch + 1}/{config["train"]["num_epochs"]}')
    logger.info(f'  Best Metric:      {best_metric_name}')
    logger.info(f'  Best Value:       {best_metric:.4f}')
    logger.info(f'  Best Epoch:       {best_epoch}')
    logger.info(f'  Final Train Loss: {train_loss:.4f}')
    if val_results:
        logger.info(f'  Final Val Loss:   {val_results["val_loss"]:.4f}')
    logger.info('=' * 60)
    logger.info('📁 Saved Files:')
    logger.info(f'  Best Model:       {os.path.join(config["save"]["checkpoint_dir"], "best_model.pth")}')
    logger.info(f'  Checkpoints:      {config["save"]["checkpoint_dir"]}')
    logger.info(f'  Logs:             {config["save"]["log_dir"]}')
    logger.info(f'  TensorBoard:      {tensorboard_dir}')
    logger.info('=' * 60)
    logger.info('🚀 Next Steps:')
    logger.info(f'  1. View TensorBoard: tensorboard --logdir={tensorboard_dir}')
    logger.info(f'  2. Test best model:  python test_rgbb_tiny.py --checkpoint {os.path.join(config["save"]["checkpoint_dir"], "best_model.pth")}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()