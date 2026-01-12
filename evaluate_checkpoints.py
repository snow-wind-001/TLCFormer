#!/usr/bin/env python3
"""
评估检查点脚本
比较最佳权重和最新权重的识别效果，并生成详细的分析报告
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.osformer import OSFormer
from datasets.rgbt_tiny_coco import RGBTTinyCOCODataset, collate_fn
from torch.utils.data import DataLoader
from utils.visualize import visualize_detection_results, create_visualization


def setup_logger(log_file):
    """设置日志"""
    logger = logging.getLogger('eval')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def compute_iou(boxes1, boxes2):
    """
    计算两组框的IoU
    boxes1: (N, 4) [x1, y1, x2, y2]
    boxes2: (M, 4) [x1, y1, x2, y2]
    返回: (N, M) IoU矩阵
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), len(boxes2))
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def decode_predictions(predictions, img_size, feature_size, score_thresh=0.3):
    """
    解码预测结果为边界框
    
    Returns:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        labels: (N,)
    """
    cls_pred = predictions['cls']  # (B, num_classes, H, W)
    bbox_pred = predictions['bbox']  # (B, 4, H, W)
    centerness_pred = predictions.get('centerness', None)  # (B, 1, H, W)
    
    B, C, H, W = cls_pred.shape
    device = cls_pred.device
    stride = img_size / feature_size
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for b in range(B):
        cls_b = cls_pred[b]  # (num_classes, H, W)
        bbox_b = bbox_pred[b]  # (4, H, W)
        
        scores_b = torch.sigmoid(cls_b)  # (num_classes, H, W)
        
        if centerness_pred is not None:
            centerness_b = torch.sigmoid(centerness_pred[b, 0])  # (H, W)
            scores_b = scores_b * centerness_b.unsqueeze(0)
        
        max_scores, _ = scores_b.max(dim=0)  # (H, W)
        max_labels = scores_b.argmax(dim=0)  # (H, W)
        
        mask = max_scores > score_thresh
        
        if mask.sum() == 0:
            all_boxes.append(torch.zeros(0, 4, device=device))
            all_scores.append(torch.zeros(0, device=device))
            all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
            continue
        
        valid_indices = mask.nonzero(as_tuple=False)  # (N, 2) [h, w]
        valid_scores = max_scores[mask]
        valid_labels = max_labels[mask]
        
        valid_bbox = bbox_pred[b, :, mask].t()  # (N, 4)
        
        h_idx = valid_indices[:, 0].float()
        w_idx = valid_indices[:, 1].float()
        
        center_x = (w_idx + 0.5) * stride
        center_y = (h_idx + 0.5) * stride
        
        l, t, r, b = valid_bbox[:, 0], valid_bbox[:, 1], valid_bbox[:, 2], valid_bbox[:, 3]
        x1 = (center_x - l * stride).clamp(min=0, max=img_size)
        y1 = (center_y - t * stride).clamp(min=0, max=img_size)
        x2 = (center_x + r * stride).clamp(min=0, max=img_size)
        y2 = (center_y + b * stride).clamp(min=0, max=img_size)
        
        boxes_b = torch.stack([x1, y1, x2, y2], dim=1)
        
        all_boxes.append(boxes_b)
        all_scores.append(valid_scores)
        all_labels.append(valid_labels)
    
    return all_boxes, all_scores, all_labels


def calculate_ap(recalls, precisions):
    """计算 AP（使用 11-point 插值）"""
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_model(model, dataloader, device, config, class_names, score_thresh=0.3, logger=None):
    """
    详细评估模型
    
    Returns:
        results: dict with detailed metrics
    """
    model.eval()
    
    img_size = config['model']['img_size']
    feature_size = img_size // 16
    
    # 存储所有预测和GT
    all_predictions = defaultdict(list)  # class_id -> list of (score, matched)
    all_ground_truths = defaultdict(int)  # class_id -> count
    
    # 统计信息
    stats = {
        'total_images': 0,
        'total_predictions': 0,
        'total_ground_truths': 0,
        'predictions_per_image': [],
        'ground_truths_per_image': [],
        'box_sizes': [],  # 记录检测框大小
        'iou_scores': [],  # 记录IoU分数
    }
    
    # 按类别统计
    class_stats = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0,
        'predictions': 0, 'ground_truths': 0,
        'avg_iou': []
    })
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            targets_batch = batch['targets']
            
            # 前向传播（取中间帧）
            predictions = model(rgb, thermal)
            mid_frame = len(predictions) // 2
            pred = predictions[mid_frame]
            
            # 解码预测
            pred_boxes_list, pred_scores_list, pred_labels_list = decode_predictions(
                pred, img_size, feature_size, score_thresh
            )
            
            # 处理每个样本
            for b in range(len(targets_batch)):
                target = targets_batch[b]
                pred_boxes = pred_boxes_list[b].cpu()
                pred_scores = pred_scores_list[b].cpu()
                pred_labels = pred_labels_list[b].cpu()
                
                # GT boxes
                if len(target['boxes']) > 0:
                    gt_boxes = torch.tensor(target['boxes'], dtype=torch.float32)
                    gt_labels = torch.tensor(target['labels'], dtype=torch.long)
                    
                    # 转换归一化坐标到绝对坐标
                    gt_boxes[:, [0, 2]] *= img_size
                    gt_boxes[:, [1, 3]] *= img_size
                else:
                    gt_boxes = torch.zeros(0, 4)
                    gt_labels = torch.zeros(0, dtype=torch.long)
                
                stats['total_images'] += 1
                stats['total_predictions'] += len(pred_boxes)
                stats['total_ground_truths'] += len(gt_boxes)
                stats['predictions_per_image'].append(len(pred_boxes))
                stats['ground_truths_per_image'].append(len(gt_boxes))
                
                # 记录框大小
                if len(pred_boxes) > 0:
                    box_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
                    stats['box_sizes'].extend(box_areas.tolist())
                
                # 统计每个GT标签
                for label in gt_labels:
                    all_ground_truths[label.item()] += 1
                    class_stats[label.item()]['ground_truths'] += 1
                
                # 统计每个预测
                for label in pred_labels:
                    class_stats[label.item()]['predictions'] += 1
                
                # 如果没有预测或GT，跳过匹配
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    # 所有GT都是FN
                    for label in gt_labels:
                        class_stats[label.item()]['fn'] += 1
                    # 所有预测都是FP
                    for score, label in zip(pred_scores, pred_labels):
                        all_predictions[label.item()].append((score.item(), False))
                        class_stats[label.item()]['fp'] += 1
                    continue
                
                # 计算IoU矩阵
                iou_matrix = compute_iou(pred_boxes, gt_boxes)
                
                # 对每个类别进行匹配
                for class_id in range(len(class_names)):
                    # 该类别的预测
                    pred_mask = pred_labels == class_id
                    pred_class_indices = pred_mask.nonzero(as_tuple=True)[0]
                    
                    # 该类别的GT
                    gt_mask = gt_labels == class_id
                    gt_class_indices = gt_mask.nonzero(as_tuple=True)[0]
                    
                    if len(pred_class_indices) == 0:
                        # 没有预测，GT都是FN
                        class_stats[class_id]['fn'] += len(gt_class_indices)
                        continue
                    
                    if len(gt_class_indices) == 0:
                        # 没有GT，预测都是FP
                        for idx in pred_class_indices:
                            all_predictions[class_id].append((pred_scores[idx].item(), False))
                            class_stats[class_id]['fp'] += 1
                        continue
                    
                    # 提取该类别的IoU子矩阵
                    iou_sub = iou_matrix[pred_class_indices][:, gt_class_indices]
                    
                    # 贪婪匹配（IoU > 0.5）
                    matched_gt = set()
                    pred_indices_sorted = torch.argsort(pred_scores[pred_class_indices], descending=True)
                    
                    for i in pred_indices_sorted:
                        pred_idx = pred_class_indices[i]
                        score = pred_scores[pred_idx].item()
                        
                        # 找到最大IoU的GT
                        ious = iou_sub[i]
                        max_iou, max_gt_idx = ious.max(dim=0)
                        
                        if max_iou >= 0.5 and max_gt_idx.item() not in matched_gt:
                            # TP
                            all_predictions[class_id].append((score, True))
                            matched_gt.add(max_gt_idx.item())
                            class_stats[class_id]['tp'] += 1
                            class_stats[class_id]['avg_iou'].append(max_iou.item())
                            stats['iou_scores'].append(max_iou.item())
                        else:
                            # FP
                            all_predictions[class_id].append((score, False))
                            class_stats[class_id]['fp'] += 1
                    
                    # 未匹配的GT是FN
                    fn_count = len(gt_class_indices) - len(matched_gt)
                    class_stats[class_id]['fn'] += fn_count
    
    # 计算 AP 和其他指标
    aps = {}
    for class_id in range(len(class_names)):
        predictions = all_predictions[class_id]
        n_gt = all_ground_truths[class_id]
        
        if n_gt == 0:
            aps[class_id] = 0.0
            continue
        
        if len(predictions) == 0:
            aps[class_id] = 0.0
            continue
        
        # 按分数排序
        predictions.sort(key=lambda x: x[0], reverse=True)
        
        # 计算累积TP和FP
        tp = np.array([1 if matched else 0 for _, matched in predictions])
        fp = np.array([0 if matched else 1 for _, matched in predictions])
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 计算AP
        ap = calculate_ap(recalls, precisions)
        aps[class_id] = ap
    
    # 计算 mAP
    valid_aps = [ap for ap in aps.values() if ap > 0]
    mAP = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0
    
    # 计算全局Precision和Recall
    total_tp = sum(class_stats[i]['tp'] for i in range(len(class_names)))
    total_fp = sum(class_stats[i]['fp'] for i in range(len(class_names)))
    total_fn = sum(class_stats[i]['fn'] for i in range(len(class_names)))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算平均IoU
    avg_iou = np.mean(stats['iou_scores']) if len(stats['iou_scores']) > 0 else 0.0
    
    # 组织结果
    results = {
        'mAP': mAP,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'aps': aps,
        'class_stats': dict(class_stats),
        'stats': stats,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }
    
    return results


def visualize_comparison(results_best, results_latest, class_names, output_dir):
    """可视化对比结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. mAP 对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 总体指标对比
    ax = axes[0, 0]
    metrics = ['mAP', 'Precision', 'Recall', 'F1', 'Avg IoU']
    best_values = [results_best['mAP'], results_best['precision'], results_best['recall'], 
                   results_best['f1'], results_best['avg_iou']]
    latest_values = [results_latest['mAP'], results_latest['precision'], results_latest['recall'],
                     results_latest['f1'], results_latest['avg_iou']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, best_values, width, label='Best Checkpoint', alpha=0.8)
    ax.bar(x + width/2, latest_values, width, label='Latest Checkpoint', alpha=0.8)
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 每类别AP对比
    ax = axes[0, 1]
    class_ids = list(range(len(class_names)))
    best_aps = [results_best['aps'].get(i, 0) for i in class_ids]
    latest_aps = [results_latest['aps'].get(i, 0) for i in class_ids]
    
    x = np.arange(len(class_names))
    ax.bar(x - width/2, best_aps, width, label='Best Checkpoint', alpha=0.8)
    ax.bar(x + width/2, latest_aps, width, label='Latest Checkpoint', alpha=0.8)
    ax.set_ylabel('AP')
    ax.set_title('Per-Class AP Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TP/FP/FN 对比 (Best)
    ax = axes[1, 0]
    best_tp = [results_best['class_stats'][i]['tp'] for i in class_ids]
    best_fp = [results_best['class_stats'][i]['fp'] for i in class_ids]
    best_fn = [results_best['class_stats'][i]['fn'] for i in class_ids]
    
    x = np.arange(len(class_names))
    ax.bar(x - width, best_tp, width, label='TP', alpha=0.8)
    ax.bar(x, best_fp, width, label='FP', alpha=0.8)
    ax.bar(x + width, best_fn, width, label='FN', alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Best Checkpoint: TP/FP/FN per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TP/FP/FN 对比 (Latest)
    ax = axes[1, 1]
    latest_tp = [results_latest['class_stats'][i]['tp'] for i in class_ids]
    latest_fp = [results_latest['class_stats'][i]['fp'] for i in class_ids]
    latest_fn = [results_latest['class_stats'][i]['fn'] for i in class_ids]
    
    ax.bar(x - width, latest_tp, width, label='TP', alpha=0.8)
    ax.bar(x, latest_fp, width, label='FP', alpha=0.8)
    ax.bar(x + width, latest_fn, width, label='FN', alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Latest Checkpoint: TP/FP/FN per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_charts.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比图表已保存到: {output_dir / 'comparison_charts.png'}")


def generate_report(results_best, results_latest, class_names, output_file, logger):
    """生成详细的文本报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("检查点评估详细报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体对比
        f.write("### 总体指标对比\n\n")
        f.write(f"{'Metric':<20} {'Best Checkpoint':<20} {'Latest Checkpoint':<20} {'Difference':<15}\n")
        f.write("-" * 75 + "\n")
        
        metrics = [
            ('mAP', 'mAP'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1 Score', 'f1'),
            ('Avg IoU', 'avg_iou')
        ]
        
        for name, key in metrics:
            best_val = results_best[key]
            latest_val = results_latest[key]
            diff = latest_val - best_val
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            f.write(f"{name:<20} {best_val:<20.4f} {latest_val:<20.4f} {diff_str:<15}\n")
        
        f.write("\n")
        
        # 检测统计
        f.write("### 检测统计\n\n")
        f.write(f"{'Statistic':<30} {'Best':<15} {'Latest':<15}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Total TP':<30} {results_best['total_tp']:<15} {results_latest['total_tp']:<15}\n")
        f.write(f"{'Total FP':<30} {results_best['total_fp']:<15} {results_latest['total_fp']:<15}\n")
        f.write(f"{'Total FN':<30} {results_best['total_fn']:<15} {results_latest['total_fn']:<15}\n")
        f.write(f"{'Total Images':<30} {results_best['stats']['total_images']:<15} {results_latest['stats']['total_images']:<15}\n")
        f.write(f"{'Total Predictions':<30} {results_best['stats']['total_predictions']:<15} {results_latest['stats']['total_predictions']:<15}\n")
        f.write(f"{'Total Ground Truths':<30} {results_best['stats']['total_ground_truths']:<15} {results_latest['stats']['total_ground_truths']:<15}\n")
        f.write(f"{'Avg Pred/Image':<30} {np.mean(results_best['stats']['predictions_per_image']):<15.2f} {np.mean(results_latest['stats']['predictions_per_image']):<15.2f}\n")
        f.write(f"{'Avg GT/Image':<30} {np.mean(results_best['stats']['ground_truths_per_image']):<15.2f} {np.mean(results_latest['stats']['ground_truths_per_image']):<15.2f}\n")
        
        f.write("\n")
        
        # 每类别详细统计
        f.write("### 每类别详细统计\n\n")
        for class_id, class_name in enumerate(class_names):
            f.write(f"#### 类别 {class_id}: {class_name}\n\n")
            
            best_stats = results_best['class_stats'][class_id]
            latest_stats = results_latest['class_stats'][class_id]
            best_ap = results_best['aps'].get(class_id, 0)
            latest_ap = results_latest['aps'].get(class_id, 0)
            
            f.write(f"{'Metric':<25} {'Best':<15} {'Latest':<15}\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'AP':<25} {best_ap:<15.4f} {latest_ap:<15.4f}\n")
            f.write(f"{'TP':<25} {best_stats['tp']:<15} {latest_stats['tp']:<15}\n")
            f.write(f"{'FP':<25} {best_stats['fp']:<15} {latest_stats['fp']:<15}\n")
            f.write(f"{'FN':<25} {best_stats['fn']:<15} {latest_stats['fn']:<15}\n")
            f.write(f"{'Predictions':<25} {best_stats['predictions']:<15} {latest_stats['predictions']:<15}\n")
            f.write(f"{'Ground Truths':<25} {best_stats['ground_truths']:<15} {latest_stats['ground_truths']:<15}\n")
            
            if len(best_stats['avg_iou']) > 0:
                f.write(f"{'Avg IoU (matched)':<25} {np.mean(best_stats['avg_iou']):<15.4f} ", end='')
            else:
                f.write(f"{'Avg IoU (matched)':<25} {'N/A':<15} ", end='')
            
            if len(latest_stats['avg_iou']) > 0:
                f.write(f"{np.mean(latest_stats['avg_iou']):<15.4f}\n")
            else:
                f.write(f"{'N/A':<15}\n")
            
            f.write("\n")
        
        # 问题分析
        f.write("="*80 + "\n")
        f.write("### 问题分析\n")
        f.write("="*80 + "\n\n")
        
        # 1. mAP 低的原因
        f.write("#### 1. mAP 较低的可能原因：\n\n")
        
        best_mAP = results_best['mAP']
        best_precision = results_best['precision']
        best_recall = results_best['recall']
        
        if best_mAP < 0.3:
            f.write(f"⚠️ mAP = {best_mAP:.4f} 确实较低，主要问题可能包括：\n\n")
            
            if best_recall < 0.3:
                f.write(f"1. **召回率过低** (Recall={best_recall:.4f}):\n")
                f.write(f"   - FN 数量: {results_best['total_fn']} (漏检)\n")
                f.write(f"   - 模型可能过于保守，置信度阈值太高\n")
                f.write(f"   - 建议: 降低 score_thresh (当前 0.3 → 尝试 0.1-0.2)\n\n")
            
            if best_precision < 0.3:
                f.write(f"2. **精确度过低** (Precision={best_precision:.4f}):\n")
                f.write(f"   - FP 数量: {results_best['total_fp']} (误检)\n")
                f.write(f"   - 模型产生太多低质量检测\n")
                f.write(f"   - 建议: 提高 score_thresh 或增加训练\n\n")
            
            # 检查每类别AP
            low_ap_classes = [(i, name, results_best['aps'].get(i, 0)) 
                             for i, name in enumerate(class_names) 
                             if results_best['aps'].get(i, 0) < 0.2]
            
            if low_ap_classes:
                f.write(f"3. **某些类别表现特别差**:\n")
                for class_id, class_name, ap in low_ap_classes:
                    stats = results_best['class_stats'][class_id]
                    f.write(f"   - {class_name}: AP={ap:.4f}, ")
                    f.write(f"TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}\n")
                f.write(f"   - 建议: 针对这些类别增加训练数据或调整损失权重\n\n")
            
            # 检查预测数量
            avg_pred = np.mean(results_best['stats']['predictions_per_image'])
            avg_gt = np.mean(results_best['stats']['ground_truths_per_image'])
            
            if avg_pred < avg_gt * 0.5:
                f.write(f"4. **预测数量过少**:\n")
                f.write(f"   - 平均预测/图像: {avg_pred:.2f}\n")
                f.write(f"   - 平均GT/图像: {avg_gt:.2f}\n")
                f.write(f"   - 模型产生的候选框太少\n")
                f.write(f"   - 建议: 降低置信度阈值或检查模型训练\n\n")
            
            elif avg_pred > avg_gt * 2:
                f.write(f"4. **预测数量过多**:\n")
                f.write(f"   - 平均预测/图像: {avg_pred:.2f}\n")
                f.write(f"   - 平均GT/图像: {avg_gt:.2f}\n")
                f.write(f"   - 模型产生过多低质量检测\n")
                f.write(f"   - 建议: 提高置信度阈值或加强训练\n\n")
        
        # 2. 检查点对比分析
        f.write("\n#### 2. 最佳权重 vs 最新权重：\n\n")
        
        mAP_diff = results_latest['mAP'] - results_best['mAP']
        
        if abs(mAP_diff) < 0.01:
            f.write("✅ 两个检查点性能几乎相同\n")
            f.write("   - 训练可能已经收敛\n")
            f.write("   - 或者都处于较差的局部最优\n\n")
        elif mAP_diff > 0.01:
            f.write(f"✅ 最新权重更好 (mAP 提升 {mAP_diff:.4f})\n")
            f.write("   - 建议使用最新权重\n\n")
        else:
            f.write(f"⚠️ 最佳权重更好 (mAP 下降 {abs(mAP_diff):.4f})\n")
            f.write("   - 训练可能过拟合或不稳定\n")
            f.write("   - 建议使用最佳权重或重新训练\n\n")
        
        # 3. 改进建议
        f.write("\n#### 3. 改进建议：\n\n")
        
        f.write("**短期建议（调整推理参数）**:\n")
        f.write("1. 调整置信度阈值:\n")
        f.write("   - 当前: 0.3\n")
        if best_recall < 0.3:
            f.write("   - 建议尝试: 0.1, 0.15, 0.2 (提高召回率)\n")
        elif best_precision < 0.3:
            f.write("   - 建议尝试: 0.4, 0.5 (提高精确度)\n")
        else:
            f.write("   - 建议尝试: 0.2-0.4 范围内调整\n")
        f.write("\n")
        
        f.write("2. 添加 NMS (非极大值抑制):\n")
        f.write("   - 减少重复检测\n")
        f.write("   - 建议 NMS 阈值: 0.5\n\n")
        
        f.write("**中期建议（重新训练）**:\n")
        if best_mAP < 0.15:
            f.write("1. 检查训练是否正常:\n")
            f.write("   - 查看损失曲线是否下降\n")
            f.write("   - 检查是否有 NaN 或异常值\n")
            f.write("   - 验证数据加载是否正确\n\n")
        
        f.write("2. 调整训练超参数:\n")
        f.write("   - 降低学习率 (当前 1e-4 → 5e-5)\n")
        f.write("   - 增加训练轮数\n")
        f.write("   - 调整损失权重\n\n")
        
        f.write("3. 数据增强:\n")
        f.write("   - 增加数据增强强度\n")
        f.write("   - 添加针对小目标的数据增强\n\n")
        
        f.write("**长期建议（模型改进）**:\n")
        f.write("1. 检查模型架构是否适合数据集\n")
        f.write("2. 考虑预训练权重\n")
        f.write("3. 分析失败案例，针对性改进\n\n")
    
    logger.info(f"✅ 详细报告已保存到: {output_file}")
    
    # 打印到控制台
    print("\n" + "="*80)
    print("快速总结")
    print("="*80)
    print(f"\n最佳检查点:")
    print(f"  mAP: {results_best['mAP']:.4f}")
    print(f"  Precision: {results_best['precision']:.4f}")
    print(f"  Recall: {results_best['recall']:.4f}")
    print(f"  F1: {results_best['f1']:.4f}")
    
    print(f"\n最新检查点:")
    print(f"  mAP: {results_latest['mAP']:.4f}")
    print(f"  Precision: {results_latest['precision']:.4f}")
    print(f"  Recall: {results_latest['recall']:.4f}")
    print(f"  F1: {results_latest['f1']:.4f}")
    
    print(f"\n差异:")
    print(f"  mAP: {results_latest['mAP'] - results_best['mAP']:+.4f}")
    print(f"  Precision: {results_latest['precision'] - results_best['precision']:+.4f}")
    print(f"  Recall: {results_latest['recall'] - results_best['recall']:+.4f}")
    
    print("\n" + "="*80)
    print(f"详细报告已保存到: {output_file}")
    print("="*80 + "\n")


def main():
    # 配置
    config_path = './configs/rgbt_tiny_config.yaml'
    checkpoint_dir = './checkpoints/rgbt_tiny'
    output_dir = './evaluation_results'
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(output_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("开始检查点评估")
    logger.info("="*80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    logger.info("加载验证数据集...")
    val_dataset = RGBTTinyCOCODataset(
        root_dir=config['data']['root_dir'],
        split='test',
        num_frames=config['model']['num_frames'],
        img_size=config['model']['img_size'],
        modality=config['data']['modality']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    logger.info(f"验证集样本数: {len(val_dataset)}")
    
    # 类别名称
    class_names = config['classes']['names']
    logger.info(f"类别数: {len(class_names)}")
    logger.info(f"类别: {class_names}")
    
    # 加载模型
    logger.info("创建模型...")
    model = OSFormer(
        num_classes=config['model']['num_classes'],
        num_frames=config['model']['num_frames'],
        img_size=config['model']['img_size']
    ).to(device)
    
    # 评估最佳检查点
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    logger.info(f"\n{'='*80}")
    logger.info(f"评估最佳检查点: {best_checkpoint_path}")
    logger.info(f"{'='*80}")
    
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ 加载最佳检查点 (Epoch {checkpoint.get('epoch', 'unknown')})")
        
        results_best = evaluate_model(
            model, val_loader, device, config, class_names,
            score_thresh=0.3, logger=logger
        )
        logger.info(f"最佳检查点 mAP: {results_best['mAP']:.4f}")
    else:
        logger.error(f"❌ 最佳检查点不存在: {best_checkpoint_path}")
        return
    
    # 评估最新检查点
    latest_checkpoint_path = None
    for epoch in range(100, 0, -1):  # 从高到低查找
        candidate = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        if os.path.exists(candidate):
            latest_checkpoint_path = candidate
            break
    
    if latest_checkpoint_path is None:
        logger.warning("⚠️ 未找到其他检查点，仅评估最佳检查点")
        results_latest = results_best
    else:
        logger.info(f"\n{'='*80}")
        logger.info(f"评估最新检查点: {latest_checkpoint_path}")
        logger.info(f"{'='*80}")
        
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ 加载最新检查点 (Epoch {checkpoint.get('epoch', 'unknown')})")
        
        results_latest = evaluate_model(
            model, val_loader, device, config, class_names,
            score_thresh=0.3, logger=logger
        )
        logger.info(f"最新检查点 mAP: {results_latest['mAP']:.4f}")
    
    # 生成可视化和报告
    logger.info("\n生成对比图表...")
    visualize_comparison(results_best, results_latest, class_names, output_dir)
    
    logger.info("\n生成详细报告...")
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    generate_report(results_best, results_latest, class_names, report_file, logger)
    
    # 保存结果为JSON
    results_json = {
        'best_checkpoint': {
            'mAP': float(results_best['mAP']),
            'precision': float(results_best['precision']),
            'recall': float(results_best['recall']),
            'f1': float(results_best['f1']),
            'avg_iou': float(results_best['avg_iou']),
            'total_tp': int(results_best['total_tp']),
            'total_fp': int(results_best['total_fp']),
            'total_fn': int(results_best['total_fn'])
        },
        'latest_checkpoint': {
            'mAP': float(results_latest['mAP']),
            'precision': float(results_latest['precision']),
            'recall': float(results_latest['recall']),
            'f1': float(results_latest['f1']),
            'avg_iou': float(results_latest['avg_iou']),
            'total_tp': int(results_latest['total_tp']),
            'total_fp': int(results_latest['total_fp']),
            'total_fn': int(results_latest['total_fn'])
        }
    }
    
    json_file = os.path.join(output_dir, 'results.json')
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"✅ 结果已保存到: {json_file}")
    
    logger.info("\n" + "="*80)
    logger.info("评估完成！")
    logger.info("="*80)
    
    print(f"\n📁 所有结果已保存到: {output_dir}/")
    print(f"  - 评估报告: {report_file}")
    print(f"  - 对比图表: {os.path.join(output_dir, 'comparison_charts.png')}")
    print(f"  - JSON 结果: {json_file}")
    print(f"  - 日志文件: {log_file}")


if __name__ == "__main__":
    main()

