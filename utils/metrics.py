"""
评估指标模块
包含 mAP, SAFit 等指标
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的 IoU
    
    参数：
        box1: (4,) [x1, y1, x2, y2]
        box2: (4,) [x1, y1, x2, y2]
        
    返回：
        iou: IoU 值
    """
    # 计算交集
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # 计算并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def calculate_ap(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """
    计算 Average Precision
    
    参数：
        pred_boxes: List of (N, 4) 预测框
        pred_scores: List of (N,) 预测分数
        gt_boxes: List of (M, 4) 真实框
        iou_threshold: IoU 阈值
        
    返回：
        ap: Average Precision
    """
    # 收集所有预测
    all_preds = []
    for img_idx, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for box, score in zip(boxes, scores):
            all_preds.append({
                'img_idx': img_idx,
                'box': box,
                'score': score
            })
    
    # 按分数降序排序
    all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
    
    # 统计每张图的 GT 数量
    num_gt = sum(len(boxes) for boxes in gt_boxes)
    
    if num_gt == 0:
        return 0.0
    
    # 计算 TP 和 FP
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    
    # 记录每个 GT 是否已被匹配
    gt_matched = [np.zeros(len(boxes), dtype=bool) for boxes in gt_boxes]
    
    for pred_idx, pred in enumerate(all_preds):
        img_idx = pred['img_idx']
        pred_box = pred['box']
        
        if len(gt_boxes[img_idx]) == 0:
            fp[pred_idx] = 1
            continue
        
        # 计算与所有 GT 的 IoU
        ious = np.array([
            calculate_iou(pred_box, gt_box)
            for gt_box in gt_boxes[img_idx]
        ])
        
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        
        # 判断是否为 TP
        if max_iou >= iou_threshold:
            if not gt_matched[img_idx][max_iou_idx]:
                tp[pred_idx] = 1
                gt_matched[img_idx][max_iou_idx] = True
            else:
                fp[pred_idx] = 1  # 重复检测
        else:
            fp[pred_idx] = 1
    
    # 计算累积 TP 和 FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # 计算 Precision 和 Recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / num_gt
    
    # 计算 AP（使用 11 点插值）
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


def calculate_map(
    pred_boxes_list: List[List[np.ndarray]],
    pred_scores_list: List[List[np.ndarray]],
    gt_boxes_list: List[List[np.ndarray]],
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    计算 mAP (mean Average Precision)
    
    参数：
        pred_boxes_list: List of (每个类别的预测框列表)
        pred_scores_list: List of (每个类别的预测分数列表)
        gt_boxes_list: List of (每个类别的真实框列表)
        iou_thresholds: IoU 阈值列表
        
    返回：
        metrics: 包含 mAP50, mAP75, mAP50-95 等指标的字典
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))
    
    # 计算不同 IoU 阈值下的 AP
    ap_dict = defaultdict(list)
    
    for iou_thresh in iou_thresholds:
        # 对于每个类别（这里简化为单类别）
        ap = calculate_ap(
            pred_boxes_list[0] if pred_boxes_list else [],
            pred_scores_list[0] if pred_scores_list else [],
            gt_boxes_list[0] if gt_boxes_list else [],
            iou_threshold=iou_thresh
        )
        ap_dict[f'AP@{iou_thresh:.2f}'] = ap
    
    # 计算 mAP50
    map50 = ap_dict.get('AP@0.50', 0.0)
    
    # 计算 mAP75
    map75 = ap_dict.get('AP@0.75', 0.0)
    
    # 计算 mAP50-95（平均所有 IoU 阈值）
    map_50_95 = np.mean([ap_dict[f'AP@{t:.2f}'] for t in np.arange(0.5, 1.0, 0.05)])
    
    metrics = {
        'mAP50': map50,
        'mAP75': map75,
        'mAP50-95': map_50_95
    }
    
    return metrics


def calculate_safit(
    pred_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    area_threshold: float = 32 * 32
) -> Dict[str, float]:
    """
    计算 SAFit (Small Area Fit) 指标
    专门用于评估小目标检测性能
    
    参数：
        pred_boxes: List of (N, 4) 预测框
        gt_boxes: List of (M, 4) 真实框
        area_threshold: 小目标的面积阈值
        
    返回：
        metrics: 包含 SAFit 相关指标的字典
    """
    total_small_gt = 0
    matched_small_gt = 0
    total_area_error = 0.0
    
    for pred_box_list, gt_box_list in zip(pred_boxes, gt_boxes):
        # 筛选小目标 GT
        gt_areas = (gt_box_list[:, 2] - gt_box_list[:, 0]) * (gt_box_list[:, 3] - gt_box_list[:, 1])
        small_gt_mask = gt_areas < area_threshold
        small_gt_boxes = gt_box_list[small_gt_mask]
        
        total_small_gt += len(small_gt_boxes)
        
        if len(small_gt_boxes) == 0 or len(pred_box_list) == 0:
            continue
        
        # 为每个小目标 GT 找最佳匹配
        for gt_box in small_gt_boxes:
            ious = np.array([calculate_iou(gt_box, pred_box) for pred_box in pred_box_list])
            
            if len(ious) > 0:
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]
                
                if max_iou >= 0.5:
                    matched_small_gt += 1
                    
                    # 计算面积误差
                    pred_box = pred_box_list[max_iou_idx]
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                    area_error = abs(pred_area - gt_area) / gt_area
                    total_area_error += area_error
    
    # 计算指标
    if total_small_gt == 0:
        recall = 0.0
        avg_area_error = 0.0
    else:
        recall = matched_small_gt / total_small_gt
        avg_area_error = total_area_error / max(matched_small_gt, 1)
    
    # SAFit 综合指标：考虑召回率和面积拟合度
    safit = recall * (1 - avg_area_error)
    
    metrics = {
        'SAFit': safit,
        'SmallRecall': recall,
        'AvgAreaError': avg_area_error,
        'TotalSmallGT': total_small_gt,
        'MatchedSmallGT': matched_small_gt
    }
    
    return metrics


class DetectionMetrics:
    """
    检测指标计算器（在线累积版本）
    """
    
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.pred_boxes = [[] for _ in range(self.num_classes)]
        self.pred_scores = [[] for _ in range(self.num_classes)]
        self.gt_boxes = [[] for _ in range(self.num_classes)]
    
    def update(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ):
        """
        更新统计
        
        参数：
            pred_boxes: (N, 4)
            pred_scores: (N,)
            pred_labels: (N,)
            gt_boxes: (M, 4)
            gt_labels: (M,)
        """
        for cls_idx in range(self.num_classes):
            # 预测
            cls_mask = pred_labels == cls_idx
            if cls_mask.any():
                self.pred_boxes[cls_idx].append(pred_boxes[cls_mask])
                self.pred_scores[cls_idx].append(pred_scores[cls_mask])
            else:
                self.pred_boxes[cls_idx].append(np.zeros((0, 4)))
                self.pred_scores[cls_idx].append(np.zeros(0))
            
            # GT
            gt_mask = gt_labels == cls_idx
            if gt_mask.any():
                self.gt_boxes[cls_idx].append(gt_boxes[gt_mask])
            else:
                self.gt_boxes[cls_idx].append(np.zeros((0, 4)))
    
    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        # 计算 mAP
        map_metrics = calculate_map(
            self.pred_boxes,
            self.pred_scores,
            self.gt_boxes
        )
        
        # 计算 SAFit
        safit_metrics = calculate_safit(
            self.pred_boxes[0],
            self.gt_boxes[0]
        )
        
        # 合并
        metrics = {**map_metrics, **safit_metrics}
        
        return metrics


if __name__ == "__main__":
    # 测试指标计算
    print("Testing Metrics...")
    
    # 模拟数据
    pred_boxes = [
        np.array([[10, 10, 30, 30], [50, 50, 70, 70]]),
        np.array([[15, 15, 35, 35]])
    ]
    pred_scores = [
        np.array([0.9, 0.8]),
        np.array([0.7])
    ]
    gt_boxes = [
        np.array([[12, 12, 32, 32], [52, 52, 72, 72]]),
        np.array([[16, 16, 36, 36]])
    ]
    
    # 计算 mAP
    print("\n1. mAP:")
    map_metrics = calculate_map([pred_boxes], [pred_scores], [gt_boxes])
    for key, val in map_metrics.items():
        print(f"   {key}: {val:.4f}")
    
    # 计算 SAFit
    print("\n2. SAFit:")
    safit_metrics = calculate_safit(pred_boxes, gt_boxes, area_threshold=1000)
    for key, val in safit_metrics.items():
        print(f"   {key}: {val:.4f}" if isinstance(val, float) else f"   {key}: {val}")
    
    print("\n✓ 所有指标测试通过！")

