"""
可视化工具
用于在训练过程中可视化检测结果
"""

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import io
from PIL import Image


def visualize_detection_results(
    rgb_frames: torch.Tensor,
    thermal_frames: torch.Tensor,
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict],
    class_names: List[str],
    score_thresh: float = 0.3,
    max_samples: int = 4,
    mid_frame_only: bool = True
) -> List[np.ndarray]:
    """
    可视化检测结果
    
    参数：
        rgb_frames: (B, T, 3, H, W) RGB帧
        thermal_frames: (B, T, 1, H, W) 热红外帧  
        predictions: List of dict，模型预测
        targets: List of dict，真实标注
        class_names: 类别名称列表
        score_thresh: 分数阈值
        max_samples: 最多可视化样本数
        mid_frame_only: 是否只可视化中间帧
        
    返回:
        vis_images: 可视化图像列表（numpy数组，RGB格式）
    """
    B, T, _, H, W = rgb_frames.shape
    mid_frame = T // 2 if mid_frame_only else 0
    
    vis_images = []
    num_samples = min(B, max_samples)
    
    for b in range(num_samples):
        # 选择要可视化的帧
        frame_idx = mid_frame
        
        # 获取RGB和Thermal图像
        rgb_img = rgb_frames[b, frame_idx].cpu().permute(1, 2, 0).numpy()
        thermal_img = thermal_frames[b, frame_idx, 0].cpu().numpy()
        
        # 反归一化
        rgb_img = np.clip(rgb_img * 255, 0, 255).astype(np.uint8)
        thermal_img = np.clip(thermal_img * 255, 0, 255).astype(np.uint8)
        
        # 🔥 修复：predictions是List[Dict]，每个dict的形状是(B, C, H', W')
        # 需要根据frame_idx选择对应的预测
        if frame_idx < len(predictions):
            pred = predictions[frame_idx]
            cls_pred = pred['cls'][b]  # (num_classes+1, H', W')
            bbox_pred = pred['bbox'][b]  # (4, H', W')
            centerness_pred = pred.get('centerness', None)
            if centerness_pred is not None:
                centerness_pred = centerness_pred[b]  # (1, H', W')
            
            # 🔥 修复：排除背景类（最后一个通道）
            # num_classes是前景类数量，模型输出是num_classes+1
            num_fg_classes = cls_pred.shape[0] - 1  # 7个前景类
            cls_pred_fg = cls_pred[:num_fg_classes]  # 只取前7个通道
            
            # 解码预测（添加centerness）
            pred_boxes, pred_scores, pred_labels = decode_predictions(
                cls_pred_fg, bbox_pred, centerness_pred, H, W, score_thresh
            )
        else:
            # 如果frame_idx超出范围，使用空预测
            pred_boxes = np.zeros((0, 4))
            pred_scores = np.zeros(0)
            pred_labels = np.zeros(0, dtype=np.int64)
        
        # 获取GT
        if b < len(targets):
            target = targets[b]
            gt_boxes_norm = target.get('boxes', [])  # 归一化坐标 [x1, y1, x2, y2]
            gt_labels = target.get('labels', [])
            
            # 🔥 修复：将归一化的GT boxes转换为像素坐标
            if isinstance(gt_boxes_norm, torch.Tensor) and len(gt_boxes_norm) > 0:
                gt_boxes_norm = gt_boxes_norm.cpu().numpy()
                gt_boxes = gt_boxes_norm * np.array([W, H, W, H])  # 转换为像素坐标
            elif isinstance(gt_boxes_norm, list) and len(gt_boxes_norm) > 0:
                gt_boxes = np.array(gt_boxes_norm) * np.array([W, H, W, H])
            else:
                gt_boxes = np.zeros((0, 4))
            
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu().numpy()
            elif isinstance(gt_labels, list):
                gt_labels = np.array(gt_labels)
        else:
            gt_boxes = np.zeros((0, 4))
            gt_labels = np.array([], dtype=np.int64)
        
        # 创建可视化图像
        vis_img = create_visualization(
            rgb_img, thermal_img,
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            class_names, H, W
        )
        
        vis_images.append(vis_img)
    
    return vis_images


def decode_predictions(
    cls_pred: torch.Tensor,
    bbox_pred: torch.Tensor,
    centerness_pred: torch.Tensor,  # ← 新增
    img_h: int,
    img_w: int,
    score_thresh: float = 0.05,  # 🔥 从0.3改为0.05 ⭐⭐⭐
    nms_thresh: float = 0.5,  # ← 新增
    max_detections: int = 100  # 🆕 每张图最多保留的检测框数量 ⭐⭐⭐
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解码预测结果（修复：添加centerness和NMS）
    
    参数：
        cls_pred: (num_classes, H', W') 分类预测
        bbox_pred: (4, H', W') 边界框预测
        centerness_pred: (1, H', W') 中心度预测
        img_h, img_w: 原始图像尺寸
        score_thresh: 分数阈值
        nms_thresh: NMS IoU阈值
        
    返回：
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        labels: (N,)
    """
    device = cls_pred.device
    num_channels, H, W = cls_pred.shape
    
    # 🔥 关键修复：排除背景类（模型输出8个通道，前7个是前景类） ⭐⭐⭐
    num_classes = min(num_channels, 7)  # 只取前7个前景类
    cls_pred_fg = cls_pred[:num_classes]  # (7, H, W)
    
    # ✅ 修复：计算最终分数 = cls_score * centerness
    cls_scores = torch.sigmoid(cls_pred_fg)  # (7, H, W)
    
    if centerness_pred is not None:
        centerness_scores = torch.sigmoid(centerness_pred)  # (1, H, W)
        # 每个类别的分数都乘以 centerness
        final_scores = cls_scores * centerness_scores  # (num_classes, H, W)
    else:
        final_scores = cls_scores
    
    # 找到所有高于阈值的位置
    max_scores, max_labels = final_scores.max(dim=0)  # (H, W)
    mask = max_scores > score_thresh
    
    if mask.sum() == 0:
        # 没有检测到目标
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    
    # 获取检测位置
    indices = mask.nonzero(as_tuple=False)  # (N, 2) [h, w]
    selected_scores = max_scores[mask].cpu().numpy()
    selected_labels = max_labels[mask].cpu().numpy()
    
    # 解码边界框（FCOS风格：l, t, r, b）
    stride = img_h / H
    boxes = []
    
    for idx in indices:
        h_idx, w_idx = idx[0].item(), idx[1].item()
        
        # 获取边界框预测
        l = bbox_pred[0, h_idx, w_idx].item()
        t = bbox_pred[1, h_idx, w_idx].item()
        r = bbox_pred[2, h_idx, w_idx].item()
        b = bbox_pred[3, h_idx, w_idx].item()
        
        # 转换为像素坐标
        cx = (w_idx + 0.5) * stride
        cy = (h_idx + 0.5) * stride
        
        x1 = cx - l * stride
        y1 = cy - t * stride
        x2 = cx + r * stride
        y2 = cy + b * stride
        
        # 裁剪到图像范围内
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)
        
        boxes.append([x1, y1, x2, y2])
    
    boxes = np.array(boxes) if len(boxes) > 0 else np.zeros((0, 4))
    
    # ✅ 添加 NMS
    if len(boxes) > 0:
        try:
            from torchvision.ops import nms
            boxes_tensor = torch.from_numpy(boxes).float().to(device)
            scores_tensor = torch.from_numpy(selected_scores).float().to(device)
            
            keep_indices = nms(boxes_tensor, scores_tensor, nms_thresh)
            keep_indices = keep_indices.cpu().numpy()
            
            boxes = boxes[keep_indices]
            selected_scores = selected_scores[keep_indices]
            selected_labels = selected_labels[keep_indices]
        except Exception as e:
            # 如果NMS失败，继续使用原始结果
            print(f"Warning: NMS failed: {e}")
    
    # 🆕 限制最大检测框数量 ⭐⭐⭐
    if len(boxes) > max_detections:
        # 按分数排序，只保留前 max_detections 个
        top_k_indices = np.argsort(selected_scores)[::-1][:max_detections]
        boxes = boxes[top_k_indices]
        selected_scores = selected_scores[top_k_indices]
        selected_labels = selected_labels[top_k_indices]
    
    return boxes, selected_scores, selected_labels


def create_visualization(
    rgb_img: np.ndarray,
    thermal_img: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: List,
    gt_labels: List,
    class_names: List[str],
    img_h: int,
    img_w: int
) -> np.ndarray:
    """
    创建可视化图像（RGB + Thermal 并排显示）
    
    返回：
        vis_img: (H, W*2, 3) RGB图像
    """
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # RGB image + prediction boxes
    ax1.imshow(rgb_img)
    ax1.set_title('RGB + Predictions', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw prediction boxes (red)
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Label text
        label_text = f'{class_names[label]}: {score:.2f}'
        ax1.text(
            x1, y1 - 5, label_text,
            color='red', fontsize=10, weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='red')
        )
    
    # Draw GT boxes (green dashed)
    for box, label in zip(gt_boxes, gt_labels):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(box, list):
            box = np.array(box)
        
        # Convert normalized coordinates to pixel coordinates
        if box.max() <= 1.0:
            box = box * np.array([img_w, img_h, img_w, img_h])
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
        )
        ax1.add_patch(rect)
        
        # GT label
        gt_text = f'GT: {class_names[label]}'
        ax1.text(
            x1, y2 + 15, gt_text,
            color='lime', fontsize=9, weight='bold',
            bbox=dict(facecolor='black', alpha=0.5, pad=2)
        )
    
    # Thermal image + prediction boxes
    ax2.imshow(thermal_img, cmap='hot')
    ax2.set_title('Thermal + Predictions', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Draw prediction boxes (cyan)
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor='cyan', facecolor='none'
        )
        ax2.add_patch(rect)
    
    # Draw GT boxes (green dashed)
    for box in gt_boxes:
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        if isinstance(box, list):
            box = np.array(box)
        
        if box.max() <= 1.0:
            box = box * np.array([img_w, img_h, img_w, img_h])
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
        )
        ax2.add_patch(rect)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Predictions'),
        Line2D([0], [0], color='lime', lw=2, linestyle='--', label='Ground Truth')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 转换为numpy数组（兼容不同matplotlib版本）
    fig.canvas.draw()
    
    # 尝试使用buffer_rgba或tostring_rgb
    try:
        # 新版本matplotlib
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img = img[:, :, :3]  # 只取RGB通道
    except AttributeError:
        try:
            # 旧版本matplotlib
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # 备用方案：使用PIL
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = np.array(Image.open(buf))
            img = img[:, :, :3]  # 只取RGB通道
    
    plt.close(fig)
    
    return img


def images_to_tensorboard_grid(images: List[np.ndarray], nrow: int = 2) -> torch.Tensor:
    """
    将图像列表转换为TensorBoard网格
    
    参数：
        images: 图像列表，每个为 (H, W, 3) numpy数组
        nrow: 每行图像数
        
    返回：
        grid: (3, H_total, W_total) tensor
    """
    if len(images) == 0:
        return torch.zeros(3, 100, 100)
    
    # 转换为tensor
    tensors = []
    for img in images:
        # (H, W, 3) -> (3, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    
    # 创建网格
    from torchvision.utils import make_grid
    grid = make_grid(tensors, nrow=nrow, padding=10, normalize=False)
    
    return grid


if __name__ == "__main__":
    # 测试代码
    print("测试可视化工具...")
    
    # 模拟数据
    B, T, H, W = 2, 5, 640, 640
    rgb_frames = torch.randn(B, T, 3, H, W)
    thermal_frames = torch.randn(B, T, 1, H, W)
    
    # 模拟预测
    predictions = []
    for t in range(T):
        pred = {
            'cls': torch.randn(B, 7, 40, 40),
            'bbox': torch.randn(B, 4, 40, 40),
        }
        predictions.append(pred)
    
    # 模拟目标
    targets = [
        {'boxes': [[0.1, 0.1, 0.3, 0.3]], 'labels': [0]},
        {'boxes': [[0.5, 0.5, 0.7, 0.7]], 'labels': [1]}
    ]
    
    class_names = ['ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane']
    
    # 可视化
    vis_images = visualize_detection_results(
        rgb_frames, thermal_frames, predictions, targets, class_names
    )
    
    print(f"生成了 {len(vis_images)} 张可视化图像")
    print(f"图像形状: {vis_images[0].shape}")
    
    # 转换为网格
    grid = images_to_tensorboard_grid(vis_images)
    print(f"网格形状: {grid.shape}")
    
    print("\n✓ 可视化工具测试通过！")

