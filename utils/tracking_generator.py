#!/usr/bin/env python3
"""
自动生成 Tracking ID
通过目标框位置的相对关系（IoU + 中心距离）匹配跨帧目标
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment


def compute_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框之间的 IoU
    
    参数:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]
    
    返回:
        iou_matrix: (N, M) IoU 矩阵
    """
    # 扩展维度以进行广播
    boxes1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    boxes2 = boxes2.unsqueeze(0)  # (1, M, 4)
    
    # 计算交集
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算各自面积
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # 计算 IoU
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    # 确保返回 2D 矩阵 (N, M)
    if iou.dim() == 3:
        iou = iou.squeeze()
    
    return iou


def compute_center_distance(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框中心点之间的归一化距离
    
    参数:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]
    
    返回:
        distance_matrix: (N, M) 距离矩阵
    """
    # 计算中心点
    center1 = torch.stack([
        (boxes1[:, 0] + boxes1[:, 2]) / 2,
        (boxes1[:, 1] + boxes1[:, 3]) / 2
    ], dim=1)  # (N, 2)
    
    center2 = torch.stack([
        (boxes2[:, 0] + boxes2[:, 2]) / 2,
        (boxes2[:, 1] + boxes2[:, 3]) / 2
    ], dim=1)  # (M, 2)
    
    # 扩展维度
    center1 = center1.unsqueeze(1)  # (N, 1, 2)
    center2 = center2.unsqueeze(0)  # (1, M, 2)
    
    # 计算欧氏距离
    distance = torch.sqrt(((center1 - center2) ** 2).sum(dim=-1))
    
    return distance


def match_boxes_hungarian(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    iou_threshold: float = 0.3,
    distance_threshold: float = 0.1,
    iou_weight: float = 0.7,
    distance_weight: float = 0.3
) -> List[Tuple[int, int]]:
    """
    使用匈牙利算法匹配两帧之间的目标
    
    参数:
        boxes1: (N, 4) 第一帧的边界框
        boxes2: (M, 4) 第二帧的边界框
        iou_threshold: IoU 阈值
        distance_threshold: 距离阈值（归一化）
        iou_weight: IoU 权重
        distance_weight: 距离权重
    
    返回:
        matches: [(idx1, idx2), ...] 匹配对列表
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return []
    
    # 计算 IoU 和距离
    iou_matrix = compute_iou_batch(boxes1, boxes2)  # (N, M)
    distance_matrix = compute_center_distance(boxes1, boxes2)  # (N, M)
    
    # 归一化距离到 [0, 1]
    # 距离越小越好，转换为相似度（1 - normalized_distance）
    max_distance = torch.sqrt(torch.tensor(2.0))  # 最大可能距离（对角线）
    normalized_distance = distance_matrix / max_distance
    distance_similarity = 1.0 - normalized_distance
    
    # 综合得分（越大越好）
    score_matrix = iou_weight * iou_matrix + distance_weight * distance_similarity
    
    # 转换为成本矩阵（越小越好）
    cost_matrix = -score_matrix.cpu().numpy()
    
    # 确保 iou_matrix 和 distance_matrix 是 2D
    if iou_matrix.dim() == 0:
        iou_matrix = iou_matrix.unsqueeze(0).unsqueeze(0)
    elif iou_matrix.dim() == 1:
        iou_matrix = iou_matrix.unsqueeze(0)
    
    if distance_matrix.dim() == 0:
        distance_matrix = distance_matrix.unsqueeze(0).unsqueeze(0)
    elif distance_matrix.dim() == 1:
        distance_matrix = distance_matrix.unsqueeze(0)
    
    # 匈牙利算法
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 过滤低质量匹配
    matches = []
    for i, j in zip(row_indices, col_indices):
        iou = iou_matrix[i, j].item()
        distance = distance_matrix[i, j].item()
        
        # 检查阈值
        if iou >= iou_threshold and distance <= distance_threshold:
            matches.append((i, j))
    
    return matches


def generate_tracking_ids(
    frames_boxes: List[torch.Tensor],
    frames_labels: List[torch.Tensor],
    iou_threshold: float = 0.3,
    distance_threshold: float = 0.1
) -> List[torch.Tensor]:
    """
    为一个序列生成 tracking_id
    
    参数:
        frames_boxes: List of (N_t, 4) 每帧的边界框
        frames_labels: List of (N_t,) 每帧的类别标签
        iou_threshold: IoU 匹配阈值
        distance_threshold: 距离匹配阈值
    
    返回:
        frames_track_ids: List of (N_t,) 每帧的 tracking_id
    """
    num_frames = len(frames_boxes)
    
    if num_frames == 0:
        return []
    
    # 初始化第一帧的 tracking_id
    frames_track_ids = []
    next_track_id = 0
    
    # 第一帧：分配新的 tracking_id
    first_frame_track_ids = torch.arange(len(frames_boxes[0]), dtype=torch.long)
    frames_track_ids.append(first_frame_track_ids)
    next_track_id = len(frames_boxes[0])
    
    # 逐帧匹配
    for t in range(1, num_frames):
        boxes_prev = frames_boxes[t - 1]
        boxes_curr = frames_boxes[t]
        labels_prev = frames_labels[t - 1]
        labels_curr = frames_labels[t]
        track_ids_prev = frames_track_ids[t - 1]
        
        # 初始化当前帧的 tracking_id（全部为 -1，表示未匹配）
        track_ids_curr = torch.full((len(boxes_curr),), -1, dtype=torch.long)
        
        if len(boxes_prev) == 0 or len(boxes_curr) == 0:
            # 如果有一帧为空，当前帧所有目标分配新 ID
            for i in range(len(boxes_curr)):
                track_ids_curr[i] = next_track_id
                next_track_id += 1
            frames_track_ids.append(track_ids_curr)
            continue
        
        # 按类别分别匹配（同一类别的目标才能匹配）
        unique_labels = torch.unique(torch.cat([labels_prev, labels_curr]))
        
        for label in unique_labels:
            # 当前类别的目标索引
            prev_indices = (labels_prev == label).nonzero(as_tuple=True)[0]
            curr_indices = (labels_curr == label).nonzero(as_tuple=True)[0]
            
            if len(prev_indices) == 0 or len(curr_indices) == 0:
                continue
            
            # 提取该类别的边界框
            boxes_prev_class = boxes_prev[prev_indices]
            boxes_curr_class = boxes_curr[curr_indices]
            
            # 匹配
            matches = match_boxes_hungarian(
                boxes_prev_class,
                boxes_curr_class,
                iou_threshold=iou_threshold,
                distance_threshold=distance_threshold
            )
            
            # 分配 tracking_id
            for prev_idx, curr_idx in matches:
                global_prev_idx = prev_indices[prev_idx].item()
                global_curr_idx = curr_indices[curr_idx].item()
                
                # 继承前一帧的 tracking_id
                track_ids_curr[global_curr_idx] = track_ids_prev[global_prev_idx]
        
        # 为未匹配的目标分配新 ID
        unmatched_indices = (track_ids_curr == -1).nonzero(as_tuple=True)[0]
        for idx in unmatched_indices:
            track_ids_curr[idx] = next_track_id
            next_track_id += 1
        
        frames_track_ids.append(track_ids_curr)
    
    return frames_track_ids


def compute_offset_from_tracking(
    frames_boxes: List[torch.Tensor],
    frames_track_ids: List[torch.Tensor],
    feature_size: int = 40,
    img_size: int = 640
) -> List[torch.Tensor]:
    """
    根据 tracking_id 计算相邻帧的 offset
    
    参数:
        frames_boxes: List of (N_t, 4) 每帧的边界框 [x1, y1, x2, y2] (归一化)
        frames_track_ids: List of (N_t,) 每帧的 tracking_id
        feature_size: 特征图大小
        img_size: 输入图像大小
    
    返回:
        frames_offsets: List of Dict 每帧的 offset 信息
            - 'offset_map': (2, H, W) offset 目标
            - 'valid_mask': (H, W) 有效位置 mask
    """
    num_frames = len(frames_boxes)
    frames_offsets = []
    
    stride = img_size / feature_size
    
    for t in range(num_frames - 1):
        # 当前帧和下一帧
        boxes_t = frames_boxes[t]
        boxes_t1 = frames_boxes[t + 1]
        track_ids_t = frames_track_ids[t]
        track_ids_t1 = frames_track_ids[t + 1]
        
        # 初始化 offset map
        offset_map = torch.zeros(2, feature_size, feature_size)
        valid_mask = torch.zeros(feature_size, feature_size, dtype=torch.bool)
        
        if len(boxes_t) == 0 or len(boxes_t1) == 0:
            frames_offsets.append({
                'offset_map': offset_map,
                'valid_mask': valid_mask
            })
            continue
        
        # 找到跨帧匹配的目标
        for i, track_id in enumerate(track_ids_t):
            # 在下一帧中找到相同的 track_id
            matched_indices = (track_ids_t1 == track_id).nonzero(as_tuple=True)[0]
            
            if len(matched_indices) == 0:
                continue
            
            # 假设只有一个匹配（通常情况）
            j = matched_indices[0].item()
            
            # 计算中心点
            box_t = boxes_t[i]
            box_t1 = boxes_t1[j]
            
            center_x_t = (box_t[0] + box_t[2]) / 2
            center_y_t = (box_t[1] + box_t[3]) / 2
            center_x_t1 = (box_t1[0] + box_t1[2]) / 2
            center_y_t1 = (box_t1[1] + box_t1[3]) / 2
            
            # 计算偏移（归一化坐标）
            offset_x = center_x_t1 - center_x_t
            offset_y = center_y_t1 - center_y_t
            
            # 转换到特征图坐标
            grid_x = int(center_x_t * feature_size)
            grid_y = int(center_y_t * feature_size)
            
            # 边界检查
            if 0 <= grid_x < feature_size and 0 <= grid_y < feature_size:
                # 偏移归一化到特征图尺度
                offset_map[0, grid_y, grid_x] = offset_x * feature_size
                offset_map[1, grid_y, grid_x] = offset_y * feature_size
                valid_mask[grid_y, grid_x] = True
        
        frames_offsets.append({
            'offset_map': offset_map,
            'valid_mask': valid_mask
        })
    
    # 最后一帧没有 offset
    frames_offsets.append({
        'offset_map': torch.zeros(2, feature_size, feature_size),
        'valid_mask': torch.zeros(feature_size, feature_size, dtype=torch.bool)
    })
    
    return frames_offsets


def visualize_tracking(
    frames_boxes: List[torch.Tensor],
    frames_track_ids: List[torch.Tensor],
    frames_labels: List[torch.Tensor],
    output_path: str = 'tracking_visualization.png'
):
    """
    可视化跟踪结果
    
    参数:
        frames_boxes: List of (N_t, 4) 每帧的边界框
        frames_track_ids: List of (N_t,) 每帧的 tracking_id
        frames_labels: List of (N_t,) 每帧的类别标签
        output_path: 输出路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm
    
    num_frames = len(frames_boxes)
    
    fig, axes = plt.subplots(1, num_frames, figsize=(4*num_frames, 4))
    if num_frames == 1:
        axes = [axes]
    
    # 为每个 track_id 分配一个颜色
    all_track_ids = torch.cat(frames_track_ids)
    unique_track_ids = torch.unique(all_track_ids)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_track_ids)))
    track_id_to_color = {tid.item(): colors[i] for i, tid in enumerate(unique_track_ids)}
    
    for t in range(num_frames):
        ax = axes[t]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Frame {t}')
        
        boxes = frames_boxes[t]
        track_ids = frames_track_ids[t]
        labels = frames_labels[t]
        
        for i, (box, track_id, label) in enumerate(zip(boxes, track_ids, labels)):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            
            color = track_id_to_color[track_id.item()]
            
            rect = Rectangle((x1, y1), w, h, linewidth=2, 
                           edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 添加 track_id 标签
            ax.text(x1, y1-0.01, f'ID{track_id.item()}', 
                   color=color, fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 跟踪可视化已保存到: {output_path}")


def analyze_tracking_quality(
    frames_boxes: List[torch.Tensor],
    frames_track_ids: List[torch.Tensor]
) -> Dict:
    """
    分析跟踪质量
    
    返回:
        stats: 跟踪统计信息
    """
    num_frames = len(frames_boxes)
    
    # 统计每个 track_id 出现的帧数
    all_track_ids = torch.cat(frames_track_ids)
    unique_track_ids = torch.unique(all_track_ids)
    
    track_lengths = []
    for track_id in unique_track_ids:
        length = sum([(track_ids == track_id).any().item() for track_ids in frames_track_ids])
        track_lengths.append(length)
    
    # 统计匹配率（相邻帧之间）
    match_rates = []
    for t in range(num_frames - 1):
        track_ids_t = frames_track_ids[t]
        track_ids_t1 = frames_track_ids[t + 1]
        
        # 有多少当前帧的 track_id 在下一帧中出现
        matched = sum([1 for tid in track_ids_t if tid in track_ids_t1])
        match_rate = matched / len(track_ids_t) if len(track_ids_t) > 0 else 0
        match_rates.append(match_rate)
    
    stats = {
        'num_tracks': len(unique_track_ids),
        'avg_track_length': np.mean(track_lengths),
        'max_track_length': max(track_lengths),
        'min_track_length': min(track_lengths),
        'avg_match_rate': np.mean(match_rates) if match_rates else 0,
        'track_lengths': track_lengths,
        'match_rates': match_rates
    }
    
    return stats


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("测试 Tracking ID 生成")
    print("="*80)
    
    # 模拟 5 帧数据
    frames_boxes = [
        torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]]),  # 帧 0: 2 个目标
        torch.tensor([[0.12, 0.11, 0.22, 0.21], [0.52, 0.51, 0.62, 0.61]]),  # 帧 1: 轻微移动
        torch.tensor([[0.14, 0.12, 0.24, 0.22], [0.54, 0.52, 0.64, 0.62], [0.8, 0.8, 0.9, 0.9]]),  # 帧 2: +新目标
        torch.tensor([[0.16, 0.13, 0.26, 0.23], [0.82, 0.81, 0.92, 0.91]]),  # 帧 3: 目标1消失
        torch.tensor([[0.84, 0.82, 0.94, 0.92]]),  # 帧 4: 只剩1个
    ]
    
    frames_labels = [
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([1]),
    ]
    
    # 生成 tracking_id
    print("\n生成 Tracking ID...")
    frames_track_ids = generate_tracking_ids(frames_boxes, frames_labels)
    
    print("\n每帧的 Tracking ID:")
    for t, track_ids in enumerate(frames_track_ids):
        print(f"  帧 {t}: {track_ids.tolist()}")
    
    # 分析跟踪质量
    print("\n跟踪质量分析:")
    stats = analyze_tracking_quality(frames_boxes, frames_track_ids)
    print(f"  总轨迹数: {stats['num_tracks']}")
    print(f"  平均轨迹长度: {stats['avg_track_length']:.2f} 帧")
    print(f"  最长轨迹: {stats['max_track_length']} 帧")
    print(f"  平均匹配率: {stats['avg_match_rate']:.2%}")
    
    # 计算 offset
    print("\n计算 Offset...")
    frames_offsets = compute_offset_from_tracking(frames_boxes, frames_track_ids)
    
    for t, offset_info in enumerate(frames_offsets):
        offset_map = offset_info['offset_map']
        valid_mask = offset_info['valid_mask']
        num_valid = valid_mask.sum().item()
        
        if num_valid > 0:
            print(f"  帧 {t} -> {t+1}: {num_valid} 个有效 offset")
            # 显示非零 offset
            valid_indices = valid_mask.nonzero(as_tuple=True)
            for i in range(len(valid_indices[0])):
                y, x = valid_indices[0][i].item(), valid_indices[1][i].item()
                offset_x = offset_map[0, y, x].item()
                offset_y = offset_map[1, y, x].item()
                print(f"    位置 ({x}, {y}): offset=({offset_x:.4f}, {offset_y:.4f})")
    
    # 可视化
    print("\n生成可视化...")
    visualize_tracking(frames_boxes, frames_track_ids, frames_labels, 'test_tracking.png')
    
    print("\n✅ 测试完成！")

