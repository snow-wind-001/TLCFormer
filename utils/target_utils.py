"""
目标格式转换工具
将数据集的标注格式转换为模型损失函数需要的格式
"""

import torch
import numpy as np
from typing import List, Dict, Optional


def convert_targets_for_loss(
    targets_batch: List[Dict],
    num_frames: int,
    img_size: int,
    feature_size: int,
    device: torch.device,
    use_tracking_offset: bool = False
) -> List[Dict]:
    """
    将数据集标注转换为损失函数格式
    
    参数：
        targets_batch: List[Dict]，长度为 B（batch size）
            每个 Dict 包含 'boxes' 和 'labels'，是该样本所有帧的标注
        num_frames: 帧数 T
        img_size: 原始图像尺寸
        feature_size: 特征图尺寸
        device: 设备
        
    返回：
        frame_targets: List[Dict]，长度为 T（帧数）
            每个 Dict 包含该帧所有样本的标注
    """
    batch_size = len(targets_batch)
    stride = img_size / feature_size
    
    # 初始化每帧的目标
    frame_targets = []
    
    for t in range(num_frames):
        # 为该帧创建批次级标注
        cls_maps = []
        bbox_maps = []
        valid_masks = []
        centerness_maps = []  # 新增
        
        for b in range(batch_size):
            target = targets_batch[b]
            boxes = target['boxes']  # List of [x1, y1, x2, y2] (normalized)
            labels = target['labels']  # List of category_id
            
            # 创建该样本的分类和边界框 map
            # 🔥 改进v3: YOLO式正负样本分配
            # - ignore: -100 (远离目标的区域)
            # - 负样本: num_classes (背景类，目标周围区域)
            # - 正样本: 0~num_classes-1 (前景类，目标中心区域)
            cls_map = torch.full((feature_size, feature_size), -100, dtype=torch.long)
            bbox_map = torch.zeros(4, feature_size, feature_size, dtype=torch.float32)
            valid_mask = torch.zeros(feature_size, feature_size, dtype=torch.bool)
            centerness_map = torch.zeros(1, feature_size, feature_size, dtype=torch.float32)
            
            # 遍历所有目标
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                
                # 转换为特征图坐标
                x1_feat = x1 * feature_size
                y1_feat = y1 * feature_size
                x2_feat = x2 * feature_size
                y2_feat = y2 * feature_size
                
                # 计算中心点
                cx = (x1_feat + x2_feat) / 2
                cy = (y1_feat + y2_feat) / 2
                grid_x = int(cx)
                grid_y = int(cy)
                
                # 从中心点计算bbox（保证l,t,r,b都是正数）
                l = cx - x1_feat
                t = cy - y1_feat
                r = x2_feat - cx
                b = y2_feat - cy
                
                # 计算centerness
                min_lr = min(l, r)
                max_lr = max(l, r) + 1e-6
                min_tb = min(t, b)
                max_tb = max(t, b) + 1e-6
                centerness = ((min_lr / max_lr) * (min_tb / max_tb)) ** 0.5
                
                # 🔥 关键改进：YOLO式正负样本半径
                # positive_radius: 正样本区域（中心）
                # negative_radius: 负样本区域（周围）
                # 目标：正负比例 1:10~15
                box_size = max(x2_feat - x1_feat, y2_feat - y1_feat)
                if box_size < 2.0:  # 超小目标（<16像素）
                    positive_radius = 0  # 中心点
                    negative_radius = 4  # 9×9负样本区域
                elif box_size < 4.0:  # 小目标（<32像素）
                    positive_radius = 1  # 3×3正样本
                    negative_radius = 5  # 11×11负样本区域
                else:  # 中大目标
                    positive_radius = 2  # 5×5正样本
                    negative_radius = 7  # 15×15负样本区域
                
                # 步骤1: 先标记负样本区域（背景）
                # 这些位置会参与训练，但标签为背景类
                for dy in range(-negative_radius, negative_radius + 1):
                    for dx in range(-negative_radius, negative_radius + 1):
                        px = grid_x + dx
                        py = grid_y + dy
                        
                        # 边界检查
                        if 0 <= px < feature_size and 0 <= py < feature_size:
                            # 只有当前是ignore时才标记为背景
                            # （避免覆盖其他目标的正样本）
                            if cls_map[py, px] == -100:
                                # num_classes是背景类的ID（例如7个类，背景是第8类）
                                # 注意：需要在模型输出层添加一个背景通道
                                cls_map[py, px] = 7  # 假设有7个前景类，背景类ID=7
                
                # 步骤2: 再标记正样本区域（前景）
                # 这会覆盖步骤1中心区域的背景标签
                # 🔥🔥🔥 严重BUG修复：每个点计算自己的ltrb
                for dy in range(-positive_radius, positive_radius + 1):
                    for dx in range(-positive_radius, positive_radius + 1):
                        px = grid_x + dx
                        py = grid_y + dy
                        
                        # 边界检查
                        if 0 <= px < feature_size and 0 <= py < feature_size:
                            # 🔥 关键修复：每个点计算自己的中心
                            px_center = px + 0.5
                            py_center = py + 0.5
                            
                            # 🔥 关键修复：相对于该点的中心计算ltrb
                            l_point = px_center - x1_feat
                            t_point = py_center - y1_feat
                            r_point = x2_feat - px_center
                            b_point = y2_feat - py_center
                            
                            # 确保该点在box内部（ltrb都为正）
                            if l_point > 0 and t_point > 0 and r_point > 0 and b_point > 0:
                                # 设置前景类标签
                                cls_map[py, px] = label
                                
                                # 🔥 修复：使用该点自己的ltrb
                                bbox_map[0, py, px] = l_point
                                bbox_map[1, py, px] = t_point
                                bbox_map[2, py, px] = r_point
                                bbox_map[3, py, px] = b_point
                                
                                # 🔥 修复：重新计算该点的centerness
                                min_lr = min(l_point, r_point)
                                max_lr = max(l_point, r_point) + 1e-6
                                min_tb = min(t_point, b_point)
                                max_tb = max(t_point, b_point) + 1e-6
                                centerness_point = ((min_lr / max_lr) * (min_tb / max_tb)) ** 0.5
                                centerness_map[0, py, px] = centerness_point
                                
                                # 设置有效掩码
                                valid_mask[py, px] = True
            
            cls_maps.append(cls_map)
            bbox_maps.append(bbox_map)
            valid_masks.append(valid_mask)
            centerness_maps.append(centerness_map)
        
        # 堆叠为批次
        frame_target = {
            'cls': torch.stack(cls_maps, dim=0).to(device),  # (B, H, W)
            'bbox': torch.stack(bbox_maps, dim=0).to(device),  # (B, 4, H, W)
            'valid': torch.stack(valid_masks, dim=0).to(device),  # (B, H, W)
            'centerness': torch.stack(centerness_maps, dim=0).to(device)  # (B, 1, H, W)
        }
        
        frame_targets.append(frame_target)
    
    # 计算offset target（相邻帧之间的偏移）
    for t in range(num_frames - 1):
        offset_maps = []
        
        for b in range(batch_size):
            offset_map = torch.zeros(2, feature_size, feature_size, dtype=torch.float32)
            
            target = targets_batch[b]
            
            # 检查是否有 tracking_id 和 offset_info
            if use_tracking_offset and 'frames_offsets' in target:
                # 使用预计算的 offset
                offset_info = target['frames_offsets'][t]
                offset_map = offset_info['offset_map'].clone()
            else:
                # 简化版本：硬编码为 0
                # 如果没有启用 tracking 或没有 offset 信息，使用默认值
                valid_mask_t = frame_targets[t]['valid'][b]  # (H, W)
                offset_map[0, valid_mask_t] = 0.0  # dx
                offset_map[1, valid_mask_t] = 0.0  # dy
            
            offset_maps.append(offset_map)
        
        # 添加offset到当前帧的target
        frame_targets[t]['offset'] = torch.stack(offset_maps, dim=0).to(device)  # (B, 2, H, W)
    
    # 最后一帧没有offset（没有下一帧）
    # 不添加offset字段，在loss计算中会被跳过
    
    return frame_targets


def convert_targets_simple(
    targets_batch: List[Dict],
    num_frames: int,
    device: torch.device
) -> List[Dict]:
    """
    简化版本：直接复制标注到每一帧
    （假设所有帧的标注相同，适用于静态场景或序列标注）
    
    参数：
        targets_batch: List[Dict]，长度为 B
        num_frames: 帧数 T
        device: 设备
        
    返回：
        frame_targets: List[Dict]，长度为 T
    """
    frame_targets = []
    
    for t in range(num_frames):
        # 每帧使用相同的标注（简化处理）
        frame_target = {
            'boxes_list': [],  # 存储每个样本的 boxes
            'labels_list': []  # 存储每个样本的 labels
        }
        
        for target in targets_batch:
            frame_target['boxes_list'].append(target['boxes'])
            frame_target['labels_list'].append(target['labels'])
        
        frame_targets.append(frame_target)
    
    return frame_targets


if __name__ == "__main__":
    # 测试转换
    print("测试目标格式转换...")
    
    # 模拟数据
    targets_batch = [
        {
            'boxes': [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]],
            'labels': [0, 1]
        },
        {
            'boxes': [[0.2, 0.2, 0.4, 0.4]],
            'labels': [2]
        }
    ]
    
    device = torch.device('cpu')
    frame_targets = convert_targets_for_loss(
        targets_batch,
        num_frames=5,
        img_size=640,
        feature_size=40,
        device=device
    )
    
    print(f"转换后帧数: {len(frame_targets)}")
    print(f"第一帧 cls shape: {frame_targets[0]['cls'].shape}")
    print(f"第一帧 bbox shape: {frame_targets[0]['bbox'].shape}")
    print(f"第一帧 valid shape: {frame_targets[0]['valid'].shape}")
    print(f"第一帧有效位置数: {frame_targets[0]['valid'].sum().item()}")
    
    print("\n✓ 目标格式转换测试通过！")


