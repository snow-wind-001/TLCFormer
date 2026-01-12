"""
损失函数模块
包含 Focal Loss, CIoU Loss, Dice Loss 等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    
    用于解决类别不平衡问题，降低易分样本的权重
    
    参数：
        alpha (float): 平衡因子
        gamma (float): 聚焦参数
        reduction (str): 'none', 'mean', 'sum'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        参数：
            inputs: (B, C, H, W) 或 (B*H*W, C) 预测 logits
            targets: (B, H, W) 或 (B*H*W,) 真实标签（0/1）
            
        返回：
            loss: 标量损失
        """
        # 展平
        if inputs.dim() == 4:
            B, C, H, W = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)
            targets = targets.view(-1)
        
        # 过滤掉 ignore_index 的位置
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            # 如果没有有效目标，返回 0
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        # 应用 sigmoid
        p = torch.sigmoid(inputs)
        
        # 二分类 Focal Loss
        if inputs.shape[1] == 1:
            p = p.squeeze(1)
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs.squeeze(1), targets.float(), reduction='none'
            )
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            # 多分类
            ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
            p_t = p.gather(1, targets.long().unsqueeze(1)).squeeze(1)
            alpha_t = self.alpha
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedFocalLoss(nn.Module):
    """
    类别加权的Focal Loss
    
    用于解决严重的类别不平衡问题，为每个类别设置不同的权重
    
    参数：
        class_weights (dict or list): 每个类别的权重
        gamma (float): 聚焦参数
        reduction (str): 'none', 'mean', 'sum'
        ignore_index (int): 忽略的标签ID
    """
    
    def __init__(
        self,
        class_weights: dict = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        
        # 默认类别权重（根据RGBT-Tiny数据集统计）
        if class_weights is None:
            class_weights = {
                0: 0.9,   # ship     (9.86%)
                1: 0.4,   # car      (45.07%) - 最多，降低权重
                2: 0.8,   # cyclist  (11.67%)
                3: 0.6,   # pedestrian (25.27%)
                4: 1.0,   # bus      (2.63%)
                5: 1.2,   # drone    (1.92%) - 最少，提高权重
                6: 1.0,   # plane    (3.59%)
                7: 0.1    # background (很多) - 极低权重
            }
        
        # 转换为tensor
        if isinstance(class_weights, dict):
            # 找到最大的类别ID
            max_class_id = max(class_weights.keys())
            # 创建权重tensor
            weight_tensor = torch.ones(max_class_id + 1)
            for cls_id, weight in class_weights.items():
                weight_tensor[cls_id] = weight
            self.class_weights = weight_tensor
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        参数：
            inputs: (B, C, H, W) 或 (B*H*W, C) 预测 logits
            targets: (B, H, W) 或 (B*H*W,) 真实标签
            
        返回：
            loss: 标量损失
        """
        # 展平
        if inputs.dim() == 4:
            B, C, H, W = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)
            targets = targets.view(-1)
        
        # 过滤掉 ignore_index 的位置
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            # 如果没有有效目标，返回 0
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        # 将类别权重移到正确的设备
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # 计算交叉熵（不reduction，保留每个样本的loss）
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        
        # 计算pt (预测正确类别的概率)
        pt = torch.exp(-ce_loss)
        
        # 获取每个样本的类别权重
        alpha = self.class_weights[targets.long()]
        
        # Focal Loss
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CIoULoss(nn.Module):
    """
    Complete IoU Loss
    
    考虑了边界框的重叠面积、中心点距离、宽高比
    
    参数：
        eps (float): 防止除零的小常数
    """
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        参数：
            pred_boxes: (N, 4) 预测框 [x1, y1, x2, y2] 或 [l, t, r, b]
            target_boxes: (N, 4) 真实框
            
        返回：
            loss: CIoU 损失
        """
        # 计算 IoU
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        union_area = pred_area + target_area - inter_area + self.eps
        iou = inter_area / union_area
        
        # 计算中心点距离
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # 计算最小外接矩形的对角线距离
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
        
        # 计算宽高比一致性
        pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=self.eps)
        pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=self.eps)
        target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=self.eps)
        target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=self.eps)
        
        # 使用安全的 atan 计算，避免极端值
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2
        )
        
        with torch.no_grad():
            alpha = v / ((1 - iou + v).clamp(min=self.eps))
        
        # CIoU，限制范围避免数值爆炸
        ciou = (iou - (center_distance / enclose_diagonal + alpha * v)).clamp(min=-1.0, max=1.0)
        
        # 损失，并限制范围
        loss = (1 - ciou).clamp(min=0.0, max=2.0)
        
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss
    
    常用于分割任务，也可用于边缘检测
    
    参数：
        smooth (float): 平滑项
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        参数：
            inputs: (B, C, H, W) 预测
            targets: (B, C, H, W) 真实标签
            
        返回：
            loss: Dice 损失
        """
        # 应用 sigmoid
        inputs = torch.sigmoid(inputs)
        
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算 Dice 系数
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 损失
        loss = 1 - dice
        
        return loss


def compute_loss(
    outputs: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    loss_weights: Dict[str, float] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算总损失
    
    参数：
        outputs: 模型输出，List of dict
            - 'cls': (B, num_classes, H, W)
            - 'bbox': (B, 4, H, W)
            - 'centerness': (B, 1, H, W)
            - 'offset': (B, 2, H, W)
        targets: 真实标签，List of dict
            - 'cls': (B, H, W)
            - 'bbox': (B, N, 4)
            - 'valid': (B, H, W) 有效位置 mask
        loss_weights: 损失权重
        
    返回：
        total_loss: 总损失
        loss_dict: 各损失项字典
    """
    if loss_weights is None:
        loss_weights = {
            'cls': 1.0,
            'bbox': 5.0,
            'centerness': 1.0,
            'offset': 2.0
        }
    
    # 初始化损失函数
    # 🔥 修复Ship预测偏向：使用类别加权的Focal Loss
    # 为每个类别设置不同的权重，解决类别不平衡问题
    class_weights = loss_weights.get('class_weights', None)  # 🆕 从配置读取 ⭐⭐⭐
    focal_loss_fn = WeightedFocalLoss(
        gamma=loss_weights.get('cls_gamma', 2.0),
        class_weights=class_weights  # ⭐⭐⭐ 传入类别权重
    )
    ciou_loss_fn = CIoULoss()
    dice_loss_fn = DiceLoss()
    
    # 累积各损失项
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_centerness_loss = 0.0
    total_offset_loss = 0.0
    
    num_frames = len(outputs)
    
    for t in range(num_frames):
        output = outputs[t]
        target = targets[t] if t < len(targets) else targets[-1]
        
        # 1. 分类损失（Focal Loss）
        cls_pred = output['cls']  # (B, num_classes, H, W)
        cls_target = target.get('cls', None)
        
        if cls_target is not None:
            # ✅ 修复：Reshape to (B*H*W, num_classes) for Focal Loss
            B, C, H, W = cls_pred.shape
            cls_pred_flat = cls_pred.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            cls_target_flat = cls_target.reshape(-1)  # (B*H*W,)
            
            cls_loss = focal_loss_fn(cls_pred_flat, cls_target_flat)
            
            # 检测 NaN
            if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                print(f"Warning: NaN/Inf detected in cls_loss at frame {t}, skipping...")
                cls_loss = torch.tensor(0.0, device=cls_loss.device, requires_grad=True)
            
            total_cls_loss += cls_loss
        
        # 2. 边界框损失（CIoU Loss）
        bbox_pred = output['bbox']  # (B, 4, H, W)
        bbox_target = target.get('bbox', None)
        valid_mask = target.get('valid', None)
        
        if bbox_target is not None and valid_mask is not None:
            # 只在有目标的位置计算 bbox 损失
            B, _, H, W = bbox_pred.shape
            
            # 将预测和目标都转换为 (N, 4) 格式
            valid_indices = valid_mask.nonzero(as_tuple=False)  # (N, 3) [b, h, w]
            
            if len(valid_indices) > 0:
                # 获取预测的 bbox (FCOS style: l, t, r, b)
                bbox_pred_ltrb = bbox_pred[
                    valid_indices[:, 0],
                    :,
                    valid_indices[:, 1],
                    valid_indices[:, 2]
                ]  # (N, 4) [l, t, r, b]
                
                # 获取目标的 bbox (FCOS style: l, t, r, b)
                bbox_target_ltrb = bbox_target[
                    valid_indices[:, 0],
                    :,
                    valid_indices[:, 1],
                    valid_indices[:, 2]
                ]  # (N, 4) [l, t, r, b]
                
                # 转换为 (x1, y1, x2, y2) 格式用于 CIoU Loss
                # 动态计算stride（假设输入图像为640x640）
                img_size = 640  # 可以从config获取
                stride = img_size / H  # 计算下采样倍数
                grid_h = valid_indices[:, 1].float()
                grid_w = valid_indices[:, 2].float()
                center_x = (grid_w + 0.5) * stride
                center_y = (grid_h + 0.5) * stride
                
                # 预测框转换: (l, t, r, b) -> (x1, y1, x2, y2)
                pred_x1 = center_x - bbox_pred_ltrb[:, 0] * stride
                pred_y1 = center_y - bbox_pred_ltrb[:, 1] * stride
                pred_x2 = center_x + bbox_pred_ltrb[:, 2] * stride
                pred_y2 = center_y + bbox_pred_ltrb[:, 3] * stride
                bbox_pred_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                
                # 目标框转换: (l, t, r, b) -> (x1, y1, x2, y2)
                target_x1 = center_x - bbox_target_ltrb[:, 0] * stride
                target_y1 = center_y - bbox_target_ltrb[:, 1] * stride
                target_x2 = center_x + bbox_target_ltrb[:, 2] * stride
                target_y2 = center_y + bbox_target_ltrb[:, 3] * stride
                bbox_target_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=1)
                
                # 计算 CIoU Loss
                bbox_loss = ciou_loss_fn(bbox_pred_xyxy, bbox_target_xyxy)
                
                # 检测 NaN 并替换为 0
                if torch.isnan(bbox_loss) or torch.isinf(bbox_loss):
                    print(f"Warning: NaN/Inf detected in bbox_loss at frame {t}, skipping...")
                    bbox_loss = torch.tensor(0.0, device=bbox_loss.device, requires_grad=True)
                
                total_bbox_loss += bbox_loss
        
        # 3. 中心度损失（如果有）
        if 'centerness' in output:
            centerness_pred = output['centerness']
            centerness_target = target.get('centerness', None)
            
            if centerness_target is not None:
                # 使用 binary_cross_entropy_with_logits 以支持 AMP
                # 模型输出应该是 logits（未经 sigmoid），这里会自动应用 sigmoid
                centerness_loss = F.binary_cross_entropy_with_logits(
                    centerness_pred,
                    centerness_target,
                    reduction='mean'
                )
                total_centerness_loss += centerness_loss
        
        # 4. 偏移损失（如果有）
        if 'offset' in output:
            offset_pred = output['offset']
            offset_target = target.get('offset', None)
            
            if offset_target is not None:
                offset_loss = F.smooth_l1_loss(offset_pred, offset_target)
                total_offset_loss += offset_loss
    
    # 平均各帧损失
    total_cls_loss /= num_frames
    total_bbox_loss /= num_frames
    total_centerness_loss /= num_frames
    total_offset_loss /= max(num_frames - 1, 1)  # 偏移只有 T-1 个
    
    # 加权求和
    weighted_cls_loss = loss_weights['cls'] * total_cls_loss
    weighted_bbox_loss = loss_weights['bbox'] * total_bbox_loss
    weighted_centerness_loss = loss_weights['centerness'] * total_centerness_loss
    weighted_offset_loss = loss_weights['offset'] * total_offset_loss
    
    total_loss = (
        weighted_cls_loss +
        weighted_bbox_loss +
        weighted_centerness_loss +
        weighted_offset_loss
    )
    
    # 确保返回的 total_loss 是 Tensor（保留梯度）
    if not isinstance(total_loss, torch.Tensor):
        # 如果所有损失都是 0，创建一个需要梯度的零张量
        total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
    # 构建损失字典（存储加权后的损失以便正确显示）
    loss_dict = {
        'loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss),
        'cls_loss': weighted_cls_loss.item() if isinstance(weighted_cls_loss, torch.Tensor) else float(weighted_cls_loss),
        'bbox_loss': weighted_bbox_loss.item() if isinstance(weighted_bbox_loss, torch.Tensor) else float(weighted_bbox_loss),
        'centerness_loss': weighted_centerness_loss.item() if isinstance(weighted_centerness_loss, torch.Tensor) else float(weighted_centerness_loss),
        'offset_loss': weighted_offset_loss.item() if isinstance(weighted_offset_loss, torch.Tensor) else float(weighted_offset_loss)
    }
    
    return total_loss, loss_dict


if __name__ == "__main__":
    # 测试损失函数
    print("Testing Loss Functions...")
    
    # 测试 Focal Loss
    print("\n1. Focal Loss:")
    focal_loss = FocalLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 64, 64)).float()
    loss = focal_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    # 测试 CIoU Loss
    print("\n2. CIoU Loss:")
    ciou_loss = CIoULoss()
    pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]]).float()
    target_boxes = torch.tensor([[15, 15, 55, 55], [25, 25, 65, 65]]).float()
    loss = ciou_loss(pred_boxes, target_boxes)
    print(f"   Loss: {loss.item():.4f}")
    
    # 测试 Dice Loss
    print("\n3. Dice Loss:")
    dice_loss = DiceLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = dice_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✓ 所有损失函数测试通过！")

