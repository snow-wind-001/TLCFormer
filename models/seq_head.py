"""
Sequence Regression Head 模块
用于多帧联合检测和轨迹关联
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class LightConv(nn.Module):
    """
    轻量级卷积模块
    使用深度可分离卷积降低计算量
    
    参数：
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3
    ):
        super().__init__()
        padding = kernel_size // 2
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class OffsetPredictor(nn.Module):
    """
    跨帧偏移预测器
    预测目标在相邻帧之间的位置偏移，用于轨迹关联
    
    参数：
        in_channels (int): 输入通道数
        num_frames (int): 帧数
    """
    
    def __init__(
        self,
        in_channels: int,
        num_frames: int = 5
    ):
        super().__init__()
        self.num_frames = num_frames
        
        # 偏移预测分支（预测 x, y 偏移）
        self.offset_head = nn.Sequential(
            LightConv(in_channels, in_channels // 2),
            LightConv(in_channels // 2, in_channels // 4),
            nn.Conv2d(in_channels // 4, 2, 1)  # 2: (dx, dy)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        参数：
            features: List of (B, C, H, W)，长度为 T
            
        返回：
            offsets: (B, T-1, 2, H, W) 相邻帧之间的偏移
        """
        offsets = []
        
        for t in range(len(features) - 1):
            # 计算相邻帧特征差异
            feat_diff = features[t+1] - features[t]  # (B, C, H, W)
            
            # 预测偏移
            offset = self.offset_head(feat_diff)  # (B, 2, H, W)
            offsets.append(offset)
        
        # 堆叠
        offsets = torch.stack(offsets, dim=1)  # (B, T-1, 2, H, W)
        
        return offsets


class SequenceRegressionHead(nn.Module):
    """
    序列回归头（按论文修改）
    
    功能：
    1. 接收 Neck 输出的精炼特征 F0
    2. 预测每帧的分类和边界框
    3. 预测跨帧偏移矩阵用于轨迹关联
    
    参数：
        in_channels (int): 输入特征通道数 (F0 的通道数)
        num_classes (int): 类别数（对于二分类检测，通常为 1）
        num_frames (int): 输入帧数
        anchor_free (bool): 是否使用 anchor-free 方式
    """
    
    def __init__(
        self,
        in_channels: int = 256,  # F0 的通道数
        num_classes: int = 1,
        num_frames: int = 5,
        anchor_free: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.anchor_free = anchor_free
        
        # 不再需要 FPN 融合！Neck 已经完成了特征融合
        # 直接在 F0 上进行检测
        
        # 分类头（每个位置预测是否有目标）
        # 🔥 改进：增加背景类通道（num_classes+1）
        # 例如：7个前景类 + 1个背景类 = 8个输出通道
        self.cls_head = nn.Sequential(
            LightConv(in_channels, in_channels),
            LightConv(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, num_classes + 1, 1)  # +1 for background
        )
        
        # 边界框回归头（预测 ltrb 或 xywh）
        bbox_out_dim = 4
        self.bbox_head = nn.Sequential(
            LightConv(in_channels, in_channels),
            LightConv(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, bbox_out_dim, 1),
            nn.ReLU()  # 保证边界框参数为正
        )
        
        # 中心度预测（用于 anchor-free，提升边界框质量）
        if anchor_free:
            self.centerness_head = nn.Sequential(
                LightConv(in_channels, in_channels // 2),
                nn.Conv2d(in_channels // 2, 1, 1)
                # 注意：不使用 Sigmoid，输出 logits 以支持 AMP
                # 损失函数使用 binary_cross_entropy_with_logits
                # 推理时需要手动应用 sigmoid
            )
        
        # 偏移预测器（用于跨帧关联）
        self.offset_predictor = OffsetPredictor(in_channels, num_frames)
        
    def forward(
        self, 
        F0: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        前向传播（按论文修改）
        
        参数：
            F0: (B, C0, H, W) Neck 输出的精炼特征
                
        返回：
            outputs: List of dict，长度为 T（帧数）
                每个 dict 包含：
                - 'cls': (B, num_classes, H, W) 分类 logits
                - 'bbox': (B, 4, H, W) 边界框预测
                - 'centerness': (B, 1, H, W) 中心度（如果 anchor_free）
                - 'offset': (B, 2, H, W) 到下一帧的偏移（除最后一帧）
        """
        # 直接在 F0 上进行预测（不需要 FPN 融合）
        
        # 分类预测
        cls_pred = self.cls_head(F0)  # (B, num_classes, H, W)
        
        # 边界框预测
        bbox_pred = self.bbox_head(F0)  # (B, 4, H, W)
        
        # 中心度预测（如果使用 anchor-free）
        if self.anchor_free:
            centerness_pred = self.centerness_head(F0)  # (B, 1, H, W)
        else:
            centerness_pred = None
        
        # 构建输出（每帧）
        # 注意：这里假设所有帧共享相同的检测头
        # 实际应用中，可能需要为每帧独立预测
        outputs = []
        for t in range(self.num_frames):
            output = {
                'cls': cls_pred,
                'bbox': bbox_pred,
            }
            if centerness_pred is not None:
                output['centerness'] = centerness_pred
            outputs.append(output)
        
        # 预测跨帧偏移（用于轨迹关联）
        # 使用同一特征预测所有帧的偏移
        offsets = self.offset_predictor([F0] * self.num_frames)
        
        # 将偏移添加到输出
        for t in range(self.num_frames - 1):
            outputs[t]['offset'] = offsets[:, t]  # (B, 2, H, W)
        
        return outputs


class AnchorFreeDecoder:
    """
    Anchor-Free 解码器
    将网络输出解码为最终的边界框
    
    参数：
        score_thresh (float): 分数阈值
        nms_thresh (float): NMS 阈值
    """
    
    def __init__(
        self,
        score_thresh: float = 0.3,
        nms_thresh: float = 0.5
    ):
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
    
    def decode_single_frame(
        self,
        cls_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        centerness_pred: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解码单帧预测
        
        参数：
            cls_pred: (H, W) 或 (1, H, W)
            bbox_pred: (4, H, W)
            centerness_pred: (1, H, W)
            
        返回：
            boxes: (N, 4) xyxy 格式
            scores: (N,)
            labels: (N,)
        """
        if cls_pred.dim() == 3:
            cls_pred = cls_pred.squeeze(0)
        if centerness_pred is not None and centerness_pred.dim() == 3:
            centerness_pred = centerness_pred.squeeze(0)
        
        H, W = cls_pred.shape
        device = cls_pred.device
        
        # 应用 sigmoid
        scores = torch.sigmoid(cls_pred)  # (H, W)
        
        # 如果有中心度，先应用sigmoid再乘以分数
        if centerness_pred is not None:
            # centerness_pred 是 logits，需要先应用 sigmoid
            centerness_scores = torch.sigmoid(centerness_pred.squeeze(0))
            scores = scores * centerness_scores
        
        # 筛选高分位置
        mask = scores > self.score_thresh
        if mask.sum() == 0:
            # 没有检测到目标
            return (
                torch.zeros(0, 4, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device)
            )
        
        # 获取位置和分数
        indices = mask.nonzero(as_tuple=False)  # (N, 2) [h_idx, w_idx]
        selected_scores = scores[mask]  # (N,)
        
        # 获取对应的边界框预测
        bbox_pred_selected = bbox_pred[:, mask]  # (4, N)
        bbox_pred_selected = bbox_pred_selected.t()  # (N, 4)
        
        # 解码边界框（FCOS 风格：l, t, r, b）
        h_indices = indices[:, 0].float()
        w_indices = indices[:, 1].float()
        
        # 假设 stride = 原图尺寸 / 特征图尺寸
        # 这里简化：直接使用特征图坐标
        x_center = w_indices
        y_center = h_indices
        
        l, t, r, b = bbox_pred_selected.unbind(dim=1)
        
        x1 = x_center - l
        y1 = y_center - t
        x2 = x_center + r
        y2 = y_center + b
        
        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)
        
        # NMS
        keep = self._nms(boxes, selected_scores, self.nms_thresh)
        
        boxes = boxes[keep]
        scores = selected_scores[keep]
        labels = torch.zeros(len(keep), dtype=torch.long, device=device)
        
        return boxes, scores, labels
    
    @staticmethod
    def _nms(
        boxes: torch.Tensor, 
        scores: torch.Tensor, 
        iou_threshold: float
    ) -> torch.Tensor:
        """简单的 NMS 实现"""
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)


if __name__ == "__main__":
    # 测试代码
    print("Testing SequenceRegressionHead...")
    
    # 创建模型
    seq_head = SequenceRegressionHead(
        in_channels_list=[384, 768],
        num_classes=1,
        num_frames=5,
        anchor_free=True
    )
    
    # 模拟多尺度特征输入
    B = 2
    F3 = torch.randn(B, 384, 40, 40)
    F4 = torch.randn(B, 768, 20, 20)
    features = [F3, F4]
    
    print(f"输入特征:")
    for i, feat in enumerate(features):
        print(f"  F{i+3}: {feat.shape}")
    
    # 前向传播
    outputs = seq_head(features)
    
    print(f"\n输出预测（每帧）:")
    for t, output in enumerate(outputs):
        print(f"  Frame {t}:")
        for key, val in output.items():
            print(f"    {key}: {val.shape}")
    
    # 验证
    assert len(outputs) == 5, "输出帧数错误"
    assert 'cls' in outputs[0], "缺少分类预测"
    assert 'bbox' in outputs[0], "缺少边界框预测"
    
    print("\n✓ SequenceRegressionHead 测试通过！")
    
    # 测试解码器
    print("\nTesting AnchorFreeDecoder...")
    decoder = AnchorFreeDecoder(score_thresh=0.3, nms_thresh=0.5)
    
    cls_pred = outputs[0]['cls'][0]  # (1, H, W)
    bbox_pred = outputs[0]['bbox'][0]  # (4, H, W)
    centerness_pred = outputs[0]['centerness'][0] if 'centerness' in outputs[0] else None
    
    boxes, scores, labels = decoder.decode_single_frame(
        cls_pred, bbox_pred, centerness_pred
    )
    
    print(f"检测结果:")
    print(f"  Boxes: {boxes.shape}")
    print(f"  Scores: {scores.shape}")
    print(f"  Labels: {labels.shape}")
    
    print("\n✓ AnchorFreeDecoder 测试通过！")

