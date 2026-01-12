"""
Feature Refinement Neck 模块
按照论文描述实现特征精炼和背景抑制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BackgroundSuppressionModule(nn.Module):
    """
    背景抑制模块
    
    功能：
    1. 使用 attention 机制生成前景/背景注意力图
    2. 抑制背景区域的特征响应
    3. 增强前景（目标）区域的特征
    
    参数：
        channels (int): 输入特征通道数
        reduction (int): 通道压缩比例
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        
        # 注意力生成分支
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // reduction, 3, padding=1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()  # 输出 [0, 1] attention map
        )
        
        # 特征增强分支
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: (B, C, H, W) 输入特征
            
        返回：
            out: (B, C, H, W) 背景抑制后的特征
        """
        # 生成前景注意力图
        # 高响应区域 → 前景，低响应区域 → 背景
        att_map = self.attention(x)  # (B, 1, H, W)
        
        # 加权特征（抑制背景）
        x_weighted = x * att_map
        
        # 特征增强
        x_enhanced = self.enhance(x_weighted)
        
        # 残差连接
        out = x + x_enhanced
        
        return out


class FeatureRefinementNeck(nn.Module):
    """
    特征精炼 Neck 模块
    
    按照论文 Section 3.3 描述实现：
    "utilizing upsampling and convolutional layers to refine F into features F0 
    with background suppression mechanism"
    
    功能：
    1. 接收 VPA 输出的多尺度特征 [F1, F2, F3, F4]
    2. 通过 FPN 风格的融合 + 上采样 + 卷积精炼
    3. 应用背景抑制机制
    4. 输出精炼后的特征 F0 (H/16, W/16)
    
    参数：
        in_channels_list (List[int]): 输入特征的通道数列表 [C1, C2, C3, C4]
        out_channels (int): 输出特征通道数 C0
        use_background_suppression (bool): 是否使用背景抑制
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        use_background_suppression: bool = True
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        
        # 1. Lateral convolutions (1x1 卷积调整通道数)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.lateral_convs.append(lateral_conv)
        
        # 2. Refinement convolutions (3x3 卷积精炼特征)
        self.refine_convs = nn.ModuleList()
        for i in range(self.num_levels):
            refine_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.refine_convs.append(refine_conv)
        
        # 3. Background Suppression Module
        if use_background_suppression:
            self.bg_suppression = BackgroundSuppressionModule(out_channels)
        else:
            self.bg_suppression = None
        
        # 4. Final Refinement (生成 F0)
        self.final_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模块权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        参数：
            features: List of [F1, F2, F3, F4]
                F1: (B, C1, H/4, W/4)
                F2: (B, C2, H/8, W/8)
                F3: (B, C3, H/16, W/16)
                F4: (B, C4, H/32, W/32)
                
        返回：
            F0: (B, C0, H/16, W/16) 精炼后的特征
        """
        assert len(features) == self.num_levels, \
            f"期望 {self.num_levels} 个特征层，得到 {len(features)} 个"
        
        # Step 1: Lateral convolutions (调整通道数)
        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            lateral = lateral_conv(feat)
            laterals.append(lateral)
        
        # Step 2: Top-down pathway (自顶向下融合)
        # 从最高层开始，逐层上采样并融合
        for i in range(self.num_levels - 1, 0, -1):
            # 上采样高层特征
            _, _, H, W = laterals[i-1].shape
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i],
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # Step 3: Refinement convolutions (精炼每一层)
        refined = []
        for i, (lateral, refine_conv) in enumerate(zip(laterals, self.refine_convs)):
            refined_feat = refine_conv(lateral)
            refined.append(refined_feat)
        
        # Step 4: 选择中间层作为 F0
        # 论文中 F0 的尺寸是 H/16, 对应 F3 (index 2)
        F0 = refined[2]  # (B, C0, H/16, W/16)
        
        # Step 5: Background Suppression
        if self.bg_suppression is not None:
            F0 = self.bg_suppression(F0)
        
        # Step 6: Final Refinement
        F0 = self.final_refine(F0)
        
        # 🔥 Step 7: 额外上采样 (stride 16→8) 用于小目标检测
        # 原因：97%的目标 <32²像素，stride=16时特征太小
        # 上采样后：640×512 → 80×64特征图，12px目标 → 1.5px特征 ✅
        F0 = F.interpolate(F0, scale_factor=2, mode='bilinear', align_corners=False)
        # 现在 F0: (B, C0, H/8, W/8) instead of (B, C0, H/16, W/16)
        
        return F0


class FeatureRefinementNeckV2(nn.Module):
    """
    特征精炼 Neck 模块 V2
    
    改进版本：融合所有层的特征到 H/16 尺度
    
    功能：
    1. 将所有特征上采样/下采样到 H/16 尺度
    2. 融合所有尺度的信息
    3. 应用背景抑制
    4. 输出 F0
    
    参数：
        in_channels_list (List[int]): 输入特征的通道数列表
        out_channels (int): 输出特征通道数
        use_background_suppression (bool): 是否使用背景抑制
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        use_background_suppression: bool = True
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        
        # 调整每层通道数
        self.adapt_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            adapt_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.adapt_convs.append(adapt_conv)
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * self.num_levels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 背景抑制
        if use_background_suppression:
            self.bg_suppression = BackgroundSuppressionModule(out_channels)
        else:
            self.bg_suppression = None
        
        # 最终精炼
        self.final_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        参数：
            features: List of [F1, F2, F3, F4]
                
        返回：
            F0: (B, C0, H/16, W/16)
        """
        # 目标尺寸：H/16 (对应 F3)
        target_size = features[2].shape[-2:]
        
        # 调整所有特征到相同尺寸和通道数
        adapted = []
        for i, (feat, adapt_conv) in enumerate(zip(features, self.adapt_convs)):
            # 调整通道
            feat = adapt_conv(feat)
            
            # 调整尺寸
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            adapted.append(feat)
        
        # 融合所有特征
        fused = torch.cat(adapted, dim=1)  # (B, C0*4, H/16, W/16)
        F0 = self.fusion_conv(fused)  # (B, C0, H/16, W/16)
        
        # 背景抑制
        if self.bg_suppression is not None:
            F0 = self.bg_suppression(F0)
        
        # 最终精炼
        F0 = self.final_refine(F0)
        
        return F0


if __name__ == "__main__":
    # 测试代码
    print("Testing FeatureRefinementNeck...")
    
    # 创建模型
    neck = FeatureRefinementNeck(
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        use_background_suppression=True
    )
    
    # 模拟输入特征金字塔
    B = 2
    F1 = torch.randn(B, 96, 160, 160)   # H/4
    F2 = torch.randn(B, 192, 80, 80)    # H/8
    F3 = torch.randn(B, 384, 40, 40)    # H/16
    F4 = torch.randn(B, 768, 20, 20)    # H/32
    features = [F1, F2, F3, F4]
    
    print(f"\n输入特征:")
    for i, feat in enumerate(features):
        print(f"  F{i+1}: {feat.shape}")
    
    # 前向传播
    with torch.no_grad():
        F0 = neck(features)
    
    print(f"\n输出特征:")
    print(f"  F0: {F0.shape}")
    
    # 验证
    assert F0.shape == (B, 256, 40, 40), f"F0 形状错误: {F0.shape}"
    
    print("\n✓ FeatureRefinementNeck 测试通过！")
    
    # 测试 V2
    print("\n" + "="*60)
    print("Testing FeatureRefinementNeckV2...")
    
    neck_v2 = FeatureRefinementNeckV2(
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        use_background_suppression=True
    )
    
    with torch.no_grad():
        F0_v2 = neck_v2(features)
    
    print(f"\n输出特征 (V2):")
    print(f"  F0: {F0_v2.shape}")
    
    assert F0_v2.shape == (B, 256, 40, 40), f"F0 形状错误: {F0_v2.shape}"
    
    print("\n✓ FeatureRefinementNeckV2 测试通过！")
    
    # 计算参数量
    num_params_v1 = sum(p.numel() for p in neck.parameters())
    num_params_v2 = sum(p.numel() for p in neck_v2.parameters())
    
    print(f"\n参数量对比:")
    print(f"  V1: {num_params_v1 / 1e6:.2f}M")
    print(f"  V2: {num_params_v2 / 1e6:.2f}M")
    
    print("\n✓ 所有测试通过！")

