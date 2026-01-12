"""
TLCFormer / OSFormer: Temporal-Local-Contrast Transformer for Infrared Video Small Object Detection
完整模型实现，集成所有模块

核心改进（相比原 OSFormer）：
1. MADA (Motion-Aware Difference Attention): 替代 Doppler Filter，使用帧差而非 FFT
2. DLCM (Deep Local Contrast Module): 局部对比度增强，利用小目标的局部极值特性
3. Hybrid Energy-Preserving Mixer: 在 VPA 中使用 Max-Mean 混合池化

物理先验：
- 利用 |I_t - I_{t-1}| 对抗云层/背景杂波（背景不动，目标动）
- 利用 L_max / L_mean 对抗低信噪比（目标是局部极值）
- 利用 MaxPool 对抗下采样能量损失（防止网络层级加深时目标消失）
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from .cube_encoding import CubeEncoding
from .vpa import VariedSizePatchAttention
from .mada import MotionAwareDifferenceAttention
from .dlcm import DeepLocalContrastModule, DLCMForCube
from .neck import FeatureRefinementNeck
from .seq_head import SequenceRegressionHead

# 保留 Doppler Filter 以支持向后兼容
try:
    from .doppler_filter import DopplerAdaptiveFilter
except ImportError:
    DopplerAdaptiveFilter = None


class OSFormer(nn.Module):
    """
    TLCFormer / OSFormer 主模型
    
    架构流程：
    1. Cube Encoding: 将视频序列编码为 4D cube (B, C, H, W, S)
    2. MADA: 运动感知差分注意力（替代 Doppler Filter）
    3. DLCM: 深度局部对比度增强（可选）
    4. VPA Encoder: 多尺度特征提取（使用 Hybrid Mixer）
    5. Feature Refinement Neck: 特征精炼与背景抑制
    6. Sequence Head: 多帧检测和轨迹关联
    
    参数：
        num_frames (int): 输入视频帧数 T
        sample_frames (int): 采样帧数 S
        img_size (int): 图像尺寸
        num_classes (int): 类别数
        embed_dim (int): VPA 嵌入维度
        depths (List[int]): VPA 各阶段深度
        use_mada (bool): 是否使用 MADA（替代 Doppler）
        use_dlcm (bool): 是否使用 DLCM
        use_doppler (bool): 是否使用旧版 Doppler Filter（向后兼容）
        anchor_free (bool): 是否使用 anchor-free 检测
        mada_alpha (float): MADA 缩放因子
        dlcm_beta (float): DLCM 融合权重
    """
    
    def __init__(
        self,
        num_frames: int = 5,
        sample_frames: int = 3,
        img_size: int = 640,
        num_classes: int = 1,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        use_mada: bool = True,
        use_dlcm: bool = True,
        use_doppler: bool = False,  # 默认禁用旧版 Doppler
        anchor_free: bool = True,
        dropout: float = 0.1,
        mada_alpha: float = 0.5,
        dlcm_beta: float = 0.5
    ):
        super().__init__()
        self.num_frames = num_frames
        self.sample_frames = sample_frames
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_mada = use_mada
        self.use_dlcm = use_dlcm
        self.use_doppler = use_doppler and not use_mada  # MADA 优先于 Doppler
        
        # 1. Cube Encoding 模块
        self.cube_encoder = CubeEncoding(
            num_frames=num_frames,
            sample_frames=sample_frames,
            img_size=img_size,
            normalize=True
        )
        
        # 2. MADA: Motion-Aware Difference Attention（替代 Doppler）
        if use_mada:
            self.mada = MotionAwareDifferenceAttention(
                num_frames=sample_frames,
                in_channels=2,  # 灰度 + 热红外
                alpha=mada_alpha
            )
        
        # 3. DLCM: Deep Local Contrast Module
        if use_dlcm:
            self.dlcm = DLCMForCube(
                in_channels=2,
                kernel_inner=3,
                kernel_outer=9,
                use_soft_contrast=False,
                beta=dlcm_beta
            )
        
        # 向后兼容：旧版 Doppler Adaptive Filter
        if self.use_doppler and DopplerAdaptiveFilter is not None:
            self.doppler_filter = DopplerAdaptiveFilter(
                img_size=img_size,
                num_frames=sample_frames,
                learn_filter=True,
                filter_type='adaptive'
            )
        
        # 3. Varied-Size Patch Attention
        # 输入通道数 = cube 通道数 (2) * 采样帧数 (S)
        in_channels = 2 * sample_frames
        
        self.vpa_encoder = VariedSizePatchAttention(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            img_size=img_size,
            patch_size=4,
            drop_rate=dropout
        )
        
        # 计算多尺度特征的通道数
        # 假设 depths = [2, 2, 6, 2]，则通道数为 [96, 192, 384, 768]
        num_stages = len(depths)
        feature_dims = [embed_dim * (2 ** i) for i in range(num_stages)]
        
        # 4. Feature Refinement Neck (按论文添加)
        neck_out_channels = 256
        self.neck = FeatureRefinementNeck(
            in_channels_list=feature_dims,  # [96, 192, 384, 768]
            out_channels=neck_out_channels,
            use_background_suppression=True
        )
        
        # 5. Sequence Regression Head
        # 使用 Neck 输出的 F0 特征
        self.seq_head = SequenceRegressionHead(
            in_channels=neck_out_channels,  # 256
            num_classes=num_classes,
            num_frames=num_frames,
            anchor_free=anchor_free
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        rgb_frames: torch.Tensor,
        thermal_frames: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        前向传播（TLCFormer 改进流程）
        
        参数：
            rgb_frames: (B, T, 3, H, W) RGB 视频序列
            thermal_frames: (B, T, 1, H, W) 热红外视频序列
            
        返回：
            outputs: List of dict，长度为 T（帧数）
                每个 dict 包含：
                - 'cls': (B, num_classes, H', W') 分类预测
                - 'bbox': (B, 4, H', W') 边界框预测
                - 'centerness': (B, 1, H', W') 中心度（如果 anchor_free）
                - 'offset': (B, 2, H', W') 跨帧偏移（除最后一帧）
        """
        # 1. Cube Encoding
        cube = self.cube_encoder(rgb_frames, thermal_frames)  # (B, 2, H, W, S)
        
        # 2. MADA: 运动感知差分注意力（替代 Doppler Filter）
        if self.use_mada:
            cube = self.mada(cube)  # (B, 2, H, W, S)
        elif self.use_doppler and hasattr(self, 'doppler_filter'):
            # 向后兼容：旧版 Doppler Filter
            cube = self.doppler_filter(cube)  # (B, 2, H, W, S)
        
        # 3. DLCM: 深度局部对比度增强
        if self.use_dlcm:
            cube = self.dlcm(cube)  # (B, 2, H, W, S)
        
        # 4. VPA Encoder - 多尺度特征提取（使用 Hybrid Mixer）
        features = self.vpa_encoder(cube)  # List of (B, C_i, H_i, W_i): [F1, F2, F3, F4]
        
        # 5. Feature Refinement Neck - 特征精炼 + 背景抑制
        F0 = self.neck(features)  # (B, 256, H/16, W/16)
        
        # 6. Sequence Regression Head - 检测头
        outputs = self.seq_head(F0)  # List of dict
        
        return outputs
    
    def get_loss(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        loss_weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算损失
        
        参数：
            outputs: 模型输出
            targets: 真实标签
            loss_weights: 各损失项权重
            
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
        
        # 导入损失函数
        from ..utils.loss import compute_loss
        
        total_loss, loss_dict = compute_loss(
            outputs, targets, loss_weights
        )
        
        return total_loss, loss_dict


class OSFormerConfig:
    """TLCFormer / OSFormer 配置类"""
    
    def __init__(self, **kwargs):
        # 模型配置
        self.num_frames = kwargs.get('num_frames', 5)
        self.sample_frames = kwargs.get('sample_frames', 3)
        self.img_size = kwargs.get('img_size', 640)
        self.num_classes = kwargs.get('num_classes', 1)
        self.embed_dim = kwargs.get('embed_dim', 96)
        self.depths = kwargs.get('depths', [2, 2, 6, 2])
        
        # TLCFormer 新增配置
        self.use_mada = kwargs.get('use_mada', True)  # 使用 MADA 替代 Doppler
        self.use_dlcm = kwargs.get('use_dlcm', True)  # 使用 DLCM
        self.use_doppler = kwargs.get('use_doppler', False)  # 旧版 Doppler（向后兼容）
        self.mada_alpha = kwargs.get('mada_alpha', 0.5)  # MADA 缩放因子
        self.dlcm_beta = kwargs.get('dlcm_beta', 0.5)  # DLCM 融合权重
        
        self.anchor_free = kwargs.get('anchor_free', True)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # 训练配置
        self.lr = kwargs.get('lr', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 0.05)
        self.batch_size = kwargs.get('batch_size', 8)
        self.num_epochs = kwargs.get('num_epochs', 50)
        
        # 损失权重
        self.loss_weights = kwargs.get('loss_weights', {
            'cls': 1.0,
            'bbox': 5.0,
            'centerness': 1.0,
            'offset': 2.0
        })
    
    def to_dict(self):
        """转换为字典"""
        return {
            'num_frames': self.num_frames,
            'sample_frames': self.sample_frames,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'depths': self.depths,
            'use_mada': self.use_mada,
            'use_dlcm': self.use_dlcm,
            'use_doppler': self.use_doppler,
            'mada_alpha': self.mada_alpha,
            'dlcm_beta': self.dlcm_beta,
            'anchor_free': self.anchor_free,
            'dropout': self.dropout,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'loss_weights': self.loss_weights
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        return cls(**config_dict)


# TLCFormer 别名（向后兼容）
TLCFormerConfig = OSFormerConfig


def build_osformer(config: OSFormerConfig = None, **kwargs) -> OSFormer:
    """
    构建 TLCFormer / OSFormer 模型
    
    参数：
        config: 模型配置（可选）
        **kwargs: 直接传递的配置参数（如果 config 为 None）
        
    返回：
        model: OSFormer 模型
    """
    if config is None:
        # 如果没有提供 config，从 kwargs 创建
        if kwargs:
            config = OSFormerConfig(**kwargs)
        else:
            config = OSFormerConfig()
    
    model = OSFormer(
        num_frames=config.num_frames,
        sample_frames=config.sample_frames,
        img_size=config.img_size,
        num_classes=config.num_classes,
        embed_dim=config.embed_dim,
        depths=config.depths,
        use_mada=config.use_mada,
        use_dlcm=config.use_dlcm,
        use_doppler=config.use_doppler,
        anchor_free=config.anchor_free,
        dropout=config.dropout,
        mada_alpha=config.mada_alpha,
        dlcm_beta=config.dlcm_beta
    )
    
    return model


# 别名（向后兼容和新命名）
build_tlcformer = build_osformer
TLCFormer = OSFormer


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Testing TLCFormer (OSFormer with MADA + DLCM + Hybrid Mixer)")
    print("=" * 60)
    
    # 创建配置（使用新的 TLCFormer 特性）
    config = OSFormerConfig(
        num_frames=5,
        sample_frames=3,
        img_size=640,
        num_classes=7,  # RGBT-Tiny 有 7 个类别
        embed_dim=96,
        depths=[2, 2, 6, 2],
        use_mada=True,   # 启用 MADA
        use_dlcm=True,   # 启用 DLCM
        use_doppler=False,  # 禁用旧版 Doppler
        mada_alpha=0.5,
        dlcm_beta=0.5
    )
    
    print("\n模型配置:")
    print(f"  use_mada: {config.use_mada}")
    print(f"  use_dlcm: {config.use_dlcm}")
    print(f"  use_doppler: {config.use_doppler}")
    print(f"  mada_alpha: {config.mada_alpha}")
    print(f"  dlcm_beta: {config.dlcm_beta}")
    
    # 构建模型
    model = build_osformer(config)
    
    # 模拟输入
    B, T, H, W = 2, 5, 640, 640
    rgb_frames = torch.randn(B, T, 3, H, W)
    thermal_frames = torch.randn(B, T, 1, H, W)
    
    print(f"\n输入:")
    print(f"  RGB: {rgb_frames.shape}")
    print(f"  Thermal: {thermal_frames.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(rgb_frames, thermal_frames)
    
    print(f"\n输出（每帧预测）:")
    for t, output in enumerate(outputs):
        print(f"  Frame {t}:")
        for key, val in output.items():
            print(f"    {key}: {val.shape}")
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"  总参数量: {num_params / 1e6:.2f}M")
    print(f"  可训练参数: {num_trainable / 1e6:.2f}M")
    
    # 验证新模块
    print("\n核心模块验证:")
    print(f"  ✓ MADA: {hasattr(model, 'mada')}")
    print(f"  ✓ DLCM: {hasattr(model, 'dlcm')}")
    print(f"  ✓ VPA (Hybrid Mixer): {hasattr(model, 'vpa_encoder')}")
    
    print("\n🎉 TLCFormer 测试通过！")
    print("\n改进总结:")
    print("  1. MADA: 帧差运动注意力，替代 FFT 多普勒滤波")
    print("  2. DLCM: 局部对比度增强，利用小目标极值特性")
    print("  3. Hybrid Mixer: Max-Mean 混合池化，保留小目标能量")

