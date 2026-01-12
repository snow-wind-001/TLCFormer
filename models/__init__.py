"""
TLCFormer / OSFormer 模型组件包

核心模块：
- CubeEncoding: 视频序列编码为 4D Cube
- MADA: Motion-Aware Difference Attention（运动感知差分注意力）
- DLCM: Deep Local Contrast Module（局部对比度增强）
- VPA: Varied-Size Patch Attention（使用 Hybrid Mixer）
- OSFormer/TLCFormer: 主模型
"""

from .cube_encoding import CubeEncoding
from .vpa import VariedSizePatchAttention, VPABlock, HybridTokenMixer
from .mada import MotionAwareDifferenceAttention, MADALight
from .dlcm import DeepLocalContrastModule, DLCMForCube, DLCMLight
from .seq_head import SequenceRegressionHead
from .neck import FeatureRefinementNeck, BackgroundSuppressionModule
from .osformer import OSFormer, OSFormerConfig, build_osformer

# TLCFormer 别名
from .osformer import TLCFormer, TLCFormerConfig, build_tlcformer

# 向后兼容：保留 Doppler Filter
try:
    from .doppler_filter import DopplerAdaptiveFilter
except ImportError:
    DopplerAdaptiveFilter = None

__all__ = [
    # 核心编码模块
    'CubeEncoding',
    
    # TLCFormer 新增模块
    'MotionAwareDifferenceAttention',
    'MADALight',
    'DeepLocalContrastModule',
    'DLCMForCube',
    'DLCMLight',
    'HybridTokenMixer',
    
    # VPA 模块
    'VariedSizePatchAttention',
    'VPABlock',
    
    # Neck 模块
    'FeatureRefinementNeck',
    'BackgroundSuppressionModule',
    
    # 检测头
    'SequenceRegressionHead',
    
    # 主模型
    'OSFormer',
    'OSFormerConfig',
    'build_osformer',
    
    # TLCFormer 别名
    'TLCFormer',
    'TLCFormerConfig',
    'build_tlcformer',
    
    # 向后兼容
    'DopplerAdaptiveFilter'
]
