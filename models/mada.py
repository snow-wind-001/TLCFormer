"""
Motion-Aware Difference Attention (MADA) 模块
基于时空差分的运动注意力机制，用于抑制静态背景噪声

替代原有的 Doppler Adaptive Filter，使用帧差而非 FFT 分离运动目标

算法原理：
1. 时域梯度计算：D_pre = |I_t - I_{t-1}|, D_next = |I_{t+1} - I_t|
2. 运动显著图：M_raw = D_pre ⊙ D_next（取前后差分的交集）
3. 注意力权重：A_motion = σ(F_motion(M_raw))
4. 特征加权：I'_t = I_t · (1 + α · A_motion)

物理先验：
- 小目标运动连续，背景静止
- 只有在连续两帧都存在的变化才被视为可靠运动
- 残差结构确保静止目标也能保留原始特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MotionAwareDifferenceAttention(nn.Module):
    """
    Motion-Aware Difference Attention (MADA)

    核心思想：利用小目标的运动连续性，通过帧差计算动态注意力掩码

    算法流程：
    1. 计算相邻帧的绝对差分：D_pre = |I_t - I_{t-1}|, D_next = |I_{t+1} - I_t|
    2. 生成运动显著图：M_raw = D_pre ⊙ D_next（取前后差分的交集）
    3. 通过轻量卷积网络映射为注意力权重：A_motion = σ(F_motion(M_raw))
    4. 特征加权：I'_t = I_t · (1 + α · A_motion)

    优势：
    - 不依赖 FFT，避免高频背景噪声干扰
    - 显式利用时间差分捕捉运动
    - 计算高效，适合实时检测

    参数：
        num_frames (int): 输入帧数 S（通常为3）
        in_channels (int): 输入通道数（通常为2：灰度+热红外）
        alpha (float): 初始缩放因子，可学习
    """

    def __init__(
        self,
        num_frames: int = 3,
        in_channels: int = 2,
        alpha: float = 0.5
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels

        # 可学习的缩放因子
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # 运动特征提取网络 F_motion
        # 输入: 运动显著图 (B, C, H, W)
        # 输出: 注意力权重 (B, C, H, W)
        self.motion_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出 [0, 1] 的注意力权重
        )

        # 跨通道融合（用于多通道输入时增强运动响应）
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def compute_temporal_gradient(
        self,
        frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算时域梯度（相邻帧差分）

        参数：
            frames: (B, S, C, H, W) 输入帧序列，S 通常为 3

        返回：
            D_pre: (B, C, H, W) 前向差分 |I_t - I_{t-1}|
            D_next: (B, C, H, W) 后向差分 |I_{t+1} - I_t|
        """
        B, S, C, H, W = frames.shape
        assert S >= 3, f"需要至少3帧输入，当前为 {S} 帧"

        # 提取 t-1, t, t+1 帧（取中间3帧或所有帧）
        if S == 3:
            I_pre = frames[:, 0]   # (B, C, H, W)
            I_mid = frames[:, 1]   # 中间帧 t
            I_next = frames[:, 2]
        else:
            # 如果多于3帧，取中间3帧
            mid_idx = S // 2
            I_pre = frames[:, mid_idx - 1]
            I_mid = frames[:, mid_idx]
            I_next = frames[:, mid_idx + 1]

        # 计算绝对差分
        D_pre = torch.abs(I_mid - I_pre)   # |I_t - I_{t-1}|
        D_next = torch.abs(I_next - I_mid)  # |I_{t+1} - I_t|

        return D_pre, D_next

    def forward(
        self,
        cube: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        参数：
            cube: (B, C, H, W, S) Cube 张量，其中 S 为时间维度
            return_attention: 是否返回注意力图（用于可视化）

        返回：
            enhanced_cube: (B, C, H, W, S) 增强后的 Cube
            或 (enhanced_cube, attention_map) 如果 return_attention=True
        """
        B, C, H, W, S = cube.shape

        # 转换为 (B, S, C, H, W) 格式以便处理
        frames = cube.permute(0, 4, 1, 2, 3).contiguous()  # (B, S, C, H, W)

        # 1. 计算时域梯度
        D_pre, D_next = self.compute_temporal_gradient(frames)  # (B, C, H, W)

        # 2. 生成运动显著图（哈达玛积）
        # 只有在连续两帧都存在的变化才被视为可靠运动
        M_raw = D_pre * D_next  # (B, C, H, W)

        # 3. 生成注意力权重
        # 先进行通道融合
        M_fused = self.channel_fusion(M_raw)  # (B, C, H, W)
        A_motion = self.motion_net(M_fused)   # (B, C, H, W) in [0, 1]

        # 4. 获取中间帧进行增强
        mid_idx = S // 2
        I_mid = frames[:, mid_idx]  # (B, C, H, W)

        # 5. 残差增强：I'_t = I_t · (1 + α · A_motion)
        # 使用 clamp 限制 alpha 范围，防止数值不稳定
        alpha_clamped = torch.clamp(self.alpha, 0.0, 2.0)
        I_enhanced = I_mid * (1 + alpha_clamped * A_motion)

        # 6. 替换中间帧，其他帧保持不变
        frames_enhanced = frames.clone()
        frames_enhanced[:, mid_idx] = I_enhanced

        # 转回原始格式 (B, C, H, W, S)
        cube_enhanced = frames_enhanced.permute(0, 2, 3, 4, 1).contiguous()

        if return_attention:
            return cube_enhanced, A_motion
        return cube_enhanced


class MADALight(nn.Module):
    """
    轻量级 MADA 模块（用于实时检测）

    简化版，减少卷积层数
    """

    def __init__(
        self,
        num_frames: int = 3,
        in_channels: int = 2,
        alpha: float = 0.5
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # 简化的运动特征网络
        self.motion_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """
        轻量级前向传播

        参数：
            cube: (B, C, H, W, S)

        返回：
            enhanced_cube: (B, C, H, W, S)
        """
        B, C, H, W, S = cube.shape

        # 转换格式
        frames = cube.permute(0, 4, 1, 2, 3).contiguous()

        # 提取帧
        mid_idx = S // 2
        if S >= 3:
            I_pre = frames[:, mid_idx - 1]
            I_mid = frames[:, mid_idx]
            I_next = frames[:, mid_idx + 1]
        else:
            # 帧数不足时，返回原始数据
            return cube

        # 计算运动显著图
        D_pre = torch.abs(I_mid - I_pre)
        D_next = torch.abs(I_next - I_mid)
        M_raw = D_pre * D_next

        # 生成注意力并增强
        A_motion = self.motion_net(M_raw)
        alpha_clamped = torch.clamp(self.alpha, 0.0, 2.0)
        I_enhanced = I_mid * (1 + alpha_clamped * A_motion)

        # 更新中间帧
        frames_enhanced = frames.clone()
        frames_enhanced[:, mid_idx] = I_enhanced

        return frames_enhanced.permute(0, 2, 3, 4, 1).contiguous()


if __name__ == "__main__":
    # 测试代码
    print("Testing Motion-Aware Difference Attention (MADA)...")

    # 测试基本 MADA
    mada = MotionAwareDifferenceAttention(num_frames=3, in_channels=2, alpha=0.5)

    # 模拟输入
    B, C, H, W, S = 2, 2, 640, 640, 3
    cube = torch.randn(B, C, H, W, S)

    print(f"输入 Cube: {cube.shape}")

    # 前向传播
    cube_enhanced = mada(cube)

    print(f"输出增强 Cube: {cube_enhanced.shape}")
    print(f"α (可学习): {mada.alpha.item():.4f}")

    assert cube_enhanced.shape == (B, C, H, W, S), "输出形状错误"
    print("✓ MADA 基本测试通过！")

    # 测试带注意力返回
    cube_enhanced, attention = mada(cube, return_attention=True)
    print(f"注意力图: {attention.shape}")

    # 测试轻量级版本
    print("\n测试 MADALight...")
    mada_light = MADALight(num_frames=3, in_channels=2, alpha=0.5)
    cube_light = mada_light(cube)
    print(f"轻量级输出: {cube_light.shape}")

    print("\n🎉 所有 MADA 模块测试通过！")
