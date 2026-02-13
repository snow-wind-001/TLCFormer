"""
Deep Local Contrast Module (DLCM)
局部对比度增强模块，用于增强红外小目标的信杂比（SCR）

基于物理先验：红外小目标在局部邻域内是极值点

算法原理：
1. 背景估计：μ_bg = AvgPool_{9×9}(X) - 使用外层邻域估计背景
2. 目标强度：L_max = MaxPool_{3×3}(X) - 使用内层邻域提取极值
3. 对比度响应：C = L_max² / (μ_bg + ε) 或 C = ReLU(X - μ_bg)
4. 残差融合：X_out = X + β · C

物理先验：
- 红外小目标定义为局部区域内的极大值点
- 背景相对均匀，目标与背景有显著对比度
- 通过局部对比度可以抑制背景杂波
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DeepLocalContrastModule(nn.Module):
    """
    Deep Local Contrast Module (DLCM)

    核心思想：利用红外小目标的"局部突异性"，增强信杂比（SCR）

    算法流程：
    1. 背景估计：使用外层邻域 (9x9) 的平均池化估计局部背景 μ_bg
    2. 目标强度：使用内层邻域 (3x3) 的最大池化提取潜在目标 L_max
    3. 对比度计算：C = L_max² / (μ_bg + ε) 或 C = ReLU(X - μ_bg)
    4. 残差融合：X_out = X + β · C

    参数：
        in_channels (int): 输入通道数
        kernel_inner (int): 内层邻域大小（目标区域），默认 3x3
        kernel_outer (int): 外层邻域大小（背景区域），默认 9x9
        use_soft_contrast (bool): 是否使用软对比度（ReLU差分形式）
        beta (float): 初始融合权重，可学习
    """

    def __init__(
        self,
        in_channels: int,
        kernel_inner: int = 3,
        kernel_outer: int = 9,
        use_soft_contrast: bool = False,
        beta: float = 0.5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_inner = kernel_inner
        self.kernel_outer = kernel_outer
        self.use_soft_contrast = use_soft_contrast

        # 可学习的融合权重
        self.beta = nn.Parameter(torch.tensor(beta))

        # 对比度增强卷积（学习更好的对比度表示）
        self.contrast_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 自适应权重调整（根据输入特征动态调整增强强度）
        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, max(in_channels // 4, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 4, 1), in_channels, 1),
            nn.Sigmoid()
        )

    def estimate_background(self, x: torch.Tensor) -> torch.Tensor:
        """
        背景估计

        使用较大的邻域（9x9）的平均池化估计局部背景强度
        小目标（1-4像素）被稀释，主要保留背景信息

        参数：
            x: (B, C, H, W) 输入特征

        返回：
            mu_bg: (B, C, H, W) 背景估计
        """
        pad = self.kernel_outer // 2
        mu_bg = F.avg_pool2d(
            x,
            kernel_size=self.kernel_outer,
            stride=1,
            padding=pad
        )
        return mu_bg

    def estimate_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        目标强度估计

        使用较小的邻域（3x3）的最大池化提取潜在目标的最高能量
        保留小目标的极值信息

        参数：
            x: (B, C, H, W) 输入特征

        返回：
            L_max: (B, C, H, W) 目标强度估计
        """
        pad = self.kernel_inner // 2
        L_max = F.max_pool2d(
            x,
            kernel_size=self.kernel_inner,
            stride=1,
            padding=pad
        )
        return L_max

    def compute_contrast(
        self,
        x: torch.Tensor,
        mu_bg: torch.Tensor,
        L_max: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比度响应

        参数：
            x: (B, C, H, W) 原始输入
            mu_bg: (B, C, H, W) 背景估计
            L_max: (B, C, H, W) 目标强度

        返回：
            C: (B, C, H, W) 对比度图
        """
        eps = 1e-6

        if self.use_soft_contrast:
            # 软性差分形式：C = ReLU(X - μ_bg)
            C = F.relu(x - mu_bg)
        else:
            # 比率形式：C = L_max² / (μ_bg + ε)
            # 增强比背景亮的点，抑制比背景暗的点
            C = (L_max ** 2) / (mu_bg.abs() + eps)

            # 归一化到合理范围（防止数值爆炸）
            C = torch.clamp(C, 0, 100)

        return C

    def forward(
        self,
        x: torch.Tensor,
        return_contrast: bool = False
    ) -> torch.Tensor:
        """
        前向传播

        参数：
            x: (B, C, H, W) 输入特征图
            return_contrast: 是否返回对比度图（用于可视化）

        返回：
            x_out: (B, C, H, W) 对比度增强后的特征图
            或 (x_out, C) 如果 return_contrast=True
        """
        identity = x  # 残差连接

        # 1. 背景估计
        mu_bg = self.estimate_background(x)  # (B, C, H, W)

        # 2. 目标强度估计
        L_max = self.estimate_target(x)  # (B, C, H, W)

        # 3. 计算对比度响应
        C_raw = self.compute_contrast(x, mu_bg, L_max)  # (B, C, H, W)

        # 4. 对比度增强（通过卷积网络学习更好的对比度表示）
        C_enhanced = self.contrast_enhance(C_raw) * C_raw

        # 5. 自适应权重调整
        # 根据全局统计信息动态调整增强强度
        adaptive_beta = self.adaptive_weight(x)  # (B, C, 1, 1)
        beta_weighted = self.beta * adaptive_beta  # (B, C, 1, 1)

        # 6. 残差融合：X_out = X + β · C
        # clamp beta 防止数值不稳定
        beta_clamped = torch.clamp(beta_weighted, 0.0, 1.0)
        x_out = identity + beta_clamped * C_enhanced

        if return_contrast:
            return x_out, C_raw
        return x_out


class DLCMForCube(nn.Module):
    """
    适配 Cube 输入的 DLCM 模块

    用于处理 (B, C, H, W, S) 格式的时空 Cube 数据
    对每个时间帧分别应用 DLCM
    """

    def __init__(
        self,
        in_channels: int,
        kernel_inner: int = 3,
        kernel_outer: int = 9,
        use_soft_contrast: bool = False,
        beta: float = 0.5
    ):
        super().__init__()
        self.dlcm = DeepLocalContrastModule(
            in_channels=in_channels,
            kernel_inner=kernel_inner,
            kernel_outer=kernel_outer,
            use_soft_contrast=use_soft_contrast,
            beta=beta
        )

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            cube: (B, C, H, W, S) 输入 Cube

        返回：
            cube_out: (B, C, H, W, S) 增强后的 Cube
        """
        B, C, H, W, S = cube.shape

        # 对每个时间帧分别应用 DLCM
        enhanced_frames = []
        for s in range(S):
            frame = cube[:, :, :, :, s]  # (B, C, H, W)
            frame_enhanced = self.dlcm(frame)
            enhanced_frames.append(frame_enhanced)

        # 重组为 Cube
        cube_out = torch.stack(enhanced_frames, dim=-1)  # (B, C, H, W, S)

        return cube_out


class DLCMLight(nn.Module):
    """
    轻量级 DLCM 模块（用于实时检测）

    简化版，计算更快
    """

    def __init__(
        self,
        in_channels: int,
        kernel_inner: int = 3,
        kernel_outer: int = 9,
        beta: float = 0.5
    ):
        super().__init__()
        self.kernel_inner = kernel_inner
        self.kernel_outer = kernel_outer

        self.beta = nn.Parameter(torch.tensor(beta))

        # 简化的对比度增强（只有一层卷积）
        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（轻量版）

        参数：
            x: (B, C, H, W)

        返回：
            x_out: (B, C, H, W)
        """
        # 背景估计
        pad_bg = self.kernel_outer // 2
        mu_bg = F.avg_pool2d(x, self.kernel_outer, stride=1, padding=pad_bg)

        # 目标强度
        pad_tgt = self.kernel_inner // 2
        L_max = F.max_pool2d(x, self.kernel_inner, stride=1, padding=pad_tgt)

        # 简化的对比度计算：差分形式
        C = F.relu(L_max - mu_bg)

        # 轻量增强
        C_enhanced = self.enhance(C)

        # 残差融合
        beta_clamped = torch.clamp(self.beta, 0.0, 1.0)
        x_out = x + beta_clamped * C_enhanced

        return x_out


if __name__ == "__main__":
    # 测试代码
    print("Testing Deep Local Contrast Module (DLCM)...")

    # 测试基本 DLCM
    dlcm = DeepLocalContrastModule(
        in_channels=2,
        kernel_inner=3,
        kernel_outer=9,
        use_soft_contrast=False,
        beta=0.5
    )

    # 模拟输入
    B, C, H, W = 2, 2, 640, 640
    x = torch.randn(B, C, H, W)

    print(f"输入特征: {x.shape}")

    # 前向传播
    x_enhanced = dlcm(x)

    print(f"输出增强特征: {x_enhanced.shape}")
    print(f"β (可学习): {dlcm.beta.item():.4f}")

    assert x_enhanced.shape == (B, C, H, W), "输出形状错误"
    print("✓ DLCM 基本测试通过！")

    # 测试带对比度返回
    x_enhanced, contrast = dlcm(x, return_contrast=True)
    print(f"对比度图: {contrast.shape}")

    # 测试 Cube 版本
    print("\n测试 DLCMForCube...")
    dlcm_cube = DLCMForCube(in_channels=2, beta=0.5)
    cube = torch.randn(2, 2, 640, 640, 3)
    cube_enhanced = dlcm_cube(cube)
    print(f"Cube 输入: {cube.shape}")
    print(f"Cube 输出: {cube_enhanced.shape}")

    # 测试轻量级版本
    print("\n测试 DLCMLight...")
    dlcm_light = DLCMLight(in_channels=2, beta=0.5)
    x_light = dlcm_light(x)
    print(f"轻量级输出: {x_light.shape}")

    print("\n🎉 所有 DLCM 模块测试通过！")
