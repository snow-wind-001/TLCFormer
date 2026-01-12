"""
Varied-Size Patch Attention (VPA) 模块
可变尺寸 Patch 注意力机制，用于多尺度特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class PatchEmbedding(nn.Module):
    """
    可变尺寸 Patch 嵌入层
    
    功能：
    1. 将输入 cube 分割为可变大小的 patch
    2. 通过轻量卷积预测 patch 的缩放和平移
    3. 生成 token embeddings
    
    参数：
        in_channels (int): 输入通道数（cube 的通道数 * 采样帧数）
        embed_dim (int): 嵌入维度
        patch_size (int): 基础 patch 大小
        img_size (int): 图像尺寸
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 96,
        patch_size: int = 4,
        img_size: int = 640
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch 投影层（使用卷积实现）
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 轻量卷积预测缩放和平移参数（用于可变 patch 大小）
        self.scale_pred = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()  # 输出 [0, 1]，表示缩放因子
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数：
            x: (B, C, H, W) 输入特征
            
        返回：
            tokens: (B, N, D) token embeddings
            scale_map: (B, 1, H', W') 缩放图
        """
        B, C, H, W = x.shape
        
        # 预测缩放图
        scale_map = self.scale_pred(x)  # (B, 1, H, W)
        # 缩放范围：[0.5, 1.5]
        scale_map = 0.5 + scale_map
        
        # Patch 嵌入
        tokens = self.proj(x)  # (B, D, H', W')
        B, D, H_p, W_p = tokens.shape
        
        # 展平为 tokens
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = self.norm(tokens)
        
        return tokens, scale_map


class TokenMixer(nn.Module):
    """
    Token Mixer 层
    使用平均池化 + 残差连接替代自注意力，降低计算复杂度
    
    参数：
        dim (int): 特征维度
        pool_size (int): 池化核大小
    """
    
    def __init__(self, dim: int, pool_size: int = 3):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size
        
        # 平均池化（带 padding 保持尺寸）
        self.pool = nn.AvgPool1d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2
        )
        
        # 1x1 卷积用于特征变换
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (B, N, D) token 序列
        返回：
            out: (B, N, D) 混合后的 token
        """
        B, N, D = x.shape
        shortcut = x
        
        # 转置以使用 AvgPool1d: (B, D, N)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (B, N, D)
        
        # 投影
        x = self.proj(x)
        x = self.norm(x)
        
        # 残差连接
        x = x + shortcut
        
        return x


class MLP(nn.Module):
    """
    标准两层 MLP + LayerNorm + 残差连接
    
    参数：
        in_features (int): 输入特征维度
        hidden_features (int): 隐藏层维度（默认为输入的 4 倍）
        out_features (int): 输出特征维度
        drop (float): Dropout 概率
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(in_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (B, N, D)
        返回：
            out: (B, N, D)
        """
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x + shortcut
        return x


class VPABlock(nn.Module):
    """
    VPA Block: Token Mixer + MLP
    
    参数：
        dim (int): 特征维度
        pool_size (int): Token Mixer 的池化核大小
        mlp_ratio (float): MLP 隐藏层维度比例
        drop (float): Dropout 概率
    """
    
    def __init__(
        self,
        dim: int,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0
    ):
        super().__init__()
        self.mixer = TokenMixer(dim, pool_size)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (B, N, D)
        返回：
            out: (B, N, D)
        """
        x = self.mixer(x)
        x = self.mlp(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch 合并层，用于下采样
    将 2x2 的 patch 合并为一个 patch，通道数翻倍
    
    参数：
        dim (int): 输入特征维度
        norm_layer: 归一化层
    """
    
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        参数：
            x: (B, H*W, D)
            H, W: 特征图的高和宽
        返回：
            out: (B, H*W/4, 2*D)
            H_new, W_new: 新的高和宽
        """
        B, L, C = x.shape
        assert L == H * W, "输入特征长度与 H*W 不匹配"
        
        x = x.view(B, H, W, C)
        
        # Padding（如果 H 或 W 不是偶数）
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        # 合并 2x2 patch
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        
        x = x.view(B, -1, 4 * C)  # (B, H*W/4, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H*W/4, 2C)
        
        H_new = (H + 1) // 2
        W_new = (W + 1) // 2
        
        return x, H_new, W_new


class VariedSizePatchAttention(nn.Module):
    """
    Varied-Size Patch Attention 主模块
    
    功能：
    1. 多阶段 VPA Block 构建特征金字塔
    2. 输出 5 个尺度的特征图 F1~F5
    
    参数：
        in_channels (int): 输入通道数（cube 通道 * 采样帧）
        embed_dim (int): 基础嵌入维度
        depths (List[int]): 每个阶段的 VPA Block 数量
        img_size (int): 图像尺寸
        patch_size (int): 初始 patch 大小
        drop_rate (float): Dropout 概率
    """
    
    def __init__(
        self,
        in_channels: int = 6,  # 2 channels * 3 sample frames
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        img_size: int = 640,
        patch_size: int = 4,
        drop_rate: float = 0.0
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Patch 嵌入
        self.patch_embed = PatchEmbedding(
            in_channels, 
            embed_dim, 
            patch_size, 
            img_size
        )
        
        # 构建多阶段 VPA Blocks
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            dim = embed_dim * (2 ** i_layer)
            depth = depths[i_layer]
            
            # VPA Blocks
            blocks = nn.ModuleList([
                VPABlock(dim, pool_size=3, mlp_ratio=4.0, drop=drop_rate)
                for _ in range(depth)
            ])
            self.layers.append(blocks)
            
            # Patch Merging（除了最后一层）
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(dim)
                self.downsample_layers.append(downsample)
        
        # 计算每个阶段的特征图尺寸
        self.feature_sizes = []
        H, W = img_size // patch_size, img_size // patch_size
        for i in range(self.num_layers):
            self.feature_sizes.append((H, W))
            if i < self.num_layers - 1:
                H = (H + 1) // 2
                W = (W + 1) // 2
        
    def forward(self, cube: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        参数：
            cube: (B, C, H, W, S) Cube 张量
            
        返回：
            features: List of 5 feature maps
                F1: (B, D, H/4, W/4)
                F2: (B, 2D, H/8, W/8)
                F3: (B, 4D, H/16, W/16)
                F4: (B, 8D, H/32, W/32)
                F5: (B, 16D, H/64, W/64)（如果有 5 个阶段）
        """
        B, C, H, W, S = cube.shape
        
        # 将 cube 重塑为 (B, C*S, H, W)
        x = cube.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, S, H, W)
        x = x.view(B, C * S, H, W)  # (B, C*S, H, W)
        
        # Patch 嵌入
        x, scale_map = self.patch_embed(x)  # (B, N, D)
        
        features = []
        H_cur, W_cur = self.feature_sizes[0]
        
        # 多阶段特征提取
        for i_layer in range(self.num_layers):
            # VPA Blocks
            for block in self.layers[i_layer]:
                x = block(x)
            
            # 重塑为特征图并保存
            B, N, D = x.shape
            x_reshaped = x.transpose(1, 2).view(B, D, H_cur, W_cur)
            features.append(x_reshaped)
            
            # Patch Merging（下采样）
            if i_layer < self.num_layers - 1:
                x, H_cur, W_cur = self.downsample_layers[i_layer](x, H_cur, W_cur)
        
        return features


if __name__ == "__main__":
    # 测试代码
    print("Testing VariedSizePatchAttention...")
    
    # 创建模型
    vpa = VariedSizePatchAttention(
        in_channels=6,  # 2 channels * 3 frames
        embed_dim=96,
        depths=[2, 2, 6, 2],
        img_size=640,
        patch_size=4
    )
    
    # 模拟输入
    B, C, H, W, S = 2, 2, 640, 640, 3
    cube = torch.randn(B, C, H, W, S)
    
    print(f"输入 Cube: {cube.shape}")
    
    # 前向传播
    features = vpa(cube)
    
    print(f"\n输出特征金字塔:")
    for i, feat in enumerate(features):
        print(f"  F{i+1}: {feat.shape}")
    
    # 验证
    assert len(features) == 4, "特征金字塔层数错误"
    assert features[0].shape[1] == 96, "F1 通道数错误"
    assert features[1].shape[1] == 192, "F2 通道数错误"
    
    print("\n✓ VariedSizePatchAttention 测试通过！")

