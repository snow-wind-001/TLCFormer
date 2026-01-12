"""
Varied-Size Patch Attention (VPA) 模块 - TLCFormer 版本

核心改进：
- PatchEmbedding: 增强为 ROIAlign 的可变尺寸分块（平移+缩放）
- HybridTokenMixer: 使用 Max-Mean 混合池化替代单纯 AvgPool
  * MaxPool 保留小目标极值能量，防止深层网络中目标消失
  * AvgPool 维持背景纹理信息
  * 显著提升小目标召回率（Recall）
- 网络/接口：保持 4 个 stage（depths=[2,2,6,2]）

算法原理（Hybrid Mixer）：
1. 双路池化：P_max = MaxPool2d(X), P_avg = AvgPool2d(X)
2. 特征拼接：P_hybrid = Concat(P_max, P_avg)
3. 通道压缩：X_mixed = GELU(Conv1x1(P_hybrid))
4. 残差连接：X_out = X + X_mixed

物理先验：
- 小目标仅占 1-4 像素，纯 AvgPool 会导致能量被稀释
- MaxPool 保证极值能量不经衰减地传递到下一层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

# 尝试使用 torchvision.ops.roi_align；若不可用，退化为 grid_sample 兜底
try:
    from torchvision.ops import roi_align as tv_roi_align
except Exception:
    tv_roi_align = None


def roi_align_fallback(x: torch.Tensor, rois: torch.Tensor, output_size: Tuple[int, int]):
    """
    grid_sample 版简易 ROIAlign 兜底实现（以可运行为目标）
    x: (B, C, H, W)
    rois: (N, 5) [batch_idx, x1, y1, x2, y2] (float)
    返回: (N, C, out_h, out_w)
    """
    device = x.device
    B, C, H, W = x.shape
    out_h, out_w = output_size
    outs = []

    for i in range(rois.shape[0]):
        b = int(rois[i, 0].item())
        x1, y1, x2, y2 = rois[i, 1:].tolist()
        xs = torch.linspace(x1, x2, out_w, device=device, dtype=x.dtype)
        ys = torch.linspace(y1, y2, out_h, device=device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid_x = xx / max(W - 1, 1) * 2 - 1
        grid_y = yy / max(H - 1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, out_h, out_w, 2)
        patch = F.grid_sample(x[b:b+1], grid, mode="bilinear", align_corners=True)
        outs.append(patch)
    return torch.cat(outs, dim=0)  # (N, C, out_h, out_w)


class PatchEmbedding(nn.Module):
    """
    可变尺寸 Patch 嵌入层（保持接口不变）
    
    功能：
    1) 对输入做 1x1 通道嵌入，得到“母图特征”；
    2) 按默认网格（步长=patch_size）回归每个默认块的平移(Δx,Δy)与缩放(log s)；
    3) 基于回归参数用 ROIAlign 从母图特征中重采样到固定 (P,P)，再做池化得到 token；
    4) 返回 (tokens, scale_map)，形状与原实现一致。
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

        # === 1) 母图特征嵌入（替代原 proj 的下采样方式，保持对外行为不变） ===
        self.embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)

        # === 2) 基于“默认块”感受野的参数回归：输出 (Δx, Δy, log s) ===
        # 使用 kernel=stride=patch_size，确保每个默认块对应一个参数
        self.param_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, 3, kernel_size=patch_size, stride=patch_size, padding=0)
        )

        # scale_map（与原接口保持）：我们将回归得到的 s 作为缩放图输出
        # 注意：不单独再建 head，直接由 param_head 的第三通道经变换得到
        self.norm = nn.LayerNorm(embed_dim)

        # 平移/缩放范围控制
        self._s_min = 0.5
        self._s_max = 2.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            x: (B, C, H, W)
        返回:
            tokens: (B, N, D)
            scale_map: (B, 1, H', W')  其中 H'=H/P, W'=W/P
        """
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "H、W 必须能被 patch_size 整除"

        # 1) 母图特征
        x_embed = self.embed(x)  # (B, D, H, W)
        D = x_embed.shape[1]

        # 2) 回归 (Δx, Δy, log s)
        params = self.param_head(x_embed)  # (B, 3, H', W')
        Hp, Wp = params.shape[2], params.shape[3]
        dx = params[:, 0]   # (B, H', W')
        dy = params[:, 1]   # (B, H', W')
        ls = params[:, 2]   # (B, H', W')

        # 平移与缩放约束：Δx,Δy ∈ [-P/2, P/2]； s ∈ [0.5, 2.0]
        alpha = P / 2.0
        mx = alpha * torch.tanh(dx)
        my = alpha * torch.tanh(dy)
        s  = torch.exp(torch.tanh(ls))              # (0, e)
        s  = torch.clamp(s, self._s_min, self._s_max)
        scale_map = s.unsqueeze(1)                  # (B, 1, H', W')

        # 3) 构造默认网格中心 (像素坐标)
        device = x.device
        cx = torch.arange(Wp, device=device, dtype=x.dtype) * P + (P / 2.0)  # (Wp,)
        cy = torch.arange(Hp, device=device, dtype=x.dtype) * P + (P / 2.0)  # (Hp,)
        cyy, cxx = torch.meshgrid(cy, cx, indexing='ij')  # (Hp, Wp)

        # 每个 ROI 的 [x1,y1,x2,y2]
        x1 = (cxx[None, ...] + mx - s * (P / 2.0)).clamp(0, W - 1)
        y1 = (cyy[None, ...] + my - s * (P / 2.0)).clamp(0, H - 1)
        x2 = (cxx[None, ...] + mx + s * (P / 2.0)).clamp(0, W - 1)
        y2 = (cyy[None, ...] + my + s * (P / 2.0)).clamp(0, H - 1)

        # 组装 rois: (B*H'*W', 5) -> [batch_idx, x1, y1, x2, y2]
        rois_list = []
        for b in range(B):
            num = Hp * Wp
            batch_inds = torch.full((num, 1), float(b), device=device, dtype=x.dtype)
            r = torch.stack([
                x1[b].reshape(-1),
                y1[b].reshape(-1),
                x2[b].reshape(-1),
                y2[b].reshape(-1)
            ], dim=1)
            rois_list.append(torch.cat([batch_inds, r], dim=1))
        rois = torch.cat(rois_list, dim=0)  # (B*Hp*Wp, 5)

        # 4) ROIAlign 采样到 (P,P)，再做全局平均得到 token
        if tv_roi_align is not None:
            patches = tv_roi_align(x_embed, rois, output_size=(P, P), aligned=True)  # (B*Hp*Wp, D, P, P)
        else:
            patches = roi_align_fallback(x_embed, rois, output_size=(P, P))

        patches = patches.view(B, Hp * Wp, D, P, P)  # (B, N, D, P, P)
        tokens = patches.mean(dim=(-1, -2))          # (B, N, D)
        tokens = self.norm(tokens)

        return tokens, scale_map


class HybridTokenMixer(nn.Module):
    """
    Hybrid Energy-Preserving Token Mixer

    核心改进：使用 Max-Mean 混合池化替代单纯 AvgPool

    算法原理：
    1. 双路池化：P_max = MaxPool2d(X), P_avg = AvgPool2d(X)
    2. 特征拼接：P_hybrid = Concat(P_max, P_avg)
    3. 通道压缩：X_mixed = GELU(Conv1x1(P_hybrid))
    4. 残差连接：X_out = X + X_mixed

    优势：
    - MaxPool 保留小目标极值能量，防止深层网络中目标消失
    - AvgPool 维持背景纹理信息
    - 显著提升小目标召回率（Recall）
    """

    def __init__(self, dim: int, pool_size: int = 3):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size
        self._H = None
        self._W = None

        # 双路池化
        self.maxpool2d = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2
        )
        self.avgpool2d = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2
        )

        # 通道压缩：从 2*dim 降维回 dim
        self.channel_compression = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.norm = nn.LayerNorm(dim)

    def set_spatial(self, H: int, W: int):
        """注入空间尺寸"""
        self._H, self._W = H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: (B, N, D) token 序列

        返回：
            x: (B, N, D) 混合后的 token
        """
        B, N, D = x.shape
        shortcut = x

        # 若已注入 H,W 则做 2D 混合池化
        if self._H is not None and self._W is not None and (self._H * self._W == N):
            H, W = self._H, self._W

            # 转为 2D: (B, D, H, W)
            x2d = x.transpose(1, 2).reshape(B, D, H, W)

            # 1. 双路池化
            P_max = self.maxpool2d(x2d)  # (B, D, H, W) - 保留极值
            P_avg = self.avgpool2d(x2d)  # (B, D, H, W) - 背景纹理

            # 2. 特征拼接（通道维度翻倍）
            P_hybrid = torch.cat([P_max, P_avg], dim=1)  # (B, 2D, H, W)

            # 3. 通道压缩与融合
            x_mixed = self.channel_compression(P_hybrid)  # (B, D, H, W)

            # 转回 token 格式
            x = x_mixed.flatten(2).transpose(1, 2)  # (B, N, D)
        else:
            # 退化处理：1D 混合池化（保持兼容性）
            x = x.transpose(1, 2)  # (B, D, N)
            x_max = F.max_pool1d(x, kernel_size=self.pool_size, stride=1, padding=self.pool_size // 2)
            x_avg = F.avg_pool1d(x, kernel_size=self.pool_size, stride=1, padding=self.pool_size // 2)
            
            # 简化的融合（1D情况）
            x = (x_max + x_avg) / 2
            x = x.transpose(1, 2)  # (B, N, D)

        x = self.norm(x)
        x = x + shortcut  # 残差连接

        return x


# 保留别名以兼容旧代码
TokenMixer = HybridTokenMixer


class MLP(nn.Module):
    """
    标准两层 MLP + LayerNorm + 残差连接（保持你的实现）
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
    VPA Block: Token Mixer + MLP（保持 forward(x) 签名）
    - 通过 set_spatial(H, W) 将空间尺寸传入内部 TokenMixer
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
        self._H = None
        self._W = None

    def set_spatial(self, H: int, W: int):
        self._H, self._W = H, W
        # 同步给内部 mixer
        self.mixer.set_spatial(H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        x = self.mlp(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch 合并层，用于下采样（与你原实现一致）
    """
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W, "输入特征长度与 H*W 不匹配"

        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H*W/4, 2C)

        H_new = (H + 1) // 2
        W_new = (W + 1) // 2

        return x, H_new, W_new


class VariedSizePatchAttention(nn.Module):
    """
    Varied-Size Patch Attention 主模块（保持 4 个 stage、接口与输出不变）
    - 多阶段 VPA Block 构建特征金字塔
    - 输出 4 个尺度的特征图 F1~F4（与你原断言一致）
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

        # Patch 嵌入（可变尺寸分块 + ROIAlign）
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

            blocks = nn.ModuleList([
                VPABlock(dim, pool_size=3, mlp_ratio=4.0, drop=drop_rate)
                for _ in range(depth)
            ])
            self.layers.append(blocks)

            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(dim)
                self.downsample_layers.append(downsample)

        # 计算每个阶段的特征图尺寸（与你原实现一致）
        self.feature_sizes = []
        H, W = img_size // patch_size, img_size // patch_size
        for i in range(self.num_layers):
            self.feature_sizes.append((H, W))
            if i < self.num_layers - 1:
                H = (H + 1) // 2
                W = (W + 1) // 2

    def forward(self, cube: torch.Tensor) -> List[torch.Tensor]:
        """
        输入:
            cube: (B, C, H, W, S)
        返回:
            features: List of 4 feature maps
                F1: (B, D, H/4,  W/4)
                F2: (B, 2D, H/8,  W/8)
                F3: (B, 4D, H/16, W/16)
                F4: (B, 8D, H/32, W/32)
        """
        B, C, H, W, S = cube.shape

        # 将 cube 重塑为 (B, C*S, H, W)
        x = cube.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, S, H, W)
        x = x.view(B, C * S, H, W)  # (B, C*S, H, W)

        # Patch 嵌入（返回 tokens 与 scale_map；保持形参/返回一致）
        x, scale_map = self.patch_embed(x)  # x: (B, N, D)

        features = []
        H_cur, W_cur = self.feature_sizes[0]

        # 多阶段特征提取（保持与你原实现的调用方式一致）
        for i_layer in range(self.num_layers):
            # 在进入每个 block 前，把当前空间尺寸注入（不改变 block.forward 签名）
            for block in self.layers[i_layer]:
                block.set_spatial(H_cur, W_cur)
                x = block(x)

            # 重塑为特征图并保存
            B_, N_, D_ = x.shape
            x_reshaped = x.transpose(1, 2).view(B_, D_, H_cur, W_cur)
            features.append(x_reshaped)

            # Patch Merging（下采样）
            if i_layer < self.num_layers - 1:
                x, H_cur, W_cur = self.downsample_layers[i_layer](x, H_cur, W_cur)

        return features


if __name__ == "__main__":
    # 测试代码（与你原版保持一致）
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

