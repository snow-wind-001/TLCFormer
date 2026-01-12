"""
Doppler Adaptive Filter 模块
基于 FFT 的频域自适应滤波器，用于增强运动目标特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DopplerAdaptiveFilter(nn.Module):
    """
    Doppler Adaptive Filter 模块
    
    功能：
    1. 对输入 cube 进行时空域 FFT 变换
    2. 在频域使用可学习的带通滤波器增强运动目标
    3. 逆 FFT 回到空域
    4. 与原始特征融合
    
    原理：
    - 小目标运动产生特定的 Doppler 频率特征
    - 通过频域滤波可以抑制静态背景，增强运动目标
    
    参数：
        img_size (int): 图像尺寸
        num_frames (int): 时间帧数 S
        learn_filter (bool): 是否学习滤波器参数
        filter_type (str): 滤波器类型 ('bandpass', 'highpass', 'adaptive')
    """
    
    def __init__(
        self,
        img_size: int = 640,
        num_frames: int = 3,
        learn_filter: bool = True,
        filter_type: str = 'adaptive'
    ):
        super().__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        self.learn_filter = learn_filter
        self.filter_type = filter_type
        
        # 可学习的频域滤波器 L: (1, 1, H, W, S)
        if learn_filter:
            # 初始化为高斯带通滤波器
            filter_init = self._init_bandpass_filter(img_size, num_frames)
            self.filter_weights = nn.Parameter(filter_init)
        else:
            # 固定带通滤波器
            filter_fixed = self._init_bandpass_filter(img_size, num_frames)
            self.register_buffer('filter_weights', filter_fixed)
        
        # 融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 原始特征权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # 滤波特征权重
        
        # 用于特征增强的卷积
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Sigmoid()
        )
        
    def _init_bandpass_filter(
        self, 
        img_size: int, 
        num_frames: int
    ) -> torch.Tensor:
        """
        初始化带通滤波器
        
        返回：
            filter: (1, 1, H, W, S) 频域滤波器
        """
        H, W, S = img_size, img_size, num_frames
        
        # 创建频率网格
        freq_h = torch.fft.fftfreq(H).view(-1, 1, 1)  # (H, 1, 1)
        freq_w = torch.fft.fftfreq(W).view(1, -1, 1)  # (1, W, 1)
        freq_s = torch.fft.fftfreq(S).view(1, 1, -1)  # (1, 1, S)
        
        # 计算径向频率
        freq_spatial = torch.sqrt(freq_h**2 + freq_w**2)  # (H, W, 1)
        freq_temporal = torch.abs(freq_s)  # (1, 1, S)
        
        # 高斯带通滤波器
        # 空间：保留中高频（抑制低频背景）
        spatial_filter = torch.exp(-((freq_spatial - 0.15) ** 2) / (2 * 0.1 ** 2))
        # 时间：保留低中频（运动频率）
        temporal_filter = torch.exp(-(freq_temporal ** 2) / (2 * 0.2 ** 2))
        
        # 组合滤波器
        filter_3d = spatial_filter * temporal_filter  # (H, W, S)
        filter_3d = filter_3d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, S)
        
        return filter_3d
    
    def apply_fft_filter(
        self, 
        cube: torch.Tensor
    ) -> torch.Tensor:
        """
        应用 FFT 滤波
        
        参数：
            cube: (B, C, H, W, S) 输入 cube
            
        返回：
            filtered: (B, C, H, W, S) 滤波后的 cube
        """
        B, C, H, W, S = cube.shape
        
        # 对每个通道分别处理
        filtered_channels = []
        
        for c in range(C):
            cube_c = cube[:, c]  # (B, H, W, S)
            
            # 2D 空域 + 1D 时域 FFT
            # 先对空间维度做 2D FFT
            fft_spatial = torch.fft.fft2(cube_c, dim=(1, 2))  # (B, H, W, S)
            
            # 再对时间维度做 1D FFT
            fft_full = torch.fft.fft(fft_spatial, dim=3)  # (B, H, W, S)
            
            # 应用滤波器（频域相乘）
            fft_filtered = fft_full * self.filter_weights.squeeze(0).squeeze(0)  # (B, H, W, S)
            
            # 逆 FFT：先时间后空间
            ifft_temporal = torch.fft.ifft(fft_filtered, dim=3)  # (B, H, W, S)
            ifft_full = torch.fft.ifft2(ifft_temporal, dim=(1, 2))  # (B, H, W, S)
            
            # 取实部
            filtered_c = ifft_full.real  # (B, H, W, S)
            filtered_channels.append(filtered_c)
        
        # 合并通道
        filtered = torch.stack(filtered_channels, dim=1)  # (B, C, H, W, S)
        
        return filtered
    
    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            cube: (B, C, H, W, S) 输入 cube
            
        返回：
            enhanced: (B, C, H, W, S) 增强后的 cube
        """
        B, C, H, W, S = cube.shape
        
        # 保存原始输入
        cube_original = cube
        
        # FFT 滤波
        cube_filtered = self.apply_fft_filter(cube)
        
        # 计算每个时间步的增强权重（基于空间特征）
        enhanced_frames = []
        for s in range(S):
            # 原始帧和滤波帧
            frame_orig = cube_original[:, :, :, :, s]  # (B, C, H, W)
            frame_filt = cube_filtered[:, :, :, :, s]  # (B, C, H, W)
            
            # 使用卷积预测增强权重
            weight_map = self.enhance_conv(frame_orig)  # (B, C, H, W)
            
            # 加权融合
            frame_enhanced = (
                torch.sigmoid(self.alpha) * frame_orig + 
                torch.sigmoid(self.beta) * frame_filt * weight_map
            )
            
            enhanced_frames.append(frame_enhanced)
        
        # 堆叠时间维度
        enhanced = torch.stack(enhanced_frames, dim=-1)  # (B, C, H, W, S)
        
        return enhanced


class DopplerFilterWithDownsample(nn.Module):
    """
    带下采样的 Doppler Filter
    用于在特征金字塔的不同层级应用 Doppler 滤波
    
    参数：
        img_size (int): 输入特征图尺寸
        num_frames (int): 时间帧数
        in_channels (int): 输入通道数
        downsample_factor (int): 下采样因子
    """
    
    def __init__(
        self,
        img_size: int,
        num_frames: int,
        in_channels: int = 2,
        downsample_factor: int = 1
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.img_size = img_size // downsample_factor
        
        # Doppler 滤波器
        self.doppler_filter = DopplerAdaptiveFilter(
            img_size=self.img_size,
            num_frames=num_frames,
            learn_filter=True,
            filter_type='adaptive'
        )
        
        # 下采样层（如果需要）
        if downsample_factor > 1:
            self.downsample = nn.AvgPool2d(
                kernel_size=downsample_factor,
                stride=downsample_factor
            )
        else:
            self.downsample = nn.Identity()
        
        # 通道调整（如果输入通道数不是 2）
        if in_channels != 2:
            self.channel_proj = nn.Conv2d(in_channels, 2, 1)
        else:
            self.channel_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (B, C, H, W, S) 或 (B, C, H, W)
            
        返回：
            out: (B, C, H', W', S) 或 (B, C, H', W')
        """
        # 判断输入是否包含时间维度
        if x.dim() == 5:
            B, C, H, W, S = x.shape
            has_time = True
        else:
            B, C, H, W = x.shape
            has_time = False
            S = 3  # 默认
            x = x.unsqueeze(-1).repeat(1, 1, 1, 1, S)
        
        # 通道投影
        if C != 2:
            x_list = []
            for s in range(S):
                x_s = self.channel_proj(x[:, :, :, :, s])
                x_list.append(x_s)
            x = torch.stack(x_list, dim=-1)
        
        # 下采样
        if self.downsample_factor > 1:
            x_list = []
            for s in range(S):
                x_s = self.downsample(x[:, :, :, :, s])
                x_list.append(x_s)
            x = torch.stack(x_list, dim=-1)
        
        # Doppler 滤波
        x = self.doppler_filter(x)
        
        # 如果原始输入没有时间维度，去掉时间维度
        if not has_time:
            x = x.mean(dim=-1)  # 平均所有时间帧
        
        return x


if __name__ == "__main__":
    # 测试代码
    print("Testing DopplerAdaptiveFilter...")
    
    # 创建模型
    doppler = DopplerAdaptiveFilter(
        img_size=640,
        num_frames=3,
        learn_filter=True
    )
    
    # 模拟输入
    B, C, H, W, S = 2, 2, 640, 640, 3
    cube = torch.randn(B, C, H, W, S)
    
    print(f"输入 Cube: {cube.shape}")
    
    # 前向传播
    enhanced = doppler(cube)
    
    print(f"输出 Enhanced Cube: {enhanced.shape}")
    print(f"增强范围: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    # 验证
    assert enhanced.shape == cube.shape, "输出形状错误"
    
    print("\n✓ DopplerAdaptiveFilter 测试通过！")
    
    # 测试带下采样版本
    print("\nTesting DopplerFilterWithDownsample...")
    doppler_ds = DopplerFilterWithDownsample(
        img_size=640,
        num_frames=3,
        in_channels=96,
        downsample_factor=4
    )
    
    # 模拟特征输入
    feat = torch.randn(2, 96, 160, 160, 3)
    feat_enhanced = doppler_ds(feat)
    print(f"输入特征: {feat.shape}")
    print(f"输出特征: {feat_enhanced.shape}")
    
    print("\n✓ DopplerFilterWithDownsample 测试通过！")

