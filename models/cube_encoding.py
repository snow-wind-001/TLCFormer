"""
Cube Encoding 模块
将视频序列编码为 4D Cube 张量 (C, H, W, S)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import cv2


class CubeEncoding(nn.Module):
    """
    Cube Encoding 模块
    
    功能：
    1. 将 T 帧视频序列编码为 4D cube 张量
    2. 支持 RGBT 双模态输入（可见光 + 热红外）
    3. 时间维度均匀采样 S 帧
    
    参数：
        num_frames (int): 输入视频帧数 T
        sample_frames (int): 采样帧数 S（从 T 帧中均匀采样）
        img_size (int): 图像尺寸
        normalize (bool): 是否归一化
    """
    
    def __init__(
        self, 
        num_frames: int = 5,
        sample_frames: int = 3,
        img_size: int = 640,
        normalize: bool = True
    ):
        super().__init__()
        self.num_frames = num_frames
        self.sample_frames = sample_frames
        self.img_size = img_size
        self.normalize = normalize
        
        # 计算采样索引（均匀采样）
        if sample_frames > 1:
            self.sample_indices = np.linspace(0, num_frames - 1, sample_frames, dtype=int)
        else:
            self.sample_indices = np.array([num_frames // 2])
        
        # 归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def rgb_to_gray(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        将 RGB 图像转换为灰度图
        
        参数：
            rgb: (B, 3, H, W) RGB 图像
        返回：
            gray: (B, 1, H, W) 灰度图
        """
        # 使用标准灰度转换权重
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device).view(1, 3, 1, 1)
        gray = (rgb * weights).sum(dim=1, keepdim=True)
        return gray
    
    def forward(
        self, 
        rgb_frames: torch.Tensor, 
        thermal_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数：
            rgb_frames: (B, T, 3, H, W) 可见光视频序列
            thermal_frames: (B, T, 1, H, W) 热红外视频序列
            
        返回：
            cube: (B, 2, H, W, S) Cube 张量（采样帧，用于主干网络）
                - cube[:, 0, :, :, :]: 灰度通道
                - cube[:, 1, :, :, :]: 热红外通道
            all_frames: (B, 2, H, W, T) 全部帧张量（用于 OffsetPredictor 时间特征提取）
                - all_frames[:, 0, :, :, :]: 灰度通道
                - all_frames[:, 1, :, :, :]: 热红外通道
        """
        B, T, C, H, W = rgb_frames.shape
        assert T == self.num_frames, f"输入帧数 {T} 与设置 {self.num_frames} 不匹配"
        
        # 归一化 RGB 帧
        if self.normalize:
            rgb_frames = (rgb_frames - self.mean) / self.std
        
        # 转换 RGB 为灰度
        gray_frames = []
        for t in range(T):
            gray = self.rgb_to_gray(rgb_frames[:, t])  # (B, 1, H, W)
            gray_frames.append(gray)
        gray_frames = torch.stack(gray_frames, dim=1)  # (B, T, 1, H, W)
        
        # 归一化热红外帧
        if self.normalize:
            thermal_frames = (thermal_frames - thermal_frames.mean(dim=(2, 3, 4), keepdim=True)) / \
                           (thermal_frames.std(dim=(2, 3, 4), keepdim=True) + 1e-6)
        
        # 构建全帧张量: (B, 2, H, W, T) — 用于 OffsetPredictor
        gray_all = gray_frames.squeeze(2).permute(0, 2, 3, 1)        # (B, H, W, T)
        thermal_all = thermal_frames.squeeze(2).permute(0, 2, 3, 1)  # (B, H, W, T)
        all_frames = torch.stack([gray_all, thermal_all], dim=1)      # (B, 2, H, W, T)
        
        # 时间采样
        gray_sampled = gray_frames[:, self.sample_indices]  # (B, S, 1, H, W)
        thermal_sampled = thermal_frames[:, self.sample_indices]  # (B, S, 1, H, W)
        
        # 构建 Cube 张量: (B, 2, H, W, S) — 用于主干网络
        gray_sampled = gray_sampled.squeeze(2).permute(0, 2, 3, 1)  # (B, H, W, S)
        thermal_sampled = thermal_sampled.squeeze(2).permute(0, 2, 3, 1)  # (B, H, W, S)
        
        cube = torch.stack([gray_sampled, thermal_sampled], dim=1)  # (B, 2, H, W, S)
        
        return cube, all_frames
    
    def __call__(self, rgb_frames: torch.Tensor, thermal_frames: torch.Tensor) -> torch.Tensor:
        """支持 transform 管道调用"""
        return self.forward(rgb_frames, thermal_frames)


class CubeEncodingTransform:
    """
    用于数据加载的 Cube Encoding Transform
    支持从文件路径列表构建 Cube
    """
    
    def __init__(
        self,
        num_frames: int = 5,
        sample_frames: int = 3,
        img_size: int = 640,
        normalize: bool = True
    ):
        self.num_frames = num_frames
        self.sample_frames = sample_frames
        self.img_size = img_size
        self.normalize = normalize
        
        # 计算采样索引
        if sample_frames > 1:
            self.sample_indices = np.linspace(0, num_frames - 1, sample_frames, dtype=int)
        else:
            self.sample_indices = np.array([num_frames // 2])
    
    def load_rgb_frame(self, path: str) -> np.ndarray:
        """加载 RGB 图像"""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img.astype(np.float32) / 255.0
    
    def load_thermal_frame(self, path: str) -> np.ndarray:
        """加载热红外图像"""
        import tifffile
        img = tifffile.imread(path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        # 归一化到 [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img.astype(np.float32)
    
    def __call__(
        self,
        rgb_paths: List[str],
        thermal_paths: List[str]
    ) -> torch.Tensor:
        """
        从文件路径构建 Cube
        
        参数：
            rgb_paths: 长度为 T 的 RGB 图像路径列表
            thermal_paths: 长度为 T 的热红外图像路径列表
            
        返回：
            cube: (2, H, W, S) Cube 张量
        """
        assert len(rgb_paths) == self.num_frames
        assert len(thermal_paths) == self.num_frames
        
        # 加载所有帧
        rgb_frames = []
        thermal_frames = []
        
        for rgb_path, thermal_path in zip(rgb_paths, thermal_paths):
            rgb_frame = self.load_rgb_frame(rgb_path)
            thermal_frame = self.load_thermal_frame(thermal_path)
            
            rgb_frames.append(rgb_frame)
            thermal_frames.append(thermal_frame)
        
        rgb_frames = np.stack(rgb_frames, axis=0)  # (T, H, W, 3)
        thermal_frames = np.stack(thermal_frames, axis=0)  # (T, H, W)
        
        # 采样
        rgb_sampled = rgb_frames[self.sample_indices]  # (S, H, W, 3)
        thermal_sampled = thermal_frames[self.sample_indices]  # (S, H, W)
        
        # 转换为灰度
        gray_sampled = np.dot(rgb_sampled[..., :3], [0.299, 0.587, 0.114])  # (S, H, W)
        
        # 转换为 torch 张量
        gray_tensor = torch.from_numpy(gray_sampled).permute(1, 2, 0)  # (H, W, S)
        thermal_tensor = torch.from_numpy(thermal_sampled).permute(1, 2, 0)  # (H, W, S)
        
        # 构建 Cube: (2, H, W, S)
        cube = torch.stack([gray_tensor, thermal_tensor], dim=0)
        
        return cube


if __name__ == "__main__":
    # 测试代码
    print("Testing CubeEncoding...")
    
    # 测试 nn.Module 版本
    encoder = CubeEncoding(num_frames=5, sample_frames=3, img_size=640)
    
    # 模拟输入
    B, T, H, W = 2, 5, 640, 640
    rgb_frames = torch.randn(B, T, 3, H, W)
    thermal_frames = torch.randn(B, T, 1, H, W)
    
    cube, all_frames = encoder(rgb_frames, thermal_frames)
    print(f"输入: RGB {rgb_frames.shape}, Thermal {thermal_frames.shape}")
    print(f"输出 Cube: {cube.shape}")
    print(f"Cube 范围: [{cube.min():.3f}, {cube.max():.3f}]")
    
    assert cube.shape == (B, 2, H, W, 3), "Cube 形状错误"
    print("✓ CubeEncoding 测试通过！")

