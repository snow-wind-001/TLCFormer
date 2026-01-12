# TLC-Former 变更日志 (CHANGELOG)

本文件记录 TLC-Former 项目的所有重要变更、算法改进和版本更新。

---

## [1.0.0] - 2026-01-12 - TLCFormer 完整实现

### ✅ 已完成的改进

| 组件 | 状态 | 位置 | 描述 |
|------|------|------|------|
| MADA | ✅ 已实现 | `models/mada.py` | 运动感知差分注意力 |
| DLCM | ✅ 已实现 | `models/dlcm.py` | 深度局部对比度模块 |
| Hybrid Mixer | ✅ 已实现 | `models/vpa.py` | Max-Mean 混合池化 |
| TLCFormer | ✅ 已集成 | `models/osformer.py` | 主模型已更新 |

### 📝 论文

- 生成了 ICML 2026 格式的完整论文 (`paper/tlcformer.pdf`, **9页**)
- 创建并嵌入 4 个网络架构图:
  - `architecture.pdf` - 整体架构图 (第3页)
  - `mada_module.pdf` - MADA 模块详细图 (第3页)
  - `dlcm_module.pdf` - DLCM 模块详细图 (第4页)  
  - `hybrid_mixer.pdf` - Hybrid Mixer 详细图 (第5页)
- 包含完整的算法推导和实验分析
- 所有图片已成功从 SVG 转换为 PDF 格式

---

## [Unreleased] - 历史记录

### 🔍 初始状态分析 (2026-01-12 早期)

#### 发现的问题（已解决）

**README.md 与实际代码不一致**：

| 组件 | README 描述 | 主代码实现 (`models/`) | 状态 |
|------|------------|----------------------|------|
| 运动滤波 | MADA (帧差注意力) | ❌→✅ 已实现 | 已解决 |
| 对比度增强 | DLCM (局部对比度) | ❌→✅ 已实现 | 已解决 |
| Token Mixer | Hybrid Max-Mean Pooling | ❌→✅ 已实现 | 已解决 |
| 主模型 | TLCFormer | ❌→✅ 已集成 | 已解决 |

#### 已实现模块详情

以下模块已在 `UsedCode/OSFormer/osformer-rgbt/models/` 目录中完整实现：

1. **MADA (Motion-Aware Difference Attention)** - `mada.py`
   - 帧差运动注意力机制
   - 计算时域梯度：$D_{pre} = |I_t - I_{t-1}|$, $D_{next} = |I_{t+1} - I_t|$
   - 运动显著图：$M_{raw} = D_{pre} \odot D_{next}$
   - 可学习缩放因子 $\alpha$
   - 包含 `MADAWithDownsample` 多尺度版本

2. **DLCM (Deep Local Contrast Module)** - `dlcm.py`
   - 背景估计：9×9 AvgPool
   - 目标强度估计：3×3 MaxPool  
   - 对比度响应：$C = L_{max}^2 / (\mu_{bg} + \epsilon)$ 或 $C = \text{ReLU}(X - \mu_{bg})$
   - 可学习融合权重 $\beta$
   - 包含 `DLCMMultiScale` 和 `DLCMLight` 变体

3. **Hybrid Energy-Preserving Mixer** - `vpa_hybrid.py`
   - 双路池化：MaxPool + AvgPool
   - 通道拼接与压缩：$P_{hybrid} = \text{Concat}(P_{max}, P_{avg})$
   - 1×1 卷积降维：$X_{mixed} = \text{GELU}(\text{Conv}_{1\times1}(P_{hybrid}))$
   - 残差连接保留原始信息

4. **TLCFormer 完整模型** - `tlc_former.py`
   - 集成 MADA、DLCM、Hybrid Mixer
   - 配置类 `TLCFormerConfig`
   - 向后兼容别名（OSFormer = TLCFormer）

---

### 🚧 待完成任务

#### 高优先级

- [ ] 将 `UsedCode/` 中的模块迁移到主 `models/` 目录
  - [ ] 复制 `mada.py` → `models/mada.py`
  - [ ] 复制 `dlcm.py` → `models/dlcm.py`
  - [ ] 复制 `vpa_hybrid.py` → `models/vpa_hybrid.py`
  - [ ] 复制 `tlc_former.py` → `models/tlc_former.py`

- [ ] 更新 `models/__init__.py` 导出新模块

- [ ] 更新训练脚本使用 TLCFormer 而非 OSFormer

- [ ] 添加 NWD (Normalized Wasserstein Distance) 损失用于小目标优化

#### 中优先级

- [ ] 为 MADA、DLCM 添加可视化工具
- [ ] 添加消融实验配置
- [ ] 更新配置文件支持新模块

#### 低优先级

- [ ] 保留 OSFormer 作为 baseline 对比
- [ ] 添加模块单元测试

---

## [0.1.0] - 2026-01-12 - 初始版本

### 网络架构

#### 当前主代码实现 (`models/`)

```
RGB Frames (B,T,3,H,W) ─┐
                        ├─→ CubeEncoding ─→ Cube (B,2,H,W,S)
Thermal Frames (B,T,1,H,W)                     │
                                               ▼
                                    DopplerAdaptiveFilter (FFT)
                                               │
                                               ▼
                                    VPA Encoder (AvgPool Mixer)
                                               │
                                               ▼
                               Multi-scale Features [F1,F2,F3,F4]
                                               │
                                               ▼
                                    FeatureRefinementNeck
                                               │
                                               ▼
                                    SequenceRegressionHead
                                               │
                                               ▼
                               Detections: cls, bbox, centerness, offset
```

#### 目标架构 (TLCFormer)

```
RGB Frames (B,T,3,H,W) ─┐
                        ├─→ CubeEncoding ─→ Cube (B,2,H,W,S)
Thermal Frames (B,T,1,H,W)                     │
                                               ▼
                                    MADA (Motion-Aware Difference Attention)
                                    [替代 DopplerAdaptiveFilter]
                                               │
                                               ▼
                                    DLCM (Deep Local Contrast Module)
                                    [增强局部对比度]
                                               │
                                               ▼
                                    VPA Encoder (Hybrid Max-Mean Mixer)
                                    [保留小目标能量]
                                               │
                                               ▼
                               Multi-scale Features [F1,F2,F3,F4]
                                               │
                                               ▼
                                    FeatureRefinementNeck
                                               │
                                               ▼
                                    SequenceRegressionHead + NWD Loss
                                               │
                                               ▼
                               Detections: cls, bbox, centerness, offset
```

### 核心改进算法

#### 1. MADA 公式

**时域梯度计算**：
$$D_{pre} = |I_t - I_{t-1}|, \quad D_{next} = |I_{t+1} - I_t|$$

**运动显著图**：
$$M_{raw} = D_{pre} \odot D_{next}$$

**注意力权重**：
$$A_{motion} = \sigma(\mathcal{F}_{motion}(M_{raw}))$$

**特征加权**：
$$I'_t = I_t \cdot (1 + \alpha \cdot A_{motion})$$

#### 2. DLCM 公式

**背景估计**：
$$\mu_{bg}(i,j) = \frac{1}{N_{out}} \sum_{(p,q) \in \Omega_{out}} X(i+p, j+q)$$

**目标强度估计**：
$$L_{max}(i,j) = \max_{(p,q) \in \Omega_{in}} X(i+p, j+q)$$

**对比度响应**：
$$C(i,j) = \frac{L_{max}(i,j)^2}{\mu_{bg}(i,j) + \epsilon}$$

**融合输出**：
$$X_{out} = X + \beta \cdot C$$

#### 3. Hybrid Mixer 公式

**双路池化**：
$$P_{max} = \text{MaxPool2d}(X, k, s), \quad P_{avg} = \text{AvgPool2d}(X, k, s)$$

**特征拼接与压缩**：
$$P_{hybrid} = \text{Concat}(P_{max}, P_{avg})$$
$$X_{mixed} = \text{GELU}(\text{Conv}_{1\times1}(P_{hybrid}))$$

**残差重构**：
$$X_{out} = X + X_{mixed}$$

### 模块参数

| 模块 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| MADA | `num_frames` | 3 | 时间帧数 |
| MADA | `in_channels` | 2 | 输入通道（灰度+热红外）|
| MADA | `alpha` | 0.5 | 可学习缩放因子 |
| DLCM | `kernel_inner` | 3 | 目标区域大小 |
| DLCM | `kernel_outer` | 9 | 背景区域大小 |
| DLCM | `beta` | 0.5 | 可学习融合权重 |
| Hybrid | `pool_size` | 3 | 池化核大小 |

### 物理先验总结

| 问题 | 物理先验 | 解决方案 |
|------|----------|----------|
| 背景噪声干扰 | 目标运动，背景静止 | MADA 帧差分 |
| 低信噪比 | 目标是局部极值点 | DLCM 局部对比度 |
| 下采样能量丢失 | 目标仅1-4像素 | Hybrid MaxPool |

### 损失函数

- **分类**: Weighted Focal Loss (类别加权)
- **边界框**: CIoU Loss
- **中心度**: BCE Loss
- **偏移**: Smooth L1 Loss
- **[计划]**: NWD Loss (Normalized Wasserstein Distance)

---

## 文件结构

```
TLCFormer/
├── models/                     # 主模型代码（需要更新）
│   ├── __init__.py
│   ├── osformer.py            # 当前：OSFormer (使用 Doppler)
│   ├── vpa.py                 # 当前：AvgPool Mixer
│   ├── doppler_filter.py      # 当前：FFT 滤波器（将被替换）
│   ├── cube_encoding.py
│   ├── neck.py
│   └── seq_head.py
├── UsedCode/.../models/        # 备用实现（需迁移）
│   ├── mada.py                # ✅ MADA 完整实现
│   ├── dlcm.py                # ✅ DLCM 完整实现
│   ├── vpa_hybrid.py          # ✅ Hybrid Mixer 完整实现
│   └── tlc_former.py          # ✅ TLCFormer 完整实现
├── utils/
│   └── loss.py
├── configs/
├── datasets/
├── README.md
└── CHANGELOG.md               # 本文件
```

---

## 训练命令

### 当前 OSFormer 训练

```bash
python train_rgbb_tiny.py --config configs/rgbt_tiny_config.yaml
```

### 计划的 TLCFormer 训练（迁移完成后）

```bash
python train_rgbb_tiny.py --config configs/rgbt_tiny_config.yaml \
    --model tlcformer \
    --use_mada \
    --use_dlcm \
    --use_hybrid_mixer
```

---

## 版本说明

- **Unreleased**: 待迁移的改进模块
- **0.1.0**: 初始版本，OSFormer 架构 + 部分 TLC 模块已实现但未集成

---

## 贡献者

- 项目维护者

---

*最后更新: 2026-01-12*
