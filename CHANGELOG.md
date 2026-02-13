# TLC-Former 变更日志 (CHANGELOG)

本文件记录 TLC-Former 项目的所有重要变更、算法改进和版本更新。

---

## [1.1.2] - 2026-02-13 - 仓库精简：移除 paper 文件夹

### 变更

- **移除 `paper/` 文件夹**：LaTeX 源码、PDF、架构图等不再纳入仓库，便于代码仓库聚焦实现与文档。
- **文档更新**：README.md、README_CN.md 中已删除对 `paper/` 的引用，论文说明改为“未包含在本仓库中”。

---

## [1.1.1] - 2026-02-13 - 修复 Offset 数量不匹配：从全部 T 帧提取时间特征

### 问题

上一版 (1.1.0) 修复了 OffsetPredictor 全零问题，但 offset 数量仅 S-1=2（采样帧数-1），
而期望 T-1=4（总帧数-1）。原因：`TemporalFeatureExtractor` 从 Cube `(B,2,H,W,S=3)` 提取，
只有 3 个采样帧，无法覆盖全部 5 帧。

### 修复（方案 4）

**核心思路**：让 `TemporalFeatureExtractor` 从原始全部 T=5 帧中提取特征，而非从 Cube 的 S=3 帧。

| 文件 | 变更 |
|------|------|
| `models/cube_encoding.py` | `CubeEncoding.forward` 返回值从 `cube` 改为 `(cube, all_frames)`，新增 `all_frames: (B,2,H,W,T)` |
| `models/osformer.py` | `OSFormer.forward` 接收 `(cube, all_frames)`，将 `all_frames` 传给 `seq_head` |
| `models/seq_head.py` | `SequenceRegressionHead.forward` 参数从 `cube` 改为 `all_frames` |

**数据流**：
```
CubeEncoding → cube (B,2,H,W,S=3)   → MADA → DLCM → VPA → Neck → F0 (检测用)
             → all_frames (B,2,H,W,T=5) ────────────────────────→ TemporalFeatureExtractor
                                                                    → T 个帧特征
                                                                    → OffsetPredictor
                                                                    → T-1=4 个 offset
```

### 验证结果

| 检查项 | 修复前 (v1.1.0) | 修复后 (v1.1.1) |
|--------|----------------|----------------|
| Offset 数量 | 2/4 (S-1) | **4/4 (T-1)** |
| 覆盖帧对 | 0→2, 2→4 (跳帧) | **0→1, 1→2, 2→3, 3→4 (连续)** |
| Offset 非零 | 全非零 | 全非零 |
| Offset 互不相同 | 是 | 是 |
| 主干网络 | 不变 | 不变 |
| 额外内存 | ~0 | ~13MB (可忽略) |
| 参数量 | 30.01M | 30.01M (不变) |

---

## [1.1.0] - 2026-02-13 - 修复 OffsetPredictor 帧间偏移预测失效问题

### Bug 修复

**OffsetPredictor 输出全零（致命缺陷）**

- **问题**：`SequenceRegressionHead.forward` 中调用 `self.offset_predictor([F0] * self.num_frames)` 将同一个 F0 复制多次传入 OffsetPredictor。由于 OffsetPredictor 通过 `features[t+1] - features[t]` 计算帧间差，相同的输入导致差值恒为零，所有 offset 预测完全无效。
- **根因**：5帧输入经 Cube Encoding 采样为 S=3 帧后合并为通道维度 `(B, 2*S, H, W)`，经 VPA + Neck 后变为单帧特征 F0，时间信息丢失。
- **修复**：新增 `TemporalFeatureExtractor` 模块，从 Cube 张量中为每个采样帧独立提取下采样特征，传入 OffsetPredictor 进行真实的帧间差异计算。

### 网络架构变更

| 组件 | 位置 | 变更 |
|------|------|------|
| TemporalFeatureExtractor | `models/seq_head.py` | **新增**：轻量级时间特征提取器（0.073M 参数，仅占总量 0.24%） |
| SequenceRegressionHead | `models/seq_head.py` | **修改**：`forward` 新增 `cube` 参数，新增 `sample_frames`/`cube_channels`/`target_stride` 初始化参数 |
| OffsetPredictor | `models/seq_head.py` | **不变**：逻辑本身正确，只是之前接收了错误的输入 |
| OSFormer | `models/osformer.py` | **修改**：`forward` 传入 `cube` 到 `seq_head`；初始化时传入 `sample_frames` 等参数 |

### TemporalFeatureExtractor 详情

```
Cube (B, 2, H, W, S) → 分帧提取 → S 个 (B, 2, H, W)
                                      ↓ 逐步下采样（stride-2 卷积 x3）
                                    S 个 (B, 256, H/8, W/8)
                                      ↓ OffsetPredictor 帧差计算
                                    (B, S-1, 2, H/8, W/8) offset
```

- 输入通道：2（灰度+热红外）
- 下采样层：Conv2d(2→8, s=2) → Conv2d(8→32, s=2) → Conv2d(32→128, s=2) → Conv2d(128→256, 1x1)
- 参数量：0.073M
- 向后兼容：`cube=None` 时退化为旧行为

### 验证结果

| 检查项 | 修改前 | 修改后 |
|--------|--------|--------|
| Frame 0 offset sum | 0.000000 | >1000 (非零) |
| Frame 1 offset sum | 0.000000 | >1000 (非零) |
| 输出维度 | 正确 | 正确（不变） |
| 总参数量 | ~29.94M | ~30.01M (+0.24%) |

### 文档更新

- 更新 README.md 项目结构和架构说明
- 更新 CHANGELOG.md 添加本次修复记录

---

## [1.0.0] - 2026-01-12 - TLCFormer 完整实现与发布

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

### 📚 文档

- 创建了英文 README.md（详细项目说明，使用指南）
- 创建了中文 README_CN.md（完整中文翻译）
- 包含安装、使用、配置、引用等完整信息

### 🚀 发布

- 已推送到 GitHub: https://github.com/snow-wind-001/TLCFormer.git
- 使用 `main` 分支
- 配置了完整的 `.gitignore` 文件

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
