"""
OSFormer 评估脚本
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import OSFormer, OSFormerConfig
from models.seq_head import AnchorFreeDecoder
from datasets import RGBTTinyDataset, collate_fn
from utils.metrics import DetectionMetrics, calculate_map, calculate_safit


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate OSFormer')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str, default='./data/RGBT-Tiny',
                        help='数据集根目录')
    parser.add_argument('--split', type=str, default='val',
                        help='评估数据集分割 (val/test)')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作线程数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='评估设备')
    
    # 解码参数
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='分数阈值')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='NMS 阈值')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    parser.add_argument('--vis_dir', type=str, default='./visualizations',
                        help='可视化结果保存目录')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='可视化样本数量')
    
    # 保存参数
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存结果')
    parser.add_argument('--results_file', type=str, default='./results.txt',
                        help='结果保存文件')
    
    args = parser.parse_args()
    return args


def visualize_detections(
    rgb_frame: np.ndarray,
    thermal_frame: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    save_path: str
):
    """
    可视化检测结果
    
    参数：
        rgb_frame: (H, W, 3) RGB 图像
        thermal_frame: (H, W) 热红外图像
        pred_boxes: (N, 4) 预测框
        pred_scores: (N,) 预测分数
        gt_boxes: (M, 4) 真实框
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # RGB + 预测框
    axes[0].imshow(rgb_frame)
    axes[0].set_title('RGB + Predictions')
    axes[0].axis('off')
    
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color='red', linewidth=2
        )
        axes[0].add_patch(rect)
        axes[0].text(
            x1, y1 - 5, f'{score:.2f}',
            color='red', fontsize=10, weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, pad=2)
        )
    
    # RGB + GT 框
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color='green', linewidth=2, linestyle='--'
        )
        axes[0].add_patch(rect)
    
    # Thermal + 预测框
    axes[1].imshow(thermal_frame, cmap='hot')
    axes[1].set_title('Thermal + Predictions')
    axes[1].axis('off')
    
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color='cyan', linewidth=2
        )
        axes[1].add_patch(rect)
    
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color='lime', linewidth=2, linestyle='--'
        )
        axes[1].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def evaluate(model, dataloader, decoder, args):
    """评估模型"""
    model.eval()
    
    metrics = DetectionMetrics(num_classes=1)
    
    # 用于可视化
    vis_samples = []
    
    pbar = tqdm(dataloader, desc='Evaluating')
    
    for batch_idx, (rgb_frames, thermal_frames, targets_list) in enumerate(pbar):
        rgb_frames = rgb_frames.to(args.device)
        thermal_frames = thermal_frames.to(args.device)
        
        # 前向传播
        outputs = model(rgb_frames, thermal_frames)
        
        B = rgb_frames.shape[0]
        mid_frame = model.num_frames // 2
        
        for b in range(B):
            # 使用中间帧的预测
            output = outputs[mid_frame]
            
            cls_pred = output['cls'][b]
            bbox_pred = output['bbox'][b]
            centerness_pred = output.get('centerness', None)
            if centerness_pred is not None:
                centerness_pred = centerness_pred[b]
            
            # 解码
            boxes, scores, labels = decoder.decode_single_frame(
                cls_pred, bbox_pred, centerness_pred
            )
            
            # GT
            targets = targets_list[b][mid_frame]
            gt_boxes = targets['boxes'].cpu().numpy()
            gt_labels = targets['labels'].cpu().numpy()
            
            # 更新指标
            if len(boxes) > 0:
                pred_boxes_np = boxes.cpu().numpy()
                pred_scores_np = scores.cpu().numpy()
                pred_labels_np = labels.cpu().numpy()
            else:
                pred_boxes_np = np.zeros((0, 4))
                pred_scores_np = np.zeros(0)
                pred_labels_np = np.zeros(0, dtype=np.int64)
            
            metrics.update(
                pred_boxes_np,
                pred_scores_np,
                pred_labels_np,
                gt_boxes,
                gt_labels
            )
            
            # 保存用于可视化
            if args.visualize and len(vis_samples) < args.num_vis:
                # 获取原始图像
                rgb_img = rgb_frames[b, mid_frame].cpu().permute(1, 2, 0).numpy()
                thermal_img = thermal_frames[b, mid_frame, 0].cpu().numpy()
                
                # 反归一化（如果需要）
                rgb_img = np.clip(rgb_img * 255, 0, 255).astype(np.uint8)
                thermal_img = np.clip(thermal_img * 255, 0, 255).astype(np.uint8)
                
                vis_samples.append({
                    'rgb': rgb_img,
                    'thermal': thermal_img,
                    'pred_boxes': pred_boxes_np,
                    'pred_scores': pred_scores_np,
                    'gt_boxes': gt_boxes
                })
    
    # 计算指标
    results = metrics.compute()
    
    return results, vis_samples


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 从检查点恢复配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = OSFormerConfig.from_dict(config_dict)
    else:
        # 使用默认配置
        config = OSFormerConfig()
    
    # 创建模型
    model = OSFormer(
        num_frames=config.num_frames,
        sample_frames=config.sample_frames,
        img_size=config.img_size,
        embed_dim=config.embed_dim,
        use_doppler=config.use_doppler
    )
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载完成 (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 创建数据集
    dataset = RGBTTinyDataset(
        root_dir=args.data_root,
        split=args.split,
        num_frames=config.num_frames,
        img_size=config.img_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建解码器
    decoder = AnchorFreeDecoder(
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh
    )
    
    # 评估
    print("\n开始评估...")
    results, vis_samples = evaluate(model, dataloader, decoder, args)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"{key:20s}: {val:.4f}")
        else:
            print(f"{key:20s}: {val}")
    print("=" * 50)
    
    # 保存结果
    if args.save_results:
        with open(args.results_file, 'w') as f:
            f.write("OSFormer Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {args.data_root} ({args.split})\n")
            f.write(f"Score Threshold: {args.score_thresh}\n")
            f.write(f"NMS Threshold: {args.nms_thresh}\n")
            f.write("=" * 50 + "\n\n")
            
            for key, val in results.items():
                if isinstance(val, float):
                    f.write(f"{key:20s}: {val:.4f}\n")
                else:
                    f.write(f"{key:20s}: {val}\n")
        
        print(f"\n结果已保存到: {args.results_file}")
    
    # 可视化
    if args.visualize and len(vis_samples) > 0:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"\n生成可视化结果...")
        
        for idx, sample in enumerate(vis_samples):
            save_path = os.path.join(args.vis_dir, f'sample_{idx:03d}.png')
            visualize_detections(
                sample['rgb'],
                sample['thermal'],
                sample['pred_boxes'],
                sample['pred_scores'],
                sample['gt_boxes'],
                save_path
            )
        
        print(f"可视化结果已保存到: {args.vis_dir}")


if __name__ == '__main__':
    main()

