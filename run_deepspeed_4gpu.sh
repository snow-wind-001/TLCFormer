#!/bin/bash

# OSFormer DeepSpeed 4-GPU Training Script
# 使用ZeRO-2优化的多卡训练

set -e

# 配置参数
NUM_GPUS=4
CONFIG="./configs/rgbt_tiny_config.yaml"
DEEPSPEED_CONFIG="./configs/deepspeed_config.json"
MASTER_PORT=29500

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================================="
echo "🚀 OSFormer DeepSpeed Multi-GPU Training"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  GPUs:              $NUM_GPUS"
echo "  Config:            $CONFIG"
echo "  DeepSpeed Config:  $DEEPSPEED_CONFIG"
echo "  Master Port:       $MASTER_PORT"
echo ""

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
    echo -e "${YELLOW}⚠️  Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available${NC}"
    echo "   Will use $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# 显示GPU信息
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "❌ DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

echo "=================================================="
echo "🏃 Starting Training"
echo "=================================================="
echo ""

# 使用deepspeed启动器
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_deepspeed.py \
    --config $CONFIG \
    --deepspeed_config $DEEPSPEED_CONFIG \
    "$@"

# 训练完成
echo ""
echo "=================================================="
echo "✅ Training completed or stopped"
echo "=================================================="
echo ""
echo "📊 Check results:"
echo "  - Checkpoints: ./checkpoints/rgbt_tiny/"
echo "  - TensorBoard: tensorboard --logdir=./runs/rgbt_tiny_deepspeed"
echo ""

