#!/bin/bash
# TTFM 24代训练系统启动脚本
# 适用于Ubuntu 24.04 + RTX 5080

echo "================================================"
echo "🚀 TTFM 24代训练系统 - Ubuntu 24.04"
echo "📍 RTX 5080 8bit训练环境"
echo "================================================"

# 激活conda环境
echo "⚙️ 激活PyTorch环境..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# 设置环境变量
export BNB_CUDA_VERSION=128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0

# 显示环境信息
echo ""
echo "📊 环境信息:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import bitsandbytes as bnb; print(f'  bitsandbytes: {bnb.__version__}')"

# 检查训练状态
echo ""
echo "📈 训练状态:"
if [ -f "/mnt/d/models/留存模型/Tiger信任训练/24代系统/training_state.json" ]; then
    echo "  发现上次训练记录:"
    cat /mnt/d/models/留存模型/Tiger信任训练/24代系统/training_state.json | python -m json.tool
    echo ""
    echo "  将从断点继续训练..."
else
    echo "  无训练记录，将从头开始"
fi

# 切换到脚本目录
cd /home/tiger/tiger_trust_project/scripts

# 启动训练
echo ""
echo "🎯 开始训练..."
echo "================================================"
python dierdai_fixed3.py

echo ""
echo "✅ 训练结束"