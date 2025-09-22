#!/bin/bash
# RTX 50系列深度学习环境快速配置脚本
# 适用于Ubuntu/WSL2环境

echo "================================================"
echo "🚀 RTX 50系列深度学习环境一键配置"
echo "📦 支持: RTX 5080/5090等sm_120架构显卡"
echo "================================================"

# 检查Python版本
echo "📍 检查Python版本..."
if ! command -v python3.10 &> /dev/null; then
    echo "⚠️ 未找到Python 3.10，正在安装..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev
fi

# 检查是否已有conda
if [ -d "$HOME/miniconda3" ]; then
    echo "✅ 检测到已安装的Miniconda"
else
    echo "📥 下载并安装Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

# 初始化conda
echo "🔧 配置conda环境..."
source $HOME/miniconda3/etc/profile.d/conda.sh

# 创建pytorch环境
if conda env list | grep -q "pytorch"; then
    echo "✅ pytorch环境已存在"
    conda activate pytorch
else
    echo "📦 创建pytorch环境..."
    conda create -n pytorch python=3.10.18 -y
    conda activate pytorch
fi

# 安装PyTorch nightly (关键!)
echo "🔥 安装PyTorch nightly (RTX 50系列必需)..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装其他依赖
echo "📚 安装深度学习依赖..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # 如果没有requirements.txt，安装核心包
    pip install transformers==4.56.1
    pip install peft==0.17.1
    pip install bitsandbytes==0.47.0
    pip install datasets==4.1.1
    pip install accelerate==1.10.1
    pip install numpy scipy scikit-learn
fi

# 设置环境变量
echo "⚙️ 配置环境变量..."
cat >> ~/.bashrc << 'EOF'

# RTX 50系列环境变量
export BNB_CUDA_VERSION=128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0

# 快速激活pytorch环境
alias rtx50="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate pytorch"
EOF

# 创建激活脚本
cat > activate_env.sh << 'EOF'
#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
export BNB_CUDA_VERSION=128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
echo "✅ RTX 50环境已激活"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
EOF

chmod +x activate_env.sh

# 验证安装
echo ""
echo "🔍 验证环境..."
python -c "import torch; print(f'✅ PyTorch版本: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch未正确安装"
python -c "import transformers; print(f'✅ Transformers版本: {transformers.__version__}')" 2>/dev/null || echo "❌ Transformers未安装"
python -c "import torch; print(f'✅ CUDA可用: {torch.cuda.is_available()}')" 2>/dev/null

# 检测GPU
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "✅ 检测到GPU: $GPU_NAME"

    # 检查是否是RTX 50系列
    if [[ $GPU_NAME == *"50"* ]]; then
        echo "🎉 完美！检测到RTX 50系列显卡"
    else
        echo "⚠️ 注意：此环境专为RTX 50系列优化，你的GPU是: $GPU_NAME"
    fi
else
    echo "⚠️ 未检测到CUDA设备"
fi

echo ""
echo "================================================"
echo "✨ 环境配置完成！"
echo ""
echo "📖 使用方法："
echo "1. 激活环境: source activate_env.sh"
echo "2. 或使用别名: rtx50"
echo "3. 运行测试: python test_environment.py"
echo "4. 开始训练: python scripts/train_example.py"
echo ""
echo "💡 提示：如遇问题，请查看 QUICK_START.md"
echo "================================================"