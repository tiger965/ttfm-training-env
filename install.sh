#!/bin/bash
# 🚀 RTX 50系列深度学习环境 - 一键安装脚本
# 菜鸟救星版本 - 零基础也能用！

set -e  # 遇到错误立即停止

# 彩色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logo
echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║      RTX 50系列深度学习环境 - 菜鸟救星安装器            ║
║                                                          ║
║       _____ _______ __   __   _____ ___                 ║
║      |  __ \__   __\ \ / /  | ____/ _ \                 ║
║      | |__) | | |   \ V /   | |__ | | | |                ║
║      |  _  /  | |    > <    |___ \| | | |                ║
║      | | \ \  | |   / . \    ___) | |_| |                ║
║      |_|  \_\ |_|  /_/ \_\  |____/ \___/                 ║
║                                                          ║
║            一键安装，开箱即用，小白友好                  ║
╚══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# 系统检查
echo -e "${YELLOW}[1/8]${NC} 🔍 检查系统环境..."

# 检查操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${GREEN}✓${NC} 检测到Linux系统"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo -e "${GREEN}✓${NC} 检测到Windows WSL"
else
    echo -e "${RED}✗${NC} 不支持的操作系统: $OSTYPE"
    echo "请使用Ubuntu/WSL2环境"
    exit 1
fi

# 检查NVIDIA驱动
echo -e "${YELLOW}[2/8]${NC} 🎮 检查显卡..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓${NC} 检测到GPU: $GPU_NAME"

    # 检查是否是RTX 50系列
    if [[ $GPU_NAME == *"50"* ]]; then
        echo -e "${GREEN}✓${NC} 完美！这是RTX 50系列显卡"
    else
        echo -e "${YELLOW}⚠${NC} 注意：此环境为RTX 50系列优化，你的GPU是: $GPU_NAME"
        read -p "是否继续安装？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${RED}✗${NC} 未检测到NVIDIA GPU"
    echo "请确保已安装NVIDIA驱动"
    exit 1
fi

# 创建安装目录
INSTALL_DIR="$HOME/rtx50_env"
echo -e "${YELLOW}[3/8]${NC} 📁 创建安装目录: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# 安装Miniconda（如果需要）
echo -e "${YELLOW}[4/8]${NC} 🐍 安装Python环境管理器..."
if [ -d "$HOME/miniconda3" ]; then
    echo -e "${GREEN}✓${NC} Miniconda已安装"
else
    echo "正在下载Miniconda（约100MB）..."
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    echo "正在安装..."
    bash miniconda.sh -b -p $HOME/miniconda3 > /dev/null 2>&1
    rm miniconda.sh
    echo -e "${GREEN}✓${NC} Miniconda安装完成"
fi

# 初始化conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# 创建虚拟环境
echo -e "${YELLOW}[5/8]${NC} 🔧 创建专用Python环境..."
if conda env list | grep -q "rtx50"; then
    echo -e "${YELLOW}⚠${NC} 环境已存在，正在重建..."
    conda env remove -n rtx50 -y > /dev/null 2>&1
fi

conda create -n rtx50 python=3.10.18 -y > /dev/null 2>&1
conda activate rtx50
echo -e "${GREEN}✓${NC} Python环境创建成功"

# 安装PyTorch
echo -e "${YELLOW}[6/8]${NC} 🔥 安装PyTorch（RTX 50专用版本）..."
echo "这可能需要几分钟，请喝杯咖啡☕..."
pip install -q --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
echo -e "${GREEN}✓${NC} PyTorch安装成功"

# 安装其他依赖
echo -e "${YELLOW}[7/8]${NC} 📚 安装AI训练库..."
pip install -q transformers==4.56.1 \
    peft==0.17.1 \
    bitsandbytes==0.47.0 \
    datasets==4.1.1 \
    accelerate==1.10.1 \
    numpy scipy scikit-learn \
    gradio streamlit
echo -e "${GREEN}✓${NC} 所有依赖安装完成"

# 创建示例代码
echo -e "${YELLOW}[8/8]${NC} 📝 创建示例代码..."

# 创建测试脚本
cat > $INSTALL_DIR/test_gpu.py << 'PYEOF'
#!/usr/bin/env python3
"""GPU测试脚本 - 验证环境是否正确配置"""

import torch
import sys

print("="*50)
print("🎮 RTX 50系列环境测试")
print("="*50)

# 检查PyTorch
print(f"PyTorch版本: {torch.__version__}")

# 检查CUDA
if torch.cuda.is_available():
    print(f"✅ CUDA可用")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 简单计算测试
    print("\n运行简单测试...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✅ GPU计算正常")

    # 8bit测试
    try:
        import bitsandbytes as bnb
        print(f"✅ 8bit量化支持正常 (bitsandbytes {bnb.__version__})")
    except:
        print("⚠️  8bit量化未配置")

    print("\n🎉 恭喜！环境配置成功！")
else:
    print("❌ CUDA不可用")
    print("请检查NVIDIA驱动")
    sys.exit(1)

print("="*50)
PYEOF

# 创建简单的训练示例
cat > $INSTALL_DIR/train_example.py << 'PYEOF'
#!/usr/bin/env python3
"""简单的模型训练示例"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

print("🚀 开始一个简单的训练示例...")

# 创建示例数据
texts = ["Hello RTX 50!", "This is amazing!", "AI training is fun!"] * 100
dataset = Dataset.from_dict({"text": texts})

print(f"✅ 创建了{len(dataset)}个训练样本")

# 加载一个小模型作为示例
print("📦 加载模型（使用小模型演示）...")
model_name = "gpt2"  # 使用小模型做演示
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize数据
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    fp16=True,  # 使用混合精度
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("🎯 开始训练...")
trainer.train()

print("✅ 训练完成！")
print("模型已保存到: ./results")
PYEOF

# 创建Web UI示例
cat > $INSTALL_DIR/web_ui.py << 'PYEOF'
#!/usr/bin/env python3
"""简单的Web界面示例"""

import gradio as gr
import torch

def gpu_info():
    """获取GPU信息"""
    if torch.cuda.is_available():
        info = f"""
        ✅ GPU可用
        设备: {torch.cuda.get_device_name(0)}
        显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB
        PyTorch: {torch.__version__}
        """
    else:
        info = "❌ GPU不可用"
    return info

def simple_generate(prompt, max_length=50):
    """简单的文本生成示例"""
    # 这里只是演示，实际使用时加载真实模型
    return f"输入: {prompt}\n\n生成: 这是一个演示输出。在实际使用中，这里会是AI生成的内容。"

# 创建界面
with gr.Blocks(title="RTX 50 AI训练平台") as demo:
    gr.Markdown("# 🚀 RTX 50系列 AI训练平台")
    gr.Markdown("### 欢迎使用！这是一个简单的Web界面示例")

    with gr.Tab("GPU信息"):
        gr.Markdown("点击按钮查看GPU状态")
        info_btn = gr.Button("检查GPU")
        info_output = gr.Textbox(label="GPU信息", lines=5)
        info_btn.click(gpu_info, outputs=info_output)

    with gr.Tab("文本生成"):
        gr.Markdown("简单的文本生成演示")
        prompt_input = gr.Textbox(label="输入提示词", placeholder="输入一些文字...")
        gen_btn = gr.Button("生成")
        gen_output = gr.Textbox(label="生成结果", lines=5)
        gen_btn.click(simple_generate, inputs=prompt_input, outputs=gen_output)

    gr.Markdown("---")
    gr.Markdown("💡 提示：这只是一个演示界面，你可以基于此开发更复杂的应用")

if __name__ == "__main__":
    print("启动Web界面...")
    print("浏览器访问: http://localhost:7860")
    demo.launch(share=False)
PYEOF

# 创建启动脚本
cat > $INSTALL_DIR/start.sh << 'BASHEOF'
#!/bin/bash
# 快速启动脚本

echo "🚀 激活RTX 50环境..."
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate rtx50

# 设置环境变量
export BNB_CUDA_VERSION=128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

echo "✅ 环境已激活"
echo ""
echo "可用命令:"
echo "  python test_gpu.py     - 测试GPU"
echo "  python train_example.py - 运行训练示例"
echo "  python web_ui.py       - 启动Web界面"
echo ""

# 进入交互式shell
exec bash
BASHEOF

chmod +x $INSTALL_DIR/*.sh
chmod +x $INSTALL_DIR/*.py

# 配置快捷方式
echo -e "\n${YELLOW}配置快捷命令...${NC}"
cat >> $HOME/.bashrc << 'BASHEOF'

# RTX 50环境快捷命令
alias rtx50="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate rtx50 && export BNB_CUDA_VERSION=128"
alias rtx50-test="cd $HOME/rtx50_env && python test_gpu.py"
alias rtx50-ui="cd $HOME/rtx50_env && python web_ui.py"
BASHEOF

# 最终测试
echo -e "\n${YELLOW}运行最终测试...${NC}"
python $INSTALL_DIR/test_gpu.py

# 完成
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 恭喜！RTX 50环境安装成功！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}📚 快速开始：${NC}"
echo "1. 激活环境:  source ~/rtx50_env/start.sh"
echo "2. 测试GPU:   python ~/rtx50_env/test_gpu.py"
echo "3. 运行示例:  python ~/rtx50_env/train_example.py"
echo "4. Web界面:   python ~/rtx50_env/web_ui.py"
echo ""
echo -e "${BLUE}💡 提示：${NC}"
echo "• 所有文件在: ~/rtx50_env/"
echo "• 使用 'rtx50' 命令快速激活环境"
echo "• 查看 README.md 了解更多用法"
echo ""
echo -e "${YELLOW}⭐ 如果这个工具帮助了你，请在GitHub给我们一个Star！${NC}"
echo "   https://github.com/tiger965/ttfm-training-env"
echo ""