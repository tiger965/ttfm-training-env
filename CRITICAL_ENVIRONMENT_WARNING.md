# ⚠️ 关键环境警告 - 绝对不要修改！

## 🚨 极度重要说明

**这是经过无数次崩溃和重装后得到的唯一稳定环境配置！**

### ❌ 绝对禁止的操作：

1. **不要升级任何包**
2. **不要升级pip**
3. **不要升级PyTorch**
4. **不要升级CUDA**
5. **不要升级WSL**
6. **不要更改Python版本**
7. **不要安装conflicting packages**

### 💀 历史教训：

- 2025-09-20: 升级了某个包导致整个环境崩溃，花费10+小时重装
- 2025-09-21: 安装psutil后环境再次崩溃，又花费12+小时恢复

## 🔒 锁定版本信息

### 系统环境
```
OS: Ubuntu 24.04 LTS (WSL2)
WSL版本: 2.3.26.0
内核: 6.6.87.2-microsoft-standard-WSL2
Windows: Windows 11
GPU: NVIDIA GeForce RTX 5080 Laptop GPU (16GB)
CUDA: 12.8 (驱动577.03)
架构: sm_120 (必须用PyTorch nightly)
```

### Python环境
```
Python: 3.10.18 (绝对不要改)
Conda: 24.11.2
环境名: pytorch (不要改名)
```

### 核心依赖（精确版本）
```
torch==2.10.0.dev20250921+cu128  # 必须是nightly build！
transformers==4.56.1
peft==0.17.1
bitsandbytes==0.47.0
datasets==4.1.1
accelerate==1.10.1
numpy==2.2.6
psutil==7.1.0
```

### 关键修复代码（必须包含）
```python
# RTX 5080 PEFT修复
import peft.tuners.tuners_utils
peft.tuners.tuners_utils.BaseTuner._cast_adapter_dtype = lambda *args, **kwargs: None

# Qwen3模型注册
from transformers.models.auto import configuration_auto, modeling_auto
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
configuration_auto.CONFIG_MAPPING._extra_content['qwen3'] = Qwen2Config
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['qwen3'] = 'Qwen2ForCausalLM'
```

### 环境变量（必须设置）
```bash
export BNB_CUDA_VERSION=128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0
```

## ✅ 验证环境

运行以下命令验证环境：
```bash
cd /home/tiger/tiger_trust_project
python test_environment.py
```

应该显示：
- PyTorch: 2.10.0.dev20250921+cu128
- CUDA可用: True
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- 所有依赖: ✅

## 🔥 恢复方法（如果环境损坏）

1. 从Git恢复：
```bash
git clone https://github.com/tiger/ttfm-training-env.git
cd ttfm-training-env
```

2. 重新安装环境：
```bash
# 安装Miniconda (精确路径)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3

# 创建环境
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n pytorch python=3.10.18 -y
conda activate pytorch

# 安装PyTorch nightly (关键！)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装其他依赖
pip install -r requirements.txt
```

## 📝 最后更新

- 日期：2025-09-21 22:40
- 验证：环境稳定运行中
- 训练速度：3-4秒/步（正常）

---

**记住：这个环境来之不易，任何改动都可能导致灾难性后果！**