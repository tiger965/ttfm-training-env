# TTFM Training Environment Backup

## 📌 重要说明
这是TTFM 24代训练系统的**稳定环境备份**，包含所有必要的配置和依赖。

## 🖥️ 系统配置
- **OS**: Ubuntu 24.04 LTS (WSL2)
- **GPU**: RTX 5080 16GB (sm_120架构)
- **CUDA**: 12.8
- **Python**: 3.10.18

## 📦 核心依赖
```
PyTorch: 2.10.0.dev20250921+cu128 (nightly build)
Transformers: 4.56.1
PEFT: 0.17.1
bitsandbytes: 0.47.0
datasets: 4.1.1
accelerate: 1.10.1
```

## ⚠️ RTX 5080关键修复
必须包含以下修复代码：

```python
# 1. PEFT dtype修复
import peft.tuners.tuners_utils
peft.tuners.tuners_utils.BaseTuner._cast_adapter_dtype = lambda *args, **kwargs: None

# 2. Qwen3模型注册
from transformers.models.auto import configuration_auto, modeling_auto
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
configuration_auto.CONFIG_MAPPING._extra_content['qwen3'] = Qwen2Config
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['qwen3'] = 'Qwen2ForCausalLM'
```

## 🚀 环境安装

### 1. 安装Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
```

### 2. 创建环境
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n pytorch python=3.10.18 -y
conda activate pytorch
```

### 3. 安装PyTorch (关键！必须用nightly)
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. 安装其他依赖
```bash
pip install -r requirements.txt
```

### 5. 设置环境变量
```bash
export BNB_CUDA_VERSION=128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

## ✅ 验证环境
```bash
cd /home/tiger/tiger_trust_project
python test_environment.py
```

## 🎯 启动训练
```bash
cd /home/tiger/tiger_trust_project
./启动训练.sh
```

## ⚠️ 重要警告
1. **绝对不要升级pip或任何主要包**
2. **不要更改PyTorch版本**
3. **必须使用nightly build支持sm_120**
4. **必须包含RTX 5080修复代码**

## 📝 备份日期
2025-09-21 22:00

---
**维护者**: Tiger
**状态**: ✅ 已验证稳定