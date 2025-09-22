# 🎯 RTX 50系列显卡深度学习环境 (sm_120架构)

## 📢 致RTX 50系列用户的福音

如果你正在为RTX 5080/5090等50系列显卡配置深度学习环境而苦恼，这个仓库将帮助你节省大量时间！

### 🎁 这个环境解决了什么问题？

RTX 50系列显卡使用全新的**sm_120架构**，目前主流PyTorch稳定版还不支持。很多人遇到：

**特别说明**: 在Ubuntu 22.04上配置RTX 50环境特别困难，经过无数次尝试才成功。升级到Ubuntu 24.04后，环境配置变得容易很多。强烈建议使用Ubuntu 24.04！

- ❌ `CUDA error: no kernel image is available for execution on the device`
- ❌ bitsandbytes量化训练失败
- ❌ PEFT/LoRA训练时dtype转换错误
- ❌ 模型加载极慢或失败
- ❌ 训练速度异常缓慢

**这个环境完美解决了所有这些问题！**

### ✅ 支持的显卡

- NVIDIA RTX 5080 (16GB) ✅ 已测试
- NVIDIA RTX 5080 Laptop ✅ 已测试
- NVIDIA RTX 5090 (理论支持)
- 所有sm_120架构的显卡

### 🚀 核心特性

1. **8bit量化训练** - 节省50%显存，16GB显卡可训练8B模型
2. **LoRA/PEFT微调** - 支持最新的参数高效微调
3. **稳定训练速度** - 3-4秒/步（batch_size=2）
4. **Qwen/LLaMA支持** - 支持主流大模型

### 📦 关键组件版本

```python
torch==2.10.0.dev20250921+cu128  # PyTorch nightly (必须!)
transformers==4.56.1
peft==0.17.1
bitsandbytes==0.47.0  # 8bit量化
datasets==4.1.1
accelerate==1.10.1
CUDA: 12.8
Python: 3.10.18
```

### 🔧 快速开始

#### 1. 克隆仓库
```bash
git clone https://github.com/tiger965/ttfm-training-env.git
cd ttfm-training-env
```

#### 2. 安装环境

**方法A: 使用conda (推荐)**
```bash
# 安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# 创建环境
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n pytorch python=3.10.18 -y
conda activate pytorch

# 安装PyTorch nightly (关键!)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装其他依赖
pip install -r requirements.txt
```

**方法B: 使用pip**
```bash
# 创建虚拟环境
python3.10 -m venv rtx50_env
source rtx50_env/bin/activate

# 安装PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装其他依赖
pip install -r requirements.txt
```

#### 3. 关键修复代码

在你的训练脚本开头添加：

```python
# RTX 50系列 PEFT修复
import peft.tuners.tuners_utils
peft.tuners.tuners_utils.BaseTuner._cast_adapter_dtype = lambda *args, **kwargs: None

# 如果使用Qwen3模型
from transformers.models.auto import configuration_auto, modeling_auto
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
configuration_auto.CONFIG_MAPPING._extra_content['qwen3'] = Qwen2Config
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['qwen3'] = 'Qwen2ForCausalLM'
```

#### 4. 环境变量
```bash
export BNB_CUDA_VERSION=128
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

#### 5. 验证环境
```bash
python test_environment.py
```

应该看到：
```
✅ PyTorch: 2.10.0.dev20250921+cu128
✅ CUDA可用: True
✅ GPU: NVIDIA GeForce RTX 50xx
✅ 显存: xx.xGB
```

### 📝 示例：8bit量化训练

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 8bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8"
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)
```

### ⚠️ 重要提醒

1. **必须使用PyTorch nightly版本** - 稳定版不支持sm_120
2. **不要随意升级包** - 可能破坏兼容性
3. **Python必须是3.10.x** - 其他版本可能有问题
4. **Windows用户** - 建议使用WSL2

### 🐛 常见问题

**Q: 为什么必须用nightly版本？**
A: RTX 50系列的sm_120架构只在PyTorch nightly中支持，稳定版要等到2025年中才会支持。

**Q: 可以用其他CUDA版本吗？**
A: 建议使用CUDA 12.8，其他版本未充分测试。

**Q: 训练速度慢怎么办？**
A: 检查batch_size是否为2，关闭tf32，使用fp16_opt_level="O2"。

**Q: bitsandbytes报错？**
A: 确保设置了`export BNB_CUDA_VERSION=128`。

### 🤝 贡献

欢迎提交Issue和PR！如果这个环境帮助了你，请点个Star ⭐

### 📜 许可

MIT License - 自由使用和修改

### 🙏 致谢

感谢所有RTX 50系列早期用户的测试和反馈。这个环境是在经历多次环境崩溃和重装后总结出的最佳实践。

---

**如果这个仓库帮你节省了时间，请分享给其他RTX 50系列用户！**

最后更新：2025-09-21
测试环境：Ubuntu 24.04 + RTX 5080 16GB + WSL2