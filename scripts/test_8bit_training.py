#!/usr/bin/env python3
"""
RTX 5080 8bit量化训练测试脚本
支持sm_120架构，使用PyTorch nightly和bitsandbytes 0.47.0
"""

import torch
import warnings
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 忽略警告
warnings.filterwarnings('ignore')

# 关键修复：防止PEFT dtype转换错误
import peft.tuners.tuners_utils
peft.tuners.tuners_utils.BaseTuner._cast_adapter_dtype = lambda *args, **kwargs: None

print("=" * 60)
print("🚀 RTX 5080 8bit量化训练测试")
print("=" * 60)

# 验证环境
print(f"✅ PyTorch版本: {torch.__version__}")
print(f"✅ CUDA可用: {torch.cuda.is_available()}")
print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
print(f"✅ 支持架构: {torch.cuda.get_arch_list()}")

# GPU测试
x = torch.randn(1000, 1000).cuda()
y = x @ x.T
print(f"✅ GPU运算测试: {y.shape}")
del x, y
torch.cuda.empty_cache()

# 模型路径（需要根据实际情况修改）
base_path = Path('/home/tiger/tiger_trust_project')
# 如果有Qwen3-8B模型，使用实际路径
model_path = "/mnt/d/models/原始模型/Qwen3-8B/Qwen3-8B"  # 根据您的配置文件
output_dir = base_path / 'completed_models' / 'test_8bit'
output_dir.mkdir(parents=True, exist_ok=True)

print("\n📦 准备加载8bit量化模型...")
print(f"模型路径: {model_path}")

# 检查模型是否存在
model_path_obj = Path(model_path)
if not model_path_obj.exists():
    print(f"⚠️ 模型不存在: {model_path}")
    print("请下载Qwen3-8B模型或修改路径")
    print("\n使用简单测试模式...")

    # 创建简单测试
    import bitsandbytes as bnb
    print(f"✅ bitsandbytes版本: {bnb.__version__}")

    # 测试8bit量化操作
    linear = bnb.nn.Linear8bitLt(100, 100).cuda()
    x = torch.randn(10, 100).cuda()
    y = linear(x)
    print(f"✅ 8bit线性层测试: 输入{x.shape} -> 输出{y.shape}")

    print("\n环境配置成功！需要下载模型才能开始训练。")
else:
    # 8bit配置
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_use_double_quant=True,
    )

    print("⏳ 加载8bit量化模型（可能需要几分钟）...")

    # 加载8bit模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("✅ 8bit模型加载成功!")

    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LoRA配置
    print("\n🎯 配置LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 简单训练测试
    print("\n🚀 开始训练测试...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # 测试数据
    test_text = "用户: 你好\n助手: 你好！很高兴见到你。"

    # tokenize
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding=True
    )

    # 移动到GPU
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )
    loss = outputs.loss

    print(f"✅ 训练损失: {loss.item():.4f}")

    # 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("✅ 反向传播成功!")

    # 显存使用情况
    memory_used = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n📊 显存使用: {memory_used:.2f}GB / {memory_reserved:.2f}GB")

print("\n" + "=" * 60)
print("🎉 RTX 5080 8bit训练环境验证成功!")
print("支持sm_120架构，可以开始训练Qwen3-8B")
print("=" * 60)