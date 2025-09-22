#!/usr/bin/env python3
"""
测试训练环境是否正常
"""
import sys
import os

# 添加RTX 5080修复
import peft.tuners.tuners_utils
peft.tuners.tuners_utils.BaseTuner._cast_adapter_dtype = lambda *args, **kwargs: None

print("="*50)
print("训练环境检查")
print("="*50)

# 1. 检查Python版本
print(f"\n1. Python版本: {sys.version}")

# 2. 检查PyTorch
try:
    import torch
    print(f"\n2. PyTorch检查:")
    print(f"   版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
except Exception as e:
    print(f"   ❌ PyTorch错误: {e}")

# 3. 检查关键库
libraries = ['transformers', 'peft', 'bitsandbytes', 'datasets', 'accelerate']
print(f"\n3. 关键库检查:")
for lib in libraries:
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {lib}: {version}")
    except ImportError:
        print(f"   ❌ {lib}: 未安装")

# 4. 检查模型路径
print(f"\n4. 路径检查:")
model_path = "/mnt/d/models/原始模型/Qwen3-8B/Qwen3-8B"
save_path = "/mnt/d/models/留存模型/Tiger信任训练/24代系统"
sample_path = "/home/tiger/tiger_trust_project/scripts/样本生成器"

paths = [
    ("模型路径", model_path),
    ("保存路径", save_path),
    ("样本生成器", sample_path)
]

for name, path in paths:
    if os.path.exists(path):
        print(f"   ✅ {name}: {path}")
    else:
        print(f"   ❌ {name}: 不存在 - {path}")

# 5. 检查训练状态
print(f"\n5. 训练进度:")
state_file = f"{save_path}/training_state.json"
if os.path.exists(state_file):
    import json
    with open(state_file, 'r') as f:
        state = json.load(f)
    print(f"   当前代数: 第{state['generation']}代")
    print(f"   当前变体: 第{state['variant']}个")
    if state.get('best_scores'):
        print(f"   最佳分数: {state['best_scores']}")
else:
    print(f"   无训练记录")

# 6. 检查样本生成器
print(f"\n6. 样本生成器:")
sys.path.append('/home/tiger/tiger_trust_project/scripts/样本生成器')
try:
    from sample_generator import BalancedGenerationTrainingSystem
    print(f"   ✅ 样本生成器导入成功")
    generator = BalancedGenerationTrainingSystem()
    samples = generator.generate_generation_samples(1, 0)
    print(f"   ✅ 生成测试样本: {len(samples)}个")
except Exception as e:
    print(f"   ❌ 样本生成器错误: {e}")

print("\n" + "="*50)
print("✅ 环境检查完成！")
print("="*50)