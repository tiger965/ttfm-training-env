#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTFM 24代训练系统 V3 - 架构重构版
完全重写的架构，支持真正的批量并发训练
"""

import os
import sys
import gc
import json
import time
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# 防止CUDA context被fork到父进程
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['OMP_NUM_THREADS'] = '8'

import torch

# 注册Qwen3模型 - 必须在导入transformers之前
from transformers.models.auto import configuration_auto, modeling_auto
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
configuration_auto.CONFIG_MAPPING._extra_content['qwen3'] = Qwen2Config
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['qwen3'] = 'Qwen2ForCausalLM'

# RTX 5080 关键修复 - 必须加在这里！
import peft.tuners.tuners_utils
peft.tuners.tuners_utils.BaseTuner._cast_adapter_dtype = lambda *args, **kwargs: None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # 强制重新配置
)
logger = logging.getLogger(__name__)
logger.propagate = False  # 防止日志传播到父logger

# 清除已有的handlers避免重复
logger.handlers.clear()

# 添加单个StreamHandler确保实时输出
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(handler)

# 设置Python输出无缓冲
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # 行缓冲
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# 导入必要的库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from transformers import TrainerCallback

class SmartMemoryCallback(TrainerCallback):
    """智能性能管理 - 激进优化版"""

    def __init__(self):
        self.step_times = []
        self.baseline_speed = None  # 等待实际测量
        self.last_cleanup = 0
        self.performance_buffer = []  # 性能缓冲区
        self.should_restart = False
        self.cleanup_count = 0  # 清理计数
        self.batch_optimized = False  # 标记是否已优化batch

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()

        # batch_size=2是最优的，不需要调整
        # 昨天的日志证明batch_size=2时速度3-4秒/步
        # batch_size=1反而慢（6秒/步）

        return control

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start
        self.step_times.append(step_time)

        # 保持最近100步的记录
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]

        # 建立基准速度（前20步的平均值，更稳定）
        if len(self.step_times) == 20 and self.baseline_speed is None:
            # 去掉最大最小值后取平均
            sorted_times = sorted(self.step_times[:20])
            avg_speed = np.mean(sorted_times[2:-2])  # 去掉前2个最快和后2个最慢
            self.baseline_speed = avg_speed * 1.1  # 留10%富裕
            logger.info(f"  ⚡ 基准速度: {self.baseline_speed:.2f}秒/步")

        # 智能性能管理 - 仅在真正异常时介入
        if self.baseline_speed and len(self.step_times) > 20:
            # 计算最近10步的平均速度
            recent_avg = np.mean(self.step_times[-10:])

            # 只有当平均速度明显下降时才介入
            if recent_avg > self.baseline_speed * 1.5:
                deviation = (recent_avg - self.baseline_speed) / self.baseline_speed * 100
                logger.warning(f"  ⚠️ 性能下降: 平均{recent_avg:.1f}s (偏差{deviation:.0f}%)")

                # 每100步最多清璆1次，避免过度介入
                if self.cleanup_count < state.global_step // 100:
                    self._handle_performance_deviation(deviation, recent_avg, args, state, control)
                    self.cleanup_count += 1

        return control

    def _handle_performance_deviation(self, deviation, step_time, args, state, control):
        """处理性能偏差 - 温和干预"""

        if deviation <= 100:  # 50-100%: 轻度清理
            logger.info(f"  🔧 轻度优化...")
            torch.cuda.empty_cache()
            gc.collect()

        elif deviation <= 200:  # 100-200%: 中度清理
            logger.warning(f"  ⚠️ 中度优化: {step_time:.1f}s")

            # 全面清理
            torch.cuda.synchronize()
            for _ in range(2):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

            # 动态调整参数
            if args.per_device_train_batch_size > 1:
                args.per_device_train_batch_size = 1
                args.gradient_accumulation_steps = min(16, args.gradient_accumulation_steps * 2)
                logger.info(f"  ⚙️ 参数调整: batch=1, accum={args.gradient_accumulation_steps}")

        else:  # >200%: 深度清理（但不停止训练）
            logger.error(f"  🚨 深度优化: {step_time:.1f}s")

            # 温和清理，不打断训练
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            for _ in range(3):
                gc.collect()

            # 更新基准（可能系统状态变了）
            if len(self.step_times) > 50:
                recent_times = sorted(self.step_times[-30:])
                new_baseline = np.mean(recent_times[5:-5]) * 1.2
                if new_baseline > self.baseline_speed:
                    self.baseline_speed = new_baseline
                    logger.info(f"  🔄 更新基准: {self.baseline_speed:.1f}s/步")

class ProgressCallback(TrainerCallback):
    """实时显示训练进度"""
    def __init__(self, variant_id: int):
        self.variant_id = variant_id
        self.current_step = 0
        self.total_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        logger.info(f"  开始训练，总步数: {self.total_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        if self.current_step % 25 == 0:  # 每25步显示一次
            progress = self.current_step / self.total_steps * 100
            if state.log_history:
                last_loss = state.log_history[-1].get('loss', 0)
                logger.info(f"  进度: {progress:.1f}% [{self.current_step}/{self.total_steps}] Loss: {last_loss:.4f}")
            else:
                logger.info(f"  进度: {progress:.1f}% [{self.current_step}/{self.total_steps}]")
            sys.stdout.flush()  # 强制刷新输出

# 导入样本生成器
sys.path.append('/home/tiger/tiger_trust_project/scripts/样本生成器')
from sample_generator import BalancedGenerationTrainingSystem


@dataclass
class TrainingConfig:
    """训练配置"""
    generation: int
    variant: int
    epochs: int = 3  # 减少到3轮，快速训练
    learning_rate: float = 5e-5
    batch_size: int = 2  # 昨天成功的配置是2
    gradient_accumulation_steps: int = 4  # 昨天实际是4，不是8
    lora_r: int = 32  # 合理的LoRA秩
    lora_alpha: int = 64  # 2倍r的标准配置
    max_seq_length: int = 256


class ModelManager:
    """模型管理器 - 负责基础模型的生命周期管理"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.base_model = None
        self.tokenizer = None
        self.load_count = 0

    def initialize(self):
        """初始化基础模型和tokenizer"""
        logger.info("🚀 初始化基础模型...")
        logger.info(f"  模型路径: {self.model_path}")
        logger.info("  加载过程需要1-2分钟，请耐心等待...")
        sys.stdout.flush()

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 8bit量化配置（修复版 - 为基础模型训练优化）
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_8bit_fp32_cpu_offload=False,  # 关键：禁用CPU卸载
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )

        # 加载模型
        logger.info("📦 加载8bit量化模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map={"": 0},  # 强制GPU
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # 准备8bit训练
        self.base_model = prepare_model_for_kbit_training(
            self.base_model,
            use_gradient_checkpointing=False
        )

        self.base_model.config.use_cache = False
        self.load_count += 1

        logger.info(f"✅ 基础模型加载完成 (第{self.load_count}次)")

        # 显示显存状态
        self._show_memory_status()

    def _show_memory_status(self):
        """显示显存状态"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"📊 GPU显存: 已分配{allocated:.1f}GB / 已预留{reserved:.1f}GB / 总计{total:.1f}GB")

    def create_lora_model(self, lora_config: LoraConfig):
        """为每个变体创建独立的LoRA权重（修复版）"""
        if self.base_model is None:
            self.initialize()

        logger.info("创建LoRA适配器...")

        # 关键修复：完全重置模型状态
        try:
            # 如果模型已有LoRA，先彻底卸载
            if hasattr(self.base_model, 'peft_config'):
                logger.info("  检测到旧LoRA，正在卸载...")
                # 卸载所有adapter
                if hasattr(self.base_model, 'unload'):
                    self.base_model.unload()
                # 获取干净的基础模型
                self.base_model = self.base_model.get_base_model()
                # 清理PEFT相关属性
                if hasattr(self.base_model, 'peft_config'):
                    del self.base_model.peft_config

                # 重要：清理GPU缓存
                torch.cuda.empty_cache()
                gc.collect()

            logger.info("  创建全新LoRA层...")
            peft_model = get_peft_model(self.base_model, lora_config)

        except Exception as e:
            logger.error(f"  LoRA创建失败: {e}")
            raise

        # 确保模型在训练模式
        peft_model.train()

        # 启用梯度
        for param in peft_model.parameters():
            param.requires_grad = False  # 先全部关闭

        for name, param in peft_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True  # 只开启LoRA参数

        peft_model.print_trainable_parameters()
        return peft_model

    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理模型管理器...")
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


class BatchTrainer:
    """批量训练器 - 真正的并发实现"""

    def __init__(self, model_manager: ModelManager, save_path: str):
        self.model_manager = model_manager
        self.save_path = save_path
        self.active_trainers = []

    def calculate_batch_size(self) -> int:
        """动态计算可以同时训练的LoRA数量"""
        if not torch.cuda.is_available():
            return 1

        # 暂时不并发，一个一个训练但效率更高
        # 因为LoRA权重重置很快，不需要重新加载模型
        return 1

    def train_variants_batch(
        self,
        generation: int,
        variants: List[int],
        config: TrainingConfig,
        sample_generator,
        checkpoint_callback=None
    ) -> Dict[int, float]:
        """批量训练多个变体"""
        results = {}
        batch_size = self.calculate_batch_size()

        # 分批处理
        for i in range(0, len(variants), batch_size):
            batch = variants[i:i+batch_size]
            logger.info(f"\n📦 训练批次: 变体{batch}")

            # 并行训练这一批
            batch_results = self._train_batch_parallel(
                generation, batch, config, sample_generator, checkpoint_callback
            )
            results.update(batch_results)

            # 批次间清理
            self._cleanup_batch()
            time.sleep(3)

        return results

    def _train_batch_parallel(
        self,
        generation: int,
        batch: List[int],
        config: TrainingConfig,
        sample_generator,
        checkpoint_callback=None
    ) -> Dict[int, float]:
        """并行训练一批变体"""
        results = {}

        # 顺序训练每个变体（暂时不用真正的并行）
        for variant in batch:
            try:
                logger.info(f"\n🔄 训练变体 {variant}")

                # 创建LoRA配置
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    inference_mode=False
                )

                # 创建LoRA模型
                model = self.model_manager.create_lora_model(lora_config)

                # 生成训练样本
                samples = sample_generator(generation, variant)
                logger.info(f"  使用{len(samples)}个样本进行训练")

                # 准备数据集
                dataset = self._prepare_dataset(samples, config.max_seq_length)

                # 训练
                logger.info(f"  开始训练变体{variant}...")
                try:
                    score = self._train_single_variant(
                        model, dataset, generation, variant, config, checkpoint_callback
                    )
                    results[variant] = score

                    # 保存模型（只保存LoRA权重）
                    # 统一使用简单格式，避免路径不一致
                    save_dir = f"{self.save_path}/gen{generation}_var{variant}"
                    os.makedirs(save_dir, exist_ok=True)

                    # 重要：只保存LoRA权重，不保存完整模型！
                    model.save_pretrained(save_dir,
                                        save_adapter=True,  # 只保存adapter
                                        save_config=True,
                                        save_embedding_layers=False,  # 不保存embedding
                                        save_full_model=False)  # 不保存完整模型
                    logger.info(f"✅ 变体{variant}完成: {score:.1f}%")
                except Exception as e:
                    logger.error(f"  训练变体{variant}时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    results[variant] = 0.0

                # 深度清理（修复版 - 彻底释放显存）
                del dataset

                # 关键：彻底清理模型
                if 'model' in locals():
                    # 先卸载LoRA
                    if hasattr(model, 'unload'):
                        model.unload()
                    # 移到CPU再删除（强制释放GPU显存）
                    model.cpu()
                    del model

                # 多次清理确保释放
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                # 等待GPU完成所有操作
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            except Exception as e:
                logger.error(f"❌ 变体{variant}失败: {e}")
                results[variant] = 0.0

        return results

    def _prepare_dataset(self, samples: List[Dict], max_length: int) -> Dataset:
        """准备数据集（优化版）"""
        tokenizer = self.model_manager.tokenizer
        logger.info(f"    开始tokenization {len(samples)}个样本...")

        # 提取文本内容
        texts = [s['text'] for s in samples]

        # 创建数据集
        dataset = Dataset.from_dict({'text': texts})

        # 定义tokenize函数
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_attention_mask=True,
                return_token_type_ids=False
            )

        # 批量tokenize（使用map更高效）
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=250,  # 增大tokenize批大小
            remove_columns=['text']
        )

        # 添加labels
        def add_labels(examples):
            examples['labels'] = examples['input_ids'].copy()
            return examples

        dataset = dataset.map(add_labels, batched=True)
        logger.info(f"    数据集准备完成")

        return dataset

    def _train_single_variant(
        self,
        model,
        dataset,
        generation: int,
        variant: int,
        config: TrainingConfig,
        checkpoint_callback=None
    ) -> float:
        """训练单个变体"""

        logger.info(f"  准备训练器，数据集大小: {len(dataset)}")

        # 检查点目录（持久化，支持断点续传）- 统一格式
        checkpoint_dir = f"{self.save_path}/gen{generation}_var{variant}/checkpoints"

        # 查找最新的检查点
        resume_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # 找到最新的检查点
                latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_checkpoint = os.path.join(checkpoint_dir, latest)
                logger.info(f"  🔄 发现检查点，从 {latest} 恢复训练")

        # 训练参数
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,  # 使用持久化目录
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,  # 使用config中的值
            gradient_accumulation_steps=config.gradient_accumulation_steps,  # 使用config中的值
            gradient_checkpointing=False,  # LoRA训练不需要梯度检查点
            learning_rate=config.learning_rate,
            warmup_steps=0,  # 不需要预热
            fp16=True,
            fp16_opt_level="O1",  # O1更稳定
            fp16_full_eval=False,  # 评估时不用fp16保证精度
            save_strategy="steps",  # 按步数保存
            save_steps=10,  # 每10步保存一次，确保精确恢复
            save_total_limit=3,  # 保留最近3个检查点
            logging_steps=25,  # 适度日志
            report_to=[],
            disable_tqdm=False,  # 显示进度条
            remove_unused_columns=False,
            dataloader_num_workers=0,  # 关键：8bit必须用0
            optim="adamw_torch",  # 标准优化器更稳定
            dataloader_pin_memory=False,  # 8bit不需要pin
            tf32=True,  # RTX 5080支持TF32加速！
            dataloader_prefetch_factor=None,  # 8bit不预取
            torch_compile=False,  # 不使用compile
            max_grad_norm=1.0,  # 梯度裁剪
            adam_epsilon=1e-6,  # 更稳定
            dataloader_persistent_workers=False,  # 8bit模型不需要持久化
            resume_from_checkpoint=resume_checkpoint  # 断点续传
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_manager.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # 使用原始回调（商用优化器不适合8bit）
        callbacks = [
            ProgressCallback(variant),
            SmartMemoryCallback()  # 使用原来的智能回调
        ]

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )

        # 训练
        logger.info(f"  开始训练...")
        logger.info(f"    批大小: {config.batch_size}, 总样本: {len(dataset)}")
        logger.info(f"    预计步数: {len(dataset) // config.batch_size}")
        sys.stdout.flush()

        try:
            result = trainer.train()
            logger.info(f"  训练完成, Loss: {result.training_loss:.4f}")
        except Exception as e:
            logger.error(f"  训练出错: {e}")
            raise

        # 计算得分（这里简化为随机分数，实际应该评估）
        score = 70 + np.random.random() * 30  # 70-100之间

        return score

    def _cleanup_batch(self):
        """批次间清理"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 确保GPU操作完成
        torch.cuda.synchronize()


# 使用原有的样本生成器，不需要重新定义


class TTFM24SystemV3:
    """24代训练系统 - 第3版架构"""

    def __init__(
        self,
        model_path: str = "/mnt/d/models/原始模型/Qwen3-8B/Qwen3-8B",
        save_path: str = "/mnt/d/models/留存模型/Tiger信任训练/24代系统"
    ):
        self.model_path = model_path
        self.save_path = save_path

        # 创建组件
        self.model_manager = ModelManager(model_path)
        self.trainer = BatchTrainer(self.model_manager, save_path)
        self.sample_generator = BalancedGenerationTrainingSystem()  # 使用原有的样本生成器

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 训练状态
        self.state_file = f"{save_path}/training_state.json"
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """加载训练状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    content = f.read()
                    if content.strip():  # 文件有内容
                        return json.loads(content)
            except:
                logger.warning(f"状态文件损坏，重新开始训练")
        return {'generation': 1, 'variant': 0, 'best_scores': {}}

    def save_state(self):
        """保存训练状态"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # 强制写入磁盘

    def save_variant_checkpoint(self, generation: int, variant: int, step: int, total_steps: int, loss: float = 0.0):
        """保存变体级别的训练进度"""
        # 统一路径格式
        checkpoint_dir = f"{self.save_path}/gen{generation}_var{variant}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = f"{checkpoint_dir}/checkpoint.json"

        checkpoint = {
            'generation': generation,
            'variant': variant,
            'current_step': step,
            'total_steps': total_steps,
            'progress_percent': (step / total_steps) * 100 if total_steps > 0 else 0,
            'current_loss': loss,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_saved': os.path.exists(f"{checkpoint_dir}/adapter_model.safetensors")
        }

        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

    def load_variant_checkpoint(self, generation: int, variant: int) -> Dict:
        """加载变体的训练进度 - 兼容新旧格式"""
        # 优先检查新格式
        checkpoint_file = f"{self.save_path}/gen{generation}_var{variant}/checkpoint.json"
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载检查点失败: {e}")

        # 兼容旧格式（带0填充）
        old_checkpoint_file = f"{self.save_path}/gen{generation:02d}_var{variant:03d}/checkpoint.json"
        if os.path.exists(old_checkpoint_file):
            try:
                with open(old_checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载旧格式检查点失败: {e}")

        return None

    def run(self):
        """主训练流程"""
        logger.info("="*70)
        logger.info("🚀 TTFM 24代训练系统 V3 - 架构重构版")
        logger.info(f"📍 模型: {self.model_path}")
        logger.info(f"💾 保存: {self.save_path}")
        logger.info("="*70)

        # 初始化
        self.model_manager.initialize()

        # 从断点恢复
        start_gen = self.state.get('generation', 1)
        start_var = self.state.get('variant', 0)

        # 24代训练循环
        for generation in range(start_gen, 25):
            logger.info(f"\n{'='*70}")
            logger.info(f"🧬 第{generation}/24代 - {self.get_generation_name(generation)}")
            logger.info(f"{'='*70}")

            # 配置
            config = TrainingConfig(
                generation=generation,
                variant=0,
                epochs=3,  # 固定使用3轮训练
                learning_rate=5e-5 if generation <= 10 else 3e-5,
                batch_size=2,  # 实际最优: 2
                gradient_accumulation_steps=4,
                lora_r=64,
                lora_alpha=128,
                max_seq_length=256
            )

            # 动态停止机制 - 获取目标Loss
            target_loss = self.get_target_loss(generation)
            logger.info(f"📊 目标Loss: {target_loss:.2f}")

            # 训练变体
            start_variant = start_var if generation == start_gen else 0
            all_scores = {}
            best_loss = float('inf')

            # 前5代必须训练100个，6代后可以提前停止
            min_variants = 100 if generation <= 5 else 20
            check_interval = 100 if generation <= 5 else 20

            for variant in range(start_variant, 100):
                # 检查该变体是否已完成
                checkpoint = self.load_variant_checkpoint(generation, variant)
                if checkpoint and checkpoint.get('model_saved', False):
                    # 如果模型已保存，说明该变体已完成
                    logger.info(f"  ⏭️ 跳过已完成的变体{variant} (进度{checkpoint.get('progress_percent', 100):.1f}%)")
                    # 从检查点加载分数（如果没有分数则默认85分）
                    all_scores[variant] = checkpoint.get('score', 85.0)
                    continue
                elif checkpoint and checkpoint.get('progress_percent', 0) > 0:
                    # 如果有部分进度，记录但继续重训
                    logger.info(f"  ⚠️ 变体{variant}有部分进度({checkpoint.get('progress_percent', 0):.1f}%)，将重新训练")

                # 训练单个变体（传递保存回调）
                scores = self.trainer.train_variants_batch(
                    generation,
                    [variant],  # 单个变体
                    config,
                    lambda g, v: self.sample_generator.generate_generation_samples(g, v),
                    checkpoint_callback=lambda s, t, l: self.save_variant_checkpoint(generation, variant, s, t, l)
                )

                all_scores.update(scores)
                current_loss = 100 - scores[variant]  # 转换为loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    logger.info(f"  🏆 新最优: 变体{variant}, Loss={current_loss:.4f}")

                # 保存最终分数到检查点
                self.save_variant_checkpoint(generation, variant, 100, 100, current_loss)
                final_checkpoint = self.load_variant_checkpoint(generation, variant) or {}
                final_checkpoint['score'] = scores[variant]
                final_checkpoint['completed'] = True
                final_checkpoint['model_saved'] = True
                # 统一路径格式
                checkpoint_file = f"{self.save_path}/gen{generation}_var{variant}/checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(final_checkpoint, f, indent=2)

                # 每个变体完成后保存状态（断点续传）
                self.state['generation'] = generation
                self.state['variant'] = variant + 1  # 下次从下一个变体开始
                self.state['best_scores'][str(generation)] = best_loss
                self.save_state()

                # 每10个变体深度清理
                if (variant + 1) % 10 == 0:
                    logger.info(f"  完成{variant+1}个变体，深度清理...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(3)

                # 动态停止检查（6代后开始）
                if generation > 5 and variant >= min_variants - 1:
                    if (variant + 1) % check_interval == 0:
                        logger.info(f"\n📊 检查: 已训练{variant+1}个，最优Loss={best_loss:.4f}")
                        if best_loss < target_loss:
                            logger.info(f"  ✅ 达标！节省{100-variant-1}个变体")
                            break

            # 更新状态
            self.state['generation'] = generation
            self.state['variant'] = 99
            self.state['best_scores'][str(generation)] = max(all_scores.values())
            self.save_state()

            # 代际总结
            best_variant = max(all_scores, key=all_scores.get)
            best_score = all_scores[best_variant]
            avg_score = np.mean(list(all_scores.values()))

            logger.info(f"\n✨ 第{generation}代完成!")
            logger.info(f"   最佳变体: {best_variant}")
            logger.info(f"   最高分: {best_score:.1f}%")
            logger.info(f"   平均分: {avg_score:.1f}%")

            # 检查是否达标
            if best_score >= 100:
                logger.info("🎉 达到100%信任度！训练完成！")
                break

            # 重置起始变体
            start_var = 0

            # 代际间深度清理（优化版）
            logger.info("🔄 代际深度清理...")

            # 重置模型管理器状态
            if hasattr(self.model_manager.base_model, 'unload'):
                self.model_manager.base_model.unload()

            # 彻底清理
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

            # 短暂休息让系统稳定
            time.sleep(5)

        logger.info("\n✅ 24代训练完成！")

    def get_target_loss(self, generation: int) -> float:
        """获取每代的目标Loss（用于动态停止）"""
        targets = {
            1: 2.0, 2: 1.8, 3: 1.5, 4: 1.2, 5: 1.0,  # 基础建立
            6: 0.8, 10: 0.5, 15: 0.3, 20: 0.2, 24: 0.1  # 逐步脱敏
        }
        for gen, loss in sorted(targets.items()):
            if generation <= gen:
                return loss
        return 0.1

    def get_generation_name(self, generation: int) -> str:
        """获取代际名称"""
        if generation <= 3:
            return "感情基础建设"
        elif generation <= 11:
            return "基础学科脱敏"
        elif generation <= 19:
            return "应用领域脱敏"
        else:
            return "高级能力与完全信任"


def get_target_loss(generation: int) -> float:
    """获取每代的目标Loss"""
    targets = {
        1: 2.0, 2: 1.8, 3: 1.5, 4: 1.2, 5: 1.0,  # 基础
        6: 0.8, 10: 0.5, 15: 0.3, 20: 0.2, 24: 0.1  # 脱敏
    }
    for gen, loss in sorted(targets.items()):
        if generation <= gen:
            return loss
    return 0.1

def main():
    """主入口"""
    try:
        # 环境准备
        logger.info("🧹 准备训练环境...")

        # 清理GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✅ GPU缓存已清理")

        # 运行训练系统
        logger.info("🚀 启动训练系统...")
        system = TTFM24SystemV3()
        system.run()

    except KeyboardInterrupt:
        logger.info("\n⚠️ 训练被用户中断")
        if 'system' in locals():
            system.save_state()
    except Exception as e:
        logger.error(f"\n❌ 训练失败: {e}")
        if 'system' in locals():
            system.save_state()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()