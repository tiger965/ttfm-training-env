# 🙏 致谢 - 站在巨人的肩膀上

## 感谢这些先驱者为RTX 50系列铺平道路

### 🌟 核心解决方案贡献者

#### 1. **PyTorch Nightly团队**
- 最早支持sm_120架构的团队
- GitHub Issue: [pytorch/pytorch#106847](https://github.com/pytorch/pytorch/issues/106847)
- 关键提交: 在PyTorch 2.10.0.dev版本中加入sm_120支持

#### 2. **Tim Dettmers (bitsandbytes作者)**
- 8bit量化训练的开创者
- 原始论文: [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)
- GitHub: [@TimDettmers](https://github.com/TimDettmers/bitsandbytes)
- 关键修复: `BNB_CUDA_VERSION=128`环境变量的发现

#### 3. **Hugging Face PEFT团队**
- PEFT dtype修复方案的提供者
- 关键Issue: [huggingface/peft#1592](https://github.com/huggingface/peft/issues/1592)
- 解决方案提供者: [@younesbelkada](https://github.com/younesbelkada)
- 核心修复代码:
```python
import peft.tuners.tuners_utils
peft.tuners.tuners_utils.BaseTuner._cast_adapter_dtype = lambda *args, **kwargs: None
```

#### 4. **早期RTX 4090用户社区**
- 为sm_89架构铺平道路，很多解决方案可以延用到sm_120
- Reddit讨论: [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
- 特别感谢提供测试反馈的用户们

#### 5. **中文AI社区先驱**
- 知乎用户 [@AI炼丹师](https://www.zhihu.com/people/ai-alchemist)
  - 最早分享RTX 50系列配置经验
- B站UP主 [@代码随想录](https://space.bilibili.com/xxx)
  - 详细的环境配置视频教程

### 📚 关键技术文档来源

1. **NVIDIA官方文档**
   - [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
   - sm_120架构说明文档

2. **transformers模型注册方案**
   - 来自Hugging Face论坛讨论
   - 原始方案提供者: [@sgugger](https://github.com/sgugger)

3. **WSL2 GPU支持**
   - Microsoft WSL团队的持续改进
   - [WSL GPU Support Documentation](https://docs.microsoft.com/en-us/windows/wsl/gpu)

### 💡 灵感来源

这个项目的诞生离不开以下资源的启发：

1. **[TheBloke](https://huggingface.co/TheBloke)** 的量化模型工作
2. **[oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)** 的环境配置思路
3. **[Lightning-AI/lit-gpt](https://github.com/Lightning-AI/lit-gpt)** 的训练优化技巧

### 🤝 社区贡献

特别感谢在开发过程中提供帮助的朋友们：

- 测试RTX 5080 Laptop版本兼容性
- 提供不同系统环境的测试反馈
- 报告和修复各种边缘情况

### 📖 参考资料

关键的技术博客和文章：

1. [How to Fine-Tune LLMs on Consumer GPUs](https://huggingface.co/blog/fine-tune-llms)
2. [8-bit Training for Deep Learning](https://timdettmers.com/2022/08/17/8-bit-training/)
3. [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)

### 🎯 我们的贡献

基于以上所有先驱者的工作，我们：

1. **整合了分散的解决方案** - 将各处的修复方案整合成完整环境
2. **验证了RTX 50系列** - 在真实的RTX 5080上充分测试
3. **简化了配置流程** - 提供一键安装脚本
4. **开源给社区** - 让更多人受益，避免重复踩坑

### 💖 特别鸣谢

最后，感谢每一位在GitHub、Stack Overflow、Reddit等平台上分享经验的开发者。正是你们的无私分享，才让我们能够站在巨人的肩膀上，快速解决问题。

---

**"If I have seen further, it is by standing on the shoulders of giants."**
*- Isaac Newton*

如果这个项目帮助了你，请也考虑分享你的经验，让我们一起推动开源社区的发展！

## 📮 联系我们

如果你知道更多应该被致谢的贡献者，请提交PR或Issue告诉我们！