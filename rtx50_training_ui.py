#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTX 50系列训练可视化界面
一个友好的Web界面，让AI训练变得简单
"""

import gradio as gr
import torch
import os
import json
import psutil
import subprocess
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from threading import Thread
import time

class RTX50TrainingUI:
    """RTX 50训练界面"""

    def __init__(self):
        self.training_active = False
        self.training_logs = []
        self.gpu_history = []
        self.current_loss = []

    def get_system_info(self):
        """获取系统信息"""
        info = {}

        # GPU信息
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            info['gpu_memory_used'] = f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB"
            info['pytorch_version'] = torch.__version__
            info['cuda_version'] = torch.version.cuda

            # 检查是否RTX 50系列
            if "50" in info['gpu_name']:
                info['rtx50_detected'] = "✅ RTX 50系列显卡"
            else:
                info['rtx50_detected'] = f"⚠️ 非RTX 50系列 ({info['gpu_name']})"
        else:
            info['gpu_available'] = False
            info['error'] = "未检测到GPU"

        # CPU和内存信息
        info['cpu_percent'] = f"{psutil.cpu_percent()}%"
        info['memory_percent'] = f"{psutil.virtual_memory().percent}%"

        return info

    def format_system_info(self, info):
        """格式化系统信息显示"""
        if not info.get('gpu_available'):
            return "❌ GPU不可用\n请检查NVIDIA驱动和CUDA安装"

        return f"""
### 🖥️ 系统信息
- **GPU型号**: {info['gpu_name']}
- **检测结果**: {info['rtx50_detected']}
- **显存容量**: {info['gpu_memory_total']}
- **已用显存**: {info['gpu_memory_used']}
- **PyTorch版本**: {info['pytorch_version']}
- **CUDA版本**: {info['cuda_version']}
- **CPU使用率**: {info['cpu_percent']}
- **内存使用率**: {info['memory_percent']}
"""

    def start_training(self, model_type, batch_size, learning_rate, epochs):
        """开始训练"""
        self.training_active = True
        self.training_logs = []
        self.current_loss = []

        # 这里是示例训练过程
        log_text = f"""
🚀 开始训练
- 模型类型: {model_type}
- 批次大小: {batch_size}
- 学习率: {learning_rate}
- 训练轮数: {epochs}
"""
        self.training_logs.append(log_text)

        # 模拟训练过程
        for epoch in range(int(epochs)):
            for step in range(10):  # 模拟10个步骤
                loss = 2.0 - (epoch * 0.1) - (step * 0.01) + (0.1 * torch.randn(1).item())
                self.current_loss.append(loss)

                log = f"Epoch {epoch+1}/{epochs} | Step {step+1}/10 | Loss: {loss:.4f}"
                self.training_logs.append(log)
                time.sleep(0.5)  # 模拟训练时间

                if not self.training_active:
                    self.training_logs.append("⚠️ 训练已停止")
                    return "训练已停止"

        self.training_active = False
        return "✅ 训练完成！"

    def stop_training(self):
        """停止训练"""
        self.training_active = False
        return "正在停止训练..."

    def get_training_logs(self):
        """获取训练日志"""
        if not self.training_logs:
            return "暂无训练日志"
        return "\n".join(self.training_logs[-20:])  # 显示最后20行

    def plot_loss_curve(self):
        """绘制损失曲线"""
        if not self.current_loss:
            # 返回空图
            fig = go.Figure()
            fig.add_annotation(
                text="暂无训练数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title="训练损失曲线",
                xaxis_title="步骤",
                yaxis_title="Loss",
                height=400
            )
            return fig

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(self.current_loss))),
            y=self.current_loss,
            mode='lines',
            name='Loss',
            line=dict(color='rgb(0, 123, 255)', width=2)
        ))

        fig.update_layout(
            title="训练损失曲线",
            xaxis_title="步骤",
            yaxis_title="Loss",
            height=400,
            showlegend=True
        )

        return fig

    def test_8bit_training(self):
        """测试8bit训练"""
        try:
            import bitsandbytes as bnb

            # 创建测试张量
            test_tensor = torch.randn(100, 100).cuda()

            # 测试8bit操作
            linear = bnb.nn.Linear8bitLt(100, 100).cuda()
            output = linear(test_tensor)

            return """✅ 8bit训练测试成功！

bitsandbytes正常工作
可以使用8bit量化训练节省显存
建议batch_size=2, gradient_accumulation=4"""
        except Exception as e:
            return f"""❌ 8bit训练测试失败

错误信息: {str(e)}

解决方案:
1. 确保设置环境变量: export BNB_CUDA_VERSION=128
2. 重新安装bitsandbytes: pip install bitsandbytes==0.47.0
3. 确保使用PyTorch nightly版本"""

    def create_interface(self):
        """创建Gradio界面"""

        with gr.Blocks(title="RTX 50 AI训练平台", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🚀 RTX 50系列 AI训练平台
            ### 让深度学习训练变得简单 - 专为RTX 5080/5090优化
            """)

            with gr.Tabs():
                # 系统信息标签
                with gr.TabItem("📊 系统监控"):
                    with gr.Row():
                        with gr.Column():
                            check_btn = gr.Button("🔍 检查系统", variant="primary")
                            system_info = gr.Markdown("点击按钮检查系统状态")

                        with gr.Column():
                            test_8bit_btn = gr.Button("🧪 测试8bit训练")
                            test_result = gr.Textbox(label="测试结果", lines=8)

                    check_btn.click(
                        lambda: self.format_system_info(self.get_system_info()),
                        outputs=system_info
                    )
                    test_8bit_btn.click(self.test_8bit_training, outputs=test_result)

                # 训练控制标签
                with gr.TabItem("🎯 训练控制"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 训练参数")
                            model_type = gr.Dropdown(
                                choices=["GPT-2", "LLaMA-7B", "Qwen-8B", "自定义"],
                                value="GPT-2",
                                label="模型类型"
                            )
                            batch_size = gr.Slider(1, 16, value=2, step=1, label="批次大小")
                            learning_rate = gr.Number(value=5e-5, label="学习率")
                            epochs = gr.Slider(1, 10, value=3, step=1, label="训练轮数")

                            with gr.Row():
                                start_btn = gr.Button("▶️ 开始训练", variant="primary")
                                stop_btn = gr.Button("⏸️ 停止训练", variant="stop")

                            status = gr.Textbox(label="训练状态", value="待命中...")

                        with gr.Column(scale=2):
                            gr.Markdown("### 训练日志")
                            logs = gr.Textbox(label="实时日志", lines=15, max_lines=20)
                            refresh_btn = gr.Button("🔄 刷新日志")

                    # 损失曲线
                    with gr.Row():
                        loss_plot = gr.Plot(label="损失曲线")
                        update_plot_btn = gr.Button("📈 更新图表")

                    # 绑定事件
                    start_btn.click(
                        self.start_training,
                        inputs=[model_type, batch_size, learning_rate, epochs],
                        outputs=status
                    )
                    stop_btn.click(self.stop_training, outputs=status)
                    refresh_btn.click(self.get_training_logs, outputs=logs)
                    update_plot_btn.click(self.plot_loss_curve, outputs=loss_plot)

                # 快速指南标签
                with gr.TabItem("📚 使用指南"):
                    gr.Markdown("""
                    ## 快速开始指南

                    ### 1️⃣ 环境检查
                    - 点击"系统监控"查看GPU状态
                    - 确认显示RTX 50系列显卡
                    - 测试8bit训练是否正常

                    ### 2️⃣ 开始训练
                    - 选择模型类型
                    - 设置训练参数（推荐batch_size=2）
                    - 点击"开始训练"

                    ### 3️⃣ 监控进度
                    - 查看实时日志
                    - 观察损失曲线
                    - 必要时停止训练

                    ### 💡 性能建议
                    - **RTX 5080 (16GB)**:
                        - 8B模型: batch_size=2, gradient_accumulation=4
                        - 7B模型: batch_size=4, gradient_accumulation=2

                    - **训练速度优化**:
                        - 使用fp16混合精度
                        - 关闭tf32 (可能变慢)
                        - save_steps设置为50+

                    ### ⚠️ 注意事项
                    - 必须使用PyTorch nightly版本
                    - 设置环境变量BNB_CUDA_VERSION=128
                    - 不要随意升级包版本

                    ### 🔗 相关资源
                    - [GitHub仓库](https://github.com/tiger965/ttfm-training-env)
                    - [问题反馈](https://github.com/tiger965/ttfm-training-env/issues)
                    - [使用文档](https://github.com/tiger965/ttfm-training-env/wiki)
                    """)

                # 关于标签
                with gr.TabItem("ℹ️ 关于"):
                    gr.Markdown("""
                    ## 关于RTX 50训练平台

                    这是一个专为RTX 50系列显卡优化的AI训练界面。

                    ### 🎯 特点
                    - 零配置，开箱即用
                    - 支持8bit量化训练
                    - 实时监控和可视化
                    - 小白友好的界面

                    ### 👨‍💻 作者
                    Tiger & 开源社区贡献者

                    ### 📜 许可
                    MIT License - 自由使用

                    ### 🙏 致谢
                    感谢所有为RTX 50系列适配做出贡献的开发者！

                    ---

                    **如果这个工具帮助了你，请给我们一个⭐Star！**
                    """)

            gr.Markdown("""
            ---
            <center>Made with ❤️ for RTX 50 Series Users | <a href="https://github.com/tiger965/ttfm-training-env">GitHub</a></center>
            """)

        return interface

# 主程序
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════╗
    ║     RTX 50系列 AI训练可视化平台              ║
    ║     专为深度学习新手设计                      ║
    ╚══════════════════════════════════════════════╝
    """)

    # 设置环境变量
    os.environ['BNB_CUDA_VERSION'] = '128'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 创建UI实例
    ui = RTX50TrainingUI()
    interface = ui.create_interface()

    print("🚀 启动Web界面...")
    print("📍 本地访问: http://localhost:7860")
    print("📍 公网分享: 设置share=True")
    print("\n按Ctrl+C停止服务\n")

    # 启动界面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设置为True可以获取公网链接
        inbrowser=True  # 自动打开浏览器
    )