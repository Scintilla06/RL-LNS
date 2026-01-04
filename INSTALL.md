# 环境安装指南

## 方式 1：使用 conda environment.yaml（推荐）

```bash
# 创建环境
conda env create -f environment.yaml

# 激活环境
conda activate rl-lns
```

## 方式 2：手动创建

```bash
# 1. 创建 conda 环境
conda create -n rl-lns python=3.10 -y
conda activate rl-lns

# 2. 安装 PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. 安装 PyTorch Geometric
conda install pyg pytorch-scatter pytorch-sparse -c pyg

# 4. 安装其他依赖
pip install -r requirements.txt
```

## 额外组件安装

### Gurobi（必需，用于 MILP 求解）

```bash
# 方式 A：通过 conda
conda install -c gurobi gurobi

# 方式 B：通过 pip
pip install gurobipy

# 激活 license（需要学术或商业 license）
# 访问 https://www.gurobi.com/downloads/ 获取 license
grbgetkey YOUR-LICENSE-KEY
```

### Flash Attention 2（可选，加速注意力计算）

```bash
# 需要 CUDA 11.6+ 和合适的 GPU
pip install flash-attn --no-build-isolation
```

### Unsloth（可选，2x 训练加速）

```bash
# 安装 unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## 验证安装

```bash
# 运行验证脚本
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import torch_geometric
print(f'PyG: {torch_geometric.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

import bitsandbytes
print(f'bitsandbytes: {bitsandbytes.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

try:
    import gurobipy
    print(f'Gurobi: {gurobipy.gurobi.version()}')
except:
    print('Gurobi: Not installed')

print('\\n✅ All core dependencies installed!')
"
```

## 下载模型

```bash
# 方式 A：自动下载（首次运行时）
# 代码会自动从 HuggingFace 下载到 ~/.cache/huggingface/

# 方式 B：手动下载到本地
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# 方式 C：使用 modelscope（国内更快）
pip install modelscope
modelscope download --model qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct
```

如果手动下载，修改 `configs/model.yaml` 中的模型路径：
```yaml
model:
  name: "./models/Qwen2.5-7B-Instruct"  # 本地路径
```

## 常见问题

### Q: bitsandbytes 安装失败
```bash
# Linux
pip install bitsandbytes

# macOS (不支持 CUDA 量化，需要用 CPU 或 MPS)
# 跳过 bitsandbytes，设置 load_in_4bit: false

# Windows
pip install bitsandbytes-windows
```

### Q: torch-scatter/torch-sparse 安装失败
```bash
# 确保 PyTorch 版本匹配
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### Q: CUDA out of memory
- 减小 batch_size
- 启用 gradient_checkpointing
- 使用 DeepSpeed ZeRO-3
