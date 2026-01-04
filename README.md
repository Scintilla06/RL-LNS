# Deep-Structure RL-LNS Solver

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.40+-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> 🧠 结合 **GNN 结构编码** 与 **Qwen2.5-7B 大语言模型** 的混合整数线性规划 (MILP) 神经求解器，采用 **Physics-Informed SFT** 和 **GRPO 强化学习** 进行训练。

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [数据格式](#-数据格式)
- [模型架构](#-模型架构)
- [训练流程](#-训练流程)
- [配置说明](#-配置说明)
- [项目结构](#-项目结构)
- [API 文档](#-api-文档)
- [实验结果](#-实验结果)
- [常见问题](#-常见问题)
- [引用](#-引用)
- [许可证](#-许可证)

---

## 🎯 项目简介

**RL-LNS** (Reinforcement Learning - Large Neighborhood Search) 是一个创新的 MILP 求解框架，将深度学习与传统优化算法相结合。核心思想是使用神经网络预测 MILP 问题中二元变量的最优取值，从而加速大规模组合优化问题的求解。

### 研究动机

传统的 MILP 求解器（如 Gurobi、CPLEX）在面对大规模问题时可能需要数小时甚至数天的计算时间。本项目探索了一种新范式：

1. **学习问题结构**: 通过 GNN 编码约束-变量的二部图结构
2. **利用 LLM 推理能力**: 使用 Qwen2.5-7B 作为推理骨干
3. **物理约束感知训练**: 在损失函数中显式引入约束满足和整数性惩罚
4. **强化学习微调**: 使用 GRPO 算法进一步优化解的质量

---

## ✨ 核心特性

### 🔹 双模式输入
- **GNN 模式**: 将 MILP 问题编码为二部图，通过 GNN 提取结构特征
- **Text 模式**: 将 MILP 问题序列化为文本，支持超长序列分块处理

### 🔹 先进的模型架构
- **Qwen2.5-7B-Instruct** 作为推理骨干
- **4-bit QLoRA** 量化，支持单卡 24GB 显存训练
- **FlashAttention-2** 加速注意力计算
- **禁用 RoPE**，使用 RWPE 位置编码保留图结构

### 🔹 Physics-Informed 训练
- **任务损失**: 二分类交叉熵
- **约束损失**: 惩罚约束违反
- **整数性损失**: 推动预测趋向 0/1

### 🔹 GRPO 强化学习
- **组采样**: 每个实例采样 G=16 个候选解
- **相对优势**: 基于组内排名计算优势函数
- **可行性奖励**: 显式奖励可行解

### 🔹 启发式进化 (EOH)
- 自动生成和优化 LNS 算子
- 支持多种进化策略

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         RL-LNS Solver                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   MILP      │    │    GNN      │    │   Qwen2.5   │          │
│  │  Instance   │───▶│  Tokenizer  │───▶│   Backbone  │          │
│  └─────────────┘    └─────────────┘    └──────┬──────┘          │
│         │                                      │                │
│         │          ┌─────────────┐             │                │
│         └─────────▶│    Text     │─────────────┘                │
│                    │  Tokenizer  │                              │
│                    └─────────────┘                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Prediction Heads                       │   │
│  ├──────────────┬──────────────┬──────────────┬─────────────┤   │
│  │   Primal     │  Uncertainty │    Dual      │  Multi-Task │   │
│  │    Head      │     Head     │    Head      │    Head     │   │
│  └──────────────┴──────────────┴──────────────┴─────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Solution: P(x_i = 1) for all i              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 训练流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Raw Data   │───▶│  Preprocess  │───▶│  PyG Graphs  │
│    (JSON)    │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Best Model  │◀───│     GRPO     │◀───│   SFT Model  │
│              │    │   Training   │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
                                               ▲
                                               │
                           ┌───────────────────┴───────────────────┐
                           │     Physics-Informed SFT Training     │
                           │  L = L_task + λ₁L_constr + λ₂L_int    │
                           └───────────────────────────────────────┘
```

---

## 📦 安装指南

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.10+ | 3.10 |
| CUDA | 11.8+ | 12.1+ |
| GPU 显存 | 16GB | 24GB+ |
| RAM | 32GB | 64GB+ |
| 存储空间 | 50GB | 100GB+ |

### 方式 1: Conda 环境 (推荐)

```bash
# 克隆仓库
git clone https://github.com/your-username/RL-LNS.git
cd RL-LNS

# 创建并激活环境
conda env create -f environment.yaml
conda activate rl-lns

# 安装 Gurobi (需要 License)
conda install -c gurobi gurobi

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 方式 2: 手动安装

```bash
# 创建虚拟环境
conda create -n rl-lns python=3.10 -y
conda activate rl-lns

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 安装其他依赖
pip install transformers>=4.40.0 accelerate>=0.27.0
pip install bitsandbytes>=0.42.0 peft>=0.8.0
pip install datasets wandb tqdm pyyaml scipy
pip install gurobipy  # 需要 License
```

### 安装 Flash Attention (可选但推荐)

```bash
# 需要 CUDA 11.6+ 和支持的 GPU (Ampere, Ada, Hopper)
pip install flash-attn --no-build-isolation
```

### 下载预训练模型

```bash
# 方式 A: 自动下载 (首次运行时会自动从 HuggingFace 下载)
# 无需额外操作

# 方式 B: 手动下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

---

## 🚀 快速开始

### 1. 数据预处理

```bash
# 将原始 JSON 数据转换为 PyG 图格式
python src/main.py preprocess --config configs/data.yaml
```

### 2. SFT 训练

```bash
# Physics-Informed 监督微调
python src/main.py train-sft --config configs/training.yaml
```

### 3. GRPO 训练

```bash
# 强化学习微调 (在 SFT 之后)
python src/main.py train-grpo --config configs/training.yaml
```

### 4. 推理

```bash
# 对新问题进行预测
python src/main.py infer --model outputs/sft/best --input problem.lp
```

### 5. 启发式进化 (可选)

```bash
# 自动生成 LNS 算子
python src/main.py evolve --config configs/evolution.yaml
```

### 完整示例

```python
import torch
from src.model.neuro_solver import NeuroSolver
from src.datalib.preprocess import MILPPreprocessor

# 初始化模型
solver = NeuroSolver(
    backbone="Qwen/Qwen2.5-7B-Instruct",
    mode="gnn",
    load_in_4bit=True,
    device=torch.device("cuda")
)

# 加载训练好的权重
solver.load_checkpoint("outputs/sft/best")

# 预处理 MILP 实例
preprocessor = MILPPreprocessor()
graph = preprocessor.from_lp_file("problem.lp")

# 预测
with torch.no_grad():
    output = solver(graph)
    probs = output.primal_probs  # P(x_i = 1)
    solution = (probs > 0.5).int()
```

---

## 📊 数据格式

### 输入 JSON 格式

训练数据应为 JSON 格式，每个样本包含完整的 MILP 信息：

```json
{
  "problem_id": "instance_001",
  "lp_file": "minimize\n obj: x1 + 2 x2 + 3 x3\nsubject to\n c1: x1 + x2 >= 1\n c2: x2 + x3 <= 2\nbounds\n 0 <= x1 <= 1\n 0 <= x2 <= 1\n 0 <= x3 <= 1\nbinary\n x1 x2 x3\nend",
  "optimal_objective": 1.0,
  "optimal_solution": {
    "x1": 1,
    "x2": 0,
    "x3": 0
  },
  "metadata": {
    "num_variables": 3,
    "num_constraints": 2,
    "problem_type": "set_cover"
  }
}
```

### PyG Graph 格式 (预处理后)

```python
HeteroData(
  # 变量节点
  var={
    x=[n_vars, feat_dim],      # 特征: obj系数, bounds, 类型, LP解等
    y=[n_vars],                 # 标签: 最优解
    var_types=[n_vars],         # 变量类型
    ...
  },
  
  # 约束节点
  con={
    x=[n_constrs, feat_dim],   # 特征: RHS, sense, 稀疏度等
    ...
  },
  
  # 边 (约束-变量)
  con__to__var={
    edge_index=[2, n_edges],
    edge_attr=[n_edges, edge_feat_dim],  # 系数
  },
  
  # 元信息
  obj_sense=1,                  # 1=minimize, -1=maximize
  ...
)
```

---

## 🧠 模型架构

### GNN Tokenizer

将 MILP 二部图编码为 Qwen 可接受的嵌入序列：

```python
class GNNTokenizer(nn.Module):
    """
    MILP Graph → Embedding Sequence
    
    1. Fourier Feature Encoding: 将连续特征映射到高维空间
    2. RWPE: Random Walk Positional Encoding
    3. BipartiteGNN: 约束-变量消息传递
    4. Projection: 投影到 Qwen 隐层维度 (3584)
    """
    
    # 配置
    gnn_hidden_dim: 256
    gnn_output_dim: 3584  # Match Qwen
    num_layers: 2
    conv_type: "GINEConv"  # or "GATv2Conv"
```

**特征编码**:
- **Fourier Features**: $x \to [\sin(2^k\pi x), \cos(2^k\pi x)]_{k=0}^{L-1}$
- **RWPE**: $\text{diag}(P^k)$ for $k = 1, ..., K$ where $P = D^{-1}A$

### Text Tokenizer

处理超长 MILP 文本表示：

```python
class ChunkedTextEncoder(nn.Module):
    """
    支持 >64K token 的 MILP 文本序列
    
    1. 分块: 将长序列切分为重叠块
    2. 编码: 每块独立编码
    3. 聚合: 加权平均合并块表示
    """
    
    chunk_size: 8192
    chunk_stride: 4096  # 50% 重叠
```

### Prediction Heads

多任务预测头：

| Head | 输出 | 描述 |
|------|-----|------|
| `PredictionHead` | $P(x_i=1)$ | 主预测头，输出各变量为 1 的概率 |
| `UncertaintyHead` | $\sigma_i^2$ | 预测不确定性，用于 LNS 选择 |
| `DualHead` | $\pi_j$ | 对偶变量预测 (可选) |
| `MultiTaskHead` | 组合输出 | 统一的多任务头 |

### NeuroSolver 主模块

```python
class NeuroSolver(nn.Module):
    def forward(self, batch, mode="gnn") -> SolutionOutput:
        # 1. 编码输入
        if mode == "gnn":
            embeddings = self.gnn_tokenizer(batch)
        else:
            embeddings = self.text_tokenizer(batch)
        
        # 2. Qwen 推理 (RoPE 已禁用)
        hidden = self.qwen(inputs_embeds=embeddings, position_ids=zeros)
        
        # 3. 预测
        return SolutionOutput(
            primal_probs=self.pred_head(hidden),
            uncertainty=self.uncertainty_head(hidden),
        )
```

---

## 🏋️ 训练流程

### Stage 1: Physics-Informed SFT

**目标**: 学习从 MILP 结构到最优解的映射

**损失函数**:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{constr}} + \lambda_2 \mathcal{L}_{\text{int}}$$

| 损失项 | 公式 | 作用 |
|-------|------|------|
| Task Loss | $-\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$ | 预测精度 |
| Constraint Loss | $\sum_j \max(0, Ax - b)_j$ | 约束满足 |
| Integrality Loss | $\sum_i p_i(1-p_i)$ | 推向整数 |

**默认超参数**:
- Learning Rate: 2e-4
- Batch Size: 16 (gradient accumulation)
- Epochs: 3
- $\lambda_1 = 0.1$, $\lambda_2 = 0.01$

### Stage 2: GRPO 强化学习

**目标**: 通过与求解器交互进一步优化解质量

**算法**:
1. 对每个实例采样 $G=16$ 个候选解
2. 用 Gurobi 评估每个解的质量和可行性
3. 计算组内相对优势
4. 更新策略最大化期望奖励

**奖励函数**:
$$r(x) = \begin{cases}
-c^T x & \text{if feasible} \\
-c^T x - \gamma \cdot \text{violation} & \text{otherwise}
\end{cases}$$

**默认超参数**:
- Group Size: 16
- Learning Rate: 5e-5
- KL Coefficient: 0.01
- Infeasibility Penalty: 10.0

---

## ⚙️ 配置说明

项目使用 YAML 配置文件，位于 `configs/` 目录。

### model.yaml

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  hidden_size: 3584
  load_in_4bit: true
  lora_r: 64
  lora_alpha: 128
  use_flash_attention: true
  disable_rope: true

gnn:
  hidden_dim: 256
  output_dim: 3584
  num_layers: 2
  conv_type: "GINEConv"

heads:
  hidden_dim: 1024
  enable_uncertainty: true
```

### training.yaml

```yaml
sft:
  batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  num_epochs: 3
  lambda_constraint: 0.1
  lambda_integrality: 0.01

grpo:
  group_size: 16
  learning_rate: 5.0e-5
  kl_coef: 0.01
```

### data.yaml

```yaml
raw:
  train_json: "data/train_dataset_huge.json"
  val_split_ratio: 0.1

preprocessing:
  compute_lp_relaxation: true
```

---

## 📁 项目结构

```
RL-LNS/
├── 📄 README.md              # 本文件
├── 📄 INSTALL.md             # 详细安装指南
├── 📄 LICENSE                # MIT 许可证
├── 📄 environment.yaml       # Conda 环境配置
├── 📄 requirements.txt       # Pip 依赖
│
├── 📂 configs/               # 配置文件
│   ├── model.yaml            # 模型配置
│   ├── training.yaml         # 训练配置
│   ├── data.yaml             # 数据配置
│   └── evolution.yaml        # 进化算法配置
│
├── 📂 data/                  # 数据目录
│   ├── train_dataset_huge.json
│   └── processed/            # 预处理后的数据
│       ├── train/
│       └── val/
│
├── 📂 src/                   # 源代码
│   ├── __init__.py
│   ├── main.py               # 主入口
│   │
│   ├── 📂 datalib/           # 数据处理
│   │   ├── preprocess.py     # LP 解析、图构建
│   │   └── dataset.py        # PyTorch Dataset
│   │
│   ├── 📂 model/             # 模型定义
│   │   ├── gnn_tokenizer.py  # GNN 编码器
│   │   ├── text_tokenizer.py # 文本编码器
│   │   ├── heads.py          # 预测头
│   │   └── neuro_solver.py   # 主模型
│   │
│   ├── 📂 training/          # 训练逻辑
│   │   ├── physics_loss.py   # Physics-Informed 损失
│   │   ├── sft_trainer.py    # SFT 训练器
│   │   └── grpo_loop.py      # GRPO 训练器
│   │
│   ├── 📂 evolution/         # 启发式进化
│   │   ├── operators.py      # 进化算子
│   │   └── eoh.py            # EOH 主算法
│   │
│   ├── 📂 problems/          # 问题定义
│   │   └── milp.py           # MILP 问题接口
│   │
│   ├── 📂 llm/               # LLM 接口
│   │   └── api.py            # API 调用封装
│   │
│   └── 📂 utils/             # 工具函数
│       └── __init__.py
│
├── 📂 outputs/               # 输出目录
│   ├── sft/                  # SFT 模型检查点
│   └── grpo/                 # GRPO 模型检查点
│
└── 📂 experiments/           # 实验记录
    └── qwen2.5-7b-sft-milp/
```

---

## 📚 API 文档

### NeuroSolver

```python
class NeuroSolver(nn.Module):
    """统一的 MILP 神经求解器"""
    
    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-7B-Instruct",
        mode: str = "gnn",           # "gnn" | "text" | "both"
        load_in_4bit: bool = True,
        use_flash_attention: bool = True,
        disable_rope: bool = True,
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 2,
        lora_r: int = 64,
        lora_alpha: int = 128,
        include_uncertainty: bool = True,
        device: torch.device = None,
    ):
        ...
    
    def forward(
        self,
        batch: Union[HeteroData, Dict],
        mode: Optional[str] = None,
    ) -> SolutionOutput:
        """
        前向传播
        
        Args:
            batch: 输入批次 (PyG HeteroData 或文本字典)
            mode: 覆盖默认模式
            
        Returns:
            SolutionOutput: 包含 primal_probs, uncertainty 等
        """
        ...
    
    def predict(
        self,
        graph: HeteroData,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        预测二元解
        
        Args:
            graph: 单个 MILP 图
            threshold: 分类阈值
            
        Returns:
            solution: 二元解向量
        """
        ...
```

### MILPPreprocessor

```python
class MILPPreprocessor:
    """MILP 数据预处理器"""
    
    def __init__(
        self,
        compute_lp_relaxation: bool = True,
        normalize_features: bool = True,
    ):
        ...
    
    def from_json(self, json_data: Dict) -> HeteroData:
        """从 JSON 数据构建图"""
        ...
    
    def from_lp_file(self, lp_path: str) -> HeteroData:
        """从 LP 文件构建图"""
        ...
    
    def process_sample(self, sample: Dict) -> HeteroData:
        """处理单个样本"""
        ...
```

### SFTTrainer

```python
class SFTTrainer:
    """Physics-Informed SFT 训练器"""
    
    def __init__(
        self,
        model: NeuroSolver,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        learning_rate: float = 2e-4,
        lambda_constraint: float = 0.1,
        lambda_integrality: float = 0.01,
        ...
    ):
        ...
    
    def train(self, num_epochs: int) -> Dict:
        """执行训练"""
        ...
```

### GRPOTrainer

```python
class GRPOTrainer:
    """GRPO 强化学习训练器"""
    
    def __init__(
        self,
        model: NeuroSolver,
        group_size: int = 16,
        learning_rate: float = 5e-5,
        kl_coef: float = 0.01,
        infeasibility_penalty: float = 10.0,
        ...
    ):
        ...
    
    def train(self, num_epochs: int) -> Dict:
        """执行 GRPO 训练"""
        ...
```

---

## 📈 实验结果

### 数据集

| 数据集 | 样本数 | 变量范围 | 约束范围 | 问题类型 |
|-------|-------|---------|---------|---------|
| train_dataset_huge | ~10K | 50-500 | 20-200 | 混合 |

### 性能指标

| 指标 | 描述 |
|-----|------|
| Accuracy | 变量预测准确率 |
| Feasibility Rate | 生成可行解的比例 |
| Optimality Gap | 与最优解的差距 |
| Solve Time | 预测时间 |

### 与基准对比

*待补充实验结果*

---

## ❓ 常见问题

### Q: 显存不足怎么办？

A: 尝试以下方法：
1. 确保启用 4-bit 量化 (`load_in_4bit: true`)
2. 减小 batch size 并增加 gradient accumulation
3. 使用 gradient checkpointing
4. 减小 GNN 隐层维度

### Q: 如何获取 Gurobi License？

A: Gurobi 提供免费学术 License：
1. 访问 https://www.gurobi.com/academia/academic-program-and-licenses/
2. 注册学术账号
3. 下载并激活 License

### Q: 训练时 Loss 不下降？

A: 检查以下事项：
1. 学习率是否合适 (推荐 1e-4 ~ 5e-4)
2. 数据预处理是否正确
3. 是否正确加载了预训练权重

### Q: 如何使用自定义数据？

A: 
1. 准备 JSON 格式数据 (参考数据格式章节)
2. 修改 `configs/data.yaml` 中的路径
3. 运行预处理: `python src/main.py preprocess --config configs/data.yaml`

### Q: 支持哪些 MILP 问题类型？

A: 理论上支持任意 MILP 问题，但训练数据应覆盖目标问题类型以获得最佳效果。常见支持：
- Set Covering / Packing
- Facility Location
- Vehicle Routing (简化版)
- Scheduling
- 通用 0-1 整数规划

---

## 📖 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{rl_lns_2024,
  title = {Deep-Structure RL-LNS: Neural Solver for Mixed Integer Linear Programming},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/RL-LNS}
}
```

### 相关工作

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 基座大模型
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络框架
- [Gurobi](https://www.gurobi.com/) - 商业 MILP 求解器
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

<p align="center">
  <b>如有问题，欢迎提 Issue 或 Discussion！</b>
</p>