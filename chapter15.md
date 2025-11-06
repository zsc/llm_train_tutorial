# chapter15.md — 附录与参考

## 开篇段落
本章是整个教程的“工具箱”与“知识库索引”，不引入新的核心概念。我们的目标是提供一个快速查阅的参考中心，集中收录在前续章节中反复使用的关键符号、基线超参数、配置文件模板以及核心参考文献。它存在的意义在于，当你开始一项新的训练任务，或是在调试中对某个参数的合理性产生疑问时，能够迅速回归此地，找到一个经过验证的、坚实的出发点。无论你是需要快速查找一个符号的定义，还是想为一次新的训练任务寻找一个可靠的启动配置，亦或是希望深入阅读支撑这些实践的原始论文，本章都将为你提供直接、详尽的入口。

---

## 15.1 符号表（统一约定）

在规模化实验中，一套清晰、无歧义的符号系统是保证团队沟通效率与研究可复现性的基石。下面这张表不仅定义了符号，更附加了注解，以阐明其在本书语境下的精确内涵。

| 符号              | 含义                                 | 常见取值/单位 | 注解 (Annotation) |
| --------------- | ---------------------------------- | --- | --- |
| `N_params`      | 模型非嵌入参数量                       | B (Billion, 10⁹) | **重点**：特指 Transformer block 中的参数，不包含 token embedding 和输出层权重。这是因为在 Scaling Law 的讨论中，`N_params` 与 `FLOPs` 的关系更直接，而 embedding 的大小与 `vocab_size` 相关，有时会独立分析。 |
| `T_tokens`      | 训练总 tokens（数据集规模）                  | T (Trillion, 10¹²) | 代表模型在整个训练生命周期中“看到”的 token 总量。这是计算总 `FLOPs` 和规划训练时长的核心输入。 |
| `L_ctx`         | 上下文长度                            | 4096, 8192 | 单个训练样本的最大 token 序列长度。它直接影响了单一样本的计算量和显存占用。 |
| `GB_tok`        | **Global batch（以 tokens 计）**       | 2M, 4M | **核心概念**：在一次优化器更新（`optimizer.step()`）中，模型处理的 token 总数。`GB_tok = μ_tok × A × D`。这是影响训练动态（如噪声尺度 `ρ`）和收敛性的关键超参。 |
| `μ_tok`         | 单卡 micro-batch（以 tokens 计）         | e.g., `4096 * 4` | 单个 GPU 在一次前向/后向计算中处理的 token 数量。`μ_tok = L_ctx * sequences_per_gpu`。它的上限受单卡显存（OOM）的严格约束。 |
| `A`             | 梯度累积步数 (`GB_tok / (μ_tok * D)`)  | 4, 8, 16, ... | 为了在显存有限的情况下模拟出大的 `GB_tok`，我们累积 `A` 步的梯度进行一次参数更新。这是计算与显存之间的经典权衡。 |
| `D`             | Data 并行因子                      | 64, 32, ... | 即数据并行的 world size。`D = 总 GPU 数 / (TP * PP)`。 |
| `TP`            | Tensor 并行因子                        | 2, 4, 8 | 将模型的单个大权重矩阵（如 FFN 层）切分到多个 GPU 上，以解决单 GPU 显存无法容纳模型参数的问题。通常在节点内（通过 NVLink）使用。 |
| `PP`            | Pipeline 并行因子                      | 1 | 将模型的不同层放置在不同 GPU 上。对于本教程的 Decoder-only 架构和高带宽互联环境，PP 引入的 bubble 开销较大，通常不作为首选，故默认为 1。 |
| `η`             | 学习率（峰值/基础值）                   | 1e-5 to 3e-4 | 学习率调度器中的峰值学习率。其选择与 `GB_tok` 密切相关（参考 LR Scaling 法则）。 |
| `β₁, β₂, ε`     | AdamW 优化器超参                     | 0.9, 0.95, 1e-8 | `β₂=0.95` 是 LLM 训练中的一个关键经验值，相比传统的 0.999，它对梯度历史的遗忘更快，被认为在大批量、高噪声场景下更稳定。 |
| `wd`            | Weight Decay                         | 0.01, 0.1 | LLaMA 等模型中常用的 `wd=0.1` 是一个相对较大的值，起到了很强的正则化作用，有助于防止在海量数据上过拟合。 |
| `ρ`             | 噪声尺度（Noise Scale）                  | - | 衡量梯度噪声与梯度本身大小之比，与 `GB_tok` 和学习率 `η` 紧密相关，是理解大批量训练动态的核心理论工具。 |
| `FLOPs`         | 训练浮点运算量                       | PetaFLOPs | 估算公式 `≈ 6 · N_params · T_tokens`，其中 `6` 来自 `2` (Fwd+Bwd) × `3` (矩阵乘法近似)。这是衡量总计算量的“物理”单位，独立于硬件效率。 |
| `PUE`           | 数据中心电能使用效率                     | 1.1 ~ 1.5 | 总能耗 / IT 设备能耗。值越接近 1，据中心制冷等配套设施的能效越高。 |
| `¥/GPU·h`       | GPU 小时成本（云/自建）                   | ¥ | 一个综合成本指标，自建时需考虑硬件折旧、运维人力等，云上则是直接的标价。 |
| `kWh`           | 千瓦时（电量单位）                       | - | 1 kWh 即“一度电”，是计算电费的基础。 |

---

## 15.2 默认超参清单（基线配置）

下表提供的并非“银弹”，而是经过社区广泛验证的、稳健的**起点（Starting Points）**。它们的设计哲学源于 Chinchilla-style 的计算最优原则，并在 LLaMA 系列模型的成功实践中得到了印证。当你开启一个新的训练项目时，从这里出发可以最大程度地避免早期不必要的“炼丹”。

#### **A. 模型架构 (Architecture)**
这部分参数定义了模型的“骨架”，决定了其容量和计算特性。

| 参数 (Parameter)                    | 3B 配置 (3B Config)        | 7B 配置 (7B Config)        | 13B 配置 (13B Config)      | 说明 (Notes) |
| ----------------------------------- | -------------------------- | -------------------------- | -------------------------- | --- |
| `hidden_size`                       | 3200                       | 4096                       | 5120                       | 模型的核心维度。通常是 `num_attention_heads` × `head_dim` (e.g., 32 * 128 = 4096) 的整数倍，以保证计算效率。 |
| `num_hidden_layers`                 | 32                         | 32                         | 40                         | 模型的深度。增加层数可以提升模型的表达能力，但也会增加训练难度和推理延迟。 |
| `num_attention_heads`               | 32                         | 32                         | 40                         | 注意力头数。通常与 `hidden_size` 成比例，以保持每个头的维度（`head_dim`）在一个合理范围（如 128）。 |
| `num_key_value_heads`               | 32 (MHA)                   | 8 (GQA)                    | 8 (GQA)                    | **关键优化**：Grouped-Query Attention (GQA)。通过让多组 Query 头共享同一组 Key/Value 头，显著减少了推理时 KV-Cache 的显存占用，是长上下文推理的必备技术。 |
| `intermediate_size`                 | 8640                       | 11008                      | 13824                      | SwiGLU FFN 的中间维度。LLaMA 的计算公式为 `ceil(2/3 * 4 * hidden_size)` 并向上取整到 128 的倍数，这是一个经过实验验证的、在效果和效率上平衡的选择。 |
| `vocab_size`                        | 32000 - 64000              | 32000 - 64000              | 32000 - 64000              | 词表大小，强依赖于你的语料。一个好的 `vocab_size` 应该能有效覆盖语料中的高频词和子词，同时避免过大导致 embedding 层参数过多。通常是 2 的幂或能被 128 整除。 |
| `rope_theta` (base)                 | 10000                      | 10000                      | 10000                      | RoPE 的基础频率。当使用 PI/NTK/YaRN 等 scaling 方法时，这个基础值会被动态调整以适应更长的上下文。 |
| `context_length` (`L_ctx`)          | 4096 / 8192                | 4096 / 8192                | 4096 / 8192                | 训练时的上下文长度。注意，通过 RoPE scaling 扩展到 8k 是在 4k 预训练模型基础上微调或直接从头训练，配置会有所不同。 |

#### **B. 训练与优化器 (Training & Optimizer)**
这部分参数控制着学习过程本身，直接决定了模型的收敛速度和最终性能。

| 参数 (Parameter)                    | 3B 配置 (3B Config)        | 7B 配置 (7B Config)        | 13B 配置 (13B Config)      | 说明 (Notes) |
| ----------------------------------- | -------------------------- | -------------------------- | -------------------------- | --- |
| `global_batch_size_tokens` (`GB_tok`) | 2,097,152 (2M)             | 4,194,304 (4M)             | 4,194,304 (4M)             | **训练的黄金法则**。Chinchilla 指出，大模型需要足够大的 batch size 才能有效学习。4M tokens 是一个业界公认的甜点区，能提供足够稳定的梯度信号。 |
| `learning_rate` (`η`)                 | 3.0e-4                     | 3.0e-4                     | 1.5e-4                     | 峰值学习率。遵循“大模型、小学习率”的原则。对于更大的 `GB_tok`，可以根据 `sqrt` 或线性 scaling 法则适当增大学习率，但这需要实验验证。 |
| `lr_scheduler`                      | cosine with warmup         | cosine with warmup         | cosine with warmup         | 余弦退火调度器是 LLM 训练的标配。它前期缓慢上升（warmup），然后在大部分训练时间内缓慢下降，有助于稳定收敛。 |
| `warmup_steps`                      | 2000                       | 2000                       | 2000                       | 预热步数。通常设置为总训练步数的 1-5%。过短的 warmup 可能导致初训练不稳定，过长则浪费了高学习率的有效训练时间。 |
| `min_lr_ratio`                      | 0.1                        | 0.1                        | 0.1                        | 学习率最终会衰减到 `η * min_lr_ratio`。保持一个较小的最终学习率有助于在训练末期对模型进行微调。 |
| `optimizer`                         | AdamW                      | AdamW                      | AdamW                      | AdamW 因其鲁棒性和解耦的权重衰减而成为首选。DeepSpeed 的 Paged AdamW 可以在 CPU/NVMe offload 场景下进一步减少内存碎片，提升效率。 |
| `beta1` (`β₁`)                        | 0.9                        | 0.9                        | 0.9                        | 一阶动量（梯度均值）的衰减率。 |
| `beta2` (`β₂`)                        | 0.95                       | 0.95                       | 0.95                       | **关键经验值**：二阶动量（梯度平方均值）的衰减。0.95 意味着对历史梯度的遗忘更快，这被认为在处理大规模、非平稳的 LLM 训练数据时能提供更及时的梯度方差估计，从而提高稳定性。 |
| `weight_decay` (`wd`)               | 0.1                        | 0.1                        | 0.1                        | 一个相对较强的正则化项，对于防止在 1T+ tokens 的大规模数据上过拟合至关重要。 |
| `grad_clip`                         | 1.0                        | 1.0                        | 1.0                        | 梯度裁剪的全局范数阈值。这是防止梯度爆炸、保证训练稳定的“保险丝”。 |

#### **C. 并行与内存 (Parallelism & Memory)**
这些配置是算法与硬件基础设施的接口，目标是在给定的 64x H100 集群上实现最高的训练吞吐量（tokens/s）。

| 参数 (Parameter)                    | 3B 配置 (3B Config)        | 7B 配置 (7B Config)        | 13B 配置 (13B Config)      | 说明 (Notes) |
| ----------------------------------- | -------------------------- | -------------------------- | -------------------------- | --- |
| `tensor_parallel_size` (`TP`)       | 2 or 4                     | 4 or 8                     | 8                          | 13B 模型在 H100 上，`TP=8` 是一个高效的选择，能充分利用 8-GPU 节点内的高速 NVLink 互联。对于 7B，`TP=4` 或 `TP=8` 均可，需要根据实际 profile 决定。 |
| `activation_checkpointing`          | True                       | True                       | True                       | **内存优化的核心**。通过在前向传播时丢弃中间激活值，在后向传播时重新计算，用计算时间换取巨大的显存节省。对于训练大模型来说，这几乎是必须开启的。 |
| `precision`                         | `bf16`                     | `bf16`                     | `bf16`                     | `bfloat16` 提供了与 `float32` 几乎相同的动态范围，但显存和计算量减半。相比 `float16`，它更好地避免梯度下溢问题，是 A100/H100 等现代 GPU 上的首选。 |

---

## 15.3 配置模板 (YAML/JSON)

理论最终要落地为代码可读的配置。以下模板展示了如何使用 PyTorch Lightning 组织训练逻辑，并交由 DeepSpeed 执行底层的并行与优化。

### 15.3.1 训练主配置文件 (`pretrain_7b_8k.yaml`)

这是一个用户侧的、高度集成的配置文件，定义了实验的“科学”部分：模型、数据、优化策略。

```yaml
# ------------------ 模型配置 (Model) ------------------
# 定义了模型的宏观和微观结构
model:
  name: "llama_style_7b" # 用于日志记录和模型加载
  architecture:
    hidden_size: 4096
    num_hidden_layers: 32
    num_attention_heads: 32
    num_key_value_heads: 8 # 启用 GQA, 对于推理至关重要
    intermediate_size: 11008
    vocab_size: 32000
    context_length: 8192
    
    # RoPE Scaling 配置 (可选, 用于扩展上下文)
    # 如果不使用，请注释掉或删除此部分
    rope_scaling:
      type: 'yarn'      # 可选 'pi', 'ntk-aware'
      factor: 2.0       # 从 4k 扩展到 8k, 因子为 2
      original_max_position_embeddings: 4096

# ------------------ 数据配置 (Data) ------------------
# 定义了数据从哪里来，以及如何组织
data:
  path: "cpfs://path/to/tokenized_data_shards" # 推荐使用分片的 WebDataset/Parquet
  format: "webdataset" # 或 "parquet"
  seq_len: 8192
  
  # 动态混合策略，在每个 step 动态采样
  dynamic_mixing:
    - { name: "wiki_en", weight: 0.3, path: "cpfs://path/to/wiki_shards" }
    - { name: "books", weight: 0.5, path: "cpfs://path/to/books_shards" }
    - { name: "github_code", weight: 0.2, path: "cpfs://path/to/code_shards" }
    
  num_workers: 8 # 每个 GPU 的数据加载进程数, 推荐 4-8

# ------------------ 优化器与调度器 (Optimizer & Scheduler) ------------------
# 定义了模型参数如何更新
optimizer:
  name: "AdamW" # Lightning 会自动映射到 Pytorch 的 AdamW
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  eps: 1.0e-8

scheduler:
  name: "cosine" # 内部会映射为 CosineAnnealingLR
  warmup_steps: 2000
  min_lr_ratio: 0.1 # 最终学习率为 peak_lr * 0.1

# ------------------ 训练器配置 (Trainer) ------------------
# 这是 PyTorch Lightning 的核心，负责编排整个训练流程
trainer:
  # 硬件与并行策略
  devices: 64
  num_nodes: 8
  accelerator: "gpu"
  strategy: "deepspeed" # 告诉 Lightning 使用 DeepSpeed 策略
  
  # 精度与性能
  precision: "bf16-mixed"
  
  # 训练循环控制
  max_steps: 250000 # 示例: 4M tokens/batch * 250k steps = 1T tokens
  val_check_interval: 1000 # 每 1000 步在验证集上评估一次 PPL
  
  # 日志与检查点
  log_every_n_steps: 10
  logger:
    - class_path: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: "logs/"
      name: "pretrain_7b_8k_run_01"
  
  # 将下面的 DeepSpeed JSON 文件路径传递给 Lightning
  deepspeed_config: "./configs/deepspeed_zero3_paged.json"
```

### 15.3.2 DeepSpeed 配置文件 (`deepspeed_zero3_paged.json`)

这是一个后端配置文件，告诉 DeepSpeed 如何执行内存优化和并行计算。Lightning 会智能地从主配置中填充 `"auto"` 字段。

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto", // 全局批大小(样本数), Lightning 会根据 GB_tok 和 L_ctx 计算
  "gradient_accumulation_steps": "auto",
  
  "zero_optimization": {
    "stage": 3, // 完全分片参数、梯度和优化器状态
    "offload_optimizer": {
      "device": "cpu", // 将优化器状态卸载到 CPU 内存, 节省大量显存
      "pin_memory": true // 使用锁页内存加速 CPU-GPU 数据传输
    },
    "offload_param": {
      "device": "cpu", // 将模型参数也卸载到 CPU，仅在需要时加载到 GPU
      "pin_memory": true
    },
    "overlap_comm": true, // 尽可能重叠计算和通信
    "contiguous_gradients": true, // 使用连续内存存储梯度，提高通信效率
    "sub_group_size": 1e9, // 用于参数分组通信的阈值
    "stage3_prefetch_bucket_size": "auto", // 预取参数的桶大小
    "stage3_param_persistence_threshold": "auto", // 大于此阈值的参数将保留在 GPU 上
    "stage3_max_live_parameters": 1e9 // GPU 上同时存在的最大参数量
  },

  // 虽然 Lightning 管理优化器，但 DeepSpeed 需要此配置块来启用 PagedAdamW 等特性
  "optimizer": {
    "type": "PagedAdamW", // 相比 AdamW, 能更好地管理内存碎片
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },

  // 同样，调度器由 Lightning 控制，此处仅为占位符
  "scheduler": {
    "type": "WarmupDecayLR", 
    "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto",
        "total_num_steps": "auto"
    }
  },
  
  "bf16": {
    "enabled": true
  },
  
  "gradient_clipping": 1.0,
  
  "steps_per_print": 10,
  
  "wall_clock_breakdown": false // 设 true 可获取详细的时间分析
}
```

---

## 15.4 参考论文与进一步阅读

纸上得来终觉浅，欲知此事须躬行。但行之前，站在巨人的肩膀上是必要的。

### 15.4.1 核心论文

1.  **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022 - **Chinchilla**)
    > **为何重要**：**本教程的理论基石**。它通过大规模实验推导出，在给定的计算预算下，模型参数量 `N` 和训练数据量 `T` 存在一个最优配比关系（大致为 `N ∝ T^0.5`）。这意味着，与其训练一个巨大的模型在少量数据上，不如训练一个中等大小的模型在海量数据上。本教程所有关于 `1T` tokens 的训练目标和模型规模的选择都源于此。

2.  **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023 - **LLaMA**)
    > **为何重要**：本教程模型架构的“食谱”。LLaMA 并非提出了全新的组件，而是巧妙地组合了社区中已有的最佳实践：RMSNorm（替代 LayerNorm，更稳定）、SwiGLU（替代 ReLU，效果更好）、RoPE（替代绝对位置编码，外推性更好）。这个组合被证明极为高效和强大，成为后续开源模型模仿的典范。

3.  **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022) & **FlashAttention-2** (Dao, 2023)
    > **为何重要**：**训练吞吐量的“发动机”**。通过将注意力计算的多个步骤融合成一个 CUDA kernel，并采用 tiling 技术来优化 GPU SRAM 和 HBM 之间的数据移动，FlashAttention 在不牺牲任何精度的情况下，实现了数量级的加速和显存节省。它是长上下文训练成为可能的关键技术之一。

4.  **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2020 - **DeepSpeed ZeRO**)
    > **为何重要**：**大规模训练的“杠杆”**。ZeRO 通过在数据并行的 worker 之间切分模型状态（参数、梯度、优化器状态），打破了“单 GPU 必须容纳完整模型”的限制。ZeRO-3 更是将模型参数也进行切分，使得理论上可以用任意数量的 GPU 训练任意大的模型，只要总显存足够。

### 15.4.2 进阶阅读与社区资源

1.  **RoFormer** & **YaRN**: `Su et al., 2021` 提出了 RoPE 的原始思想。`Peng et al., 2023` 的 YaRN 则是目前最有效的 RoPE Scaling 技术之一，能以极小的性能损失将模型的上下文窗口扩展数倍。
2.  **Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2017 - **AdamW**): 理解为什么 AdamW 优于传统的 Adam，尤其是在需要强正则化的 LLM 训练中。
3.  **PyTorch Lightning & DeepSpeed 官方文档**: 最佳的实践总是在官方文档中更新。特别是 Lightning 的 DeepSpeed 策略文档和 DeepSpeed 官网的配置生成器，是解决具体工程问题的首选。
4.  **Hugging Face Blog**: Hugging Face 的工程师和研究员经常发布关于 LLM 训练、量化、推理优化的高质量技术博客，内容非常贴近实践。
5.  **Lilian Weng's Blog "Lil'Log"**: OpenAI 研究员 Lilian Weng 的博客，对 LLM 相关的技术有系统性、深入浅出的梳理，是构建知识体系的绝佳材料。

---

## 本章小结
本附录章节是整个实战教程的“速查手册”与“知识图谱”。我们固化了核心的**符号系统**以保证交流的精确性，为从 3B 到 13B 的模型提供了可直接上手的**超参数基线**和设计理念，并展示了可插拔的 **YAML/JSON 配置模板**，旨在将开启新实验的“摩擦力”降至最低。最后，通过一份精心策划的**参考文献列表**，我们为希望深究其背后原理的读者铺平了道路，连接了从理论到实践的桥梁。将本章作为你日常训练工作中的常备参考，将帮助你更高效、更规范、更有信心地进行 LLM 的探索与创新。

## 常见陷阱与错误 (Gotchas)
1.  **超参的孤立调整**：最常见的错误是只修改一个参数而忽略其连锁反应。例如，将 `GB_tok` 减半，但忘记按 `sqrt` 法则相应地降低学习率 `η`，可能导致训练发散。**规则**：将超参视为一个相互关联的系统，特别是 `GB_tok`, `η`, `wd` 这“三巨头”。
2.  **Tokenizer 词表与数据不匹配**：在 CPT 阶段，使用基座模型的 tokenizer 处理一个包含大量新领域词汇（如代码、特定语言）的数据集，会导致大量的 `<unk>` token，严重损害模型学习新知识的能力。**技巧**：在 CPT 前，务必用新数据对基座 tokenizer 进行词表扩展。
3.  **RoPE Scaling 配置错误**：错误地设置 `rope_scaling` 的 `factor` 或 `original_max_position_embeddings`，或者在模型代码中未能正确应用 scaling 逻辑，会导致模型在长于原始上下文后性能急剧下降，甚至输出乱码。**调试**：编写一个单元测试，检查 scaling 后位置编码的插值或外推行为是否符合预期。
4.  **DeepSpeed 与 Lightning 的“双重管理”**：在 Lightning 的 `configure_optimizers` 中手动设置优化器和调度器，同时又在 DeepSpeed 的 JSON 文件中详细定义它们，可能会导致行为冲突或其中一方的配置被忽略。**最佳实践**：让 Lightning 作为“总指挥”，在 YAML/代码中定义优化器和调度器，DeepSpeed JSON 中的相关字段设为 `"auto"` 或仅用于指定类型（如 `PagedAdamW`）。
5.  **无声的 I/O 瓶颈**：训练看似正常运行，但通过 `nvidia-smi` 或 `nvitop` 观察到 GPU 利用率长期低于 80%。这通常不是计算问题，而是数据加载跟不上计算速度。**诊断**：使用 PyTorch Profiler 或 `cProfile` 分析数据加载 pipeline，检查 `num_workers` 设置、磁盘读取速度或网络（对于 CPFS）带宽是否达到瓶颈。

--- END OF FILE chapter15.md ---
