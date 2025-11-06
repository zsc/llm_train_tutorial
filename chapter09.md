# 第九章 — 并行与内存：Lightning + DeepSpeed 配方

## 开篇段落
欢迎来到本教程的“引擎室”。在前面的章节中，我们讨论了模型架构、数据和优化算法，而本章将把这些理论蓝图锻造成可在物理硬件上高速运转的钢铁现实。我们将直面大规模训练中最棘手的敌人——显存 OOM（Out-of-Memory），并学习如何驾驭 PyTorch Lightning 与 DeepSpeed 这对强大的组合，将一个 13B 甚至更大规模的模型，高效地部署在 64x H100 80GB 的庞大集群上。本章的目标不仅是提供一套“复制粘贴”的配置，更是要建立一个系统的决策框架，让您能够根据模型大小上下文长度和硬件拓扑，自信地设计、调试并优化出一套属于您自己的、极致性能的训练方案。

---

## 1. 显存预算：一场与“四大吞噬者”的博弈

要赢得显存之战，必先知己知彼。在一次训练迭代中，单张 GPU 的显存主要被四大块占用，理解它们的量级是后续所有优化的基础。

#### 1.1 定量分析显存占用
让我们以一个 **13B（`N_params` = 13 × 10⁹）** 的模型为例，进行一次定量计算：

1.  **模型权重（Model Parameters）**:
    *   `bf16` 格式下，每个参数占 2 字节。
    *   **显存占用**: `13B * 2 bytes/param = 26 GB`。这是最基础的开销。

2.  **梯度（Gradients）**:
    *   每个参数都需要一个对应的梯度，其数据类型通常与参数一致。
    *   **显存占用**: `13B * 2 bytes/param (bf16) = 26 GB`。

3.  **优化器状态（Optimizer States）**:
    *   这是最隐蔽也最庞大的“吞噬者”。标准的 AdamW 优化器为每参数维护两个动量状态：一阶动量（`exp_avg`）和二阶动量（`exp_avg_sq`）。
    *   为保持精度，这些状态通常以 `fp32`（4 字节）存储。
    *   **显存占用**: `13B * (4 bytes/state_1 + 4 bytes/state_2) = 13B * 8 bytes = 104 GB`。
    *   **结论**：仅优化器状态一项，就远超单张 H100 80GB 的容量。**这是我们必须使用 ZeRO 的根本原因。**

4.  **激活值（Activations）**:
    *   这是前向传播中产生的中间结果，其大小与多个因素成正比，近似公式为：
        `Mem_act ≈ L_ctx * μ_bsz * N_layers * D_hidden * (constant)`
        其中 `μ_bsz` 是单卡 micro-batch size。
    *   对于长上下文（如 8k）和多层深层模型，激活值可以轻松达到数十 GB，成为新的显存瓶颈。例如，对于一个 13B 模型（`L=40, H=5120`）在 `L_ctx=8k`，`μ_bsz=1` 的情况下，激活值可能达到 `8192 * 1 * 40 * 5120 * 10 (approx factor) ≈ 16 GB`。这个数字会随着 `μ_bsz` 线性增长。

**小结**：对于 13B 模型，在 `bf16` 下，一次迭代的理论显存峰值（未优化）至少为 `26 (weights) + 26 (grads) + 104 (optim) + Activations ≈ 156 GB + Activations`。这清晰地表明，不采用高级并行与内存优化技术，训练是完全不可行的。

#### 1.2 并行策略的“三驾马车”

| 策略                | 核心思想                                 | 优点                                       | 缺点                                                         | 适用场景                                     |
| ------------------- | ---------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| **数据并行 (DP)**   | 每卡复制完整模型，切分数据               | 实现简单，逻辑清晰                         | 显存冗余巨大，无法容纳大模型                                 | 小型或显存极其充裕的场景（几乎不再单独使用） |
| **张量并行 (TP)**   | 在卡间切分单个算子（如矩阵乘法）         | 降低单卡峰值显存（权重+激活），提高计算并行度 | 通信密集（高带宽、低延迟要求），需修改模型代码，通常限于节点内 | 模型层过大，单卡无法容纳其权重或中间激活值     |
| **流水线并行 (PP)** | 在卡间切分模型层（`stage`）              | 可扩展至巨大模型，通信量相对 TP 较少       | 存在“流水线气泡”导致硬件利用率下降，实现复杂，负载均衡困难   | 超巨型模型（>100B），或跨节点并行            |

而我们将要深入的 **DeepSpeed ZeRO**，可以看作是数据并行的“究极进化版”，它系统性地解决了 DP 的显存冗余问题。

---

## 2. DeepSpeed ZeRO：数据并行的终极形态

ZeRO（Zero Redundancy Optimizer）的核心哲学是：**在数据并行的计算流程中，将一切可以分割的状态（优化器、梯度、权重）都进行分割（Partition），只在计算需要时才通过通信临时重组**。

#### 2.1 ZeRO 的三个阶段

*   **ZeRO-Stage 1**: **分割优化器状态**
    *   **机制**: `N` 个 GPU，每个 GPU 只持有 `1/N` 的优化器状态。在 `optimizer.step()` 时，每个 GPU 通过 `All-Gather` 收集齐自己负责更新的那部分参数所对应的**完整**优化器状态，执行更新，然后丢弃。
    *   **显存节省**: 优化器状态从 `O` 减少到 `O/N`。通信量相比标准 DP 几乎不变。
    *   **适用**: 能显著降低显存，但模型权重和梯度仍在每卡复制，对于 13B 模型仍不足够。

*   **ZeRO-Stage 2**: **分割优化器状态 + 梯度**
    *   **机制**: 在 Stage 1 基础上，反向传播计算出的梯度也进行分割。每个 GPU 只保留自己负责那部分参数的梯度。传统 DP 的 `All-Reduce` 被 `Reduce-Scatter` 替代，即计算全局梯度总和后直接分发到对应的 GPU，避了每卡存储完整梯度。
    *   **显存节省**: 优化器状态和梯度都减少到 `1/N`。
    *   **适用**: 进一步节省显存，通常是 7B 以下模型在显存较紧张（如 A100 40GB）时的不错选择。

*   **ZeRO-Stage 3**: **分割优化器状态 + 梯度 + 模型权重**
    *   **机制**: 这是最彻底的模式。在任何时刻，每个 GPU 只持久化存储 `1/N` 的模型权重。
    *   **训练步骤剖析**:
        1.  **前向传播**: 在计算某一层（如 `TransformerBlock_i`）之前，所有 GPU 通过 `All-Gather` 临时获取该层的完整权重。
        2.  计算完成后，**立即释放**非自己持有的那部分权重，只保留自己的 `1/N` 分片。
        3.  **后向传播**: 流程类似，需要梯度时，先 `All-Gather` 对应层的权重。
    *   **显存节省**: 权重、梯度、优化器状态都被分割。理论上，只要集群总显存足够，可以训练任意大小的模型。

---
**ASCII 图：ZeRO-3 中一个数分片（slice）的生命周期**
```
                           +--------------------------------+
                           |           GPU_k (owns P_k)     |
                           +--------------------------------+
                                           |
(Forward Pass for Layer_i) --> All-Gather(P_i) from all GPUs --> [P_i_full] temporary
                                           |
                                  Compute(X, P_i_full)
                                           |
                           Release(P_i_full), keep only [P_k]
                                           |
(Backward Pass for Layer_i) -> All-Gather(P_i) from all GPUs --> [P_i_full] temporary
                                           |
                           Compute Grads, Reduce-Scatter(G_i) --> [G_k] stored
                                           |
(Optimizer Step) ---------> Update(P_k, G_k, OptimState_k) --> [P_k_updated] stored
```
---

**Rule-of-thumb (经验法则):**
> 对于我们的 64x H100 80GB 场景，训练 7B 及以上模型，**直接将 ZeRO-3 设为默认配置**。它提供了最大的显存优化能力，让我们能将宝贵的显存用于提升 `μ_tok` 或 `L_ctx`，从而压榨硬件性能，提升最终的 `tokens/s`。

#### 2.2 CPU/NVMe Offload：最后的救生筏（但需谨慎）

*   **Paged Optimizer & Offload**: DeepSpeed 允许将 ZeRO 分割后的状态进一步“卸载”到 CPU 内存（Offload）或 NVMe 硬盘。
*   **性能警示**: 这是一个典型的用**时间换空间**的策略，但代价极为高昂。
    *   **带宽阶梯**: H100 内部 NVLink/NVSwitch (`~900 GB/s`) >> PCIe 5.0 (`~128 GB/s`) >> 高速 NVMe (`~14 GB/s`)。
    *   将数据从 GPU 卸载到 CPU，意味着通信带宽下降了一个数量级；卸载到 NVMe 则下降了两个数量级。这会使训练从**计算密集型**转变为**I/O 密集型**，`tokens/s` 会急剧下降。
> **Rule-of-thumb**: 在 H100 集群上，**应尽一切可能避免使用 Offload**。宁可减小 micro-batch size，或增加 TP 因子，也不要轻易打开 Offload。它只适用于研究或调试目的，在追求极致性能的生产环境中几乎没有位置。

---

## 3. 张量并行（Tensor Parallelism）：在微观层面瓦解巨型算子

即便 ZeRO-3 将模型参数分片存储，但在前向/后向计算的瞬间，单层完整的权重仍需被 `All-Gather` 到每个 GPU。如果模型极宽（如 MoE 的 FFN 层），这一瞬间的显存峰值也可能导致 OOM。**张量并行 (TP)** 正是解决这一问题的利器。

它将一个大的矩阵运算（如 `Y = XA`）分解到多个 GPU 上。
*   **列并行 (Column-Parallel Linear)**:
    *   将权重矩阵 `A` 按列切分 `A = [A_1 | A_2]`, `TP=2`。
    *   `GPU_1` 计算 `Y_1 = XA_1`，`GPU_2` 计算 `Y_2 = XA_2`。
    *   前向传播需要一次 `All-Gather` 操作将 `X` 广播给所有 TP-group 内的 GPU。
    *   输出 `Y = [Y_1 | Y_2]` 是天然分片的。
    *   **适用**: Transformer 中的 **QKV 投影**和 **FFN 的第一层（up-projection）**。

*   **行并行 (Row-Parallel Linear)**:
    *   将权重矩阵 `A` 按行切分 `A = [A_1; A_2]`。
    *   输入 `X` 本身是分片的 `X = [X_1 | X_2]`（来自前一个列并行层）。
    *   `GPU_1` 计算 `Z_1 = X_1 * A_1`，`GPU_2` 计算 `Z_2 = X_2 * A_2`。
    *   后向传播需要一次 `All-Gather`。
    *   输出需要一次 `All-Reduce` 将部分结果相加：`Y = Z_1 + Z_2`。
    *   **适用**: Transformer 中的 **注意力输出投影** 和 **FFN 的第二层（down-projection）**。

**Rule-of-thumb:**
> *   **TP 规模与拓扑**: TP 引入了大量的即时通信，必须在**高速互联（NVLink/NVSwitch）的节点内部**使用。对于 8x H100 的节点，`TP` 因子通常设为 `2`, `4`, 或 `8`。跨节点 TP 性能会急剧下降。
> *   **模型适配**: 使用 TP 需要模型代码的明确支持（例如，使用 `megatron-core` 提供的 `RowParallelLinear` 和 `ColumnParallelLinear` 替换 `nn.Linear`）。
> *   **何选择**:
    *   **7B 模型**: `TP=1` (不用) 或 `TP=2` 通常足够。
    *   **13B 模型**: `TP=2` 或 `TP=4` 是一个很好的起点。`TP=4` 意味着一个 13B 模型的单层在单卡上看起来更像一个 ~3B 模型的层，显著降低了激活值和临时权重的峰值显存。
    *   **权衡**: 增加 TP 因子会降低单卡显存峰值，但也增加了通信开销。当 `tokens/s` 随着 TP 增加而下降时，说明通信开销已经超过了显存优化带来的收益。

---

## 4. 激活检查点（Activation Checkpointing）：用算力换取“无限”上下文

长序列训练中，激活值是显存的主要消耗者。**激活检查点（AC）**，或称**重计算（Recomputation）**，是一种优雅的“时间换空间”策略。

*   **原理**: 在前向传播时，对于被“检查点”包裹的模块（通常是每个 Transformer Block），我们不存储其内部产生的所有中间激活值，**只保存该模块的输入**。当反向传播需要这些活值来计算梯度时，我们**临时重新执行一次该模块的前向计算**，得到所需的激活值，用完后立即丢弃。

*   **开销与收益分析**:
    *   **收益**: 显存占用从与层数 `N_layers` 线性相关，变为近似 `O(1)`（只存一个 block 的激活），从而可以支持极长的序列。
    *   **开销**:
        *   一次标准的训练迭代包含 `1F` (前向) + `1B` (后向)。
        *   使用 AC 后，变为 `1F` + `1F_recompute` + `1B`。
        *   粗略估计，后向传播的计算量约是前向的 2 倍 (`B ≈ 2F`)。
        *   因此，理论计算开销增加了 `F_recompute / (F + B) ≈ F / (3F) ≈ 33%`。
        *   然而，这是一个值得的交换。如果不用 AC 导致 `μ_bsz` 只能设为 1，而用了 AC 后可以设为 4，那么整体的 `tokens/s` 反而会大幅提升，因为硬件并行度得到了更好的利用。

**Rule-of-thumb:**
> *   对于 `L_ctx >= 4k` 的任何训练任务，**无条件开启激活检查点**。这是现代 LLM 长序列训练的标配。
> *   在 PyTorch Lightning 中，通过 `strategy=DeepSpeedStrategy(activation_checkpointing={"partition_activations": True, "cpu_checkpointing": False})` 或直接在模型代码中使用 `torch.utils.checkpoint.checkpoint` 即可轻松启用。
> *   DeepSpeed 的 `partition_activations` 选项可以与 ZeRO-3 配合，进一步将 AC 重计算出的激活值在 DP group 内进行分割，再次降低显存。

---

## 5. 梯度累积（Gradient Accumulation）：用时间堆砌出“虚拟大批量”

Scaling Laws 指导我们需要使用百万级别的 `GB_tok`（全局批大小）。但单次迭代能放入集群总显存的数据量是有限的。**梯度累积（Gradient Accumulation Steps, GAS）** 是弥合这一差距的桥梁。

*   **机制**:
    1.  执行一次 micro-batch 的 `forward()` 和 `backward()`。
    2.  梯度被计算并存储在 `.grad` 属性中。
    3.  **跳过** `optimizer.step()` 和 `optimizer.zero_grad()`。
    4.  重复步骤 1-2 `A` 次，新计算的梯度会**累加**到已有的梯度上。
    5.  在第 `A` 次 micro-batch 之后，执行一次 `optimizer.step()`（使用累加了 `A` 次的梯度），然后 `optimizer.zero_grad()` 清零。

---
**ASCII 图：梯度累积流程**
```
Global_Step k:
  Micro_Step 1: Fwd(data_1) -> Bwd() -> Grad_Buffer += grad_1
  Micro_Step 2: Fwd(data_2) -> Bwd() -> Grad_Buffer += grad_2
  ...
  Micro_Step A: Fwd(data_A) -> Bwd() -> Grad_Buffer += grad_A
  --> optimizer.step()  // A single update using summed gradients
  --> optimizer.zero_grad()
Global_Step k+1:
  ...
```
---

*   **核心关系式**:
    `GB_tok = μ_tok × A × D`
    *   `GB_tok`: 我们的目标全局批大小（以 tokens 计），由 Scaling Law 决定。
    *   `μ_tok`: 单卡上的 micro-batch（`μ_bsz * L_ctx`），由单卡显存容量决定。
    *   `A`: 梯度累积步数，是我们需要计算的“自由”变量。
    *   `D`: 数据并行路数（`#GPUs / (TP * PP)`）。

**Rule-of-thumb:**
> 1.  **最大化 `μ_tok`**: 你的首要任务是，在给定模型、`L_ctx` 和并行策略（ZeRO, TP）下，通过实验找到能塞满 H100 显存（例如，用到 70-75GB）的**最大 `μ_bsz`**。更高的 `μ_bsz` 意味着更好的 GPU 计算核心利用率（MFU）。
> 2.  **计算 `A`**: 根据你的目标 `GB_tok`（例如 4M tokens），用上述公式反解出 `A`： `A = GB_tok / (μ_tok * D)`。`A` 的值取一个合理的整数，如 8, 16, 32。

---

## 6. Lightning + DeepSpeed 实战配方与 Profiling 工作流

现在，我们将所有组件整合成一个为 64x H100 集群训练 13B LLaMA（8k 上下文）模型的决策工作流。

**目标**: `GB_tok = 4M tokens`, 模型 `13B`, `L_ctx = 8k`。

1.  **Step 1: 基础并行策略选择 (ZeRO-3 + TP)**
    *   **ZeRO**: 毋庸置疑，选择 `deepspeed_stage_3`。
    *   **TP**: 在 8-GPU 节点内，`TP=4` 是一个攻守兼备的选择，显著降低单层计算的显存压力。
    *   **DP**: 数据并行路数 `D = #GPUs / TP = 64 / 4 = 16`。

2.  **Step 2: 启用标准内存优化**
    *   **AC**: 8k 上下文，必须启用激活检查点。
    *   **BF16**: H100 原生支持，启用 `precision="bf16-mixed"`。

3.  **Step 3: 经验性确定 Micro-Batch Size (`μ_bsz`)**
    *   这是一个**经验性搜索**过程。在单节点（8卡，TP=4）上开始实验。
    *   尝试 `μ_bsz = 1`。如果能跑通且显存占用率低（如 < 60%），尝试 `μ_bsz = 2`。
    *   持续增加 `μ_bsz`，直到发生 OOM，然后回退到上一个成功的值。
    *   假设我们最终确定 `μ_bsz = 2` 是稳定的最大值。
    *   那么，`μ_tok = μ_bsz * L_ctx = 2 * 8192 = 16,384` tokens。

4.  **Step 4: 计算梯度累积步数 (`A`)**
    *   `A = GB_tok / (μ_tok * D) = 4,194,304 / (16,384 * 16) = 16`。

5.  **Step 5: 组装 DeepSpeed & Lightning 配置**
    *   **Lightning Trainer**:
        ```python
        trainer = Trainer(
            accelerator="gpu",
            devices=64,
            num_nodes=8,
            precision="bf16-mixed",
            strategy=DeepSpeedStrategy(
                stage=3,
                tensor_parallel={"tp_size": 4},
                activation_checkpointing={"partition_activations": True},
                # ... 其他deepspeed配置
            ),
            accumulate_grad_batches=16,
        )
        ```
    *   **DeepSpeed JSON (部分)**:
        ```json
        {
          "train_micro_batch_size_per_gpu": 2,
          "bf16": {"enabled": true},
          "zero_optimization": {
            "stage": 3,
            "contiguous_gradients": true,
            "overlap_comm": true,
            "reduce_bucket_size": 5e8
          }
        }
        ```

6.  **Step 6: Profile, 分析, 迭代**
    *   启动训练后，不要只看 Loss。使用 Lightning 的 `AdvancedProfiler` 或 NVIDIA `Nsight Systems` (nsys) 进行性能剖析。
    *   **关键监控指标**:
        *   **tokens/sec/GPU**: 这是最终的效率黄金指标。
        *   **TFLOPS/GPU**: 衡量 GPU 的计算单元被利用的程度。对于 H100 `bf16`，一个好的目标是达到理论峰值（989 TFLOPS）的 50% 以上，即 ~500 TFLOPS。
        *   **Step Time Breakdown**: 分析`forward`, `backward`, `optimizer_step` 的耗时。如果 `optimizer_step` 耗时过长，可能表示 ZeRO 的通信或梯度同步存在瓶颈。如果 step 之间有很大空闲，说明是数据加载（`chapter08`）问题。
        *   **Comm/Compute Overlap**: 理想情况下，ZeRO 的通信 (`overlap_comm`) 应该与计算并行，profiler 可以验证这一点。

---

## 本章小结
本章我们构建了一套系统性的方法论，用于在 Lightning+DeepSpeed 框架下配置大规模训练。
*   我们从**定量分析显存占用**入手，明确了使用高级并行策略的必要性。
*   **ZeRO-3** 被确立为大规模训练的基石，因为它能最大化地消除显存冗余。
*   **张量并行（TP）** 作为 ZeRO-3 的补充，用于解决单层算子过大导致的瞬时显存峰值。
*   **激活检查点（AC）** 和 **梯度累积（GAS）** 是分别应对长序列和巨大全局批次挑战的标准武器。
*   最后，我们将这些技术整合进一个**六步决策工作流**，从策略选择到参数计算，再到性能剖析，形成一个完整的闭环优化流程。掌握这套流程，你将能为任何规模的模型和硬件集群，设计出高效、稳健的训练方案。

---

## 常见陷阱与错误 (Gotchas)
1.  **ZeRO-3 的“假死”与检查点 OOM**
    *   **症状**: 在保存模型检查点时（尤其是在训练结束或验证步骤），rank 0 的 GPU 突然 OOM，整个任务失败。
    *   **原因**: 标准的 `torch.save(model.state_dict())` 会尝试在 rank 0 上聚合完整的模型权重，这会瞬间摧毁 ZeRO-3 的显存优势。
    *   **解决方案**: **严格使用 DeepSpeed 的检查点保存/加载 API**。在 Lightning 中，这是自动处理的。如果需要导出为单个文件，确保在 DeepSpeed 配置中 `zero_optimization` 设置 `"stage3_gather_16bit_weights_on_model_save": true`，并确保 CPU 有足够内存。

2.  **梯度累积与学习率调度器的“步数错位”**
    *   **症状**: Loss 曲线过早地、异常迅速地收敛到一个次优点，或者完全不收敛。
    *   **原因**: 学习率调度器（如 Cosine Annealing）是按“优化器更新次数”来计算进度的。如果错误地让它按“micro-batch 次数”更新，它的周期会缩短 `A` 倍，导致学习率在训练早期就衰减到零。
    *   **解决方案**: PyTorch Lightning 内部正确处理了这一点，其 `trainer.fit()` 循环会自动以优化器步数（global step）为单位。如果你手动控制循环，务必确保 LR scheduler 的 `step()` 是在 `optimizer.step()` 之后调用的，而不是在每次 `backward()` 之后。

3.  **TP 因子与模型配置的“整除灾难”**
    *   **症状**: 训练启动时直接报错，信息通常包含 `AssertionError` 或 `ValueError`，出某个维度（如 `num_attention_heads`）无法被 `tp_size` 整除。
    *   **解决方案**: 在设计模型和并行策略时，就确保可并行化的维度是 TP 因子的整数倍。例如，如果计划使用 `TP=4`，那么注意力头数最好是 32, 40, 64 等可以被 4 整除的数。

4.  **混合精度训练中的 `inf`/`NaN` 梯度**
    *   **症状**: 训练过程中 Loss 突然变为 `NaN`。
    *   **原因**: 在 `bf16` 或 `fp16` 下，梯度的数值范围变小，可能出现下溢（变为 0）或上溢（变为 `inf`）。
    *   **解决方案**:
        *   **梯度裁剪 (Gradient Clipping)**: 在 Lightning Trainer 中设置 `gradient_clip_val=1.0` 是一个稳健的默认值。
        *   **损失缩放 (Loss Scaling)**: DeepSpeed 会自动处理动态损失缩放，但如果问题持续，可以检查其配置，确保 `loss_scaling` 相关参数是合理的。
        *   检查数据或模型初始化中是否存在数值不稳定的源头。

5.  **“沉默的颈”：GPU 利用率不足**
    *   **症状**: 训练可以正常运行，没有错误，但 `tokens/s` 远低于预期，通过 `nvidia-smi` 或 `dcgm` 观察到 GPU SM 利用率（GPU-Util）周期性地跌落到很低的值。
    *   **原因**: 这通常不是计算问题，而是**数据供给**或**CPU 瓶颈**。可能是数据加载太慢、预处理太耗时，或者是 ZeRO-3 的某些 CPU 端操作成为了瓶颈。
    *   **调试技巧**: 使用 profiler 确认 GPU 是否在等待数据。如果是，请回到 `chapter08` 优化数据加载流程。检查主进程的 CPU 使用率，如果某个核心持续 100%，可能存在瓶颈。
