# 从零到可复现：LLM 训练实战（算法向，Lightning + DeepSpeed）—**索引**

> 目标：面向 **AI Scientist** 的中文公开教程，专注算法与训练策略（**非 infra 运维**），在 **PyTorch Lightning + DeepSpeed** 框架下，复现并扩展 **LLaMA 风格纯文本模型**（3B / 7B / 13B），上下文 **4k / 8k（RoPE scaling: PI / NTK-aware / YaRN）**，覆盖**从零预训练**与 **CPT（Continued Pre-Training）**两条路径。
> 规模假设：**64 × H100 80GB**，单次训练以**完整过 1T tokens** 为基准；数据存储为 **CPFS**；主要验证指标 **验证困惑度（perplexity）**。
> 金额一律以 **人民币（¥）** 计。

---

## 教程使用方式

* **阅读路径 A（从零预训练）**：`chapter01` → `chapter03` → `chapter05` → `chapter06` → `chapter07` → `chapter09` → `chapter11` → `chapter10`
* **阅读路径 B（CPT/继续预训练）**：`chapter01` → `chapter02` → `chapter04` → `chapter06` → `chapter08`  `chapter12` → `chapter10`
* **只想快速上手跑通 7B/13B**：`chapter01`（环境与复现性）→ `chapter09`（Lightning+DeepSpeed 配方）→ `chapter11/12`（端到端配置）

---

## 读者画像与不做什么

* **面向读者**：算法研究员 / 训练负责人 / 具备 PyTorch 分布式经验的工程科学家。
* **不覆盖**：大规模数据抓取与治理流水线、K8s 集群管理、作业调度系统、存储/网络调参与成本议价细节（仅做 **TCO 级粗估**）。

---

## 计算与软件基线（建议）

* **硬件**：64× H100 80GB（SXM/NVLink/NVSwitch），IB/400GbE；PUE 假设在成本章给公式与档位。
* **软件**：PyTorch（2.x）、PyTorch Lightning（2.x）、DeepSpeed（ZeRO/Paged Optimizer）、FlashAttention v2、fused RMSNorm/SwiGLU、fused RoPE。
* **数据**：CPFS（大文件小文件均可，训练侧采用 **shard + 流式**）。
* **模型**：LLaMA 风格 Decoder-only（RMSNorm、SwiGLU、RoPE）；上下文 4k / 8k（采用 RoPE scaling: PI / NTK-aware / YaRN）。

---

## 目录（文件结构）

* **index.md**（当前文件）
* **chapter01.md — 总览与可复现环境**

  * 复现实验要求与随机性控制（seed、CUDA/Determinism、NCCL）
  * 记号/符号约定（见下表）与单位统一（tokens、FLOPs、¥、kWh）
  * 训练日志、指标与检查点（Lightning Logger、zstd 压缩、断点续训策略）
* **chapter02.md — Tokenizer 与数据预处理（BPE 优化）**

  * 语料去重与分块概览（面向 CPT/预训练的“轻治理”）
  * **BPE 训练与优化**：vocab size、字符覆盖率、合并规则、数字与空白、特殊符号
  * 训练前 **chunking/packing** 策略（packed vs un-packed）、padding 与效率
  * 构建 **.idx/.bin** 或 **Parquet** token 数据、统计分布与长度直方图
* **chapter03.md — 架构细节：LLaMA 风格与 8k 扩展**

  * 模型结构与超参搜索空间（3B/7B/13B 的典型宽深比例、头数、head_dim）
  * **RoPE scaling**（PI / NTK-aware / YaRN）适用场景与权衡
  * **FlashAttention v2**、**fused RMSNorm/SwiGLU**、**fused RoPE** 的数值与吞吐影响
  * dropout、rmsnorm ε、init、残差配比与稳定性
* **chapter04.md — Scaling Laws 深入**

  * 从 Kaplan 到 **Chinchilla-style**：计算最优点推导（参数量 N、训练 tokens T 的闭式关系）
  * **噪声尺度定律（Noise Scale）** 与学习率/批大小关系
  * 2024 年对 Chinchilla 规律的刷新与实践建议（长上下文、分布偏移）
  * 结合 1T tokens 目标推导 3B/7B/13B 的合理 compute 预算
* **chapter05.md — 大批量训练与学习率策略**

  * **Global Batch（以 tokens 计）** 的定义与可接受区间
  * **LR scaling** 经验法则：**linear / sqrt** 两派及折中
  * 迭代步数（iters）↔ global batch size ↔ **总 tokens** 的约束关系
  * **LR schedule**：cosine、linear decay、one-cycle、warmup/cooldown
  * 梯度裁剪、weight decay、正则与稳定性观测
* **chapter06.md  多数据集动态混比**

  * 混比目标：困惑度收敛、领域泛化、分布鲁棒
  * **动态混比调度**：温度采样、重要性加权、配额上限/下限、阶段性 curriculum
  * 长度分桶与 **on-the-fly** rebalancing；CPT 中「新域优先」与「基座保真」
* **chapter07.md — 优化器与数值稳定**

  * **AdamW** 主线：β、ε、decoupled weight decay、fused AdamW
  * **Lion / Adafactor / DeepSpeed Paged AdamW** 对比与适用性
  * 梯度累积、混合精度（bf16 主推）、溢出监控与损失缩放
* **chapter08.md — 数据加载与存储格式（CPFS）**

  * **WebDataset（tar+IDX 流式）**、**Parquet（PyArrow）**、**Petastorm** 的取舍
  * 预取、pin memory、异步解码、长度感知的 dynamic packing
  * shard 切分策略、CPFS 并发 IO 与吞吐压测方法
* **chapter09.md — 并行与内存：Lightning + DeepSpeed 配方**

  * ZeRO 阶段选择、**Paged Optimizer**、offload 策略
  * **Tensor Parallel 切分维度**、（可选）Pipeline Parallel 的取舍
  * **Activation Checkpoint / 重计算** 的收益与开销
  * **梯度累积步数**、micro-batch、显存预算与稳定区间
  * tokens/s 的 **Lightning + DeepSpeed** 实战加速技巧与 Profiling
* **chapter10.md — 评估：验证困惑度**

  * PPL 的定义、滑动窗口评估（长上下文）与去偏
  * 分布外评估（held-out）、CPT 的前/后对比
  * 分布漂移与「过拟合数据混比」的诊断
* **chapter11.md — 端到端：从零预训练（1T tokens）**

  * 3B / 7B / 13B 三套「可跑通」基线配置（4k/8k 各一）
  * 推荐默认超参表、失败模式与排障 checklist
  * 训练日志样例解读：loss 曲线、噪声尺度估计、学习率热力图
* **chapter12.md — 端到端：CPT / 继续预训练**

  * 与 base 预训练的差异：混比、LR/schedule、冻结策略（是否不冻结嵌入）
  * 「域拉升」与「基座保持」的策略分层
* **chapter13.md — 成本/时长粗估（¥）与 TCO**

  * 理论 **FLOPs = 6·N_params·T_tokens**（decoder-only 近似）与**算力—时间**换算
  * **GPU 小时**、**电费（kWh, PUE）**、**云 vs 自建**（折旧/摊销）的 **TCO 公式**
  * 以 **64×H100**、1T tokens 为例的**可复用估算表格**（3B/7B/13B）
* **chapter14.md — 常见问题与诊断**

  * loss 爆炸/NaN、学习率不匹配、大 batch 不收敛
  * 长上下文退化、RoPE scaling 伪影、Tokenizer 泄漏
  * 数据加载瓶颈与 hot-spot shard
* **chapter15.md — 附录与参考**

  * 符号表、默认超参清单、YAML/JSON 模板
  * 参考论文与进一步阅读（Scaling laws、RoPE scaling、FlashAttention、Lion/Adafactor 等）

---

## 符号与单位（统一约定）

| 符号              | 含义                                 |
| --------------- | ---------------------------------- |
| `N_params`      | 模型非嵌入参数量（B = 10⁹ 级别）               |
| `T_tokens`      | 训练总 tokens（本教程默认 **1T**）           |
| `L_ctx`         | 上下文长度（4k / 8k）                     |
| `GB_tok`        | **Global batch（以 tokens 计）**       |
| `μ_tok`         | 单卡 micro-batch（以 tokens 计）         |
| `A`             | 梯度累积步数                             |
| `D/TP/PP`       | Data/Tensor/Pipeline 并行因子          |
| `η`             | 学习率（peak/base）                     |
| `β₁, β₂, ε, wd` | 优化器超参（AdamW 等）                     |
| `ρ`             | 噪声尺度（Noise Scale）                  |
| `FLOPs`         | 训练浮点量，近似 **`6·N_params·T_tokens`** |
| `PUE`           | 数据中心电能使用效率                         |
| `¥/GPU·h`       | GPU 小时成本（云/自建）                     |
| `kWh`           | 用电量计量单位                            |

---

## 成本/时长粗估：可插拔公式（详细见 `chapter13`）

> 在不承诺任何外部价格的前提下，提供**一键替换参数**的估框架（**单位：¥**）：

* **训练 compute 近似**：`FLOPs_train ≈ 6 · N_params · T_tokens`
* **有效吞吐（tokens/s）**：由 `chapter09` 的 tokens/s 实测或经验区间代入
* **总时长（s）**：`T_seconds = T_tokens / tokens_per_second`
* **GPU 小时**：`GPUh = (#GPUs) · T_seconds / 3600`
* **GPU 成本**：`¥_gpu = GPUh · (¥/GPU·h)`
* **电费估算**：

  * IT 功耗 `≈ 0.7 kW × #GPUs`（以 H100 80GB 典型训练功耗量级计，可按实际测量替换）
  * 总功耗 `= IT 功耗 × PUE`
  * `¥_power = (总功耗 kW) × (T_seconds/3600) × (¥/kWh)`
* **TCO 粗估**：`¥_total ≈ ¥_gpu + ¥_power + 其它（存储/网络/运维/摊销）`

> 我们在 `chapter13` 给出 **3B/7B/13B × 4k/8k** 的计算模板（表格），可直接替换 `tokens/s`、`¥/GPU·h`、`¥/kWh`、`PUE` 得出本地/云两套结果。

---

## 关键实践要点（贯穿全书）

1. **以困惑度为核心指标**：离线 dev/val 切片固定；长上下文使用滑窗 PPL。
2. **先定总 tokens，再配 N 与 batch**：遵循 **Chinchilla-style** 预算，结合噪声尺度调 `GB_tok` 与 LR。
3. **优先吞吐的前提下保稳定**：FlashAttn v2 + fused op + bf16；必要时激活重计算；观察梯度范数与 optimizer state 占比。
4. **动态混比而非“一盘端”**：阶段性调度 + 温度采样；CPT 中保持 base 分布的“守门人”比例。
5. **度量优先**：tokens/s、有效利用率（step time breakdown）、IO 饱和度、ZeRO bucket 命中率。
6. **配置可迁移**：统一 YAML/JSON schema（附录提供），3B/7B/13B 共用一套可缩放参数。

---

## 代码与资源结构（建议）

```
.
├── index.md
├── chapter01.md ... chapter15.md
├── configs/
│   ├── pretrain_3b_4k.yaml
│   ├── pretrain_7b_8k_ntk.yaml
│   ├── cpt_13b_mix schedule.yaml
│   └── deepspeed_zero{1,2,3}_paged.json
├── scripts/
│   ├── prepare_tokenizer_bpe.py
│   ├── build_token_dataset.py
│   ├── launch_trainer.py
│   └── eval_ppl.py
└── tools/
    ├── webdataset_reader.py
    ├── parquet_streamer.py
    └── noise_scale_estimator.py
```

> 每章给出 **最小可运行片段** 与 **配置对照表**，尽量做到「复制配置即可跑」。

---

## 许可与引用

* 教程文本建议以 **CC BY 4.0** 授权；示例代码建议 **Apache-2.0**（具体以仓库为准）。
* 引用与重用请注明来源：《从零到可复现：LLM 训练实战（Lightning+DeepSpeed）》。

---

## 反馈与贡献

* 欢迎在 `chapter14` 指出你遇到的异常日志/曲线与最小复现脚本。
* 贡献指南：PR 前先对齐 **数据合规** 与 **安全红线**（不提供可复现的敏感/侵权数据样本）。

---

### 附：章节速查（超短摘要）

* `01` 环境复现与符号；`02` Tokenizer/BPE & 打包；`03` 模型与 RoPE scaling；
* `04` Scaling laws（Chinchilla/Noise scale/2024 新进展）；`05` 大 batch 与 LR；
* `06` 数据集动态混比；`07` 优化器（AdamW/Lion/Adafactor/Paged AdamW）；
* `08` 数据加载（WebDataset/Petastorm/Parquet）；`09` 并行与内存（TP/ZeRO/CKPT）；
* `10` 验证 PPL；`11` 预训练端到端；`12` CPT 端到端；`13` 成本/时长（¥）；
* `15` 附录：符号、默认超参、模板与文献。

