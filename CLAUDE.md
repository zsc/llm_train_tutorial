（交流可以用英文，所有文档中文）

## 项目目标
编写一份《从零到可复现：LLM 训练实战（算法向，Lightning + DeepSpeed）》的中文课程markdown
文件组织是 index.md + chapter1.md + ...
不写代码

###
写一本中文公开教程 markdown讨论 LLM 模型的训练细节，目标读者是 ai scientist (重点在算法侧，不在 infra)。从原理到所有重要实操细节。底下的 infra 是基于 pytorch lightning deepspeed（预计 64x H100 80GB 规模，一次训练以过完 1T token 一遍为准）
计划覆盖的模型规模：3B / 7B / 13B。模型基本结构为纯文字 token llama（基本不改，但允许使用 RoPE scaling（PI/NTK-aware/YaRN） 来支持 8k）。目标上下文长度：4k / 8k。
包括从零预训练和 CPT 两种。给出成本/时长粗估（电费、GPU 小时、云/自建机房对比）。
讨论多数据集动态混比。假设数据集在 CPFS。
包含数据集的预处理成 token，和 token 的训练时动态加载。
主要看验证困惑度。
详细讨论 scaling law（需要到 Chinchilla-style 最优解的推导与噪声尺度定律，以及2024 年对 Chinchilla law 的刷新），large batch size(**Global batch（以 tokens 计）**的目标与可受区间. 希望讨论 large-batch scaling & LR scaling 的经验法则（linear / sqrt）),讨论 iter 数和 global batch size 关系， learning rate schedule，训练 token/s 的算法优化（ pytorch lightning deepspeed 下）。包含激活检查点 / 重计算、张量并行切分维度、梯度累积步数 的目标与约束。
AdamW 为主. 对比 Lion / Adafactor 或 Paged AdamW（DeepSpeed）.
讨论如何用 bpe 优化 tokenizer.
详细讨论数据加载（WebDataset / Petastorm / Parquet）偏好。
讨论FlashAttention v2、fused RMSNorm/SwiGLU、fused RoPE、Paged Optimizer 等.
教程由 index.md + chapter1.md + chapter2.md + ... 组成。
所有金额用人民币¥。

## 章节结构要求
每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **文字论述**：以文字论述为主，适当配上公式和 ASCII 图说明。如有数学公式，用 latex. 要有 rule-of-thumb。不写代码。
3. **本章小结**：总结关键概念和公式
4. **常见陷阱与错误** (Gotchas)：每章包含该主题的常见错误和调试技巧
