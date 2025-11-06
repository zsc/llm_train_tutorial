# 第 8 章：数据加载与存储格式（CPFS）— 榨干 IO 吞吐

## 开篇段落

在 64x H100 这种规模的训练集群中，每一秒的闲置都意味着巨大的成本浪费。当模型优化、并行策略都已到位后，数据加载管道（Data Pipeline）——这个从海量存储（CPFS）到 GPU 显存的漫长旅程——便从幕后走向台前，成为决定训练效率（`tokens/s`）和硬件利用率（MFU）的终极瓶颈。一个设计不良的数据加载器，即使面对理论带宽惊人的 CPFS，也足以让价值数千万的 H100 集群陷入“计算五分钟，等待两小时”的窘境。本章的目标是解构并重塑这一关键路径，我们将深入对比 **WebDataset**、**Parquet** 和 **Petastorm** 在 LLM 预训练场景下的性能与取舍，并系统性地阐述一套优化“组合拳”，涵盖从预取、内存锁定到动态打包的每一环。最终，我们将提供在 CPFS 环境下可落地的分片策略与吞吐压测方法，确保数据流能像高压油管一样，持续、稳定地为 H100 这台性能猛兽注入燃料。

## 文字论述

### 2.1 核心矛盾：计算密集型 vs. IO 密集型

衡量 LLM 训练效率的黄金指标是 **模型 FLOPs 利用率（Model FLOPs Utilization, MFU）**，它直接反映了 GPU 硬件的有效计算时间占比。一个理想的训练任务是 **计算密集型（Compute-Bound）** 的，即 MFU 接近理论上限，GPU 核心（Tensor Core）始终在进行高强度的矩阵运算。然而，当数据供给速度跟不上计算消耗速度时，训练就会退化为 **IO 密集型（IO-Bound）**。

我们可以将一个训练步（step）的时间分解如下：
`T_step = T_data_wait + T_h2d_copy + T_compute + T_sync`

*   `T_data_wait`: 主进程等待 `DataLoader` 提供下一个 batch 的时间。
*   `T_h2d_copy`: 数据从 CPU 内存（Host）拷贝到 GPU 显存（Device）的时间。
*   `T_compute`: GPU 执行前向、后向传播和优化器更新的纯计算时间。
*   `T_sync`: 多节点/多卡间的梯度同步、集合通信时间。

我们的核心目标是，通过高效的数据加载策略，将 `T_data_wait` 压缩至接近零，并通过异步化手段将 `T_h2d_copy` 与 `T_compute` 高度重叠，从而让 `T_step ≈ T_compute + T_sync`。

让我们量化一下数据供给的压力。假设我们训练一个 7B 模型，Global Batch Size 为 4M tokens (`GB_tok`)，集群为 64xH100。在理想的 MFU 下（例如 50%），整个集群的理论计算吞吐可达 `64 * 2000 TFLOPs/s * 50% = 64 PFLOPs/s`。根据 `FLOPs ≈ 6 * N * T` 的估算，处理一个 token 约需 `6 * 7e9 = 42 GFLOPs`。因此，集群每秒需要消耗 `64e15 / 42e9 ≈ 1.5 M tokens`。如果使用 `bf16`每个 token 占 2 字节，这意味着整个集群的数据供给系统必须持续提供 `1.5M tokens/s * 2 bytes/token ≈ 3 MB/s` 的**有效数据流**。这个数字看似不大，但它要求在**每个训练 step 内，以微秒级的延迟，为所有 512 个 GPU Worker (64 nodes * 8 GPUs/node) 精准、同步地提供数据**。任何一个环节的抖动都会导致整个 global batch 的等待。

```ascii
+-------------+      +---------+      +----------------+      +---------+      +-------------+
| CPFS        |----->| Network |----->| Node RAM (Cache)|----->|   PCIe  |----->| GPU VRAM    |
| (TB/PB data)|      | (IB/RoCE)|      | (Pinned Memory)|      |  (DMA)  |      | (CUDA Tensors)|
+-------------+      +---------+      +----------------+      +---------+      +-------------+
      ^                  ^                    ^                    ^                  ^
      |                  |                    |                    |                  |
   元数据延迟          网络拥           CPU Dataloader         PCIe带宽         计算消耗
   (瓶颈 A)            (瓶颈 B)             (瓶颈 C)             (瓶颈 D)
```

我们的优化工作就是逐一攻克 A、B、C、D 四个潜在瓶颈点。

### 2.2 主流数据格式深度对比

选择正确的底层存储格式，是所有优化的起点。它决定了我们如何与瓶颈 A 和 C 进行博弈。

#### 2.2.1 WebDataset（tar+IDX 流式）：简单粗暴的性能王者

*   **核心机制**：WebDataset 将大量小文件（如每个样本一个 `tokens.bin`）聚合成一个或多个大的 `.tar` 归档文件（shards）。`DataLoader` 直接以流的方式读取 `.tar` 文件，按顺序解析出内部的样本。这巧妙地将对文件系统的数百万次小文件随机 I/O 操作，转化为了对数百个大文件的顺序 I/O，完美契合了 CPFS 这类并行文件系统为大文件顺序读优化的设计哲学。

    ```ascii
    一个 .tar shard (e.g., shard-00001.tar):
    +----------------------+
    | sample_001.bin Hdr   |
    +----------------------+
    | sample_001.bin Data  |  <-- Dataloader 流式读取
    +----------------------+
    | sample_002.bin Hdr   |
    +----------------------+
    | sample_002.bin Data  |
    +----------------------+
    | ...                  |
    +----------------------+
    ```

*   **优点**：
    1.  **极致的 IO 性能**：几乎没有解析开销。读取 `.tar` 流和直接读取原始二进制文件流的性能非常接近。
    2.  **流式处理（Streaming）**：对内存极其友好。一个 worker 只需在内存中保留当前正在处理的样本，无需加载整个数据集的索引。这对于动辄数十 TB 的数据集至关重要。
    3.  **天然契合分布式**：每个计算节点/worker 可以被分配不同的 shard 集合进行处理，节点间无需通信，扩展性极佳。
    4.  **生态简洁**：创建和读取仅需标准库或 `webdataset` 这个轻量级库。`torchdata` 虽提供了更复杂的 DataPipes API，其核心思想与 WebDataset 一致。

*   **缺点**：
    1.  **随机访问困难**：虽然可以通过外部索引（`.idx`）实现 shard 级别的跳转，但要精确跳转到 `.tar` 内部的某个样本则非常低效。但这在预训练场景中几乎不是问题，因为我们总是顺序消费数据。
    2.  **数据更新不便**：修改单个样本需要重写整个 shard。这反而强化了其作为“一次写入，多次读取”的不可变训练数据源的定位。

*   **Rule-of-thumb**：对于从零开始、以最大吞吐为目标的 LLM 预训练，**WebDataset 是无可争议的 SOTA (State-of-the-Art) 选择**。

#### 2.2.2 Parquet（PyArrow）：大数据生态的瑞士军刀

*   **核心机制**：Parquet 是一种列式存储格式。与按行存储（如 JSON Lines）不同，它将同一列的数据连续存储在一起。这使得只读取部分列的查询操作极其高效。

    ```ascii
    行式存储 (JSON):
    {"input_ids": [...], "source": "web"}
    {"input_ids": [...], "source": "book"}

    列式存储 (Parquet):
    column(input_ids): [[...], [...], ...]
    column(source):    ["web", "book", ...]
    ```

*   **优点**：
    1.  **分析友好**：在数据预处理和分析阶段，可以极快地对某个元数据字段（如 `source`）进行统计或过滤，而无需读取庞大的 `input_ids` 列。
    2.  **Schema 强制与压缩**：自带严格的 schema，保证数据一致性。其内置的字典编码、行程长度编码（RLE）等对低基数（low-cardinality）的列有很好的压缩效果。
    3.  **生态集成**：与 Apache Spark, Dask, Pandas 等大数据处理框架无缝集成，是数据仓库的事实标准之一。

*   **缺点**：
    1.  **训练时优势不明显**：LLM 训练时，我们通常需要读取 `input_ids` 这一整“行”数据，列式存储的优势无法发挥。反而，重组行为列的过程会引入微小的 CPU 开销。
    2.  **读取库依赖**：需要 `pyarrow` 或 `fastparquet` 这样的库，相比 `tarfile` 更重。

*   **Rule-of-thumb**：如果你的数据ETL（Extract, Transform, Load）流水线已经深度绑定 Spark 且产物就是 Parquet，并且重导出为 WebDataset 的成本很高，那么直接基于 `PyArrow` 读取 Parquet 是一个完全可行且性能不错的次优选择。

#### 2.2.3 Petastorm：为机器学习而生的 Parquet 封装

*   **核心机制**：Petastorm 可以看作是 Parquet 之上的一层智能加载框架。它封装了 Parquet 的读写，并提供了专为分布式机器学习设计的高级 API。
*   **优点**：
    1.  **ML 特性**：原生支持多 worker 数据分片、带缓存的 shuffling、N-grams/序列特征生成等复杂采样策略。
    2.  **谓词下推（Predicate Pushdown）**：允许在读取数据时进行过滤，只加载符合条件的行组（row groups），这对细粒度的数据筛选场景很有用。
*   **缺点**：
    1.  **过度设计（Over-engineered）**：对于 LLM 预训练这种“顺消费所有数据”的简单场景，Petastorm 的许多高级功能都用不上，反而增加了系统的复杂度和潜在的调试难度。
    2.  **性能开销**：抽象层不可避免地会带来一些性能开销，虽然通常不大，但在追求极致吞吐的场景下，任何开销都应被审视。

**决策矩阵与最终推荐**

| 特性 | WebDataset | Parquet (PyArrow) | Petastorm |
| :--- | :--- | :--- | :--- |
| **访问模式** | **顺序流式 (Sequential Streaming)** | 列式 (Columnar) | 基于 Parquet 的 ML API |
| **核心优势** | **IO吞吐最大化，系统开销最小** | 数据分析与ETL生态集成 | 复杂的采样与过滤策略 |
| **系统复杂度** | **极低** | 中等 | 高 |
| **适用场景** | **大规模、顺序、不可变数据预训练** | 数据分析与训练一体化工作流 | 推荐系统、多模态、表格数据 |
| **本教程推荐度**| ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**结论**：在 64x H100 预训练的背景下，**坚决择 WebDataset**。它的设计哲学与我们的目标——最大化顺序读吞吐——完美契合。

### 2.3 数据加载优化“组合拳”：榨干每个 CPU 周期

选定 WebDataset 后，真正的优化工作发生在 `torch.utils.data.DataLoader` 的配置和数据处理逻辑中。

1.  **多进程预取 (`num_workers` & `prefetch_factor`)**
    *   `num_workers > 0`：这是最重要的参数。它会启动多个子进程并行地加载数据。主训练进程只需从一个共享队列中取走已经准备好的 batch。
    *   **Rule-of-thumb**：`num_workers` 的最优值通常需要实验寻找，一个好的起始点是 **节点 CPU 核心数的一半**。例如，一个 96 核的 CPU 节点，可以从 `num_workers=48` 开始测试。过低会导致 CPU 瓶颈，过高则会因为进程间切换和资源竞争导致性能下降。
    *   `prefetch_factor`：该参数控制每个 worker 提前加载多少个 batch。`prefetch_factor=2` 意味着每个 worker 会维护一个大小 2 的预取队列，能更好地平滑单次数据加载的耗时波动。

2.  **固定内存与异步拷贝 (`pin_memory` & `non_blocking`)**
    *   `pin_memory=True`：这个简单的 `True` 值背后，是性能优化的关键一环。默认情况下，CPU 创建的 Tensor 位于**可分页内存（pageable memory）**中。为了将数据传输到 GPU，CUDA 驱动必须先将其拷贝到一个临时的**固定内存（pinned memory）**缓冲区，因为 GPU 的 DMA 引擎要求源内存地址在传输期间保持物理位置不变。设置 `pin_memory=True` 会让 `DataLoader` 直接在固定内存中创建 Tensor，省去了这一次冗余的内部拷贝，从而显著加快 Host-to-Device 的传输速度。
    *   `tensor.to(device, non_blocking=True)`：当与 `pin_memory=True` 配合使用时，`non_blocking=True` 使得 H2D 拷贝成为一个异步操作。CPU 发起拷贝指令后，无需等待拷贝完成就可以继续执行下一行代码（例如，准备下一个 batch 的计算）。这使得数据传输和 GPU 计算得以高效地流水线化并行。

    ```python
    # 正确的异步加载与计算流水线
    loader = DataLoader(..., pin_memory=True)
    for batch in loader:
        # 1. CPU 发起异步 H2D 拷贝
        inputs = batch["input_ids"].to(device, non_blocking=True)
        # 2. CPU 立刻返回，可以开始准备下一个 batch 的 IO
        
        # 3. GPU 在后台接收数据，一旦完成，立即开始计算
        outputs = model(inputs)
        ...
    ```

3.  **长度感知的动态打包 (Length-aware Dynamic Packing)**
    这是提升 MFU 的核心算法技巧，因为它直接减少了浪费在 `[PAD]` token 上的无效计算。
    *   **实现逻辑**：在 `collate_fn` 或 Dataset 层面实现一个打包器。它维护一个缓冲区，不断从数据源拉取样本，并将它们拼接在一起（用 `[EOS]` 分隔），直到总长度接近但不超过 `L_ctx`。
    *   **伪代码示例**:
        ```python
        class DynamicPacker:
            def __init__(self, source_iterator, max_length):
                self.source = source_iterator
                self.buffer = []
                self.max_length = max_length

            def __iter__(self):
                while True:
                    packed_sequence = []
                    current_length = 0
                    while current_length < self.max_length:
                        try:
                            # 从源获取下一个样本
                            sample = next(self.source)
                        except StopIteration:
                            # 数据源耗尽
                            if packed_sequence: yield packed_sequence
                            return
                        
                        if current_length + len(sample) > self.max_length:
                            # 当前样本放不下，先把它存起来，终止当前打包
                            self.buffer.append(sample)
                            break
                        
                        # 放入当前包
                        packed_sequence.extend(sample)
                        packed_sequence.append(EOS_TOKEN_ID) # 添加分隔符
                        current_length += len(sample) + 1

                    if packed_sequence:
                        # 可选：填充到 max_length
                        # packed_sequence.extend([PAD_TOKEN_ID] * (self.max_length - current_length))
                        yield packed_sequence
        ```
    *   **收益量化**：对于包含大量短文档的数据集（如网页、代码），动态打包能将 padding 比例从 30-50% 降低到 5% 以下，这意味着 MFU 可以获得**立竿见影的 1.2x 到 1.5x 提升**。

### 2.4 CPFS 实践：分片与压测

1.  **分片策略 (Sharding Strategy)**
    *   **分片大小 (Shard Size)**：在 CPFS 上，单个 shard 的大小是性能调优的关键参数。
        *   **太小 (< 10MB)**: 导致海量文件，给 CPFS 的元数据服务器（MDS）来巨大压力，每次文件打开、关闭、权限检查的开销会累积成显著的延迟。
        *   **太大 (> 5GB)**: 降低了并行加载的粒度。如果一个 worker 被分配到一个大 shard，它会长时间占用该文件，其他 worker 无法介入。同时，如果发生节点故障，需要重试或跳过的数据粒度也更大。
        *   **Rule-of-thumb**: 一个健康的 shard 大小范围是 **100MB 到 1GB**。对于一个 2TB (1T tokens) 的数据集，可以创建 2000 个 1GB 的 shard，或者 20000 个 100MB 的 shard。
    *   **分片数量 (Number of Shards)**：
        *   **Rule-of-thumb**: 分片总数应远大于全局数据加载 worker 的总数 (`num_nodes * num_workers_per_node`)。一个安全的法则是 `Num_Shards >= 10 * Num_Global_Workers`。这确保了数据加载的负载均衡，并且在任何时候，每个 worker 都有充足的、空闲的 shard 可供选择，避免排队等待。

2.  **独立的吞吐压测 (IO Stress Testing)**
    启动耗资巨大的正式训练前，必须对数据加载管道进行隔离压测，确保它不是瓶颈。
    *   **压测脚本**：创建一个“虚拟训练”脚本，它使用与真实训练完全相同的 `DataLoader` 配置，但在训练循环中，用一个几乎零开销的 CUDA 操作替换复杂的模型计算。
    *   **伪代码实现**:
        ```python
        import torch
        from torch.utils.data import DataLoader
        # from your_project import create_dataset, collate_fn

        # 1. 使用与真实训练完全相同的配置
        dataset = create_dataset(...)
        loader = DataLoader(dataset, batch_size=..., num_workers=..., pin_memory=True, collate_fn=...)
        
        device = torch.device("cuda")
        WARMUP_STEPS = 50
        TOTAL_STEPS = 500
        
        torch.cuda.synchronize()
        start_time = time.time()

        for i, batch in enumerate(loader):
            if i >= TOTAL_STEPS:
                break
            
            # 2. 模拟 H2D 拷贝，这是数据加载的一部分
            inputs = batch["input_ids"].to(device, non_blocking=True)
            
            # 3. 等待 H2D 拷贝完成，但不做任何计算
            torch.cuda.synchronize()

            if i == WARMUP_STEPS - 1:
                print("Warmup finished. Starting timer.")
                torch.cuda.synchronize()
                start_time = time.time()

        torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        processed_batches = TOTAL_STEPS - WARMUP_STEPS
        processed_tokens = processed_batches * GLOBAL_BATCH_SIZE_IN_TOKENS
        
        tokens_per_sec = processed_tokens / duration
        print(f"IO Stress Test Result: {tokens_per_sec=:.2f} tokens/sec")
        ```
    *   **性能诊断**:
        *   将压测出的 `tokens_per_sec` 与你根据模型规模估算的**目标计算吞吐**进行比较。
        *   **健康状态**: `IO_throughput > 1.5 * Target_compute_throughput`。这你预留了足够的安全边际。
        *   **亚健康状态**: `IO_throughput` 与 `Target_compute_throughput` 接近。这意味着 IO 随时可能成为瓶颈，需要进一步优化。
        *   **瓶颈状态**: `IO_throughput < Target_compute_throughput`。**严禁开始训练**。此时必须回头检查 `num_workers`、分片策略、存储系统健康状况或数据格式本身是否存在问题。

## 本章小结
*   大规模训练的效率瓶颈最终会从计算转移到 IO。我们的目标是通过优化将 `T_data_wait` 降至零，并异步化 `T_h2d_copy`，使训练回归“计算密集型”。
*   在主流数据格式中，**WebDataset** 以其极简设计、流式特性和对并行文件系统（CPFS）的友好性，成为 LLM 预训练场景下实现最大 IO 吞吐的**最佳选择**。
*   一套**优化的“组合拳”** 是实现高性能数据加载的必要条件：
    *   使用足量的 `num_workers` 并行化加载。
    *   开启 `pin_memory=True` 消除冗内存拷贝。
    *   配合 `non_blocking=True` 实现计算与数据传输的异步流水线。
    *   实施**长度感知的动态打包**，从根本上减少无效计算，是提升 MFU 的关键算法。
*   在 CPFS 上，采用 **100MB-1GB** 的分片大小和**远多于全局 worker 数量**的分片总数，是实现高并发、负载均衡读取的基础。
*   在正式训练前，通过**独立的吞吐压测**来量化数据加载管道的性能上限，是避免昂贵试错、进行科学决策的关键步骤。

## 常见陷阱与错误 (Gotchas)

1.  **陷阱：被“缓存”的假象所迷惑**
    *   **现象**：训练或压测在启动初期 `tokens/s` 极高，运行一段时间后骤降并稳定在一个较低水平。
    *   **原因**：这是操作系统文件缓存（OS page cache）的典型效应。初始读取的数据块被缓存到节点内存中，后续访问速度极快。当缓存被填满并开始淘汰时，才真正暴露出从 CPFS 经网络读取的真实性能。
    *   **调试技巧**：压测时必须读取足够大的数据量（例如，大于集群总内存），或者在每次运行前手动清理缓存（`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`，需谨慎操作），以获得稳态性能数据。

2.  **陷阱：盲目增加 `num_workers`**
    *   **现象**：将 `num_workers` 从 32 提升到 64，吞吐率反而下降了。
    *   **原因**：`num_workers` 并非越多越好。过多的 worker 进程会导致严重的 CPU 上下文切换开销、Python GIL 争用（尤其是在 `collate_fn` 中有复杂逻辑时），以及对存储系统造成“惊群效应”（Thundering Herd）。
    *   **调试技巧**：以节点 CPU 核心数的一半为基准，进行网格搜索（如 `[16, 24, 32, 48, 64]`），绘制 `num_workers` vs. `tokens/s` 的曲线，找到峰值点。同时使用 `htop` 监控 CPU 使用率，如果 System CPU time（红色）过高，通常是 worker 数过多的信号。

3.  **陷阱：忽视数据加载的“尾效应”**
    *   **现象**：大部分 step 很快，但周期性地出现个别 step 耗时异常长，拖慢整体进度。
    *   **原因**：可能是遇到了“热点”或慢速的 shard/存储节点。或者，如果 sharding 不均匀，某些 worker 会提前完成任务并空闲，等待处理最长 shard 的 worker。
    *   **调试技巧**：确保所有 shard 的大小和样本数量大致均匀。在 `DataLoader` 中对 shard 列表进行彻底的随机 shuffle (`torch.randperm`)。为数据加载操作添加详细的计时日志，定位到是哪个 worker 或哪个 shard 导致了延迟。

4.  **陷阱：在“最后一公里”引入 CPU 计算**
    *   **现象**：数据已经是 tokenized 的 `.bin` 文件，但吞吐还是上不去。
    *   **原因**：检查数据加载的每一步，即使是看似无害的操作。例如，在 `__getitem__` 中进行 `torch.tensor(list.from_bytes(...))` 这种类型转换，如果实现不当，也可能成为瓶颈。最常见的误是在线解压数据（如 `.gz`），或者进行任何形式的文本处理。
    *   **调试技巧**：**原则：数据加载循环中只应包含最纯粹的 IO 和内存操作**。使用 PyTorch Profiler 分析 `DataLoader` 的 CPU 时间开销，它可以精确地告诉你哪个函数调用占用了最多的时间。目标是让 `[dataloader]` 部分的 CPU 时间远低于 `[cuda]` 部分。

