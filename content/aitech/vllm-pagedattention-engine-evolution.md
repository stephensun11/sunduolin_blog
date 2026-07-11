---
title: "vLLM 详解：PagedAttention、连续批处理与 V0/V1 架构演进"
date: 2026-07-10T09:00:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM 推理优化"]
topics: ["推理引擎"]
tags: ["vLLM", "PagedAttention", "KV Cache", "Continuous Batching", "LLM Serving"]
summary: "从自回归服务的 KV Cache 碎片问题出发，讲清 PagedAttention、块表、连续批处理、前缀缓存、分块预填充，以及 vLLM 从 V0 到 V1 的核心架构变化与工程取舍。"
---

vLLM 经常被概括为“一个使用 PagedAttention 的高吞吐 LLM 推理框架”。这句话没有错，但只解释了它最早、也最知名的一层。

真正的 vLLM 是一套完整的在线推理系统：请求如何进入调度器，prefill 与 decode 如何共享 GPU，KV Cache 如何分配和复用，批次为什么能在每一步变化，多卡 worker 如何保持一致，以及 CUDA Graph、`torch.compile`、量化和推测解码怎样接入同一执行路径，都属于它要解决的问题。

本文从服务系统的瓶颈开始，逐层解释 PagedAttention 和 vLLM 的架构，并把“V0/V1 引擎代际”与 `v0.x.y` 软件版本号分开讨论。

## 先给出核心结论

1. PagedAttention 的主要价值不是让 attention 的理论 FLOPs 变少，而是把逻辑连续的 KV Cache 映射到可离散分配的物理块，减少预留、碎片和复制。
2. vLLM 的高吞吐来自一组协同机制：Paged KV、连续批处理、token 级调度、专用 kernel、前缀缓存、分块 prefill 和 CUDA Graph，不能把全部收益归因于一个 kernel。
3. continuous batching 允许完成的请求立即离开、等待的请求立即进入；它比固定请求批次更适合输出长度不可预测的在线生成。
4. V1 不是 `v1.0.0` 软件版本，而是 2025 年开始启用的新引擎架构。vLLM 的公开发行版到 2026 年仍使用 `v0.x.y` 编号。
5. PagedAttention 解决 KV 内存管理，不会消除模型权重带宽、MLP/MoE、跨卡通信或长上下文 attention 的全部成本。

## 为什么 LLM 服务比离线推理难

Decoder-only 模型通常经历两个阶段：

- **prefill**：一次处理整个 prompt，生成每一层的 KV Cache；矩阵较大，计算密集度较高。
- **decode**：每步只生成一个 token，读取权重和已有 KV，再把一个新 KV 追加到缓存；矩阵很瘦，常受显存带宽限制。

如果模型有 <span class="math-inline">\\(L\\)</span> 层、每个 token 的 KV 头数为 <span class="math-inline">\\(H_{kv}\\)</span>、每头维度为 <span class="math-inline">\\(d_h\\)</span>、数据类型占 <span class="math-inline">\\(b\\)</span> 字节，那么单个请求长度为 <span class="math-inline">\\(T\\)</span> 时，KV Cache 大小近似为：

<div class="math-display">\[
M_{KV}=2LTH_{kv}d_hb.
\]</div>

系数 2 来自 Key 和 Value。MHA 中 <span class="math-inline">\\(H_{kv}=H_q\\)</span>；GQA/MQA 通过减少 KV 头数降低常数，但缓存仍随序列长度和并发请求数线性增长。

在线服务还有三个额外困难：

1. 请求到达时间不同，无法预先组成整齐批次；
2. 输出长度未知，短请求结束后会在固定批次里留下空位；
3. 为请求预留最大长度会浪费内存，不预留又要处理不断增长和搬迁的缓存。

GPU 是否算得快只是问题的一半。另一半是：有限显存能同时容纳多少条正在生成的序列。

## 传统连续 KV Cache 的碎片问题

一种直观实现是为每条序列申请一段连续显存，并按最大上下文长度预留空间。假设请求当前只有 300 token，但系统为它预留 4096 token，那么大部分显存暂时不可使用。若只按当前长度分配，序列增长时又可能没有相邻空间，只能重新申请并复制。

这和操作系统早期的连续内存分配非常相似：

- **内部碎片**：分配单元内部没有用满；
- **外部碎片**：空闲空间总量足够，但分散在各处，无法提供一段大的连续区域；
- **过度预留**：为了避免扩容，提前锁住未来可能用到的空间。

这些浪费会直接降低可并发序列数。对 decode 而言，更大的有效 batch 往往意味着一次加载模型权重可以服务更多 token，因此 KV 利用率会进一步影响吞吐。

## PagedAttention 的核心抽象

[PagedAttention 论文](https://arxiv.org/abs/2309.06180) 借用了虚拟内存分页的思路：把每条序列的 KV Cache 切成固定 token 数的**逻辑块**，把 GPU 中可用的缓存空间切成同样大小的**物理块**。

逻辑块不需要在物理显存中连续。每条序列维护一张块表：

| 逻辑块 | 物理块 | token 范围 |
|---|---:|---|
| 0 | 7 | 0-15 |
| 1 | 2 | 16-31 |
| 2 | 11 | 32-47 |

Attention kernel 根据块表定位 Key 和 Value。对模型而言，token 仍按逻辑顺序排列；对内存管理器而言，只要找到任意空闲物理块即可扩展序列。

### 为什么固定块能减少浪费

设块大小为 <span class="math-inline">\\(B\\)</span> 个 token，长度为 <span class="math-inline">\\(T\\)</span> 的序列需要：

<div class="math-display">\[
N_{block}=\left\lceil\frac{T}{B}\right\rceil
\]</div>

个块。除最后一块外，其余块都能装满，因此每条序列的浪费少于一个块，即最多 <span class="math-inline">\\(B-1\\)</span> 个 token 槽位。相比按最大长度预留，这个上界稳定得多。

块并非越小越好。小块降低尾部浪费，却会增大块表、调度元数据和间接寻址开销；大块提高连续访问效率，却会增加内部碎片。实际系统需要结合 kernel、模型结构和平均序列长度选取。

### 分页没有把 attention 变成稀疏 attention

PagedAttention 仍然可以对全部历史 KV 执行精确 attention。它改变的是 KV 的物理布局和访问方法，不是注意力的数学定义。

因此，对长度 <span class="math-inline">\\(T\\)</span> 的单步 decode，读取历史 KV 的工作仍随 <span class="math-inline">\\(T\\)</span> 增长。分页提高了可管理性和显存利用率，并不把 full attention 的复杂度改成常数。

## Copy-on-Write 与 KV 共享

分页抽象的另一项重要能力是多个逻辑序列共享同一个物理块。

### 并行采样与 beam search

当一个 prompt 分叉成多条候选序列时，它们拥有相同前缀。传统实现可能为每条候选复制整份前缀 KV；分页实现只需让多张块表指向相同物理块，并维护引用计数。

当某个分支准备修改共享块时，系统才执行 copy-on-write：

1. 为该分支申请新物理块；
2. 复制需要保留的内容；
3. 更新该分支块表；
4. 其他分支继续引用旧块。

### 前缀缓存

不同请求也可能共享系统提示、长文档前缀或 few-shot 示例。vLLM 的 automatic prefix caching 以完整 KV 块为单位计算哈希，哈希输入包含当前块 token、前缀块哈希以及必要的附加信息。命中时，请求可以直接复用已有物理块，跳过这部分 prefill。

必须满足两个条件：

- token 序列和影响 KV 的模型输入真正相同；
- 缓存块仍驻留且未被淘汰。

前缀缓存主要减少 prefill 工作。它不会让后续 decode 的每一步不再读取这些 KV，也不能复用只“语义相似”但 token 不同的提示。

## Continuous Batching：批次为什么每一步都能变化

静态批处理会把一组请求绑在一起，直到最长请求结束。假设四条请求分别生成 20、40、100、200 个 token，那么前三条完成后，对应槽位只能等待最后一条。

连续批处理把调度边界放到每个迭代：

```text
迭代 t:   A B C D
迭代 t+1: A 完成，E 进入 -> E B C D
迭代 t+2: B 完成，F 进入 -> E F C D
```

一次迭代通常为每个 decode 请求安排一个或多个 token，也可为 prefill 请求安排一段 token。已完成请求立即释放 KV 块，等待队列中的请求随即补入，因此 batch 的请求集合可以持续变化。

### token budget 比 request count 更接近真实成本

请求数相同，不代表计算量相同：一个 8k prompt 的 prefill 和一个单 token decode 差别很大。现代调度器更倾向使用本轮可处理的 token 数作为预算，并同时考虑：

- GPU 可用 KV 块；
- 当前运行与等待请求；
- prefill/decode 优先级；
- 最大序列数和最大 batched token 数；
- 多模态 encoder cache；
- 推测解码额外 token。

这使调度决策更贴近执行器真正要处理的张量形状。

## Chunked Prefill：避免长 prompt 阻塞 decode

如果调度器一次执行完整的超长 prefill，已经在线生成的请求会等待较久，inter-token latency 出现尖峰。Chunked prefill 把长 prompt 拆成多个 token 块，让它和 decode 请求共享多个迭代。

V1 的常见策略是优先安排 decode，再用剩余 token budget 填充 prefill；超出预算的 prefill 被切块延后。这样做的效果是：

- 降低长 prompt 对在线 decode 的干扰；
- 把计算密集的 prefill 与带宽密集的 decode 放进同一批次，可能提高 GPU 利用率；
- 允许在吞吐和 token 间延迟之间通过预算参数折中。

它不是免费的。分块过小会增加调度、kernel launch 和中间状态管理开销；分块过大又会重新造成延迟尖峰。

## 一次 vLLM 请求如何流动

简化后的在线路径可以分成四层。

### 1. API 与输入处理

API server 接收 OpenAI-compatible 请求，完成鉴权、chat template、tokenization、参数校验和流式输出组织。多模态模型还要执行图像/音频预处理。

### 2. EngineCore 与调度

EngineCore 保存请求状态，决定本轮哪些请求运行、各自运行多少 token，并通过 KV cache manager 分配、复用或回收块。

### 3. Worker 与 Model Runner

Worker 在 GPU 上执行模型。张量并行时，每个 rank 持有部分权重并参与 collective communication。Model Runner 负责准备输入、选择 attention backend、执行 forward、采样并更新持久批次状态。

### 4. 输出处理

生成 token 被增量 detokenize，经过 stop 条件、logprobs 和结构化输出处理后流回客户端。

早期实现中，这些工作更容易串在同一个 Python 控制路径上。V1 把核心模型执行隔离为 EngineCore/EngineCoreProc，使 tokenization、detokenization、网络传输和 GPU 执行更容易重叠。

## V0 到 V1：为什么需要重构

2023 年的 vLLM 以 PagedAttention 和高效 batching 建立了基础。随着多模态、prefix cache、LoRA、speculative decoding、chunked prefill 和多种硬件后端不断加入，V0 的控制流逐渐承担过多特例。

[V1 架构发布说明](https://vllm.ai/blog/2025-01-27-v1-alpha-release) 把重构重点放在以下方面。

### 统一 token 调度

V1 的调度结果可抽象为“请求 ID 到本轮 token 数”的映射。prefill、decode、chunked prefill 和 speculative token 不再需要完全独立的调度路径。

这个统一模型很重要：对 GPU 执行而言，它们最终都是在已有 token 数基础上计算若干新 token，只是 attention 元数据和 logits 需求不同。

### 重新设计 KV cache manager

V1 让块分配、前缀哈希、引用和淘汰更直接地围绕统一调度工作。Prefix cache 使用基于哈希的块匹配与 LRU 风格淘汰，不再依赖 V0 中更重的缓存路径。

### worker 只接收增量变化

调度器不必每一步把全部请求状态重新发送给 worker。worker 维护自己的请求视图，控制面只发送新增、完成或长度变化等增量信息，降低 CPU 序列化和多 rank 同步开销。

### persistent batch

Model Runner 维护一个可原地更新的持久批次。请求进入、退出和 token 追加只修改必要槽位，避免每轮在 Python 中重建完整 batch。这对小 decode step 尤其重要，因为 CPU 开销很容易盖过 GPU 计算时间。

### `torch.compile` 与 piecewise CUDA Graph

CUDA Graph 能重放固定执行图，减少 Python 和 kernel launch 开销，但请求数、token 数和 attention 元数据经常变化。V1 将可编译的模型片段与动态 attention 部分分开，并针对若干 batch shape 捕获 piecewise CUDA Graph，在动态服务与低开销之间折中。

官方 V1 alpha 博客在其测试条件下报告相对未启用 multi-step scheduling 的 V0 最高约 1.7 倍吞吐提升。这个数字依赖模型、硬件、长度和配置，不能当成所有部署的固定收益。

## 版本号最容易混淆的地方

vLLM 有两套不同语义的“版本”：

| 名称 | 含义 | 示例 |
|---|---|---|
| V0 / V1 | 引擎内部架构代际 | V1 scheduler、V1 Model Runner |
| `v0.x.y` | 项目发行版 SemVer 风格标签 | `v0.23.0` |

截至 2026 年 7 月，GitHub 最新公开发行版是 [`v0.23.0`](https://github.com/vllm-project/vllm/releases)，而其中默认或可用的核心执行路径已经属于 V1。不能因为包版本仍以 `0` 开头，就认为它仍在使用 V0 引擎。

### 一条更有用的演进时间线

| 时间 | 里程碑 | 核心变化 |
|---|---|---|
| 2023 | vLLM / PagedAttention | 分页 KV、连续批处理、高显存利用率 |
| 2023-2024 | V0 功能扩展 | 分布式、量化、LoRA、多模态、prefix cache、spec decode 等逐步加入 |
| 2025 | V1 引擎 | 统一调度、重写 KV manager、persistent batch、异步 EngineCore、编译与 CUDA Graph |
| 2025-2026 | 大规模 serving 扩展 | disaggregated prefill、KV connector/offload、hybrid KV、更多 attention backend 与硬件 |
| 2026 | Model Runner V2、Rust frontend 等 | 进一步压低 Python 控制面开销并扩展新模型/新硬件路径 |

这不是每个小版本的 changelog，而是理解架构变化最有价值的节点。生产升级仍应逐版阅读 release notes，因为模型兼容、依赖和默认参数可能变化。

## 抢占、交换与分离式 prefill/decode

当 GPU KV 块耗尽时，调度器必须处理压力。可选策略包括：

- 暂停或抢占低优先级请求；
- 丢弃其 KV，稍后重新计算前缀；
- 把 KV 卸载到 CPU、磁盘或远端存储；
- 通过 connector 在 prefill worker 与 decode worker 间传输 KV。

重新计算消耗算力但不需要慢速传输；swap/offload 节省算力却消耗 PCIe、网络或存储带宽。哪种更好取决于上下文长度、互连和服务等级目标。

Disaggregated prefill 将计算密集的 prompt 阶段和带宽密集的 decode 阶段放在不同 worker 池。它可以隔离延迟并分别扩缩容，但增加 KV 传输、路由和故障处理复杂度。官方文档仍把这类能力中的部分标为实验性，部署时不能只看功能开关。

## vLLM 不会自动解决什么

### 1. 它不会降低模型参数量

每个 decode step 仍要读取大量权重。量化、tensor parallel、expert parallel 和更高效 kernel 是另一组问题。

### 2. 它不会消除长上下文计算

Paged KV 让内存更紧凑，但 full attention 仍要访问历史 Key/Value。FlashAttention、Flash-Decoding、稀疏 attention 或线性 attention 分别从不同方向处理计算与 I/O。

### 3. 高吞吐不等于低延迟

把 batch 做大通常提高 tokens/s，却可能增加排队时间和单用户 token 间延迟。应同时测 TTFT、TPOT/ITL、端到端 latency、request throughput 和 token throughput。

### 4. benchmark 结果不能脱离工作负载

vLLM 论文报告相对 FasterTransformer、Orca 等基线 2-4 倍吞吐提升，早期博客也曾报告相对 Hugging Face Transformers 更大的倍数。这些都是特定模型、请求分布、硬件和基线版本下的结果，不是框架的固定常数。

## 如何做一次可信的 vLLM 评测

至少固定并记录以下变量：

1. 模型、dtype、量化方法和 tensor/pipeline parallel 配置；
2. GPU 型号、数量、互连和 vLLM/CUDA/PyTorch 版本；
3. 输入长度与输出长度分布，而不只是平均值；
4. 到达过程、并发数、请求超时和 streaming 设置；
5. `max_num_batched_tokens`、`max_num_seqs`、KV cache dtype、prefix cache 和 chunked prefill；
6. TTFT、TPOT、P50/P95/P99 latency、tokens/s、requests/s 和峰值显存；
7. 是否发生 preemption、cache hit、KV offload 或 OOM。

离线 `--prompts` benchmark 可以测上限，不能代替具有排队和突发流量的在线压测。

## 总结

PagedAttention 把 LLM 的 KV Cache 从“每条请求一段连续且难以增长的数组”，变成“逻辑连续、物理离散、可共享、可回收的块集合”。这项抽象显著提高了显存利用率，也为 copy-on-write、prefix caching 和灵活调度提供了基础。

vLLM 的系统价值则更进一步：continuous batching 让 GPU 槽位持续被新请求填补，chunked prefill 平衡首 token 与后续 token 延迟，V1 统一了 token 调度和 KV 管理，persistent batch、增量 worker 状态、`torch.compile` 与 CUDA Graph 共同压低控制面开销。

理解 vLLM 时，最好把它看成“面向动态请求的 GPU 操作系统”，而不是单个 attention 算法。但这个类比也有边界：它仍受模型权重、attention 复杂度、通信、硬件后端和工作负载分布约束。最终是否更快，需要在真实请求分布上用完整延迟指标验证。

## 参考资料

1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), Kwon et al., SOSP 2023.
2. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm), vLLM Team, 2023.
3. [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release), vLLM Team, 2025.
4. [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/), official documentation.
5. [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/), official documentation.
6. [Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/), official documentation.
7. [Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/), official documentation.
8. [vLLM Releases](https://github.com/vllm-project/vllm/releases), official repository.
