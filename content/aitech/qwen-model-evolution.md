---
title: "Qwen 全系模型详解：从 Qwen 1 到 3.6"
date: 2026-07-13T10:10:00+08:00
draft: false
summary: "系统梳理 Qwen 1、1.5、2、2.5、3、3-Next、3.5 与 3.6，推导 GQA、DCA、QK-Norm、Gated Attention、Gated DeltaNet、高稀疏 MoE、MTP 和原生多模态。"
categories: ["AiTech"]
subcategories: ["LLM 基础技术"]
topics: ["模型演进"]
tags: ["Qwen", "GQA", "Gated Attention", "Gated DeltaNet", "MoE", "DCA", "YaRN", "QK-Norm", "Multimodal"]
---

Qwen 是最容易被“版本名”绕晕的模型家族之一。同一代里经常同时出现 Dense、MoE、Coder、Math、VL、Audio、Thinking 和 Instruct；小数版本有时换数据，有时换后训练，有时才真正更换骨干。

要读懂它，最好沿三条轴展开：

- **骨干轴**：MHA → GQA → QK-Norm → Gated Attention + Gated DeltaNet；
- **稀疏轴**：Dense → 细粒度 MoE → 高稀疏 MoE；
- **能力轴**：通用对话 → 代码/数学 → 统一思考模式 → 多模态 Agent。

本文覆盖截至 2026 年 7 月 13 日公开的 Qwen 1 到 Qwen 3.6。结论优先依据技术报告、官方博客和官方模型卡。

![Qwen 从 1 到 3.6 的版本时间线](/images/aitech/model-evolution/qwen-timeline.svg)

*图 1：Qwen 主线。最需要记住的一点是：普通 Qwen3 仍使用 GQA，Gated Attention 与 Gated DeltaNet 是 Qwen3-Next 开始的骨干升级。*

## 先学会识别 Qwen 型号

以 `Qwen3-Next-80B-A3B-Instruct` 为例：

- `Qwen3-Next`：架构家族；
- `80B`：总参数量；
- `A3B`：每个 token 大约激活 3B 参数；
- `Instruct`：完成对话后训练，不是 Base 权重；
- 若带 `Thinking`、`Coder`、`VL` 等后缀，还说明推理模式或专业分支。

`A3B` 不能用来估算完整权重显存。MoE 的全部专家仍需存储，只是每个 token 不会把它们全算一遍。

## 一张表建立版本坐标系

| 主版本 | 发布时间 | 代表模型 | 上下文与数据 | 架构/训练升级重点 |
| --- | --- | --- | --- | --- |
| Qwen 1 | 2023-08 起 | 1.8B、7B、14B、72B Dense | 2.2T-3T，8K/32K | 改进 LLaMA 骨干、152K tokenizer、工具与代码分支 |
| Qwen1.5 | 2024-02 | 0.5B-110B、MoE-A2.7B | 全系 32K | 尺寸补全、部分 GQA、首个公开 MoE、部署生态 |
| Qwen2 | 2024-06 | 0.5B-72B、57B-A14B | 约 7T，最高 128K | 全系 GQA、DCA + YaRN、MoE upcycling、30 种语言 |
| Qwen2.5 | 2024-09 | 0.5B-72B、Coder、Math | 最多 18T，128K | 数据与后训练升级，代码、数学、JSON、长生成 |
| Qwen3 | 2025-04 | 0.6B-32B、30B-A3B、235B-A22B | 36T，119 种语言，128K | QK-Norm、去 QKV bias、统一 thinking/non-thinking |
| Qwen3-Next | 2025-09 | 80B-A3B | 15T，原生 262K、可扩 1M | Gated DeltaNet + Gated Attention、高稀疏 MoE、MTP |
| Qwen3.5 | 2026-02 起 | 0.8B-397B-A17B | 原生 262K、可扩 1M，201 种语言 | 延续混合骨干，Early Fusion 原生视觉，大规模 Agent RL |
| Qwen3.6 | 2026-04 | 35B-A3B、27B | 262K/1M 扩展 | 骨干延续 3.5，重点升级仓库级编码、前端与思考状态保持 |

## Qwen 1：在 LLaMA 式骨干上补齐中文、工具和工程能力

### 骨干结构

Qwen 1 是 decoder-only Dense Transformer，关键组成包括：

- Pre-Norm + RMSNorm；
- SwiGLU FFN；
- RoPE；
- 多头注意力 MHA；
- Q、K、V 投影带 bias；
- embedding 与输出 head 不共享权重；
- RoPE 的逆频率以 FP32 保存，降低长训练中的数值误差。

这套骨干并不追求结构新奇，重点是把中文、多语言、代码和工具数据放到统一 Base 模型中。

### 152K tokenizer 为什么重要

早期中文模型常直接复用主要面向英文的词表。一个汉字可能被切成多个 byte token，同样一段中文就比英文占用更多上下文和计算。

Qwen 基于 `cl100k`/tiktoken 风格 BPE 扩充中文与多语言 token，普通词表约 151K。它还把数字按位切分，便于模型学习数值组成。

Tokenizer 不是只影响输入格式。设一段文本有 <span class="math-inline">\(C\)</span> 个字符，平均每字符产生 <span class="math-inline">\(r\)</span> 个 token，那么注意力长度约为 <span class="math-inline">\(L=rC\)</span>。稠密注意力成本近似：

<div class="math-display">\[
O(L^2)=O(r^2C^2)
\]</div>

把中文 token 化率从 1.8 降到 1.1，不只是“上下文多装一点”，二次注意力成本也会明显下降。

### 早期长上下文方案

Qwen 1 的早期预训练上下文较短，后续模型通过更大的 RoPE base、NTK-aware interpolation 与 LogN attention scaling 扩到 8K/32K。

NTK-aware interpolation 调整不同 RoPE 频率，使模型在更长位置上减少相位失真；LogN scaling 则随上下文长度放大 Query/Key 的有效尺度，缓解长序列中 softmax 分布过平。

这些属于“外推与续训”方案，还不是 Qwen2 的 DCA，也不是 Qwen3-Next 的线性注意力。

### 后训练与 Agent 雏形

Qwen 1 使用 SFT 和 PPO 式 RLHF，并把 ReAct 格式、函数调用、代码解释器等轨迹加入后训练。Code-Qwen、Math-Qwen 与 Qwen-VL 则是专业分支。

这确立了 Qwen 的长期风格：Base 骨干尽量统一，专业数据和工具模板在分支中快速迭代，成熟后再合流到下一代通用模型。

## Qwen1.5：架构变化有限，生态变化很大

Qwen1.5 补齐 0.5B、1.8B、4B、7B、14B、32B、72B，随后加入 110B，并让所有尺寸支持 32K。

### GQA 还没有普及到全系

Qwen1.5 的 32B 与 110B 使用 GQA，其他主要尺寸仍延续 MHA。到了 Qwen2，GQA 才成为所有尺寸的统一配置。

GQA 让 <span class="math-inline">\(n_q\)</span> 个 Query head 共享较少的 <span class="math-inline">\(n_{kv}\)</span> 组 K/V：

<div class="math-display">\[
C_{KV}^{GQA}=2n_{kv}d_h,qquad n_{kv}<n_q
\]</div>

它通常在 MHA 的质量与 MQA 的推理效率之间取得平衡。可结合[注意力头共享详解](/aitech/attention-mha-mqa-gqa/)阅读。

### Qwen1.5-MoE-A2.7B

这是 Qwen 首个公开 MoE：总参数约 14.3B，每个 token 激活约 2.7B。它采用细粒度专家和共享专家，用较小激活计算获得接近更大 Dense 模型的容量。

### 工程生态升级

Qwen1.5 开始原生接入 Hugging Face Transformers，不再要求 `trust_remote_code=True`；同时提供 AWQ、GPTQ、GGUF 等格式，适配 vLLM、SGLang、llama.cpp 等环境。

这提醒我们：<strong>一次有价值的模型迭代不一定要发明新公式。</strong>标准加载、量化权重和推理框架支持，直接决定开发者能否真正使用它。

## Qwen2：GQA、长上下文和 MoE 正式进入统一主线

Qwen2 包括 0.5B、1.5B、7B、72B Dense，以及 57B-A14B MoE。

### 全系 GQA

所有尺寸统一使用 GQA，KV Cache 的增长不再随 Query head 数等比例增加。其余骨干仍保留 RoPE、SwiGLU、RMSNorm、Pre-Norm 与 QKV bias。

### DCA：把长序列分块，又保留跨块路径

Dual Chunk Attention 将长序列划分为 chunk，对不同关系使用不同位置编号：

- chunk 内 token 使用局部相对位置；
- 跨 chunk 读取使用受控的位置表示；
- 相邻 chunk 保留连续性路径。

这样模型可以在训练长度之外扩展，而不会让所有相对距离直接落到从未见过的巨大数值。

需要区分：DCA 主要改变**位置编码与分块方式**，每个 Query 的内容注意力仍是标准 softmax attention；它不是线性注意力。

### YaRN：平滑扩展 RoPE

YaRN 对 RoPE 不同频率区间采用不同插值，并加入 attention scaling。Qwen2 将 RoPE base 从 10K 提高到 1M，并结合 DCA/YaRN，把 7B、72B 等模型的推理窗口扩到 128K。

位置编码基础可参考[从绝对位置到 RoPE](/aitech/positional-encoding-basics/)。

### Qwen2 MoE 与 Dense Upcycling

Qwen2-57B-A14B 使用细粒度路由专家和共享专家。训练时先得到 Dense 模型，再把 FFN 参数复制/切分成专家继续训练，称为 **dense upcycling**。

它的好处是：

- 不必从随机初始化训练整个 MoE；
- 早期通用表示直接继承 Dense checkpoint；
- 后续专家在路由中逐渐专业化。

但复制出来的专家起点相似，需要路由噪声、均衡约束和继续训练打破对称性。

### 后训练从单次偏好优化变成迭代流程

Qwen2 先做 SFT，再做离线 DPO，随后使用 reward model 对策略新生成的回答做在线优化，并周期性合并或刷新训练信号。

关键变化是分布：离线偏好数据来自旧策略或人工样本；在线阶段看到的是当前模型真实会生成的回答，能更快修正最新错误。

## Qwen2.5：骨干基本不变，数据和可控性跃迁

Qwen2.5 把高质量预训练数据扩大到最多约 18T token，并同时发布 0.5B 到 72B 的 Dense 系列、Coder 与 Math 系列。

这一代的关键提升包括：

- 更强代码生成、调试和数学推理；
- 29 种以上语言；
- 128K 输入、最长约 8K 生成；
- 更稳定的 JSON 与表格输出；
- 更强 system prompt 遵循、角色设定和长文生成；
- 对 structured data、图表文本和多种数据格式更敏感。

### Coder 与 Math 分支不是另一种 Transformer

Qwen2.5-Coder 使用约 5.5T code-related token，加入代码仓库、合成代码、执行反馈等数据。Qwen2.5-Math 则强化 Chain-of-Thought、Program-of-Thought 与工具集成推理。

它们主要是数据配比和后训练目标不同，不能因为 Coder 更会写代码就推断它采用了不同 attention。

### Qwen2.5-1M

后续 1M 分支通过长上下文继续训练、DCA/YaRN 与稀疏预填充工程扩展到百万 token。它证明传统 GQA + RoPE 系也可通过系统优化做得很长，但计算仍会随注意力矩阵增长。Qwen3-Next 因此从骨干层面引入线性注意力。

## QwQ：通向 Qwen3 的推理分支

QwQ 在 Qwen2.5 底座上强化长思维链与可验证推理，角色类似 DeepSeek-R1 对 DeepSeek 主线的作用。QwQ-32B 证明中等规模 Dense 模型也能通过大规模 RL 提升数学和代码推理。

它是后训练分支，不应写成 Qwen2.6，也没有把骨干换成 Gated Attention。QwQ 积累的 thinking 数据与训练流程，最终进入 Qwen3 的统一模式。

## Qwen3：思考模式统一，但仍是标准 GQA

Qwen3 发布 Dense 0.6B、1.7B、4B、8B、14B、32B，以及 30B-A3B、235B-A22B 两个 MoE 主力型号。

### 骨干的三个小而关键的变化

Qwen3 仍使用 GQA、RoPE、SwiGLU、RMSNorm 和 Pre-Norm，但：

1. 删除 QKV projection bias，减少不必要的自由偏移；
2. 在 Query 和 Key 上加入 QK-Norm；
3. MoE 结构调整为 128 个路由专家、每 token 激活 8 个，不再设置共享专家。

QK-Norm 先把每个 head 的 Q/K 做 RMS 类归一化：

<div class="math-display">\[
\hat q=\frac{q}{\sqrt{\operatorname{mean}(q^2)+\epsilon}},\qquad
\hat k=\frac{k}{\sqrt{\operatorname{mean}(k^2)+\epsilon}}
\]</div>

然后再计算：

<div class="math-display">\[
a_{t,s}=\operatorname{softmax}\left(\frac{\hat q_t^T\hat k_s}{\sqrt{d_h}}\right)
\]</div>

它限制 attention logits 的极端尺度，提高大规模训练稳定性。QK-Norm 是对 Q/K 向量归一化；Gated Attention 是对 attention 输出做门控，二者位置和目标不同。

### 三阶段预训练

Qwen3 训练约 36T token、覆盖 119 种语言，预训练分为：

1. 约 30T 通用 token，4K 上下文；
2. 约 5T STEM、代码和推理密集 token，仍以 4K 为主；
3. 数千亿长上下文 token，将训练窗口提升到 32K，并用 YaRN/DCA 外推到 128K。

分阶段的意义是先学广泛分布，再提高困难样本密度，最后才承担长序列的昂贵计算。

### 统一 Thinking 与 Non-Thinking

Qwen3 后训练分四阶段：长 CoT 冷启动、推理 RL、思考模式融合、通用 RL。最终同一权重可以：

- `/think`：产生较长推理过程；
- `/no_think`：直接回答，降低延迟；
- 通过 thinking budget 控制推理 token 预算。

小模型还使用大模型生成的 logits、回答与偏好数据做强到弱蒸馏。

**重要结论：Qwen3 的最大升级是训练范式，不是 Gated Attention。**

## Gated Attention：为什么 softmax 后还要再加一个门

Qwen 团队随后系统研究 30 余种 attention gate 设计。效果最稳定的方向是在 Scaled Dot-Product Attention 输出后、输出投影前，对每个 head 加查询相关的 Sigmoid gate。

### 标准 attention 有一个隐藏限制

单个 head 的输出为：

<div class="math-display">\[
o_{t,h}=\sum_{s\le t}a_{t,s,h}v_{s,h},
\qquad
\sum_{s\le t}a_{t,s,h}=1
\]</div>

softmax 权重总和为 1。即使当前 Query 没有真正值得读取的历史，head 也必须输出某种 Value 加权平均，不能自然选择“什么都不写”。

### 输出门控

由当前状态产生门值：

<div class="math-display">\[
g_{t,h}=\sigma(w_h^Tx_t+b_h)
\]</div>

再缩放 head 输出：

<div class="math-display">\[
\tilde o_{t,h}=g_{t,h}\,o_{t,h}
\]</div>

不同实现可使用 head 级标量或更细粒度向量 gate，但核心都是：**softmax 决定从哪里读，Sigmoid gate 决定读出的内容写回多少。**

![Qwen Gated Attention 的数据流](/images/aitech/model-evolution/qwen-gated-attention.svg)

*图 2：gate 在 SDPA 之后。它不改变注意力权重归一化，而是给每个 token、每个 head 一次“关闭写回”的机会。*

### 它为什么有助于稳定训练

论文实验观察到，门控可以：

- 抑制无用 head 对残差流的持续写入；
- 缓解 massive activation；
- 减轻 attention sink，即大量 token 机械关注少数特殊位置；
- 增加轻量、查询相关的非线性；
- 改善长上下文外推与训练扩展趋势。

它与 MoE gate 完全不同。Attention gate 控制一个 head 的输出强度；MoE gate 选择执行哪些 FFN 专家。

## Qwen3-Next：混合线性注意力成为新骨干

Qwen3-Next-80B-A3B 是这条架构线的首个公开模型：80B 总参数、约 3B 激活，48 层，原生 262,144 上下文，可用 YaRN 扩到约 1.01M。

### 为什么不把所有层都换成线性注意力

标准全注意力保存所有历史 K/V，能精确回看任意 token，但长上下文成本高。线性/递归注意力把历史压入固定状态，计算和缓存更省，却可能丢失逐字细节。

Qwen3-Next 采用周期：

<div class="math-display">\[
12\times\left(3\times(\text{Gated DeltaNet}\rightarrow\text{MoE})
+1\times(\text{Gated Attention}\rightarrow\text{MoE})\right)
\]</div>

也就是 75% 层做线性状态更新，25% 层保留精确全注意力。

![Qwen3-Next 混合注意力与 Qwen3.5 多模态骨干](/images/aitech/model-evolution/qwen-hybrid-architecture.svg)

*图 3：Gated DeltaNet 负责高吞吐的压缩记忆，全注意力层周期性恢复精确检索能力。*

### Gated DeltaNet 的直觉

它维护固定大小的状态矩阵 <span class="math-inline">\(S_t\)</span>。教学化的 gated delta rule 可写成：

<div class="math-display">\[
S_t=\alpha_t\odot S_{t-1}
+\beta_t\big(v_t-S_{t-1}k_t\big)k_t^T
\]</div>

其中：

- <span class="math-inline">\(\alpha_t\)</span> 是遗忘/保留门；
- <span class="math-inline">\(S_{t-1}k_t\)</span> 是旧状态对当前 key 的预测；
- <span class="math-inline">\(v_t-S_{t-1}k_t\)</span> 是需要修正的误差；
- Delta Rule 先擦除旧绑定，再写入新的 key-value 关联。

读取近似为：

<div class="math-display">\[
o_t=S_tq_t
\]</div>

递归更新每步成本与已过去长度无关，因此整段序列近似 <span class="math-inline">\(O(L)\)</span>。真实实现还包含短卷积、多头状态、归一化与并行扫描。

若需要更完整的线性注意力背景，可读[线性注意力详解](/aitech/linear-attention-guide/)。

### 高稀疏 MoE

80B-A3B 配置使用 512 个专家、每 token 选择 10 个路由专家，并额外经过 1 个共享专家；每个专家较小，因此能以约 3B 激活参数容纳 80B 总容量。

相比 Qwen3 的 128 选 8，这里专家更多、单专家更窄、激活比例更低，路由和 all-to-all 的实现也更重要。

### MTP 与稳定性改动

Qwen3-Next 加入 Multi-Token Prediction，用未来多个 token 的辅助损失提高预训练信号密度，并可配合 speculative decoding。它还采用 zero-centered、weight-decayed layer normalization 等稳定性技术，减少超深 MoE 混合骨干的漂移。

官方模型卡报告，80B-A3B 在长于 32K 的上下文中相对 Qwen3-32B 可获得明显吞吐优势；实际倍数取决于推理框架是否实现 DeltaNet kernel、MTP 草稿解码和专家并行。

## Qwen3.5：混合架构从语言模型升级为原生多模态模型

Qwen3.5 延续 Qwen3-Next 的 Gated DeltaNet + Gated Attention + MoE 主干，但从预训练开始把视觉 token 与文本 token 融入同一序列。

### Early Fusion 与外挂视觉适配器的区别

外挂式多模态模型常先训练语言模型，再接一个 vision encoder 与 projection/cross-attention adapter。Early Fusion 则在大规模预训练中就让文本、图像和视频 token 共同进入共享骨干。

优势是跨模态关系不是最后补课：网页截图、图表、视频帧与文字说明可以共同塑造表示。代价是训练数据配比、序列长度和模态冲突更难控制。

### 以 35B-A3B 为例看精确结构

Qwen3.5-35B-A3B 有 40 层：

<div class="math-display">\[
10\times\left(3\times\text{Gated DeltaNet}+1\times\text{Gated Attention}\right)
\]</div>

- Gated Attention：16 个 Q head、2 个 KV head，head dimension 256，RoPE dimension 64；
- MoE：256 个专家，每 token 路由 8 个，另有 1 个共享专家；
- 约 35B 总参数、3B 激活；
- 原生 262K，上下文可扩到约 1.01M；
- 支持多步 MTP。

Qwen3.5 家族还覆盖 397B-A17B、122B-A10B、27B、9B、4B、2B、0.8B 等尺寸，覆盖约 201 种语言。不同尺寸可能是 Dense 或 MoE，不能把 35B-A3B 的专家配置套到全家族。

### Agent RL 成为训练主角

Qwen3.5 在大量可交互环境中训练工具调用、网页/GUI、代码执行和多轮决策。模型不只对最终答案拿 reward，还要学习中间动作、状态观察和错误恢复。

这解释了为什么“多模态”与“Agent”会在同一代合流：真实 Agent 看到的往往不是纯文本，而是截图、文档、代码编辑器和工具返回值。

## Qwen3.6：骨干延续，优化真实编码与 Agent 工作流

截至本文时间，Qwen3.6 已公开 35B-A3B 与 27B 等模型。它继续使用 `qwen3_5` 系混合架构，没有再引入一种新的 attention。

主要升级放在：

- 仓库级代码理解和多文件修改；
- 前端页面生成与视觉反馈；
- 长时 Agent 任务的稳定性；
- 多轮对话中保留历史 thinking 状态，而不是每轮完全重启推理；
- 更真实的软件工程和工具使用数据。

因此 Qwen3.6 应理解为**架构成熟后的能力迭代**。版本号变大不等于每次都要更换 Transformer block。

## Coder、Math、VL 分支应该放在什么位置

| 分支 | 主要变化 | 不应误解为 |
| --- | --- | --- |
| Qwen-Coder | 代码仓库、执行反馈、FIM、Agent 工具数据 | 一定使用全新 attention |
| Qwen-Math / QwQ | 数学数据、CoT/PoT、可验证 RL | 下一代主干版本号 |
| Qwen-VL | vision encoder、视觉 token、跨模态训练 | 与同名纯文本模型参数完全通用 |
| Thinking | 推理后训练与 chat template | Base 模型天然输出同样推理格式 |
| Instruct | SFT、偏好优化、安全与工具模板 | 可直接继续预训练的原始 Base checkpoint |

阅读模型卡时应先确定**架构家族**，再确定**模态、训练阶段与专业分支**。

## 把 Qwen 演进压缩成四次关键跨越

### 1. Token 效率

<div class="math-display">\[
\text{英文中心词表}\rightarrow
\text{152K 多语言 tokenizer}\rightarrow
\text{119/201 种语言训练}
\]</div>

### 2. KV 与长上下文

<div class="math-display">\[
\text{MHA}\rightarrow
\text{GQA}\rightarrow
\text{DCA + YaRN}\rightarrow
\text{DeltaNet 线性状态 + 周期全注意力}
\]</div>

### 3. 稀疏容量

<div class="math-display">\[
\text{Dense}\rightarrow
\text{14.3B-A2.7B}\rightarrow
\text{57B-A14B}\rightarrow
\text{235B-A22B}\rightarrow
\text{80B-A3B 高稀疏 MoE}
\]</div>

### 4. 能力形态

<div class="math-display">\[
\text{Chat/Tool}\rightarrow
\text{Coder/Math}\rightarrow
\text{Think/No-think}\rightarrow
\text{原生多模态 Agent}
\]</div>

## 初学者最容易混淆的八件事

1. <strong>Qwen3 不等于 Gated Attention。</strong>普通 Qwen3 是 QK-Norm + GQA；Qwen3-Next 才采用输出门控。
2. <strong>Gated Attention 不等于 Gated DeltaNet。</strong>前者门控全注意力输出，后者维护递归线性状态。
3. <strong>Attention gate 不等于 MoE gate。</strong>一个缩放 head，一个选择 FFN 专家。
4. <strong>DCA 不等于稀疏注意力。</strong>它主要重组分块位置编码，内容 attention 仍可为稠密 softmax。
5. <strong>A3B 不等于只需加载 3B。</strong>80B 权重仍需存储，只是每 token 激活约 3B。
6. <strong>128K/1M 不等于无损回忆。</strong>线性状态会压缩历史，周期全注意力用于补回精确检索。
7. <strong>Coder/Math 不是另一个主版本。</strong>多数差异来自数据和后训练。
8. <strong>模型卡中的吞吐倍数不能跨框架照搬。</strong>kernel、量化、batch、上下文长度和专家并行都会改变结果。

## 如何按需求选 Qwen

| 需求 | 建议关注 | 原因 |
| --- | --- | --- |
| 资源很少、本地端侧 | Qwen3/3.5 小尺寸 Dense | 生态成熟，部署简单 |
| 传统服务栈、通用文本 | Qwen2.5 或 Qwen3 Dense | GQA 支持广，框架兼容好 |
| 数学/代码深度推理 | Qwen3 Thinking 或相应 Coder/Math | 统一推理训练与专业数据 |
| 超长上下文、高吞吐 | Qwen3-Next | 75% DeltaNet，原生 262K |
| 图像、视频和 GUI Agent | Qwen3.5/3.6 | Early Fusion 多模态与 Agent RL |
| 研究注意力新架构 | Qwen3-Next Base | Gated Attention、DeltaNet、MTP、稀疏 MoE 集中出现 |

部署 Qwen3-Next 及以后模型时，要确认框架支持专用线性注意力 kernel；若退化为低效通用实现，理论复杂度优势不会自动变成真实吞吐。

## 学完后的自测题

1. Qwen1.5 哪些尺寸使用 GQA？为什么不能说它已经全系 GQA？
2. DCA 和 YaRN 分别在长上下文链路上解决什么问题？
3. Qwen3 的 QK-Norm 位于哪里，为什么它不是 attention gate？
4. softmax 权重和为 1，为什么会产生“无法什么都不读”的问题？
5. Gated DeltaNet 的误差项 <span class="math-inline">\(v_t-S_{t-1}k_t\)</span> 表示什么？
6. 为什么 Qwen3-Next 每四层仍保留一层全注意力？
7. Qwen3.5 的 Early Fusion 与后接视觉 adapter 有什么训练差别？
8. Qwen3.6 为什么可以能力升级而不更换骨干？

能回答这些问题，就能从型号名称反推出 Qwen 的注意力、MoE、上下文和后训练路线。

## 官方资料

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [Qwen1.5 官方发布说明](https://qwenlm.github.io/blog/qwen1.5/)
- [Qwen1.5-MoE 官方说明](https://qwenlm.github.io/blog/qwen-moe/)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [Qwen2 官方发布说明](https://qwenlm.github.io/blog/qwen2/)
- [Qwen2.5 官方发布说明](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 官方发布说明](https://qwenlm.github.io/blog/qwen3/)
- [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
- [Qwen3-Next 官方模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen3.5 官方模型卡示例](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)
- [Qwen3.6 官方仓库](https://github.com/QwenLM/Qwen3.6)

## 总结

Qwen 的演进不是一条“参数越来越大”的直线。Qwen 1 解决多语言 token 与工具基础；Qwen1.5 完善尺寸和生态；Qwen2 把 GQA、DCA/YaRN 与 MoE 合入主线；Qwen2.5 用数据和后训练做强代码、数学与结构化输出；Qwen3 统一思考和非思考模式；Qwen3-Next 才通过 Gated Attention、Gated DeltaNet 与高稀疏 MoE 真正更换骨干；Qwen3.5/3.6 则把这套高效骨干扩成原生多模态 Agent。

如果只记一个判断顺序，请记住：**先看模型是 Dense 还是 MoE，再看 attention 是 GQA 还是混合 DeltaNet，最后看它接受了哪种专业数据和后训练。**
