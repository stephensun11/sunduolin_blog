---
title: "参数高效微调详解：LoRA、QLoRA 与适配器工程"
date: 2026-07-12T09:20:00+08:00
draft: false
summary: "推导 LoRA 的低秩更新，解释 QLoRA、DoRA、显存核算、目标模块选择、适配器合并与评测方法。"
categories: ["AiTech"]
subcategories: ["LLM训练"]
topics: ["参数高效微调"]
tags: ["LLM", "PEFT", "LoRA", "QLoRA", "DoRA", "微调"]
---

微调大模型时，真正昂贵的往往不是一次前向计算，而是为全部参数保存梯度和优化器状态。参数高效微调（PEFT）保留预训练权重，只训练一小组新增参数，从而降低训练显存和多任务存储成本。

LoRA 是其中最常用的方法，但“给模型加一个 LoRA”仍然包含很多决定：加在哪些线性层、rank 多大、是否量化底座、是否训练 embedding、怎样保存和合并，以及怎样判断效果下降来自容量不足还是数据问题。

## 全量微调为什么贵

设模型有 <span class="math-inline">\(P\)</span> 个参数。以混合精度 AdamW 为例，训练通常要保存：

- 模型权重；
- 梯度；
- 一阶矩；
- 二阶矩；
- 某些实现中的 FP32 master weight；
- 与 batch 和序列长度相关的激活。

因此不能用“参数量乘以权重字节数”估算训练显存。一个 7B 模型的 BF16 权重约 14 GB，但全量训练的模型状态远大于 14 GB，尚未包括激活与通信缓冲区。

PEFT 主要减少的是可训练参数对应的梯度和优化器状态；底座权重仍需参与前向与反向计算，因此它不会按相同比例减少计算量或激活显存。

## LoRA 的低秩假设

对预训练线性层：

<div class="math-display">\[
y=Wx,\qquad W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
\]</div>

全量微调会学习同尺寸更新 <span class="math-inline">\(\Delta W\)</span>。LoRA 冻结 <span class="math-inline">\(W\)</span>，用两个小矩阵表示更新：

<div class="math-display">\[
\Delta W=BA,
\quad A\in\mathbb{R}^{r\times d_{\text{in}}},
\quad B\in\mathbb{R}^{d_{\text{out}}\times r}
\]</div>

于是前向变成：

<div class="math-display">\[
y=Wx+\frac{\alpha}{r}BAx
\]</div>

其中 <span class="math-inline">\(r\)</span> 是 rank，<span class="math-inline">\(\alpha\)</span> 是缩放系数。新增参数量从 <span class="math-inline">\(d_{\text{out}}d_{\text{in}}\)</span> 变为：

<div class="math-display">\[
r(d_{\text{in}}+d_{\text{out}})
\]</div>

当 <span class="math-inline">\(r\)</span> 远小于层宽时，节省显著。[LoRA 原论文](https://arxiv.org/abs/2106.09685) 的核心假设是，下游适配所需的权重更新具有较低的内在秩；它不是说原始权重矩阵本身低秩。

## 初始化为什么要让增量从零开始

常见实现随机初始化 <span class="math-inline">\(A\)</span>，把 <span class="math-inline">\(B\)</span> 初始化为零。这样初始时 <span class="math-inline">\(BA=0\)</span>，模型输出与原底座一致，同时梯度可以先更新 <span class="math-inline">\(B\)</span>。

如果两个矩阵都初始化为零，早期梯度可能无法打破对称；如果增量一开始很大，则会在第一个 step 就扰动预训练表示。

LoRA dropout 只作用于适配分支，通常用于正则化。它的最佳值依赖数据规模和噪声，不应机械设置。

## 应该把 LoRA 加在哪里

Transformer 中常见候选包括：

- attention 的 `q_proj`、`k_proj`、`v_proj`、`o_proj`；
- MLP 的 gate/up/down projection；
- embedding 或 LM head；
- MoE 的专家线性层与 router。

只适配 Q、V 的成本低，是早期常见配置；覆盖所有主要线性层通常容量更强，但参数与通信更多。目标模块名称依赖模型实现，必须在启动时打印实际匹配结果。最危险的情况不是报错，而是模式一个模块都没匹配却继续训练。

embedding 和 LM head 是否训练要看任务。新增词表 token 时，至少新 token 对应的 embedding 需要学习；若冻结整个词表，又没有其他可训练路径，新 token 很难获得合理表示。

## rank、alpha 与容量

rank 决定更新子空间的最大维度。增大 rank 会增加容量，但并不保证效果单调提升：

- 小数据上可能更容易过拟合；
- 不同层需要的 rank 可能不同；
- 数据质量和学习率可能先成为瓶颈；
- 适配更多模块有时比单层加大 rank 更有效。

缩放 <span class="math-inline">\(\alpha/r\)</span> 使更新幅度不至于随 rank 线性增加。有些变体采用 <span class="math-inline">\(\alpha/\sqrt r\)</span> 等尺度；比较实验时必须记录具体实现，而不能只写 rank。

建议以小网格做消融：固定数据、总 token、seed 和评测协议，比较 rank、target modules 与学习率。只比较最终 checkpoint 容易错过过拟合，应保留验证曲线。

## LoRA 的显存账本

LoRA 节省了可训练状态，但以下部分仍存在：

| 项目 | 是否显著减少 |
| --- | --- |
| 冻结底座权重 | 否 |
| 底座权重的梯度 | 是，不保存 |
| 底座 optimizer state | 是，不保存 |
| LoRA 参数及 optimizer state | 新增，但规模较小 |
| 激活 | 通常不会按参数比例减少 |
| attention 中间量 | 取决于 kernel 和 checkpointing |

因此长上下文微调仍可能由激活显存主导。此时需要配合 FlashAttention、activation checkpointing、packing 和较小 micro-batch。

## QLoRA：量化底座，训练 LoRA

[QLoRA](https://arxiv.org/abs/2305.14314) 将冻结的预训练权重量化为 4 bit，在反向传播时让梯度穿过反量化计算流向 LoRA 参数。关键点是：

1. 底座权重是冻结的 4-bit 表示；
2. 矩阵计算通常在 BF16/FP16 等更高精度中执行；
3. LoRA 参数与 optimizer state 保持可训练精度；
4. 这不是对 4-bit 整数权重直接做普通梯度更新。

论文提出 NF4、double quantization 和 paged optimizer。

### NF4

NF4 为近似正态分布的权重设计非均匀量化点。它利用权重分布而不是简单均匀切分数轴。所谓“信息论最优”依赖论文的分布假设，不应推广成所有张量上都绝对最优。

### Double quantization

分组量化需要为每组保存 scale。double quantization 进一步量化这些量化常数，降低元数据开销。

### Paged optimizer

它借助统一内存处理偶发显存尖峰，目标是降低 OOM 风险；频繁发生主机与设备迁移仍可能损害性能，不能把它当作无限显存。

## 4 bit 是怎样存下来的

对称均匀量化可用一个简化公式理解：

<div class="math-display">\[
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{w}{s}\right),q_{\min},q_{\max}\right),
\qquad \hat w=sq
\]</div>

实际 QLoRA 的 NF4 更复杂，并按 block 保存量化信息。显存估算要包括量化值、scale、量化状态、未量化模块、LoRA、激活和临时 workspace，不能只做 <span class="math-inline">\(P\times 0.5\)</span> 字节的理想计算。

## QLoRA 与量化推理不是一回事

QLoRA 的 4-bit 底座用于低显存微调。训练完成后的部署格式取决于推理引擎：

- 可以保留适配器，运行时加载；
- 可以先反量化并合并，再用目标引擎重新量化；
- 某些框架支持量化底座加动态适配器；
- 不同量化格式的 kernel、group size 和硬件支持不同。

“训练时能加载”不代表“生产推理会更快”。速度必须在实际引擎、GPU 和 batch/序列分布上测量。

## DoRA 在改什么

[DoRA](https://arxiv.org/abs/2402.09353) 把权重分解为方向和幅度，LoRA 主要更新方向，同时单独学习幅度。其动机是让低秩适配的更新模式更接近全量微调。

它可能提高特定任务的学习能力，但会增加实现和状态复杂度。是否优于 LoRA 取决于模型、数据和预算，不能把新变体默认当成替代品。

## SFT 时怎样构造 loss

对话模板通常包含 system、user、assistant 和特殊边界 token。两种常见目标是：

- **全序列 loss**：所有非 padding token 都参与预测；
- **response-only loss**：prompt token 的 label 设为 ignore，只训练 assistant 回复。

response-only loss 更直接优化回答，但模型仍通过 prompt 的前向表示条件化输出。哪种更好取决于模板、数据和是否希望模型学习用户文本分布。

必须做一个手工样本测试：打印 token、角色边界、attention mask 和 label，确认 assistant 起始 token、EOS 与多轮对话都被正确处理。

## 数据质量比 adapter 参数量更先决定上限

LoRA 只能改变模型行为，不能把错误标签变成正确知识。微调数据应控制：

- 指令是否明确，回答是否真正完成任务；
- 风格和安全边界是否一致；
- 多轮上下文是否存在错误角色拼接；
- 长度分布是否与生产请求接近；
- 是否包含验证集或 benchmark 答案；
- 是否过度重复同一种模板。

在小而高质量的数据上，低 rank 可能已足够；在复杂领域适配中，容量不足、覆盖不足与底座能力不足需要分别诊断。

## 合并与多适配器服务

LoRA 可在部署前合并：

<div class="math-display">\[
W_{\text{merged}}=W+\frac{\alpha}{r}BA
\]</div>

合并后不再有额外 LoRA 分支，适合单任务静态部署。但要注意：

- 合并应在足够高精度中完成；
- 量化底座通常应先反量化或使用框架支持的正确流程；
- 合并后要重新保存配置与 tokenizer；
- 多个 adapter 直接相加不保证互不干扰。

动态多适配器适合共享底座服务多个任务，但会增加缓存、批处理和路由复杂度。需要压测 adapter 切换、并发和显存碎片。

## 如何评估一个适配器

至少分四层：

1. **训练正确性**：单样本过拟合、mask、梯度和保存/加载一致；
2. **任务能力**：独立测试集上的准确率、通过率或人工偏好；
3. **通用能力回归**：基础知识、指令遵循和原任务是否退化；
4. **系统指标**：吞吐、首 token 延迟、峰值显存和 adapter 切换成本。

对生成任务不要只看训练 loss。较低的 token loss 可能来自风格模仿，却未提高事实性或任务成功率。

## 常见失败模式

### target module 匹配错误

训练参数量为零或远低于预期。启动时应列出可训练参数名、数量和占比。

### 学习率照搬全量微调

LoRA 可训练参数少且初始化不同，常用学习率范围与全量微调不同。应单独扫描，而不是默认越大越好。

### 忘记保存 tokenizer 和 chat template

同一权重配不同模板会产生完全不同的提示 token，模型表现可能显著下降。

### 在测试集上挑 checkpoint

这会把测试集变成验证集。应使用验证集选 checkpoint，测试集只做最终报告。

### 合并后没有做数值回归

合并前后的 logits 在允许误差内应一致。若重新量化，还要单独评估量化误差。

## 选择方法的决策表

| 场景 | 优先考虑 |
| --- | --- |
| 有足够集群、追求最大适配容量 | 全量微调或分阶段比较 |
| 单卡/少卡、底座能加载为 BF16 | LoRA |
| 显存连 BF16 底座都放不下 | QLoRA |
| 多租户共享一个底座 | 动态多 LoRA |
| 只需少量新词或分类头 | selective tuning / head tuning |
| LoRA 容量不足且有验证证据 | 更高 rank、更多模块或 DoRA 等变体 |

## 总结

LoRA 的价值不是“低成本获得全量微调的一切”，而是把适配问题限制到一个可控的低维更新空间。QLoRA 又进一步压缩了冻结底座的存储，让有限显存也能参与微调。

正确使用它们需要同时管理数学、数据与系统：明确低秩更新发生在哪里，正确核算显存，验证模板和 loss，区分训练量化与部署量化，并用独立评测判断是否真的学会了目标任务。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
