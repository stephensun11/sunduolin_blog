---
title: "LLM 量化部署详解：精度、显存与真实加速之间"
date: 2026-07-12T10:00:00+08:00
draft: false
summary: "解释低比特量化的数学基础、GPTQ/AWQ/SmoothQuant、KV Cache 量化，以及显存降低与真实加速的区别。"
categories: ["AiTech"]
subcategories: ["部署实践"]
topics: ["模型压缩"]
tags: ["LLM", "Quantization", "GPTQ", "AWQ", "SmoothQuant", "KV Cache"]
---

量化把浮点权重或激活映射到更低精度表示。它可以减少模型文件、显存占用和内存带宽，并在硬件有对应 kernel 时提高吞吐。

但三个结果不能混为一谈：文件变小、显存降低和推理变快。一个 4-bit 模型可能显著节省存储，却因为反量化开销、kernel 不匹配或 batch 形状太小而没有加速。严谨部署必须同时验证数值质量和真实系统性能。

## 先读懂 W4A16、W8A8 与 KV8

常见记法：

- **W4A16**：权重 4 bit，激活用 FP16/BF16；
- **W8A8**：权重和激活都 8 bit；
- **W4A8**：权重 4 bit、激活 8 bit；
- **FP8**：使用具体 FP8 格式表示权重/激活，通常需要缩放；
- **KV8/KV4**：KV cache 使用 8/4 bit 等低精度。

记法只描述目标精度，不描述 group size、scale 精度、对称性、零点、异常值处理和 kernel。两个都叫 W4A16 的文件可能不兼容，也可能有不同精度和速度。

## 线性量化的基本公式

对浮点值 <span class="math-inline">\(x\)</span>，仿射量化可写成：

<div class="math-display">\[
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right)+z,
q_{\min},q_{\max}\right)
\]</div>

反量化为：

<div class="math-display">\[
\hat x=s(q-z)
\]</div>

其中 <span class="math-inline">\(s\)</span> 是 scale，<span class="math-inline">\(z\)</span> 是 zero-point。对称量化通常令 <span class="math-inline">\(z=0\)</span>，实现简单；非对称量化能更好利用偏移分布的数值范围，但元数据和 kernel 更复杂。

量化误差为 <span class="math-inline">\(e=x-\hat x\)</span>。目标不是单独最小化每个权重的误差，而是让模型输出和任务质量尽量保持。

## 粒度决定 scale 的共享范围

| 粒度 | scale 数量 | 精度与成本 |
| --- | --- | --- |
| per-tensor | 整个张量一个 | 元数据少，对 outlier 敏感 |
| per-channel | 每个输出/输入通道一个 | 精度更好，依赖 kernel 支持 |
| per-group | 每若干连续权重一个 | W4 常见折中 |
| per-token activation | 每个 token 动态 scale | 适应激活变化，但有运行时开销 |

group 越小通常越能适应局部分布，但 scale 元数据、访存和 kernel 复杂度越高。比较模型大小时要把 scale 和 zero-point 也算进去。

## 为什么 outlier 是核心难题

如果一个分组大多数值接近零，少数值很大，统一 scale 为了覆盖大值会让小值落到很少的离散格点；若缩小范围，又会裁剪大值。

Transformer 激活中某些通道会出现系统性大值。[LLM.int8()](https://arxiv.org/abs/2208.07339) 将异常特征维度用较高精度计算，其余部分使用 INT8，从而兼顾精度与内存。

异常值不是“删除掉就好”。它们可能承载重要特征，正确策略是隔离、平滑或使用更细粒度表示。

## PTQ 与 QAT

### Post-Training Quantization（PTQ）

在模型训练完成后量化，通常只需要少量校准数据，不更新或只局部调整权重。优点是便宜、快速；低 bit 下精度更依赖算法和校准集。

### Quantization-Aware Training（QAT）

训练时模拟量化和反量化，让模型适应误差。离散 round 不可导，常用 straight-through estimator 近似反向。QAT 成本更高，但在极低 bit 或敏感模型上可能更稳。

QLoRA 属于“量化冻结底座上训练适配器”，不等同于为最终整数推理做完整 QAT。

## GPTQ：用二阶信息补偿权重误差

[GPTQ](https://arxiv.org/abs/2210.17323) 是一次性权重量化方法，基于近似二阶信息逐列量化并更新未量化权重，以补偿当前量化误差。

直觉上，若层输出为 <span class="math-inline">\(Y=WX\)</span>，不应只最小化 <span class="math-inline">\(\|W-\hat W\|\)</span>，而应关注校准输入上的输出误差：

<div class="math-display">\[
\min_{\hat W}\|WX-\hat W X\|_2^2
\]</div>

输入相关的 Hessian 近似告诉算法哪些方向更敏感。GPTQ 常用于 W4A16 权重量化，适合降低解码阶段的权重带宽。

校准数据若与目标域完全不匹配，误差补偿也会偏向错误分布。

## AWQ：保护重要权重通道

[AWQ](https://arxiv.org/abs/2306.00978) 的观察是，并非所有权重同等重要；激活幅度可以帮助识别少量显著权重通道。它通过基于激活统计的缩放保护这些权重，再进行低 bit 权重量化。

AWQ 不把所有 activation 存成量化值，它是 activation-aware 的 weight quantization。名字里的 activation-aware 描述校准依据，不代表最终一定是 W4A8。

## SmoothQuant：把激活难度迁移到权重

[SmoothQuant](https://arxiv.org/abs/2211.10438) 面向 W8A8。对线性层 <span class="math-inline">\(Y=XW\)</span>，引入按通道尺度矩阵 <span class="math-inline">\(S\)</span>：

<div class="math-display">\[
Y=(XS^{-1})(SW)
\]</div>

这个变换在浮点数学上等价，却能把激活通道的异常尺度迁移到相对容易量化的权重。平滑强度控制迁移程度：过少不能处理激活 outlier，过多会让权重更难量化。

SmoothQuant 主要帮助 activation quantization；与 GPTQ/AWQ 的目标和典型部署路径不同。

## Weight-only 为什么常适合 decode

LLM 推理有两个阶段：

- **prefill**：一次处理全部输入 token，大矩阵乘通常更 compute-bound；
- **decode**：每步只生成少量 token，反复读取权重，常更 memory-bandwidth-bound。

W4A16 减少权重读取，可能显著帮助 decode。但计算前往往要反量化，能否加速取决于 fused kernel 和 batch 形状。

W8A8/FP8 同时降低权重与激活精度，更有机会利用低精度 Tensor Core 加速大 GEMM，因此可能更适合高吞吐 prefill。这里的“可能”必须通过目标硬件实测。

## KV cache 量化为什么越来越重要

每层保存 key 和 value。粗略忽略分组元数据，batch 为 <span class="math-inline">\(B\)</span>、序列长度 <span class="math-inline">\(L\)</span>、层数 <span class="math-inline">\(N\)</span>、KV 头数 <span class="math-inline">\(H_{kv}\)</span>、头维 <span class="math-inline">\(D\)</span>、每元素字节数 <span class="math-inline">\(b\)</span> 时：

<div class="math-display">\[
M_{KV}\approx 2BNLH_{kv}Db
\]</div>

前面的 2 对应 K 与 V。长上下文和高并发下，KV cache 可能超过权重显存。KV8 能近似减半 BF16 KV 的存储，KV4 更进一步。

但 KV cache 是随 token 动态生成的，scale 计算和反量化位于热路径；量化误差还会在后续所有 attention step 中被重复使用。必须按上下文长度评测，而不能只测短回答 PPL。

## 量化对象不只是一组权重

常见敏感模块包括：

- embedding 与 LM head；
- 第一层和最后一层；
- attention 输出投影；
- router 与小型控制模块；
- normalization 参数；
- logits 计算和 softmax；
- MoE 中很少被校准样本激活的专家。

很多方案对部分模块保留更高精度。最终 bit-width 应按实际存储的混合配置报告，不应把整个模型笼统称为“纯 4 bit”。

## 校准集怎样选择

校准集用于估计权重重要性、激活范围或误差补偿。它应覆盖：

- 生产语言与领域；
- 真实输入长度分布；
- 代码、数学、表格等特殊格式；
- chat template 与 system prompt；
- MoE 的不同专家路由；
- 极端但合法的长上下文。

校准集不需要像训练集一样大，但要有代表性。还必须与最终测试集隔离，否则量化参数会间接适配测试数据。

## 量化误差怎样验证

### 层级数值检查

比较浮点与量化模型的层输出、logits 余弦相似度、最大误差和 KL divergence，用于定位敏感层。

### 语言建模指标

在固定 tokenizer 和数据上比较 loss/PPL，适合快速回归，但不能代表所有任务。

### 下游能力

覆盖知识、数学、代码、长上下文、工具格式、事实性和目标领域。低平均损失不保证 instruction following 不退化。

### 生成行为

检查重复、乱码、EOS、拒答、JSON 合法性和输出长度。量化可能改变接近决策边界的 token 排序，从而放大到完全不同的长文本。

### 系统性能

记录模型加载时间、静态与峰值显存、TTFT、TPOT、请求/token 吞吐、P95/P99 和功耗。必须对齐并发与长度分布。

## 显存估算为什么总比文件大

权重理论大小约为：

<div class="math-display">\[
M_{w,\text{ideal}}=P\times\frac{b_w}{8}
\]</div>

实际还包括：

- scale 和 zero-point；
- 未量化层；
- KV cache；
- 临时反量化与 GEMM workspace；
- CUDA graph、通信 buffer 和 allocator 碎片；
- tokenizer、runtime 与多 adapter 状态。

部署前应使用实际引擎测峰值，而不是用模型文件大小决定 GPU 容量。

## 格式与 kernel 才是部署接口

GGUF、GPTQ、AWQ、bitsandbytes、FP8 checkpoint 等格式面向不同运行时。选择前要确认：

- 引擎是否原生支持该格式和 group size；
- GPU 架构是否有高效 kernel；
- tensor/expert parallel 是否兼容；
- LoRA、prefix cache、speculative decoding 是否可组合；
- 是否需要启动时转换或反量化；
- 长上下文下 KV cache 格式是否匹配。

若引擎启动时把 4-bit 权重展开为 FP16，文件虽小，运行显存和速度优势可能消失。

## 常见失败模式

### 只测 perplexity

量化可能在平均 PPL 上变化很小，却破坏少数关键任务、长上下文或结构化输出。

### 用随机文本校准领域模型

随机样本没有激活真实的敏感通道，量化参数对生产分布不可靠。

### 位数越低越划算

极低 bit 可能需要更复杂解码、更多 scale 和混合精度补丁，最终速度和质量未必更优。

### GPU 支持 INT4 就一定加速

还需要形状、布局、kernel 和推理引擎匹配。硬件指令存在不等于完整路径使用它。

### 把量化误差当成随机噪声

误差与层、通道、输入和任务相关，可能形成系统性偏差。

## 一条稳妥的选型流程

1. 冻结浮点基线、评测集和服务负载；
2. 明确瓶颈是权重显存、KV cache、带宽还是计算；
3. 选择候选 W4A16、W8A8/FP8 或 KV 量化路径；
4. 用代表性校准集生成 checkpoint；
5. 做层级数值和完整能力回归；
6. 在目标引擎与硬件上压测；
7. 对质量、成本和延迟做 Pareto 比较；
8. 保存量化工具、配置、校准数据版本和校验和。

## 总结

量化不是单一压缩按钮，而是数值表示、校准算法、kernel 和服务负载共同决定的部署方案。

W4A16 常针对权重带宽，W8A8/FP8 更强调低精度计算，KV 量化解决长上下文状态显存。无论选择 GPTQ、AWQ、SmoothQuant 还是其他方法，最终问题都相同：在真实任务上损失多少质量，在真实硬件上节省多少资源。只有两本账同时成立，量化才算成功。

## 参考资料

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
