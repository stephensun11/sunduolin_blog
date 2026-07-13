---
title: "LLM 训练时到底发生了什么：从一段文本到一次参数更新"
date: 2026-07-13T09:00:00+08:00
draft: false
summary: "沿着一批真实文本完整追踪 Tokenize、前向传播、交叉熵、反向传播、梯度同步和 AdamW 更新，解释 LLM 训练的全过程。"
categories: ["AiTech"]
subcategories: ["LLM训练"]
topics: ["训练流程"]
tags: ["LLM", "预训练", "前向传播", "反向传播", "交叉熵", "AdamW"]
---

大模型训练最容易被一句话概括成：“输入很多文本，预测下一个 token，再反向传播。”

这句话是对的，但它省略了几乎所有真正需要理解的细节：文本怎样变成张量，标签为什么向左错一位，Transformer 在每一层做什么，交叉熵怎样形成，梯度从哪里来，多张 GPU 怎样合成一次更新，以及 checkpoint 究竟保存了什么。

本文不从某个训练框架的配置文件出发，而是沿着一批数据走完整条链路。读完后，你应该能回答两个问题：

1. 一次训练 step 内部发生了什么；
2. 数万亿 token 的训练任务怎样由这些 step 组成。

## 先区分三个时间尺度

“LLM 训练流程”至少包含三个不同尺度。

### 项目尺度：几周到几个月

```text
确定目标与预算
  -> 准备数据和 tokenizer
  -> 设计模型与并行策略
  -> 小规模验证
  -> 大规模预训练
  -> 中期评测与故障恢复
  -> SFT / 偏好对齐
  -> 最终评测与发布
```

### 数据尺度：一个 epoch 或若干万亿 token

数据被分片、打乱、采样和打包。训练器持续读取 batch，直到达到预定 token 或计算预算。

### 更新尺度：几十到几千毫秒

```text
读取 micro-batch
  -> 前向传播
  -> 计算 loss
  -> 反向传播
  -> 梯度同步与累积
  -> 梯度裁剪
  -> optimizer 更新
  -> 学习率调度
```

下面先讲项目尺度，再把镜头拉近到一次更新。

## 第零步：训练前先定义“训练什么”

训练不是先启动 GPU，再看能得到什么模型。至少要提前固定：

- 模型结构：层数、隐藏维度、注意力头、FFN/MoE、词表和上下文长度；
- 训练目标：自回归语言建模、掩码建模、SFT 或偏好目标；
- 数据配方：语言、代码、数学和领域数据比例；
- 预算：训练 token、GPU 时长、峰值显存和 checkpoint 频率；
- 评测：验证损失、能力、安全和目标领域指标；
- 退出条件：预算耗尽、验证集不再改善或出现不可接受的风险。

参数量、数据量和计算量要一起规划。在固定计算预算下，一味扩大参数而不给足训练 token，可能得到“参数很多但训练不足”的模型。

## 第一步：原始文本变成训练语料

假设原始数据里有一句话：

```text
猫喜欢吃鱼。
```

真实语料还会经历正文抽取、语言识别、质量过滤、精确与近似去重、隐私处理、评测集去污染和数据配比。最终训练器读取的不是网页，而是经过版本化的数据 shard。

数据清洗不能由“看起来正常”验收。需要记录每个来源删除了多少文档、保留了多少 token，以及过滤规则对下游小模型有什么影响。更完整的数据工程可参见站内文章《LLM 预训练数据工程》。

## 第二步：Tokenizer 把文本映射为整数

为了便于演算，假设一个极小词表：

| token | id |
| --- | ---: |
| `<bos>` | 1 |
| `猫` | 12 |
| `喜欢` | 35 |
| `吃` | 48 |
| `鱼` | 27 |
| `。` | 6 |
| `<eos>` | 2 |

那么文本可能变成：

```text
[1, 12, 35, 48, 27, 6, 2]
```

真实 tokenizer 可能把“喜欢”切成一个或多个 token。模型只看到整数 id，不直接看到汉字。

Tokenizer 还决定：

- 词表大小 <span class="math-inline">\(V\)</span>；
- 不同语言的压缩率；
- embedding 和输出层的尺寸；
- 特殊 token 与对话模板；
- 一个训练样本实际占多少 token。

Tokenizer 一旦在大规模训练中途更换，id 与语义的对应关系就变了，通常不能当成普通配置热更新。

## 第三步：切分、packing 与 batch

模型每次处理固定或受限长度的序列。假设上下文长度 <span class="math-inline">\(T=6\)</span>，可以构造：

```text
input_ids = [1, 12, 35, 48, 27, 6]
labels    = [12, 35, 48, 27, 6, 2]
```

标签向左错一位，是因为每个位置都要预测下一个 token：

| 当前位置输入 | 目标 token |
| --- | --- |
| `<bos>` | `猫` |
| `猫` | `喜欢` |
| `猫 喜欢` | `吃` |
| `猫 喜欢 吃` | `鱼` |
| `猫 喜欢 吃 鱼` | `。` |
| `... 。` | `<eos>` |

在 decoder-only 模型中，通常不需要真的保存两份错位数组；训练代码可以把 logits 的前 <span class="math-inline">\(T-1\)</span> 个位置与 input 的后 <span class="math-inline">\(T-1\)</span> 个 token 对齐。

上表展示的是概念上的输入-目标对齐。实现可以在数据层提前 shift，也可以让 `labels` 先与原 token 序列同位置保存，再在计算 loss 时用 `logits[:, :-1]` 对齐 `labels[:, 1:]`。后文伪代码采用第二种，二者不能同时做，否则会重复错位。

### 多条短文本怎样装进一个序列

如果每条样本都 padding 到最大长度，会浪费计算。packing 会把多条短文本装入一个训练序列：

```text
文档 A <eos> 文档 B <eos> 文档 C <eos>
```

必须明确文档边界。若没有 EOS 或隔离 mask，模型会把两个不相关文档误认为自然连续文本。

### 一个 batch 的形状

设 micro-batch 有 <span class="math-inline">\(B=4\)</span> 条序列，每条长度 <span class="math-inline">\(T=2048\)</span>：

```text
input_ids:      [4, 2048]
attention_mask: [4, 2048] 或可隐式生成
labels:         [4, 2048]
```

这张量随后被送到 GPU。

## 第四步：Embedding 把 id 变成向量

模型有一个 embedding 矩阵：

<div class="math-display">\[
E\in\mathbb{R}^{V\times d}
\]</div>

其中 <span class="math-inline">\(V\)</span> 是词表大小，<span class="math-inline">\(d\)</span> 是隐藏维度。对每个 id 查表后：

```text
[B, T] -> [B, T, d]
```

如果 <span class="math-inline">\(B=4,T=2048,d=4096\)</span>，隐藏状态形状就是 `[4, 2048, 4096]`。

同一个 token 初始总是查到同一个向量，但经过位置编码和多层上下文交互后，它在不同句子中的隐藏状态会不同。

## 第五步：每个 Transformer block 做什么

现代 decoder-only block 常采用 PreNorm：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

一个 block 主要有两种计算。

### 1. Causal Self-Attention

线性投影得到 query、key、value：

<div class="math-display">\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]</div>

单头注意力为：

<div class="math-display">\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
\]</div>

因果 mask <span class="math-inline">\(M\)</span> 把未来位置设为负无穷，使位置 <span class="math-inline">\(t\)</span> 只能读取 <span class="math-inline">\(\le t\)</span> 的 token。否则模型训练时会偷看答案。

### 2. MLP / FFN

SwiGLU 形式常写成：

<div class="math-display">\[
\operatorname{FFN}(x)=W_{\text{down}}
\left(\operatorname{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x\right)
\]</div>

Attention 负责 token 间的信息交互，FFN 对每个 token 的表示做非线性变换。堆叠几十或上百个 block 后，张量形状通常仍是 `[B, T, d]`，但内容已被反复更新。

## 第六步：隐藏状态变成词表 logits

最后一层输出经过归一化和语言模型头：

<div class="math-display">\[
Z=HW_{\text{vocab}}^\top
\]</div>

得到：

```text
hidden_states: [B, T, d]
logits:        [B, T, V]
```

`logits[b, t, :]` 是第 <span class="math-inline">\(b\)</span> 条序列在位置 <span class="math-inline">\(t\)</span> 对整个词表的未归一化分数。

softmax 将 logits 转为概率：

<div class="math-display">\[
p_j=\frac{e^{z_j}}{\sum_{k=1}^{V}e^{z_k}}
\]</div>

训练时通常直接使用融合的 cross-entropy kernel，不必把完整概率张量长期保存。

## 第七步：交叉熵怎样给模型打分

假设某个位置的正确 token 是“鱼”，为了演算只看三个候选：

| token | 模型概率 |
| --- | ---: |
| 鱼 | 0.70 |
| 肉 | 0.20 |
| 水 | 0.10 |

该位置的负对数似然为：

<div class="math-display">\[
\ell=-\log 0.70\approx 0.357
\]</div>

如果模型只给正确 token 0.01 概率：

<div class="math-display">\[
\ell=-\log 0.01\approx 4.605
\]</div>

错误且自信的预测损失更大。整个 batch 的 loss 是所有有效目标 token 的平均或总和：

<div class="math-display">\[
\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}
\log p_\theta(y_i\mid x_i)
\]</div>

padding、被忽略的 prompt token 和无效位置不能计入 <span class="math-inline">\(N\)</span>。

### logits 梯度的直觉

对 softmax + cross-entropy，单个位置的 logits 梯度有一个漂亮形式：

<div class="math-display">\[
\frac{\partial \ell}{\partial z_j}=p_j-\mathbf{1}[j=y]
\]</div>

若正确 token“鱼”的概率是 0.70，它的梯度是 <span class="math-inline">\(0.70-1=-0.30\)</span>；“肉”的梯度是 0.20，“水”的梯度是 0.10。

梯度下降会提高正确 token 的 logit，同时压低错误 token 的 logits。这就是“模型从答案中学习”的最局部解释。

## 第八步：反向传播把责任分到每个参数

计算图记录了 logits 如何由 LM head、每层 FFN、Attention、Embedding 产生。链式法则把 loss 梯度从输出传回所有可训练参数：

<div class="math-display">\[
\frac{\partial\mathcal{L}}{\partial W_l}
=\frac{\partial\mathcal{L}}{\partial Z}
\frac{\partial Z}{\partial H_L}
\cdots
\frac{\partial H_l}{\partial W_l}
\]</div>

残差连接为梯度提供直接路径；归一化、初始化和学习率共同决定深层训练是否稳定。

### 一个参数更新的简单直觉

假设某个权重：

```text
w = 0.5000
gradient = 0.2000
learning_rate = 0.001
```

若用最简单的 SGD：

<div class="math-display">\[
w' = w-\eta g=0.5000-0.001\times0.2=0.4998
\]</div>

真实 LLM 常用 AdamW，它会维护梯度的一阶矩和二阶矩，再进行偏差修正与解耦权重衰减。因此上面的数字只解释“沿负梯度方向更新”，不是 AdamW 的完整计算。

## 第九步：AdamW 怎样更新参数

对梯度 <span class="math-inline">\(g_t\)</span>：

<div class="math-display">\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]</div>

<div class="math-display">\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]</div>

偏差修正后，简化更新为：

<div class="math-display">\[
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]</div>

一阶矩提供类似动量的平滑，二阶矩按历史梯度尺度调整每个参数的步幅。AdamW 的 weight decay 与把 L2 正则直接加进 Adam 损失并不等价。

一次 `optimizer.step()` 完成后，模型参数发生微小变化。下一批文本将看到略有不同的模型。

## 第十步：为什么要梯度累积

目标全局 batch 可能放不进显存。假设：

```text
每卡 micro-batch = 2 条序列
数据并行 GPU = 8
梯度累积步数 = 4
```

则一次 optimizer update 对应：

<div class="math-display">\[
B_{\text{global}}=2\times8\times4=64\text{ 条序列}
\]</div>

若每条序列有 2048 个有效 token，则约为 131072 token/update。

前 3 次 micro-step 只累加梯度，第 4 次才执行 optimizer update。日志里的 `step` 必须说明指 micro-step 还是 update step。

当各 micro-batch 有效 token 数不同，不能简单平均它们的 mean loss；应按整个逻辑 batch 的有效 token 数加权。

## 第十一步：多张 GPU 怎样得到同一个梯度

在 Distributed Data Parallel（DDP）中，每个数据并行 rank 保存一份模型，读取不同数据。

```text
GPU 0: batch A -> gradient g0
GPU 1: batch B -> gradient g1
GPU 2: batch C -> gradient g2
GPU 3: batch D -> gradient g3
```

反向传播期间通过 all-reduce 求和/平均：

<div class="math-display">\[
g=\frac{1}{W}\sum_{r=1}^{W}g_r
\]</div>

然后每个 rank 用相同梯度执行相同更新，参数继续保持一致。

FSDP、ZeRO、tensor parallel、pipeline parallel 和 expert parallel 会进一步切分参数、梯度、优化器状态或计算，但数学目标仍是对同一个全局 batch 求梯度。

## 混合精度期间发生了什么

大模型训练常让矩阵乘使用 BF16/FP16/FP8 等低精度，让某些累积、归一化、softmax 或优化器状态保持更高精度。

它不是把所有张量统一改成半精度。典型原因是：

- 低精度矩阵乘吞吐更高；
- 权重和激活占用更少显存与带宽；
- 高精度累积避免小误差快速放大；
- FP16 可能需要 loss scaling 防止小梯度下溢；
- BF16 范围更大，但仍可能出现 inf/NaN。

使用 gradient scaler 时，必须先 unscale 梯度，再做梯度裁剪。

## Activation checkpointing 为什么能省显存

反向传播需要前向中间激活。普通训练保存每层激活；activation checkpointing 只保存少数边界，在反向时重新计算其余激活。

```text
不重计算：更多显存，较少 FLOPs
重计算：   更少显存，更多 FLOPs
```

它减少的是激活显存，不是 optimizer state。长序列训练常同时使用 activation checkpointing、FlashAttention 和模型状态分片。

## 一次更新的 PyTorch 风格伪代码

下面假设每个 micro-batch 有相同有效 token 数：

```python
model.train()
optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(loader):
    input_ids = batch["input_ids"].to(device)  # [B, T]
    # labels 与 input_ids 同位置保存；不参与监督的位置已写成 -100。
    # 下面在 loss 中统一完成 next-token shift。
    labels = batch["labels"].to(device)        # [B, T]

    with autocast(dtype=torch.bfloat16):
        logits = model(input_ids)               # [B, T, V]
        loss = cross_entropy(
            logits[:, :-1].contiguous().view(-1, vocab_size),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        loss = loss / grad_accum_steps

    loss.backward()

    if (micro_step + 1) % grad_accum_steps == 0:
        global_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

真实分布式代码还要处理通信、混合精度 scaler、变长 token 加权、无效 batch、checkpoint 和异常恢复。

## 一个完整训练任务怎样反复执行这些 step

假设要训练 7B 模型，预算 1T token，全局每次更新 2M 有效 token。理论更新次数约为：

<div class="math-display">\[
\frac{10^{12}}{2\times10^6}=500000\text{ updates}
\]</div>

每次更新都执行前向、反向和优化器更新。随着训练推进：

- 训练 loss 总体下降，但不会严格单调；
- 学习率先 warmup，再衰减；
- 模型从局部字符和词法规律，逐渐学到结构、事实和任务模式；
- 验证集按固定间隔评测；
- checkpoint 定期保存并做恢复演练；
- 数据 loader 记录已消费 shard 和随机状态。

训练不是“完整读完一篇文章再记住它”。每个 batch 只产生一次很小的统计更新，能力是海量更新共同形成的。

## 预训练之后为什么还要 SFT

预训练目标是预测互联网文本的下一个 token。它会让模型具备语言和知识能力，却不必然学会按照用户意图回答。

SFT 仍使用 token-level cross-entropy，但输入变为对话模板：

```text
<system>你是一个可靠的技术助手</system>
<user>解释什么是梯度累积</user>
<assistant>梯度累积是...</assistant>
```

常见做法只让 assistant 回答 token 参与 loss，system 和 user token 作为条件但标签设为 ignore。

随后还可以进行 DPO、RLHF 或可验证奖励强化学习。这些阶段改变反馈形式，但底层依然包含模型采样、概率计算、梯度和参数更新。

## 训练中到底要观察什么

| 指标 | 说明 |
| --- | --- |
| train / validation loss | 拟合和泛化趋势 |
| learning rate | scheduler 是否与 update 对齐 |
| gradient norm | 梯度尖峰、爆炸或异常平坦 |
| update / weight norm | 参数实际变化幅度 |
| tokens/s | 端到端吞吐 |
| padding ratio | 数据打包效率 |
| overflow / skipped steps | 混合精度稳定性 |
| 每域 validation loss | 是否牺牲某种语言或领域 |
| GPU 与网络利用率 | 计算、数据或通信瓶颈 |

平均 loss 会掩盖局部问题。训练日志还应记录数据 shard、样本 ID 和各 rank 状态，以便从 loss spike 回溯到具体输入。

## loss 突然爆炸时怎样排查

1. 确认不是日志分母或 all-reduce 错误；
2. 检查当前 batch 是否乱码、全 mask、极端重复或数据损坏；
3. 定位第一个出现 inf/NaN 的层；
4. 检查学习率、gradient scaler 和裁剪顺序；
5. 检查恢复训练时 optimizer/scheduler 是否完整加载；
6. 从 spike 前 checkpoint 用相同数据顺序重放。

梯度裁剪可以缓解偶发尖峰，但不能修复持续过高学习率或坏数据。

## checkpoint 不是只有模型权重

要实现真正可恢复训练，至少保存：

- 模型参数；
- AdamW 一阶和二阶矩；
- scheduler 和 gradient scaler；
- 全局 update、已见 token 数；
- 数据 sampler 和 shard 游标；
- CPU/GPU 随机数状态；
- tokenizer、配置和代码版本；
- 分布式分片元数据。

只加载权重再继续跑，会丢失优化器动量和学习率时间轴，这更像开启一个新训练阶段。

## 三个常见误解

### 模型是在保存句子吗

训练目标会让模型记住部分高频或重复文本，但参数主要编码的是跨样本统计结构。去重、隐私处理和记忆评测仍然必要。

### 每个参数都对应一个知识点吗

知识通常分布在大量参数和激活模式中。同一个参数参与许多输入，不能把它直接命名为“某事实参数”。

### loss 下降就证明模型更聪明吗

loss 可能因为数据重复、泄漏或模板变简单而下降。必须结合独立验证集、去污染 benchmark 和真实任务评测。

## 总结

一次 LLM 训练更新可以压缩成六件事：

1. 文本被 tokenizer 变成 id，并打包成 batch；
2. Embedding 和 Transformer 把 `[B,T]` 变成 `[B,T,V]` 的 logits；
3. 交叉熵比较每个位置的预测与下一个 token；
4. 反向传播把误差分配到所有可训练参数；
5. 多卡同步和梯度累积形成全局梯度；
6. AdamW 用很小的步幅更新参数。

大模型能力并不是某一步突然出现的，而是几十万次更新对海量数据分布持续拟合的结果。理解这条链路后，训练配置中的 batch、sequence length、learning rate、precision 和并行策略就不再是孤立参数，而是同一个计算过程的不同控制面。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
