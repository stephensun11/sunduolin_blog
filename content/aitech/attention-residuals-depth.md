---
title: "残差注意力详解：从标准残差到 AttnRes 与 MoDA"
date: 2026-07-13T09:10:00+08:00
draft: false
summary: "解释深层 LLM 中的残差信息稀释，推导 Kimi Attention Residuals 与 Mixture-of-Depths Attention，并区分 RealFormer。"
categories: ["AiTech"]
subcategories: ["LLM最新技术"]
topics: ["深度建模"]
tags: ["LLM", "Attention Residuals", "AttnRes", "MoDA", "Residual Connection", "RealFormer"]
---

Transformer 已经用注意力解决了一个核心问题：序列中的当前 token，可以按内容选择读取哪些历史 token。

但沿着网络深度，现代 LLM 仍主要使用固定加法：每一层的输出都以单位权重写入同一条残差流。网络越来越深时，浅层特征可能被大量后续更新稀释，而且后层无法单独取回某一层的表示。

2026 年，Kimi Team 提出的 **Attention Residuals（AttnRes）** 把注意力从“序列维度”延伸到“深度维度”；同期的 **Mixture-of-Depths Attention（MoDA）** 则让每个 attention head 在一次统一 softmax 中同时读取当前层的序列 KV 和历史层的深度 KV。

它们都在解决跨层信息聚合，但机制并不相同。本文从标准残差开始推导，并把容易混淆的 RealFormer 一起讲清楚。

## 先把四个概念分开

| 名称 | 注意力发生在哪里 | 跨层保存什么 | 主要操作 |
| --- | --- | --- | --- |
| 标准残差 | 不额外做注意力 | 一个累加隐藏状态 | 固定相加 |
| RealFormer | attention score/logit | 前一层 score | score 相加后 softmax |
| AttnRes | 网络深度 | 历史层/块输出 | 深度 softmax 加权 |
| MoDA | 序列与深度的联合空间 | 历史层 depth KV | 统一 softmax 读取两类 KV |

用户提供的参考文章主要讨论后两者。标题中的“残差注意力”不能自动等同于 RealFormer。

## 标准残差连接做了两件事

设第 <span class="math-inline">\(l\)</span> 个子层的输入为 <span class="math-inline">\(h_l\)</span>，变换为 <span class="math-inline">\(f_l\)</span>。标准残差写成：

<div class="math-display">\[
h_{l+1}=h_l+f_l(h_l)
\]</div>

它最著名的作用是建立梯度捷径。对若干层求导时，Jacobian 中始终包含 identity 路径，使梯度不必完全穿过所有非线性变换。

它还有第二个作用：定义跨深度的信息聚合。展开递推：

<div class="math-display">\[
h_l=h_0+\sum_{i=0}^{l-1}f_i(h_i)
\]</div>

这意味着 embedding 和所有历史层输出都进入残差流，而且系数固定为 1。

### 为什么“固定为 1”可能有问题

随着深度增加：

- 后层只看到已经混合好的 <span class="math-inline">\(h_l\)</span>，不能单独访问某个浅层输出；
- 所有历史更新都以相同显式系数累加，不能针对 token 或层选择；
- PreNorm 残差流的尺度可能随深度增长；
- 单层新增输出在总残差流中的相对占比可能变小。

AttnRes 论文把这一现象称为 PreNorm dilution。这里要谨慎：具体增长速度和稀释程度受初始化、归一化、残差缩放、模型结构与训练动态影响，不能把它理解成所有 Transformer 都必然按同一曲线退化。

## 门控残差为什么还不够

一种自然改进是给两条路径加权：

<div class="math-display">\[
h_{l+1}=\alpha_l h_l+\beta_l f_l(h_l)
\]</div>

或者使用逐维 gate。它能控制“保留当前状态还是写入新信息”，但第 <span class="math-inline">\(l\)</span> 层仍只直接读取 <span class="math-inline">\(h_l\)</span>。

更早层的输出已经被压缩进一个固定宽度状态。若某项信息在混合中被冲淡，后层无法像 sequence attention 那样回到具体来源重新读取。

## AttnRes 的核心：沿深度做 softmax attention

Kimi Team 的思路是把历史层输出视为一组 value。

定义：

<div class="math-display">\[
v_0=h_0,\qquad v_i=f_{i-1}(h_{i-1}),\quad i\ge 1
\]</div>

第 <span class="math-inline">\(l\)</span> 层不再接收固定求和，而是：

<div class="math-display">\[
h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}v_i,
\qquad \sum_{i=0}^{l-1}\alpha_{i\to l}=1
\]</div>

每层有一个可学习 pseudo-query：

<div class="math-display">\[
q_l=w_l\in\mathbb{R}^{d}
\]</div>

历史输出同时作为 key 和 value。权重为：

<div class="math-display">\[
s_{i\to l}=w_l^\top\operatorname{RMSNorm}(v_i)
\]</div>

<div class="math-display">\[
\alpha_{i\to l}=\frac{\exp(s_{i\to l})}
{\sum_{j=0}^{l-1}\exp(s_{j\to l})}
\]</div>

RMSNorm 避免大范数历史状态仅凭尺度主导权重。pseudo-query 是每层一个可学习向量，但 score 还依赖当前 token 在各历史层的表示，因此权重仍是 token-dependent 的。

## 一个可手算的深度路由例子

假设某层可以读取 4 个来源，对一个 token 得到 score：

```text
embedding:   0.2
layer 1:     1.4
layer 2:    -0.4
layer 3:     0.8
```

softmax 后近似为：

```text
[0.149, 0.496, 0.082, 0.272]
```

于是该层输入为：

<div class="math-display">\[
h_l\approx0.149v_0+0.496v_1+0.082v_2+0.272v_3
\]</div>

这个 token 主要读取第 1 层和第 3 层输出，弱化第 2 层。另一个 token 的历史表示不同，即使使用同一个 <span class="math-inline">\(w_l\)</span>，也可能得到不同权重。

标准残差对应的是所有来源固定累加；AttnRes 把它改成归一化、可学习、依赖内容的深度选择。

## Full AttnRes 怎样插进 Transformer

在论文的记号里，attention 子层和 MLP 子层都被视为单独的 layer。因此一个 Transformer block 有两次深度聚合机会：

```text
h_attn_in = AttnRes(history, query_for_attn)
attn_out  = Attention(Norm(h_attn_in))
history.append(attn_out)

h_mlp_in = AttnRes(history, query_for_mlp)
mlp_out  = MLP(Norm(h_mlp_in))
history.append(mlp_out)
```

这段伪代码只表达数学关系。完整实现还要维护 embedding 来源、位置维度、训练激活和并行通信。

### 复杂度

若共有 <span class="math-inline">\(L\)</span> 个子层、隐藏维度 <span class="math-inline">\(d\)</span>，Full AttnRes 的深度 attention 计算约为：

<div class="math-display">\[
O(L^2d)
\]</div>

保存历史层输出为 <span class="math-inline">\(O(Ld)\)</span> 每 token。

在不使用 activation recomputation 的普通训练中，这些激活本来就可能为反向传播保留，因此额外内存很小。但大规模训练通常使用重计算和 pipeline parallel，历史输出原本可以释放，现在却需要保留并跨 stage 传递，成本会变得明显。

## Block AttnRes：把层先压成块

Block AttnRes 将 <span class="math-inline">\(L\)</span> 个子层分为 <span class="math-inline">\(N\)</span> 个 block。block 内仍使用标准残差累加：

<div class="math-display">\[
b_n=\sum_{j\in B_n}f_j(h_j)
\]</div>

跨 block 只对以下表示做深度 attention：

```text
embedding b0
已完成 block: b1, b2, ..., b(n-1)
当前 block 的 partial sum
```

这样保存和跨 stage 通信的历史从 <span class="math-inline">\(O(Ld)\)</span> 降到 <span class="math-inline">\(O(Nd)\)</span>，深度 attention 从 <span class="math-inline">\(O(L^2d)\)</span> 降到 <span class="math-inline">\(O(N^2d)\)</span> 的量级。

论文实验发现约 8 个 block 在其测试规模上能保留大部分收益。这是该论文配置下的经验结果，不是所有架构的固定最优值。

## Block AttnRes 的简化伪代码

```python
def mix_depth(completed_blocks, partial, query, rmsnorm):
    # sources: [depth, batch, sequence, hidden]
    sources = torch.stack([*completed_blocks, partial], dim=0)
    keys = rmsnorm(sources)

    # 每个 token 在 depth 维度得到一组 score
    scores = torch.einsum("d,nbtd->nbt", query, keys)
    weights = scores.softmax(dim=0)

    return torch.einsum("nbt,nbtd->btd", weights, sources)
```

关键点有三个：

1. softmax 沿 depth 维度，而不是 sequence 维度；
2. 不同 token 有不同 score；
3. attention 子层与 MLP 子层可以使用不同 query。

## 为什么 pseudo-query 不直接来自当前隐藏状态

Full AttnRes 选择每层独立参数 <span class="math-inline">\(w_l\)</span> 作为 query。这样做的一个系统优势是：多个层的跨深度权重计算不必依赖该层的顺序前向结果，可以在 block 内组织成更并行的计算。

权重仍然依赖历史 value 的内容，所以它不是静态 layer weight。它可以被理解为“这一层通常需要寻找哪类深度特征”，再由每个 token 的历史状态决定匹配程度。

## 训练和推理的工程代价

### 训练

大规模 pipeline parallel 下，后续 stage 需要访问历史 block 表示。若每次都传完整历史，会产生重复通信。论文使用跨 stage cache，只传新增 block，减少冗余。

### Prefill

长上下文下，每个 token 都有若干 block 表示，额外状态随序列长度增长。实现需要把深度混合与内存访问合理融合。

### Decode

每步只有新 token，但仍要聚合深度来源。论文采用两阶段计算和 online softmax 合并部分结果，并报告典型推理负载中小于 2% 的延迟开销。这个数字依赖论文硬件、实现和负载，不能直接外推到任意引擎。

## AttnRes 的论文结果应该怎样读

[Attention Residuals](https://arxiv.org/abs/2603.15031) 报告了三类证据：

- scaling-law 实验中，Full/Block AttnRes 相对标准残差有一致改善；
- 在论文设置下，Block AttnRes 达到的 loss 相当于标准基线使用约 1.25 倍计算量；
- 在 Kimi Linear 48B total / 3B activated 模型上训练 1.4T token 后，下游评测整体改善，隐藏状态尺度和梯度在深度上更均匀。

这些结果支持“值得进一步验证”，但论文目前是 2026 年技术报告。对其他 dense Transformer、不同 norm、不同深度和训练配方，仍需要独立复现。

## MoDA：把序列记忆与深度记忆放进一次 attention

Mixture-of-Depths Attention 的出发点相同：浅层有用信号可能在重复残差更新中被稀释。但它不是直接对完整层输出做加权求和。

MoDA 为历史层维护 depth KV。对当前位置 <span class="math-inline">\(t\)</span>、当前层 <span class="math-inline">\(l\)</span> 的 query：

- sequence KV：当前层中位置 <span class="math-inline">\(\le t\)</span> 的 token；
- depth KV：同一位置 <span class="math-inline">\(t\)</span> 在先前层写入的 KV。

将两类 key/value 拼接：

<div class="math-display">\[
K_{\text{mix}}=[K_{\text{seq}};K_{\text{depth}}],
\qquad
V_{\text{mix}}=[V_{\text{seq}};V_{\text{depth}}]
\]</div>

再做统一 softmax：

<div class="math-display">\[
O=\operatorname{softmax}\left(
\frac{QK_{\text{mix}}^\top}{\sqrt{d_h}}+M
\right)V_{\text{mix}}
\]</div>

“统一”很重要：序列来源和深度来源在同一个归一化空间中竞争，而不是先得到两份输出再固定相加。

## MoDA 的 read-operate-write 视角

论文把跨层机制分成三个动作：

```text
read:    当前层读取哪些历史状态
operate: Attention 或 FFN 做变换
write:   当前结果怎样写回深度流
```

### 标准残差

```text
read = 当前隐藏状态
operate = 当前子层
write = 加法
```

### Depth Attention

query 沿深度读取同一 token 的历史 KV，再把新的 KV 追加进 depth stream。

### MoDA

query 同时读取当前序列 KV 和历史深度 KV，并共享一次 softmax。Attention 层可以复用已有 KV；FFN 没有天然 KV，因此可增加轻量 KV projection 把 FFN 信息写入深度流。

## 为什么 MoDA 需要专门 kernel

普通 sequence attention 的 KV 在序列维度上布局规则。depth KV 需要对每个 token 读取跨层记录，朴素 PyTorch 循环会产生不连续访存和许多小操作。

MoDA 论文通过：

- 将 depth KV 按适合 FlashAttention 的方式重排；
- chunk-aware 深度缓存布局；
- sequence 和 depth 两阶段共享 online-softmax 状态；
- group-aware indexing 适配 GQA；

降低访存开销。论文报告在 64K 序列长度上达到 FlashAttention-2 约 97.3% 的效率，同时整体训练 FLOPs 开销约 3.7%。二者不是同一个指标：前者是 kernel 相对效率，后者是模型总计算增量。

## MoDA 的实验结论与边界

[MoDA 论文](https://arxiv.org/abs/2603.15619) 在 1.5B 模型实验中报告：

- 10 个验证集平均 perplexity 改善 0.2；
- 10 个下游任务平均指标提高 2.11%；
- 与 post-norm 组合优于论文测试的 pre-norm 版本。

论文还在 700M 消融中发现，加入 depth KV 即使不增加 projection 参数也有收益；为 FFN 增加 depth KV projection 进一步改善效果。

这些结论仍限于论文报告的模型规模、数据和实现。特别是 norm 选择会改变残差流动态，不能把“MoDA 必须 post-norm”推广成所有模型的定律。

## AttnRes 与 MoDA 的关键差异

| 维度 | AttnRes | MoDA |
| --- | --- | --- |
| 读取对象 | 历史层或 block 输出 | 当前 sequence KV + 历史 depth KV |
| attention 轴 | 深度 | 序列和深度的联合候选 |
| query | 每层可学习 pseudo-query | 当前 attention query |
| 归一化 | 历史深度来源间 softmax | sequence/depth 联合 softmax |
| FFN 信息 | FFN 输出直接成为深度来源 | 需要额外 KV projection 才完整写入 |
| 主要系统难点 | 历史 block 激活和 pipeline 通信 | depth KV 布局与 fused attention kernel |

二者不是简单的“一个粗粒度、一个细粒度”版本，而是不同的跨深度表示方式。

## RealFormer 又是什么

[RealFormer](https://arxiv.org/abs/2012.11747) 提出的 Residual Attention 更早，做法是把前一层的 attention logits 传到下一层：

<div class="math-display">\[
S_l=\frac{Q_lK_l^\top}{\sqrt{d_h}}+S_{l-1}
\]</div>

然后：

<div class="math-display">\[
A_l=\operatorname{softmax}(S_l)
\]</div>

它建立的是 attention score 的跨层残差，让相邻层继承注意力模式。它不让第 <span class="math-inline">\(l\)</span> 层直接对所有历史 hidden state 做 depth attention，也没有 MoDA 的 depth KV stream。

因此：

```text
RealFormer: 残差发生在 attention logits
AttnRes:    注意力决定怎样聚合历史层输出
MoDA:       当前 query 联合读取 sequence KV 与 depth KV
```

## 它们适合解决什么问题

### 更深模型的表示稀释

让深层可以重新读取浅层形成的词法、结构或局部特征，而不是只依赖被多次混合的单一状态。

### 不同子层需要不同深度来源

Attention 和 MLP 的功能不同。AttnRes 可以使用不同 pseudo-query；MoDA 中不同 head 也可以形成不同深度偏好。

### 深度扩展的新计算轴

传统扩展集中在参数量、宽度、数据和上下文长度。深度 attention 为“怎样利用更多层”提供新的结构选择。

## 什么时候不应直接采用

- 现有模型不深，残差稀释不是主要瓶颈；
- 推理引擎没有相应 kernel 或缓存布局；
- pipeline parallel 网络已经是瓶颈；
- 训练依赖激进 activation recomputation，无法承担历史状态；
- 只是微调已有 checkpoint，不能无代价改变基础架构；
- 缺少相同 FLOPs、参数量和数据下的对照实验。

对已有 LLM 来说，AttnRes/MoDA 不是可以像 LoRA 一样直接外挂的 adapter。它们改变前向图和 checkpoint 结构，通常需要预训练或较充分的继续训练。

## 实现与评测检查表

- 明确实现的是 RealFormer、AttnRes 还是 MoDA；
- 对齐 baseline 的参数、token、FLOPs 和学习率；
- 检查深度 softmax 的维度是否正确；
- AttnRes score 前是否按论文使用 RMSNorm；
- 记录每层/每 head 的深度权重分布；
- 监控隐藏状态范数和梯度范数随深度变化；
- 单独测训练显存、pipeline 通信、prefill 和 decode；
- 长上下文下检查 depth cache 增长；
- 用独立任务验证收益，不只看训练 loss；
- 报告 kernel 实现，避免把理论 FLOPs 当成真实速度。

## 总结

标准残差把所有历史层更新以固定单位权重累加，简单、高效，也为梯度提供捷径；它的局限是后层只能读取混合后的单一状态。

AttnRes 用深度 softmax 让每层按 token 选择历史层或 block 输出；MoDA 则把当前序列 KV 和同位置的历史深度 KV 放进一次统一 attention。RealFormer 虽然也叫 Residual Attention，但残差对象是 attention logits。

三者共同说明了一个变化：注意力不再只负责“从哪些 token 读取信息”，也开始负责“从哪些深度读取信息”。它是否会成为大模型的通用组件，还需要更大范围复现；但从建模角度看，深度已经从固定的层叠顺序，变成了可以学习的检索维度。

## 参考资料

- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [Attention Residuals 官方代码](https://github.com/MoonshotAI/Attention-Residuals)
- [Mixture-of-Depths Attention](https://arxiv.org/abs/2603.15619)
- [MoDA 官方代码](https://github.com/hustvl/MoDA)
- [RealFormer: Transformer Likes Residual Attention](https://arxiv.org/abs/2012.11747)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [参考文章：残差连接——Kimi 注意力残差 / 字节混合注意力](https://www.cnblogs.com/Big-Yellow/p/19760790)
