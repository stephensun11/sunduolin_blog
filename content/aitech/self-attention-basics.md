---
title: "Self-Attention 入门：模型如何决定该看哪里"
date: 2026-07-06T22:10:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM基本原理"]
topics: ["Attention"]
tags: ["Attention", "Self-Attention", "Transformer", "LLM"]
---

Self-Attention 是 Transformer 最核心的计算。

它回答的问题很直接：当模型处理某个 token 时，应该从上下文里的哪些 token 获取信息？注意力分数越高，说明当前位置越需要关注那个位置。

## 一个直觉例子

看这个句子：

```text
小明把书放进书包，因为它很重。
```

这里的“它”更可能指“书”，不是“书包”。人类可以根据语义判断引用关系，模型也需要类似能力。

Self-Attention 做的事情，就是让“它”这个位置可以和前面的“小明”“书”“书包”等位置建立联系，并根据上下文分配不同权重。

## Q、K、V 是什么

注意力机制里经常看到三个符号：

```text
Q = Query
K = Key
V = Value
```

可以用检索来类比。

Query 表示“我现在想找什么信息”。Key 表示“我这里有什么特征可以被匹配”。Value 表示“如果你关注我，真正取走的信息是什么”。

每个 token 都会生成自己的 Q、K、V。然后用 Q 和所有 K 做相似度计算，得到注意力权重，再用这些权重对 V 加权求和。

## 计算流程

简化公式如下：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

拆开看：

1. `QK^T` 计算当前位置和其他位置的匹配程度。
2. 除以 `sqrt(d)` 是为了稳定数值。
3. `softmax` 把分数变成概率分布。
4. 最后乘以 `V`，得到融合上下文后的表示。

这就是模型“看哪里”的核心。

## 为什么需要 Multi-Head Attention

单个注意力头只能从一种角度看上下文。多头注意力允许模型同时关注不同关系。

有的头可能关注语法结构，有的头可能关注实体引用，有的头可能关注局部搭配。虽然我们不应该把每个头都解释得太死，但多头机制确实给了模型更丰富的表达空间。

```text
Head 1: 关注主谓关系
Head 2: 关注指代关系
Head 3: 关注局部短语
```

多个 head 的结果会拼接起来，再经过线性层融合。

## Causal Mask

在 GPT 这类 Decoder-only 模型里，生成第 t 个 token 时不能偷看未来 token。

所以模型会使用 causal mask，让当前位置只能关注自己和之前的位置。

```text
第 1 个 token: 看 1
第 2 个 token: 看 1,2
第 3 个 token: 看 1,2,3
```

这保证了训练目标和生成过程一致：根据过去预测未来。

## Attention 的代价

Self-Attention 的计算量和序列长度平方相关。

如果上下文长度变成 2 倍，注意力矩阵大约会变成 4 倍。这就是长上下文模型在训练和推理上都更贵的原因之一。

后来很多优化，比如 FlashAttention、稀疏注意力、滑动窗口注意力，都是在尝试降低这个成本。

## 总结

Self-Attention 让模型在处理每个 token 时，动态选择上下文里的重要信息。

Q、K、V 是它的基本语言，多头注意力让模型从多个角度看文本，causal mask 则保证生成模型不能偷看未来。理解这些，就能更清楚地理解 Transformer 为什么能成为 LLM 的基础结构。
