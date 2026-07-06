---
title: "Transformer 架构入门：从 Encoder-Decoder 到 Decoder-only"
date: 2026-07-06T22:00:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM基本原理"]
topics: ["Transformer"]
tags: ["Transformer", "LLM", "架构"]
---

Transformer 是现代大语言模型的骨架。

如果只记一个结论，可以这样理解：Transformer 用注意力机制替代了传统循环结构，让模型可以并行处理序列，并在每一层里重新判断“当前 token 应该关注上下文里的哪些信息”。

## 为什么 Transformer 重要

在 Transformer 出现之前，序列建模常用 RNN、LSTM、GRU 这类结构。它们按顺序处理文本，天然适合时间序列，但训练效率受限，因为后面的 token 依赖前面的计算结果。

Transformer 的变化在于，它不再逐个时间步递推，而是一次性看见整个序列。模型通过注意力权重计算 token 之间的关系，再用多层网络不断更新表示。

这带来了两个直接好处：

1. 训练更容易并行化。
2. 长距离依赖更容易建模。

这也是后来 BERT、GPT、T5 以及各种 LLM 都建立在 Transformer 之上的原因。

## 原始 Transformer：Encoder-Decoder

论文《Attention Is All You Need》里的 Transformer 是 Encoder-Decoder 架构，最早主要服务于机器翻译。

```text
输入句子 -> Encoder -> 中间表示 -> Decoder -> 输出句子
```

Encoder 负责理解输入序列，Decoder 负责根据输入表示和已经生成的内容继续生成下一个 token。

Encoder 每层大致包括：

```text
Self-Attention
Feed Forward Network
Residual + LayerNorm
```

Decoder 每层多一个 Cross-Attention，用来关注 Encoder 的输出。

这种结构很适合“输入到输出”的任务，比如翻译、摘要、文本改写。

## Encoder-only：代表是 BERT

Encoder-only 模型只保留 Encoder，更擅长理解文本。

典型代表是 BERT。它通过 Masked Language Modeling 训练：随机遮住一些词，让模型根据上下文猜被遮住的词。

这种模型适合：

- 文本分类
- 句子匹配
- 信息抽取
- 语义向量表示

Encoder-only 的特点是可以双向看上下文。对理解任务来说，这非常自然。

## Decoder-only：现代 LLM 的主流

GPT 系列使用的是 Decoder-only 架构。

Decoder-only 模型的训练目标很简单：根据前面的 token 预测下一个 token。

```text
The cat sat on the -> mat
```

这叫自回归语言建模。它让模型天然适合生成任务：每次生成一个 token，再把这个 token 加回上下文，继续生成下一个。

现代聊天模型大多是 Decoder-only，因为它们要持续生成文本、代码、推理过程和结构化回答。

## 一个 token 在模型里经历了什么

简化来看，一个 token 会经历这些步骤：

1. tokenizer 把文本切成 token id。
2. embedding 层把 token id 变成向量。
3. 加入位置信息。
4. 多层 Transformer block 反复更新向量表示。
5. 输出层把最终向量映射到词表概率。
6. 采样或贪心选择下一个 token。

每一层 Transformer block 都在做两件事：先通过注意力整合上下文，再通过前馈网络加工特征。

## 总结

Transformer 并不是一个神秘黑盒。它的核心思想是：让序列里的每个位置都能根据任务需要动态关注其他位置。

Encoder-Decoder 擅长输入输出转换，Encoder-only 擅长理解，Decoder-only 擅长生成。理解这三种结构，就能看懂大多数 LLM 架构讨论的起点。
