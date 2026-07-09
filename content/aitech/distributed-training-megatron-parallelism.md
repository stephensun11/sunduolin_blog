---
title: "Megatron-LM：张量并行、流水线并行与 3D 并行"
date: 2026-07-09T09:30:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM训练"]
tags: ["LLM", "Megatron-LM", "张量并行", "流水线并行", "3D并行"]
summary: "用 Megatron-LM 的视角理解超大模型训练中的 TP、PP、DP，以及它们如何组合成多维并行。"
---

## Megatron-LM 要解决什么

当模型达到数百亿甚至上千亿参数时，单纯的数据并行已经不够了。

DDP 可以让多张卡处理更多数据，FSDP 和 ZeRO 可以把模型状态拆开存储，但超大模型训练还会遇到另一个问题：单个矩阵运算、单个 Transformer 层本身就可能太大，或者计算量太集中。

Megatron-LM 的目标是让超大模型能在多 GPU、多节点环境中高效训练。它通常组合三类并行：

- **张量并行**：把同一层里的矩阵运算切开。
- **流水线并行**：把模型的不同层切到不同 GPU。
- **数据并行**：复制若干组模型并行结构，处理不同数据。

如果再结合 ZeRO 或 FSDP 的状态分片，就形成更完整的多维并行训练。

## 整体结构

可以用一个简化图理解 Megatron-LM 的层次：

```text
+--------------------------------------------------------------+
|                    Megatron-LM 训练系统                       |
+--------------------------------------------------------------+
| Data Parallel：多个样本                                       |
|   - 不同并行组处理不同 batch                                  |
|   - 组间同步梯度                                              |
+--------------------------------------------------------------+
| Pipeline Parallel：层级切分                                   |
|   - 不同 GPU 负责不同 Transformer 层                          |
|   - micro-batch 在各 stage 间流动                              |
+--------------------------------------------------------------+
| Tensor Parallel：算子内切分                                   |
|   - 同一层的矩阵运算在多张 GPU 上并行计算                     |
|   - 通过 AllReduce 或 Concat 聚合结果                          |
+--------------------------------------------------------------+
```

这三类并行解决的是不同层面的瓶颈：TP 拆算子，PP 拆层，DP 拆数据。

## 张量并行：拆矩阵

Transformer 中最重的计算通常来自线性层和注意力里的矩阵乘法。张量并行会把一个大矩阵拆成多个子矩阵，让多张 GPU 一起计算。

以线性层为例：

```text
Y = XW + b
```

如果把权重矩阵 `W` 按输出维度切成 4 份：

```text
W = [W1, W2, W3, W4]
```

那么每张 GPU 只保存一部分权重，并计算自己的输出：

```text
Y1 = XW1
Y2 = XW2
Y3 = XW3
Y4 = XW4
```

最后再把结果拼接：

```text
Y = Concat(Y1, Y2, Y3, Y4)
```

张量并行可以降低单卡权重和计算压力，也能让大矩阵乘法更充分地利用多 GPU。但它需要频繁通信，因此更适合节点内高速互联环境，例如 NVLink 连接的 8 卡机器。

## 流水线并行：拆层

流水线并行把模型按层切分到不同 GPU 上。

例如一个 48 层 Transformer 可以被切成 4 个 stage：

- GPU 0：第 1 到 12 层。
- GPU 1：第 13 到 24 层。
- GPU 2：第 25 到 36 层。
- GPU 3：第 37 到 48 层。

如果只把一个 batch 顺序传过这 4 张卡，后面的 GPU 会一直等待，利用率很低。所以流水线并行会把 batch 再切成多个 micro-batch，让不同 GPU 同时处理不同 micro-batch。

| 时间步 | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | --- | --- | --- | --- |
| t1 | FWD micro-batch 1 |  |  |  |
| t2 | FWD micro-batch 2 | FWD micro-batch 1 |  |  |
| t3 | FWD micro-batch 3 | FWD micro-batch 2 | FWD micro-batch 1 |  |
| t4 | FWD micro-batch 4 | FWD micro-batch 3 | FWD micro-batch 2 | FWD micro-batch 1 |

这样多个 micro-batch 会像流水线一样在不同 stage 之间流动，从而提高 GPU 利用率。

## 1F1B 与 pipeline bubble

流水线并行有一个经典问题：开始阶段和结束阶段总会有 GPU 空等，这叫 pipeline bubble。

Megatron-LM 常用 **1F1B** 调度，也就是 one-forward-one-backward。流水线填满后，每个 stage 尽量交替执行一次前向和一次反向，让计算更均衡。

另一个常见优化是 virtual pipeline stage。它会把一个物理 GPU 上负责的层再切成更细的虚拟 stage，减少等待时间，提高调度灵活性。

## 数据并行：复制并行组

即使使用了张量并行和流水线并行，训练时通常仍然会使用数据并行。

可以把一组 TP + PP 组合看成一个“巨大模型副本”。如果资源足够，就复制多组这样的结构，每组处理不同数据。最后，各组之间同步梯度。

这时的数据并行常常会和 ZeRO 或 DeepSpeed 结合，用来进一步降低优化器状态、梯度和参数的冗余显存。

## 3D 并行怎么理解

所谓 3D 并行，可以粗略理解为三个维度一起切：

| 并行维度 | 切分对象 | 解决的问题 |
| --- | --- | --- |
| Tensor Parallel | 单层矩阵和算子 | 单个算子太大、计算太重 |
| Pipeline Parallel | Transformer 层 | 层数太多、模型纵向太深 |
| Data Parallel | 训练样本 | 提高吞吐、扩大 batch |

再叠加 ZeRO/FSDP 后，还可以把优化器状态、梯度和参数分片，进一步降低显存压力。

## 小结

Megatron-LM 的重点不是某一个单独技巧，而是把模型训练拆成多个维度来组织。

TP 让同一层变小，PP 让模型深度可拆，DP 让吞吐扩大。理解这三件事，再看大模型训练脚本里的 tensor model parallel size、pipeline model parallel size、data parallel size，就会清楚很多。
