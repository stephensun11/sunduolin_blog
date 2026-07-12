---
title: "LLM 训练优化与稳定性：从交叉熵到故障诊断"
date: 2026-07-12T09:10:00+08:00
draft: false
summary: "从交叉熵、AdamW 和学习率调度出发，系统说明混合精度、梯度控制、checkpoint 与训练故障诊断。"
categories: ["AiTech"]
subcategories: ["LLM训练"]
topics: ["训练优化"]
tags: ["LLM", "AdamW", "学习率", "混合精度", "梯度裁剪", "训练稳定性"]
---

把分布式训练程序跑起来，只说明计算图能够执行。要让一个大模型在数十亿乃至数万亿 token 上稳定收敛，还需要理解损失、优化器、数值精度、批量、调度和监控之间的耦合关系。

本文以 decoder-only 语言模型为主，给出一条从目标函数到故障诊断的完整路径。文中的超参数是分析框架，不是可以无条件复制的“标准答案”。

## 训练目标到底是什么

对 token 序列 <span class="math-inline">\(x_1,\ldots,x_T\)</span>，自回归模型分解联合概率：

<div class="math-display">\[
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{&lt;t})
\]</div>

最大似然训练等价于最小化 token 级负对数似然：

<div class="math-display">\[
\mathcal{L}(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_i)
\]</div>

这里的 <span class="math-inline">\(N\)</span> 应是参与损失的有效 token 数，而不一定是张量元素总数。padding、被 mask 的 prompt token 和跨样本隔离位置都不能错误计入分母。

### 困惑度的边界

若平均交叉熵为 <span class="math-inline">\(\mathcal{L}\)</span>，困惑度为：

<div class="math-display">\[
\operatorname{PPL}=\exp(\mathcal{L})
\]</div>

困惑度只有在 tokenizer、数据、切分方式和上下文设置一致时才适合直接比较。不同词表会改变 token 粒度，因此不能把两个模型的 PPL 数字脱离协议排序。

## 从一个训练 step 看全局

一次逻辑更新通常包含：

```text
读取并打包 batch
  -> 前向计算 logits
  -> 计算有效 token 上的 loss
  -> 反向传播
  -> 跨设备同步/规约梯度
  -> 梯度裁剪
  -> optimizer.step()
  -> scheduler.step()
  -> 清空梯度
```

使用梯度累积时，多个 micro-batch 共同构成一个 optimizer step。日志中的 `step` 必须说明是 micro step 还是 update step，否则学习率、吞吐和训练 token 数都会被误读。

全局 batch 的 token 数近似为：

<div class="math-display">\[
B_{\text{tokens}}=B_{\text{micro}}\times L\times G\times W
\]</div>

其中 <span class="math-inline">\(L\)</span> 是序列长度，<span class="math-inline">\(G\)</span> 是累积步数，<span class="math-inline">\(W\)</span> 是数据并行副本数。若样本长度不一，应记录实际有效 token，而不是用上限估算。

## AdamW 为什么是常见基线

Adam 为每个参数维护一阶和二阶矩估计：

<div class="math-display">\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]</div>

<div class="math-display">\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]</div>

完成偏差修正后，参数按归一化梯度更新。AdamW 再把权重衰减与损失梯度解耦，可写成：

<div class="math-display">\[
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]</div>

这与“把 <span class="math-inline">\(\lambda\|\theta\|_2^2\)</span> 加入损失”在 Adam 这类自适应优化器中并不等价，[AdamW 论文](https://arxiv.org/abs/1711.05101) 正是为了解决这个区别。

实践中通常不给 bias 和归一化层的 scale 参数做 weight decay，但这是一项配置选择，应通过参数组显式实现并测试，不能只靠名称模糊匹配。

## 学习率：峰值不是全部

学习率调度通常分为 warmup 和 decay。

### 为什么需要 warmup

训练早期的矩估计不稳定，网络表示也在快速变化。直接使用峰值学习率可能造成梯度尖峰和不可逆发散。线性 warmup 的简单形式是：

<div class="math-display">\[
\eta_t=\eta_{\max}\frac{t}{T_{\text{warmup}}},\quad t\le T_{\text{warmup}}
\]</div>

warmup 应按 optimizer update 或已见 token 定义，并在恢复训练时正确恢复 scheduler 状态。

### decay 选择

常见策略包括线性衰减、余弦衰减和保持常数后再衰减。余弦形式可写为：

<div class="math-display">\[
\eta_t=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right]
\]</div>

选择哪一种要与训练预算相匹配。若预计后续继续训练，把学习率提前衰减到接近零会让续训困难；若做固定预算的最终模型，末段 decay 往往有利于稳定收敛。

## batch size 与学习率的关系

增大全局 batch 会降低随机梯度噪声，但不会无限提升样本效率。过大的 batch 可能需要更高学习率或更长训练才能达到同等泛化。

“学习率随 batch 线性缩放”是经验起点，不是定律。模型规模、优化器、序列长度和数据相关性都会改变最优关系。最稳妥的方法是在小规模上扫描峰值学习率，同时观察：

- 初始下降速度；
- 梯度范数和更新范数；
- 是否出现 loss spike；
- 固定 token 预算下的验证损失。

## 混合精度究竟混了什么

[混合精度训练](https://arxiv.org/abs/1710.03740) 的目的不是简单把所有张量改成半精度，而是让吞吐、显存和数值稳定性取得平衡。

| 格式 | 指数位特点 | 典型注意事项 |
| --- | --- | --- |
| FP32 | 范围和精度较高 | 显存、带宽和算力成本高 |
| FP16 | 尾数较多，指数范围较小 | 小梯度下溢，通常需要 loss scaling |
| BF16 | 指数范围接近 FP32 | 精度较低，但大模型训练通常更稳 |
| FP8 | 更低精度、更高吞吐潜力 | 依赖硬件、缩放策略与高精度累积 |

常见做法是矩阵乘使用 BF16/FP16，累加、归一化、softmax 或部分 optimizer state 保留更高精度。具体边界取决于框架和 kernel，不能只看模型权重的 `dtype` 推断整条计算链。

### FP16 的 loss scaling

将损失乘以缩放因子 <span class="math-inline">\(S\)</span>，反向后再把梯度除以 <span class="math-inline">\(S\)</span>：

<div class="math-display">\[
\nabla_\theta(S\mathcal{L})/S=\nabla_\theta\mathcal{L}
\]</div>

数学上梯度不变，但中间值更不容易下溢。动态 loss scaling 在检测到 inf/NaN 时跳过更新并降低 scale。若先裁剪缩放后的梯度、再 unscale，裁剪阈值就失去原意；顺序必须是 unscale 后再裁剪。

## 梯度裁剪解决什么问题

全局范数裁剪把梯度向量限制在阈值 <span class="math-inline">\(c\)</span> 内：

<div class="math-display">\[
g\leftarrow g\cdot\min\left(1,\frac{c}{\|g\|_2}\right)
\]</div>

它能缓解偶发尖峰，但不能修复持续过高的学习率、坏数据或数值错误。应同时记录裁剪前范数和被裁剪比例；如果几乎每一步都裁剪，真正的更新方向和幅度已被系统性改变。

分片训练下必须计算全局范数，而不是每张卡各自裁剪局部参数。FSDP、ZeRO 等框架通常提供对应 API。

## 激活重计算与显存

反向传播需要前向激活。activation checkpointing 只保存部分边界，反向时重算中间激活，以更多计算换显存。

它与 optimizer state 分片解决的是不同问题：

- 参数/梯度/optimizer 分片减少模型状态显存；
- activation checkpointing 减少与 batch、序列长度相关的激活显存；
- sequence parallel 等技术进一步分摊特定激活。

开启重计算后吞吐下降是预期行为，应该比较“每美元训练 token”或“能否扩大有效 batch/序列”，而不是只比较单步时间。

## 初始化、归一化与残差稳定性

深层 Transformer 的残差流会累积信号。架构通常通过初始化尺度、RMSNorm/LayerNorm 位置、残差缩放和高精度计算控制数值范围。

从已有 checkpoint 续训时，不要重新初始化 optimizer、scheduler 或随机数状态，除非明确进行新的训练阶段。只加载模型权重会造成动量丢失和学习率时间轴错位。

新加 token 或模块时，应单独检查初始化分布。新 embedding 的范数若远离原词表，可能在训练初期制造异常 logits。

## 必须监控的指标

一个可靠 dashboard 不只显示 loss：

| 指标 | 解释 |
| --- | --- |
| train/validation loss | 拟合和泛化趋势 |
| 每域 loss | 数据配比或域退化 |
| learning rate | scheduler 是否正确 |
| gradient norm | 尖峰、爆炸或异常平坦 |
| update/weight norm | 实际参数变化比例 |
| tokens/s 与 MFU | 数据或计算瓶颈 |
| padding/有效 token 比 | 打包效率 |
| overflow/跳步次数 | 混合精度稳定性 |
| 各卡 step time | straggler 和通信问题 |
| 数据 shard 与样本 ID | 异常回溯 |

Model FLOPs Utilization（MFU）是估算指标，依赖对模型 FLOPs 的定义。不同项目的 MFU 公式可能不同，比较前必须统一口径。

## loss spike 的诊断顺序

发现单步 loss 突然上升时，先保留现场，不要立即删 checkpoint。

### 1. 判断是日志问题还是真实问题

检查 loss 分母、梯度累积、数据并行规约和 NaN 聚合。某个 rank 的异常可能被平均值掩盖。

### 2. 回溯数据

记录异常 step 对应的文档 ID。检查超长重复、乱码、极端 token 分布、mask 全空、非法 label 和数据损坏。

### 3. 检查数值链

定位第一个非有限值出现在哪一层，是 logits、softmax、归一化、梯度还是 optimizer state。只在最终 loss 上调用 `isfinite` 太晚。

### 4. 检查学习率与恢复状态

常见错误包括 scheduler 多走一步、恢复后 warmup 重启、梯度累积次数改变却未调学习率、optimizer state 未加载。

### 5. 从 spike 前 checkpoint 复现

使用相同数据顺序和随机状态重放。如果稳定复现，多半是确定性数据或代码问题；若位置漂移，需要进一步检查并发、通信和随机数状态。

## checkpoint 必须保存什么

可恢复训练的 checkpoint 不只是 `model.pt`：

- 模型参数；
- optimizer 一阶、二阶矩和 step；
- scheduler 状态；
- gradient scaler；
- 全局 step、已见 token 和 epoch/数据游标；
- Python、NumPy、CPU/GPU RNG 状态；
- 数据加载器和 sampler 状态；
- 配置、代码版本与 tokenizer；
- 分布式拓扑与分片元数据。

保存成功不代表可恢复。应定期在独立作业里加载 checkpoint，继续若干步，并比较恢复前后的 loss 连续性。

## 固定有效 token 数时的最小伪代码

下面的写法假设每个 micro-batch 都有相同数量的有效 token（例如完成固定长度 packing）。若有效 token 数不同，必须按整个逻辑 batch 的 token 总数加权梯度，不能简单平均各 micro-batch 的 mean loss。

```python
for micro_step, batch in enumerate(loader):
    with autocast(dtype=compute_dtype):
        logits = model(batch["input_ids"], attention_mask=batch["mask"])
        loss_sum = token_cross_entropy(logits, batch["labels"], reduction="sum")
        token_count = (batch["labels"] != IGNORE_INDEX).sum()
        assert token_count == tokens_per_micro_batch
        loss = loss_sum / token_count / grad_accum_steps

    scaler.scale(loss).backward()

    if (micro_step + 1) % grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        global_norm = clip_global_grad_norm(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

真实分布式实现还要正确规约 `loss_sum` 和 `token_count`。当各 rank 或 micro-batch 的有效 token 数不同，只平均各自的平均 loss 会产生偏差。

## 如何做小规模验证

大规模训练前至少完成四个阶段：

1. **单 batch 过拟合**：确认 mask、label shift 和 loss 能下降；
2. **单卡短跑**：检查数值、数据和 checkpoint；
3. **多卡等价性**：相同全局 batch 下比较单卡与多卡的前几步；
4. **规模化 soak test**：跑过 warmup 和一次保存/恢复，观察吞吐、通信和稳定性。

等价性不要求浮点结果逐 bit 相同，但差异应在可解释范围内，且不会随步数快速放大。

## 常见误区

### loss 下降就代表训练正确

标签错位、样本泄漏或重复数据也可能让 loss 很漂亮。必须结合 held-out 数据与下游评测。

### BF16 不会溢出

BF16 指数范围较大，但运算仍可能产生 inf/NaN，也仍有精度损失。它降低风险，不消除风险。

### 梯度累积等同于真正的大 batch

若模型含 batch 相关操作、随机性或每个 micro-batch 的 token 数不同，两者不一定完全等价。学习率和 loss 归一化也必须匹配。

### checkpoint 越频繁越安全

保存会占用 I/O 和训练时间。更关键的是原子写入、校验和、保留策略和恢复演练。

## 总结

稳定训练不是靠某个神奇优化器，而是靠一组彼此一致的约定：有效 token 上的损失、明确的全局 batch、可解释的学习率时间轴、正确的精度边界、全局梯度控制、完整 checkpoint 和可回溯的数据日志。

当异常出现时，最有价值的不是再试一组超参数，而是能把异常精确定位到某个 step、某批数据、某层数值和某次状态变更。训练系统真正成熟的标志，就是失败可以被复现和解释。

## 参考资料

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
