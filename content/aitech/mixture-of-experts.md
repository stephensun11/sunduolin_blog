---
title: "MoE 大模型详解：稀疏路由、负载均衡与工程代价"
date: 2026-07-12T09:30:00+08:00
draft: false
summary: "从稀疏路由公式出发，讲清 MoE 的专家容量、负载均衡、专家并行、显存账本与推理代价。"
categories: ["AiTech"]
subcategories: ["LLM 基础技术"]
topics: ["MoE"]
tags: ["LLM", "MoE", "Sparse Model", "Router", "Expert Parallelism", "Mixtral"]
---

Mixture of Experts（MoE）经常被概括为“用更少计算激活更多参数”。这句话抓住了稀疏性的直觉，却没有说明代价：参数仍要存储和通信，路由会造成不均衡，专家并行引入 all-to-all，训练与服务的性能也取决于 token 分布。

本文聚焦 Transformer 中最常见的稀疏前馈 MoE。它通常替换部分 dense FFN，而 attention 仍由所有 token 共享。

## 从 dense FFN 开始

一个简化的 Transformer 前馈层可写为：

<div class="math-display">\[
\operatorname{FFN}(x)=W_2\,\sigma(W_1x)
\]</div>

若使用 SwiGLU，则有两条上投影：

<div class="math-display">\[
\operatorname{SwiGLU}(x)=W_{\text{down}}
\left(\operatorname{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x\right)
\]</div>

每个 token 都经过同一组参数。扩大 FFN 宽度会同时增加参数量和每 token FLOPs。

MoE 准备 <span class="math-inline">\(E\)</span> 个专家 FFN，只为每个 token 选择少量专家。总参数量可以大幅增加，而每 token 激活参数量取决于 top-k。

## Router 怎样选择专家

对 token 表示 <span class="math-inline">\(x\)</span>，router 产生 logits：

<div class="math-display">\[
z=W_rx
\]</div>

再得到门控概率：

<div class="math-display">\[
p_i(x)=\frac{\exp(z_i)}{\sum_{j=1}^{E}\exp(z_j)}
\]</div>

选出 top-k 专家集合 <span class="math-inline">\(S(x)\)</span> 后，输出可写成：

<div class="math-display">\[
y=\sum_{i\in S(x)}\tilde p_i(x)\,E_i(x)
\]</div>

其中 <span class="math-inline">\(\tilde p_i\)</span> 可以是原概率，也可以在选中专家间重新归一化。不同实现对权重、归一化、共享专家和残差路径的定义不同。

### Top-1 与 Top-2

Top-1 每 token 激活一个专家，计算和通信较低；Top-2 激活两个专家，通常提供更多容量和路由冗余，但成本更高。

[Switch Transformer](https://arxiv.org/abs/2101.03961) 以简化的 top-1 routing 展示了大规模稀疏训练；[Mixtral](https://arxiv.org/abs/2401.04088) 则在每层专家中为每个 token 选择两个。

“总参数”与“每 token 激活参数”必须同时报告。MoE 模型的总参数量不能直接与 dense 模型参数量按相同 FLOPs 解释。

## Capacity：专家一次能收多少 token

一个 batch 有 <span class="math-inline">\(T\)</span> 个 token，<span class="math-inline">\(E\)</span> 个专家，top-k 为 <span class="math-inline">\(k\)</span>。平均每个专家接收约 <span class="math-inline">\(kT/E\)</span> 个 token。工程上会设置 capacity factor：

<div class="math-display">\[
C=\left\lceil \text{capacity\_factor}\times\frac{kT}{E}\right\rceil
\]</div>

如果某个专家被分配超过 <span class="math-inline">\(C\)</span> 个 token，系统必须决定：丢弃超额 token、转给次选专家、使用更大的动态 buffer，或做其他路由修正。

capacity factor 太小会丢 token 或降低质量；太大则浪费显存和计算，并让 straggler 更严重。这个参数与 batch token 数和路由分布有关，不能脱离运行规模设置。

## 为什么会负载不均衡

如果 router 发现某些专家在早期略占优势，更多 token 会流向它们，热门专家得到更多梯度，可能形成正反馈；其他专家则训练不足。这会同时损害模型容量和硬件利用率。

一种常见辅助损失会同时考虑“被分配比例”和“平均路由概率”。设：

- <span class="math-inline">\(f_i\)</span> 为 batch 中送给专家 <span class="math-inline">\(i\)</span> 的 token 比例；
- <span class="math-inline">\(P_i\)</span> 为专家 <span class="math-inline">\(i\)</span> 的平均路由概率。

可构造：

<div class="math-display">\[
\mathcal{L}_{\text{balance}}=E\sum_{i=1}^{E} f_iP_i
\]</div>

总目标为语言模型损失加权辅助损失。系数过小无法平衡，过大则可能迫使均匀路由，牺牲有意义的专家分工。

还常见 router z-loss，用于抑制 router logits 过大、改善数值稳定性。具体公式和归一化要以实现为准。

## Token-choice 与 Expert-choice

最常见的 token-choice routing 是“每个 token 选 top-k 专家”。它保证每个 token 获得固定计算量，但不能保证专家负载一致。

[Expert Choice Routing](https://arxiv.org/abs/2202.09368) 反过来让每个专家选择固定数量的 token，从而天然控制容量，但一个 token 可能被零个、一个或多个专家选择，训练和自回归服务的语义与实现会更复杂。

路由策略没有脱离场景的绝对优劣。训练吞吐、推理确定性、批量大小和目标硬件都会影响选择。

## Shared Expert 在解决什么

如果所有知识都通过路由专家处理，通用模式会在多个专家中重复学习。共享专家让每个 token 都经过一条 dense 路径，路由专家则学习增量和特化能力。

[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 提出更细粒度专家和 shared expert isolation，目标是提升专家专业化并减少知识冗余。

共享专家增加每 token 固定计算，因此需要和路由专家数量、宽度共同核算。它也不保证专家会自动按人类可解释领域分工。

## MoE 的训练通信

专家通常分布在不同设备上。一次 MoE 层包含：

```text
本地 token 表示
  -> router 与分桶
  -> all-to-all：token 发往专家所在设备
  -> 专家 FFN 计算
  -> all-to-all：结果返回原设备
  -> 按门控权重聚合
```

这就是 expert parallelism。与 tensor parallelism 在每个矩阵内切分不同，expert parallelism 主要按专家切分参数和 token。

通信量与 token 数、hidden size、top-k 和数据类型有关。即使理论 FLOPs 很低，跨节点 all-to-all 也可能成为瓶颈。专家放置应尽量利用节点内高速互联，并避免让路由频繁跨慢速网络。

## 与其他并行方式怎样组合

大规模 MoE 常组合：

- data parallel：复制模型处理不同 batch；
- tensor parallel：切分 attention 或单个大专家；
- pipeline parallel：按层切 stage；
- expert parallel：把不同专家放到不同 rank；
- sequence/context parallel：切分长序列激活。

并行维度乘积必须匹配 world size。更重要的是，各维度的通信发生在不同位置，不能只根据 GPU 数拼出一个可整除配置。应基于拓扑测量 all-reduce、all-to-all 和 point-to-point 的竞争。

## 为什么 MoE 训练更容易出现数值问题

路由使每个专家看到的 token 数和分布不断变化。常见风险包括：

- 某专家长时间几乎没有梯度；
- 小 batch 下路由统计方差大；
- router logits 过大导致概率饱和；
- 热门专家出现更大梯度或 overflow；
- capacity 溢出导致 token 丢弃；
- 混合精度下门控排序对微小误差敏感。

监控至少包括每专家 token 数、门控概率、溢出/丢 token 比例、辅助损失、专家梯度范数和 all-to-all 时间。只看总 loss 很难发现一个专家已经“死亡”。

## MoE 的推理不是天然更快

自回归 decode 每个 step 的 token 数可能很少。此时专家 kernel 太小，难以充分利用 GPU；动态路由又使连续批处理和缓存管理更复杂。

推理成本至少包含：

- 所有专家权重的存储或跨设备访问；
- router 计算和 token 排序；
- 小 GEMM 或 grouped GEMM；
- expert parallel 的 all-to-all；
- 热门专家造成的负载倾斜；
- 多副本部署时的专家复制成本。

因此 MoE 的优势通常是“在相近激活 FLOPs 下获得更大参数容量”，不是“任何硬件上延迟都更低”。小 batch、低延迟场景尤其需要实测。

## 参数量和显存怎样计算

设每层 dense attention 参数为 <span class="math-inline">\(P_a\)</span>，单个专家 FFN 参数为 <span class="math-inline">\(P_e\)</span>，有 <span class="math-inline">\(E\)</span> 个专家，top-k 为 <span class="math-inline">\(k\)</span>。

总参数近似：

<div class="math-display">\[
P_{\text{total}}\approx P_a+E P_e+P_{\text{router}}
\]</div>

每 token 激活参数近似：

<div class="math-display">\[
P_{\text{active}}\approx P_a+kP_e+P_{\text{router}}
\]</div>

但显存仍更接近总参数，因为未激活专家的权重也要驻留某处。若权重从 CPU 或远端动态换入，延迟通常会非常高。

## 微调 MoE 的特殊问题

全量微调要更新所有专家，但某个小数据集可能只激活少数专家，造成能力偏移。LoRA 微调也需要决定：

- 只适配共享 attention；
- 适配所有专家；
- 只适配被选中的专家；
- 是否训练 router；
- 是否保留负载均衡损失。

冻结 router 可以保持原路由分布，但可能限制领域适配；训练 router 则可能导致专家塌缩。需要同时比较任务效果和路由统计。

## 一个正确的性能比较

比较 dense 与 MoE 时至少对齐：

- 训练 token 数和数据配方；
- 每 token 前向/训练 FLOPs 的估算口径；
- 总参数与激活参数；
- 上下文长度和 batch token 数；
- GPU 型号、互联和并行策略；
- 质量指标、吞吐、延迟和峰值显存。

只说“8x7B 因此相当于 56B”并不严谨。专家参数、共享模块、top-k 和模型具体维度都会改变总参数与激活计算。

## 常见误区

### 每个专家会自动成为一个明确领域专家

模型可能形成语法、位置或抽象特征上的分工，未必对应“数学专家”“代码专家”这类人类标签。

### 未选中的专家不占任何成本

它们不参与该 token 的 FFN 计算，但仍占权重存储、checkpoint、加载和分布式管理成本。

### top-1 一定比 top-2 快一倍

总延迟还受通信、kernel 利用率和负载影响，速度不会简单按 k 线性缩放。

### 负载越均匀越好

完全均匀可能抹掉有价值的专业化。目标是避免硬件和学习塌缩，而不是禁止路由偏好。

## 实施检查表

- 明确报告总参数、激活参数、top-k 和专家数；
- router 与门控归一化有单元测试；
- capacity overflow 行为明确且可监控；
- 每专家 token、概率和梯度有可视化；
- all-to-all 在目标拓扑上做过基准测试；
- checkpoint 能正确恢复专家、router 和 optimizer 分片；
- 微调前后比较路由分布和通用能力；
- 服务压测包含真实 batch、输入/输出长度和热点请求分布。

## 总结

MoE 把模型扩展问题从“所有 token 经过所有参数”改成“router 为 token 选择少量参数”。它用稀疏激活换取更大的总容量，但也把难题转移到了路由、容量、负载和通信上。

理解 MoE 时，始终把三本账分开：总参数决定存储规模，激活参数影响理论计算，实际 token 路由和网络拓扑决定系统性能。三者缺一，任何“更大但更省”的结论都不完整。

## 参考资料

- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
