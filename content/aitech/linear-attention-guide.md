---
title: "线性注意力详解：从核技巧、递推状态到现代架构"
date: 2026-07-11T11:00:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM 基础技术"]
tags: ["LLM", "Attention", "Linear Attention", "Performer", "DeltaNet", "长上下文"]
summary: "从标准 softmax 注意力出发，完整推导核化线性注意力的并行与递推形式，解释复杂度、数值稳定性、固定状态容量和硬件效率，并梳理 Performer、GLA、DeltaNet、Based、Mamba-2 与 Kimi Linear 的关系。"
---

线性注意力最吸引人的承诺，是把随序列长度平方增长的注意力计算改成线性增长，并在自回归解码时用一个固定大小的状态替代不断变长的 KV cache。

但“把 <span class="math-inline">\\(O(N^2)\\)</span> 变成 <span class="math-inline">\\(O(N)\\)</span>”只是故事的开头。真正理解线性注意力，还需要回答几个更重要的问题：它是否仍然等价于 softmax 注意力？为什么矩阵乘法换个顺序就能降复杂度？因果掩码怎么处理？固定大小的状态会丢掉什么？为什么理论 FLOPs 更少，实际运行却不一定更快？

本文从公式和张量形状开始推导，再把理论放回 LLM 的训练、推理与硬件环境中。这里的“线性注意力”主要指**利用可分解核函数，把历史压缩为矩阵状态的狭义线性注意力**，而不是所有具有线性复杂度的稀疏、低秩或状态空间方法。

## 先给出核心结论

读完整篇文章之前，可以先记住下面六点：

1. “线性”指 attention 的计算量对序列长度 <span class="math-inline">\\(N\\)</span> 线性增长，不是说模型只能表达线性函数。
2. 线性注意力的关键不是删除 softmax，而是把相似度写成 <span class="math-inline">\\(\phi(q)^\top\phi(k)\\)</span>，再利用矩阵乘法结合律改变计算顺序。
3. 使用 `ELU + 1`、ReLU 或恒等映射时，通常是**换了一个注意力核**；Performer 的 FAVOR+ 则尝试用随机特征**近似原始 softmax 核**。两者不能混为一谈。
4. 因果线性注意力可以维护固定大小的矩阵状态，因此单步解码成本不再随历史长度增加；但它把整个前缀压缩进有限状态，精确召回能力通常弱于保存全部 KV 的 softmax 注意力。
5. <span class="math-inline">\\(O(N)\\)</span> 不自动等于墙钟时间更快。GPU 是否能使用大矩阵乘法、状态是否频繁读写 HBM、特征维度多大，以及序列长度是否超过性能交叉点，都很重要。
6. 现代方案很少只使用最朴素的线性注意力。门控、衰减、delta rule、局部窗口和少量全局注意力层，都是在弥补有限状态记忆的不足。

## 从标准 softmax 注意力开始

先看单个注意力头。设序列长度为 <span class="math-inline">\\(N\\)</span>，Query 和 Key 的维度为 <span class="math-inline">\\(d_k\\)</span>，Value 的维度为 <span class="math-inline">\\(d_v\\)</span>：

```text
Q: [N, d_k]
K: [N, d_k]
V: [N, d_v]
```

位置 <span class="math-inline">\\(i\\)</span> 对位置 <span class="math-inline">\\(j\\)</span> 的未归一化相似度为：

<div class="math-display">\[
s(q_i,k_j)
=\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right).
\]</div>

输出是所有 Value 的加权平均：

<div class="math-display">\[
o_i
=\frac{\sum_{j=1}^{N}s(q_i,k_j)v_j}
{\sum_{j=1}^{N}s(q_i,k_j)}.
\]</div>

把所有位置写成矩阵形式，就是熟悉的 scaled dot-product attention：

<div class="math-display">\[
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]</div>

这里最昂贵的中间量是 <span class="math-inline">\\(QK^\top\in\mathbb R^{N\times N}\\)</span>。每个 Query 都要和每个 Key 比较，核心计算量约为 <span class="math-inline">\\(O(N^2d_k+N^2d_v)\\)</span>，通常简写为 <span class="math-inline">\\(O(N^2d)\\)</span>。

### 训练、prefill 与 decode 的瓶颈并不相同

在训练或 prompt prefill 阶段，模型一次拿到整段序列，平方项来自所有 token 两两交互。自回归 decode 则通常保存 KV cache：生成第 <span class="math-inline">\\(t\\)</span> 个 token 时，只计算新 Query，但仍要读取并匹配前面 <span class="math-inline">\\(t\\)</span> 个 Key 和 Value，所以单步 attention 成本为 <span class="math-inline">\\(O(td)\\)</span>，KV cache 也随 <span class="math-inline">\\(t\\)</span> 线性增长。

如果连续生成 <span class="math-inline">\\(N\\)</span> 个 token，attention 部分的累计工作量仍然包含：

<div class="math-display">\[
\sum_{t=1}^{N}O(td)=O(N^2d).
\]</div>

MQA 和 GQA 会减少 KV 头数，从而降低缓存和带宽的常数，但不会改变它们关于序列长度的阶数。

### FlashAttention 解决的是 I/O，不是平方 FLOPs

[FlashAttention](https://arxiv.org/abs/2205.14135) 通过分块和 online softmax 避免把完整的 <span class="math-inline">\\(N\times N\\)</span> 注意力矩阵写回显存，显著减少 HBM 与片上 SRAM 之间的数据搬运。它计算的仍然是**精确 softmax 注意力**，核心算术复杂度仍是平方级。

因此，FlashAttention 和线性注意力不是同一条路线：前者保留算法，优化数据流；后者改变或近似注意力核，减少需要完成的计算。

## 核技巧如何把平方复杂度变成线性复杂度

先暂时忘掉 softmax，把注意力写成任意非负相似度函数 <span class="math-inline">\\(s(q,k)\\)</span>：

<div class="math-display">\[
o_i
=\frac{\sum_{j=1}^{N}s(q_i,k_j)v_j}
{\sum_{j=1}^{N}s(q_i,k_j)}.
\]</div>

假设这个相似度可以用某个 <span class="math-inline">\\(m\\)</span> 维特征映射分解：

<div class="math-display">\[
s(q,k)=\phi(q)^\top\phi(k),
\qquad \phi:\mathbb R^{d_k}\rightarrow\mathbb R^m.
\]</div>

代回注意力公式：

<div class="math-display">\[
o_i
=\frac{\sum_{j=1}^{N}\phi(q_i)^\top\phi(k_j)v_j}
{\sum_{j=1}^{N}\phi(q_i)^\top\phi(k_j)}.
\]</div>

因为 <span class="math-inline">\\(\phi(q_i)\\)</span> 与求和下标 <span class="math-inline">\\(j\\)</span> 无关，可以移到求和外面：

<div class="math-display">\[
o_i
=\frac{
\phi(q_i)^\top
\left(\sum_{j=1}^{N}\phi(k_j)v_j^\top\right)
}{
\phi(q_i)^\top
\left(\sum_{j=1}^{N}\phi(k_j)\right)
}.
\]</div>

定义两个与 Query 无关的汇总量：

<div class="math-display">\[
S=\sum_{j=1}^{N}\phi(k_j)v_j^\top
\in\mathbb R^{m\times d_v},
\qquad
z=\sum_{j=1}^{N}\phi(k_j)
\in\mathbb R^m.
\]</div>

那么每个位置的输出只需计算：

<div class="math-display">\[
o_i=\frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}.
\]</div>

这就是线性注意力最核心的推导。标准注意力先计算：

```text
(Q K^T) V
 ^^^^^^^
  N x N
```

线性注意力改变括号位置：

```text
Q (K^T V)
   ^^^^^^^
   m x d_v
```

它没有显式构造 token 与 token 的完整注意力矩阵，而是先把全部 Key-Value 对压缩为 <span class="math-inline">\\(S\\)</span> 和 <span class="math-inline">\\(z\\)</span>，再让每个 Query 读取这份汇总状态。

### 用矩阵形状做一次校验

令 <span class="math-inline">\\(\Phi_Q=\phi(Q)\in\mathbb R^{N\times m}\\)</span>，<span class="math-inline">\\(\Phi_K=\phi(K)\in\mathbb R^{N\times m}\\)</span>，则：

| 中间量 | 计算 | 形状 |
|---|---|---|
| Key-Value 状态 | <span class="math-inline">\\(S=\Phi_K^\top V\\)</span> | <span class="math-inline">\\(m\times d_v\\)</span> |
| 归一化状态 | <span class="math-inline">\\(z=\Phi_K^\top\mathbf 1\\)</span> | <span class="math-inline">\\(m\\)</span> |
| 分子 | <span class="math-inline">\\(\Phi_QS\\)</span> | <span class="math-inline">\\(N\times d_v\\)</span> |
| 分母 | <span class="math-inline">\\(\Phi_Qz\\)</span> | <span class="math-inline">\\(N\\)</span> |

计算 <span class="math-inline">\\(S\\)</span> 和输出大约需要 <span class="math-inline">\\(O(Nmd_v)\\)</span>，归一化部分需要 <span class="math-inline">\\(O(Nm)\\)</span>。当 <span class="math-inline">\\(m\\)</span> 和 <span class="math-inline">\\(d_v\\)</span> 不随 <span class="math-inline">\\(N\\)</span> 增长时，对序列长度就是线性的。

这里不能把维度藏起来后就忘掉它。若特征映射为了提高精度而让 <span class="math-inline">\\(m\\)</span> 很大，线性注意力仍可能在实际长度范围内比优化后的 softmax 注意力更慢。

## 两条不同路线：更换核与近似 softmax

“把 softmax 去掉”是一种过于粗糙的说法。核化之后至少有两条不同路线，它们优化的目标并不相同。

### 路线一：选择一个有限维特征映射

[Transformers are RNNs](https://arxiv.org/abs/2006.16236) 使用了简单的正值特征映射：

<div class="math-display">\[
\phi(x)=\operatorname{ELU}(x)+1.
\]</div>

它容易计算，结果严格为正，能让注意力权重和归一化分母保持非负。ReLU、恒等映射以及多项式特征也出现在后续工作中。

关键是：`ELU + 1` 并不与 softmax 核等价。此时模型学习的是一个新的相似度：

<div class="math-display">\[
s(q,k)=\bigl(\operatorname{ELU}(q)+1\bigr)^\top
\bigl(\operatorname{ELU}(k)+1\bigr).
\]</div>

它保留了“基于内容匹配后加权聚合”的结构，却改变了 softmax 指数核的几何性质。模型通常需要围绕这个新算子重新训练，不能指望把已训练好的 softmax attention 直接替换后性能完全不变。

### 路线二：用随机特征近似 softmax 核

softmax 注意力对应的指数点积核可以写成：

<div class="math-display">\[
k(q,k)=\exp\left(\frac{q^\top k}{\sqrt{d_k}}\right).
\]</div>

它存在特征空间表示，但精确表示通常是无限维的，无法直接得到有限成本的 exact linear attention。[Performer](https://arxiv.org/abs/2009.14794) 提出的 FAVOR+ 使用正交随机特征，把它近似为：

<div class="math-display">\[
\exp\left(\frac{q^\top k}{\sqrt{d_k}}\right)
\approx \phi(q)^\top\phi(k).
\]</div>

于是仍可使用前面的结合律。特征数 <span class="math-inline">\\(m\\)</span> 越大，随机近似通常越准确，但计算量、状态大小和显存占用也越高。FAVOR+ 强调正值、低方差的随机特征，因为普通随机特征在长序列上可能出现高方差或不稳定的归一化。

可以这样区分二者：

| 方法 | 是否保留 softmax 核 | 主要误差来源 |
|---|---|---|
| `ELU + 1`、ReLU、identity | 否，定义了新核 | 模型表达偏置发生变化 |
| Performer / FAVOR+ | 近似保留 | 有限随机特征的近似误差与方差 |

## 因果线性注意力为什么等价于一个 RNN

非因果 attention 可以一次汇总整段序列。自回归语言模型不能读取未来 token，因此位置 <span class="math-inline">\\(t\\)</span> 只能使用 <span class="math-inline">\\(1\ldots t\\)</span> 的 Key 和 Value。

把全局状态改成前缀状态：

<div class="math-display">\[
S_t=\sum_{i=1}^{t}\phi(k_i)v_i^\top,
\qquad
z_t=\sum_{i=1}^{t}\phi(k_i).
\]</div>

它们可以递推更新：

<div class="math-display">\[
S_t=S_{t-1}+\phi(k_t)v_t^\top,
\qquad
z_t=z_{t-1}+\phi(k_t).
\]</div>

当前输出为：

<div class="math-display">\[
o_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t+\varepsilon}.
\]</div>

这已经是一个标准的循环状态更新：输入当前 token，更新内部状态，再读出当前输出。状态 <span class="math-inline">\\(S_t\\)</span> 的大小为 <span class="math-inline">\\(m\times d_v\\)</span>，不会随上下文长度增长。

因此，在固定 <span class="math-inline">\\(m\\)</span> 和 <span class="math-inline">\\(d_v\\)</span> 下：

- 单步解码 attention 成本为 <span class="math-inline">\\(O(md_v)\\)</span>，与已经生成多少 token 无关；
- 整段长度 <span class="math-inline">\\(N\\)</span> 的 attention 成本为 <span class="math-inline">\\(O(Nmd_v)\\)</span>；
- 推理状态大小为 <span class="math-inline">\\(O(md_v+m)\\)</span>，关于 <span class="math-inline">\\(N\\)</span> 是常数。

这里的“常数”只表示不随上下文长度变化，并不表示状态很小。一层、多头、较大特征维度和大 batch 仍可能产生可观的矩阵状态。

### 一个两维小例子

假设特征空间只有两维，Value 是标量：

```text
phi(k1) = [1, 0], v1 = 2
phi(k2) = [0, 1], v2 = 6
```

读完两个 token 后：

<div class="math-display">\[
S_2=
\begin{bmatrix}2\\6\end{bmatrix},
\qquad
z_2=
\begin{bmatrix}1\\1\end{bmatrix}.
\]</div>

若 <span class="math-inline">\\(\phi(q)=[1,1]\\)</span>，两个槽位权重相同，输出为：

<div class="math-display">\[
o=\frac{[1,1][2,6]^\top}{[1,1][1,1]^\top}=4.
\]</div>

若 <span class="math-inline">\\(\phi(q)=[3,1]\\)</span>，第一个方向权重更高：

<div class="math-display">\[
o=\frac{3\times2+1\times6}{3+1}=3.
\]</div>

真实模型并没有两个互不干扰的硬槽位。不同 Key 的特征通常不正交，它们的外积会叠加在同一个矩阵状态里，这正是后面要讨论的记忆干扰来源。

## 并行、递推与分块：同一个算子的三种计算形态

线性注意力在论文公式里是线性的，不代表最直接的 `for` 循环就适合 GPU。现代实现通常在三种形态之间选择。

### 并行形式

非因果 attention 可直接计算 <span class="math-inline">\\(\Phi_K^\top V\\)</span> 和 <span class="math-inline">\\(\Phi_QS\\)</span>。对于因果 attention，也可以显式计算带下三角 mask 的 pairwise 结果，或者使用并行 prefix scan。

并行形式容易让 GPU 保持高占用，但显式 pairwise 形式会重新引入平方 FLOPs；prefix scan 若物化每个时间步的二维状态，也会带来很高的内存 I/O。

### 递推形式

递推形式每一步只做状态读写，FLOPs 最少，也最适合 token-by-token 解码。但训练时各时间步存在依赖，简单循环难以利用 Tensor Core，大量小算子和状态访存可能让实际速度很差。

### 分块并行形式

分块方法把长度 <span class="math-inline">\\(N\\)</span> 的序列切成长度 <span class="math-inline">\\(C\\)</span> 的块：

- 块内使用并行矩阵乘法，计算局部因果交互；
- 块间传递压缩状态，计算更早历史对当前块的贡献；
- 调节 <span class="math-inline">\\(C\\)</span>，在并行度、FLOPs、片上存储和 HBM 流量之间折中。

[GLA](https://proceedings.mlr.press/v235/yang24ab.html) 的分析中，在特征维度与隐状态维度相近的简化设置下，chunkwise 计算约为 <span class="math-inline">\\(O(NCd+Nd^2)\\)</span>。当 <span class="math-inline">\\(C=1\\)</span> 时接近递推形式，当 <span class="math-inline">\\(C=N\\)</span> 时退化到完整并行形式。

这也是现代线性注意力论文越来越重视自定义 CUDA/Triton kernel 的原因：算法复杂度只决定上限，数据放在哪里、以多大 tile 计算，才决定 GPU 真正花多少时间。

## 一份可校验的 PyTorch 教学实现

下面的代码实现了 `ELU + 1` 特征映射下的两种等价计算：

- `causal_linear_attention_reference` 显式构造下三角 pairwise 权重，便于作为正确性基准；
- `causal_linear_attention_recurrent` 只维护 <span class="math-inline">\\(S_t\\)</span> 和 <span class="math-inline">\\(z_t\\)</span>，展示真正的递推形式。

输入形状统一为 `[batch, heads, seq_len, dim]`。为了降低长序列累加误差，半精度输入使用 FP32 状态累加，最后再转回 Value 的 dtype。

```python
import torch
import torch.nn.functional as F


def positive_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def _accumulation_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


def causal_linear_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """显式 O(N^2) 参考实现，只用于校验。"""
    dtype = _accumulation_dtype(q)
    qf = positive_feature_map(q).to(dtype)
    kf = positive_feature_map(k).to(dtype)
    vf = v.to(dtype)

    # scores: [batch, heads, query_pos, key_pos]
    scores = torch.einsum("bhnm,bhsm->bhns", qf, kf)
    seq_len = q.size(2)
    mask = torch.ones(
        seq_len, seq_len, dtype=torch.bool, device=q.device
    ).tril()
    scores = scores.masked_fill(~mask, 0.0)

    numerator = torch.einsum("bhns,bhsd->bhnd", scores, vf)
    denominator = scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (numerator / denominator).to(v.dtype)


def causal_linear_attention_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """O(N) 递推实现，状态大小与 seq_len 无关。"""
    dtype = _accumulation_dtype(q)
    qf = positive_feature_map(q).to(dtype)
    kf = positive_feature_map(k).to(dtype)
    vf = v.to(dtype)

    batch, heads, seq_len, feature_dim = qf.shape
    value_dim = vf.size(-1)
    state = torch.zeros(
        batch,
        heads,
        feature_dim,
        value_dim,
        dtype=dtype,
        device=q.device,
    )
    normalizer = torch.zeros(
        batch,
        heads,
        feature_dim,
        dtype=dtype,
        device=q.device,
    )

    outputs = []
    for t in range(seq_len):
        kt = kf[:, :, t]
        vt = vf[:, :, t]
        qt = qf[:, :, t]

        # S_t = S_{t-1} + phi(k_t) v_t^T
        state = state + torch.einsum("bhm,bhd->bhmd", kt, vt)
        normalizer = normalizer + kt

        numerator = torch.einsum("bhm,bhmd->bhd", qt, state)
        denominator = torch.einsum(
            "bhm,bhm->bh", qt, normalizer
        ).unsqueeze(-1)
        outputs.append(numerator / denominator.clamp_min(eps))

    return torch.stack(outputs, dim=2).to(v.dtype)


if __name__ == "__main__":
    torch.manual_seed(7)
    q = torch.randn(2, 4, 16, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 16, 8, dtype=torch.float32)
    v = torch.randn(2, 4, 16, 12, dtype=torch.float32)

    expected = causal_linear_attention_reference(q, k, v)
    actual = causal_linear_attention_recurrent(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    print("reference and recurrent forms match")
```

这个实现强调公式正确性，不代表高性能实现。Python 循环会产生大量 kernel launch，训练时也会保存计算图中的逐步状态。生产实现应使用并行 scan 或 I/O-aware chunkwise kernel，并为反向传播设计重计算策略。

还要注意一个边界：代码先写入当前 <span class="math-inline">\\((k_t,v_t)\\)</span> 再读取，因此对应包含对角线的下三角 attention，即位置 <span class="math-inline">\\(t\\)</span> 可以读取当前位置。若任务需要严格只看 <span class="math-inline">\\(1\ldots t-1\\)</span>，应先读状态再更新。

## 复杂度不能只写一个 O(N)

设每头特征维度为 <span class="math-inline">\\(m\\)</span>，Value 维度为 <span class="math-inline">\\(d_v\\)</span>。下面只比较 attention 核心，不含 Q/K/V 投影、MLP、通信和优化器状态。

| 方法 | 整段核心计算 | 单步 decode 随历史长度的变化 | 推理时历史状态 | 是否精确 softmax |
|---|---|---|---|---|
| 朴素 softmax attention | <span class="math-inline">\\(O(N^2d)\\)</span> | <span class="math-inline">\\(O(td)\\)</span> | KV cache 为 <span class="math-inline">\\(O(Nd)\\)</span> | 是 |
| FlashAttention | <span class="math-inline">\\(O(N^2d)\\)</span> | <span class="math-inline">\\(O(td)\\)</span> | KV cache 为 <span class="math-inline">\\(O(Nd)\\)</span> | 是 |
| 核化线性注意力 | <span class="math-inline">\\(O(Nmd_v)\\)</span> | <span class="math-inline">\\(O(md_v)\\)</span> | <span class="math-inline">\\(O(md_v+m)\\)</span> | 通常不是 |
| Performer | <span class="math-inline">\\(O(Nmd_v)\\)</span> | <span class="math-inline">\\(O(md_v)\\)</span> | <span class="math-inline">\\(O(md_v+m)\\)</span> | 随机近似 |
| chunkwise 线性注意力 | 依具体门控和块长而定 | 可递推为常数 | 固定矩阵状态 | 依核而定 |

“单步为常数”是指对 <span class="math-inline">\\(t\\)</span> 为常数，仍要付出 <span class="math-inline">\\(md_v\\)</span> 的矩阵状态读写和计算。自回归生成也仍然必须逐 token 进行，所以它消除的是上下文增长带来的额外 attention 成本，不会让整个 decoder 突然变成并行生成器。

### 固定状态不一定比短 KV cache 小

做一个仅用于建立量级直觉的估算。若每头 <span class="math-inline">\\(m=d_v=d\\)</span>：

- 标准 MHA 的每头 KV cache 约有 <span class="math-inline">\\(2Nd\\)</span> 个元素；
- 基础线性注意力的矩阵状态和归一化向量约有 <span class="math-inline">\\(d^2+d\\)</span> 个元素。

两者的存储量在 <span class="math-inline">\\(N\approx(d+1)/2\\)</span> 附近相等。若 head dimension 为 128，这个粗略交叉点只有约 64 个 token。

但这不是速度结论。真实系统还要考虑 GQA 的 KV 头共享、线性注意力的多状态或扩展维度、数据类型、kernel 融合和 batch 调度。它只说明“固定大小”不等于“零成本”，并且状态设计会直接决定线性模型的效率与容量。

## 为什么理论上线性，实践中未必更快

### 1. GPU 喜欢大矩阵乘法，不喜欢细碎递推

softmax attention 虽然 FLOPs 多，但 FlashAttention 能把大量工作组织成规则的矩阵乘法，充分利用 Tensor Core。朴素线性递推每步更新一个矩阵状态，算术强度低、顺序依赖强，可能花更多时间搬数据和启动算子。

### 2. 性能存在序列长度交叉点

当 <span class="math-inline">\\(N\\)</span> 较短时，平方 attention 的矩阵乘法非常高效，而线性 attention 仍要处理 <span class="math-inline">\\(m\times d_v\\)</span> 状态。只有上下文足够长、batch 和硬件条件合适时，渐近复杂度优势才会转化成墙钟时间优势。

### 3. 特征维度决定常数

Performer 增大随机特征数可以降低近似误差，却会同步放大计算和状态。DeltaNet、GLA、KDA 等更强算子还会加入门控、卷积、归一化与额外投影。比较模型时不能只看到名字里有“linear”。

### 4. attention 不是模型的全部成本

短上下文、大 MLP 或 MoE 模型中，前馈层、路由、通信和权重读取可能占主要时间。把 attention 的复杂度降下来，不等于端到端吞吐会按相同比例提升。

### 5. 数值稳定性会改变实现

状态是大量外积的累加。长序列下，FP16/BF16 容易积累舍入误差或溢出；归一化分母也可能过小。常见处理包括：

- 让特征映射保持非负，并给分母加入 <span class="math-inline">\\(\varepsilon\\)</span>；
- 使用 FP32 累加状态；
- 对 Query、Key 做归一化或缩放；
- 对状态做分块重标定，或使用具有稳定衰减的门控；
- 在一些现代架构中删除显式分母，再通过 RMSNorm、门控和训练配方控制尺度。

最后一种做法已经不再与归一化核 attention 完全等价，阅读论文时要检查它到底保留了哪一项公式。

## 固定状态的代价：历史被压缩，而不是被完整保存

softmax attention 的 KV cache 保存每个历史 token 的独立 Key 和 Value。Query 到来时，可以重新计算它与所有历史 Key 的匹配分数。这像一份会随上下文增长的显式内容寻址存储。

基础线性注意力只保留：

<div class="math-display">\[
S_t=\sum_{i=1}^{t}\phi(k_i)v_i^\top.
\]</div>

所有键值关联以外积相加的方式叠在同一个固定矩阵中。若两个 Key 在特征空间接近，它们写入的方向也接近，读取时就会互相干扰。序列继续增长，状态大小不变，模型不可能无损保存任意多、任意精确的关联。

[Based](https://arxiv.org/abs/2402.18668) 把这件事总结为 recall 与 state size 的权衡：状态更小、生成吞吐更高，但需要精确从长上下文取回某个 token 时通常更困难。Based 因此把线性注意力与滑动窗口 attention 结合：固定状态负责全局汇总，局部窗口负责精确的近邻比较和 token shift。

这也解释了为什么“支持一百万 token”与“能准确利用一百万 token 中的任意细节”是两个不同问题。复杂度允许模型接收很长输入，只是第一步；状态容量、训练数据和检索机制决定它能保留什么。

## 从加法写入到 Delta Rule

[Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) 提供了一个很有帮助的视角：矩阵状态 <span class="math-inline">\\(S_t\\)</span> 是一组在序列内部动态更新的 fast weights，Key 是地址，Value 是希望写入的内容。

基础线性注意力只会做加法写入：

<div class="math-display">\[
S_t=S_{t-1}+k_tv_t^\top.
\]</div>

若同一个或相近的 Key 后来对应了新 Value，旧关联仍留在状态中。Delta rule 先读取当前记忆对 Key 的预测：

<div class="math-display">\[
\hat v_t=S_{t-1}^\top k_t,
\]</div>

再只写入预测误差：

<div class="math-display">\[
S_t=S_{t-1}+\beta_t k_t(v_t-\hat v_t)^\top,
\qquad 0\le\beta_t\le1.
\]</div>

若当前记忆已经能从 <span class="math-inline">\\(k_t\\)</span> 读出 <span class="math-inline">\\(v_t\\)</span>，更新接近零；若内容发生变化，模型会沿对应 Key 的方向纠正旧映射。严格推导通常还会约束或归一化 Key，确保这一步具有稳定的“擦除再写入”含义。

Delta rule 没有凭空增加状态维度，但给了模型主动修正记忆的能力。2024 年的 [DeltaNet 并行化工作](https://arxiv.org/abs/2406.06484) 又解决了这类更新难以沿序列并行训练的问题，使其能够扩展到更标准的语言建模规模。

## 门控、衰减与现代线性注意力

另一个自然改进是允许模型忘记旧状态。最一般的二维门控可以写成：

<div class="math-display">\[
S_t=G_t\odot S_{t-1}+k_tv_t^\top,
\qquad G_t\in(0,1)^{d_k\times d_v}.
\]</div>

完整矩阵门控表达力强，却很难高效训练。现代方法会限制门控结构，例如使用标量、向量、对角矩阵或低秩参数化，在表达力与硬件效率之间折中。

### GLA：数据依赖的遗忘

[Gated Linear Attention](https://proceedings.mlr.press/v235/yang24ab.html) 使用数据依赖的向量门控，让不同 Key 通道具有不同衰减速度，并设计 I/O-aware 的 chunkwise 算法。其重要贡献不只是“加了一个 gate”，而是让更有表达力的递推状态仍能被组织为适合 Tensor Core 的训练计算。

### RetNet 与 RWKV：平行训练、递推推理

[RetNet](https://arxiv.org/abs/2307.08621) 的 retention 机制同时给出 parallel、recurrent 和 chunkwise recurrent 三种形式，并用衰减控制历史贡献。[RWKV](https://arxiv.org/abs/2305.13048) 也把 time mixing 写成训练时可并行、推理时可递推的 WKV 运算。

它们和核化线性注意力共享“训练采用并行表达、推理维护固定状态”的目标，但具体归一化、位置机制和状态更新不同。把它们统称为广义线性递推架构可以，认为它们都等于 `ELU + 1` linear attention 就不准确。

### Kimi Linear：细粒度门控与混合架构

2025 年的 [Kimi Linear](https://arxiv.org/abs/2510.26692) 在 Gated DeltaNet 上提出 Kimi Delta Attention（KDA）。其递推核心把 delta update 与细粒度对角衰减结合，使模型能更精细地控制不同状态通道的保留，并使用专门的 chunkwise 算法提高硬件效率。

值得注意的是，Kimi Linear 不是“所有层都只用线性注意力”的纯架构。技术报告采用 3:1 的 KDA 与全局注意力层比例。这个设计本身传达了一个务实结论：固定状态适合承担大部分廉价的长程建模，少量 full attention 则补充精确检索和表达能力。

### Mamba-2：关系很近，但不应直接画等号

Mamba 属于 selective state space model，而不是传统核技巧线性注意力。[Mamba-2 / Structured State Space Duality](https://arxiv.org/abs/2405.21060) 证明了 SSM 与一类结构化 attention 矩阵之间存在深刻对应关系，并据此设计更高效的算法。

所以“线性 attention、门控线性 RNN、SSM”正在形成统一的代数视角，但它们的状态转移约束、输入依赖方式和可表达的 attention 矩阵仍不完全相同。理解联系很有价值，抹平差异反而会妨碍正确选型。

## 位置信息为什么仍然重要

基础状态更新是外积之和：

<div class="math-display">\[
S_t=\sum_{i=1}^{t}k_iv_i^\top.
\]</div>

加法满足交换律。若 Key 和 Value 本身不包含位置信息，把前缀中的键值对重新排序不会改变这个状态。因果边界只能告诉模型“哪些 token 已经出现”，不能自动完整表达它们的相对次序。

常见解决路线包括：

- 在 Query 和 Key 中加入可分解的位置编码；
- 使用随相对距离衰减的状态转移；
- 让数据依赖门控承担部分位置与遗忘功能；
- 在局部窗口或少量 full-attention 层中保留更直接的位置交互；
- 在进入 attention 前加入短卷积，增强局部顺序建模。

RoPE 能否直接套用，取决于具体特征映射和递推公式。对 identity feature map，旋转后的 Q/K 仍可保留清晰的相对位置结构；对非线性随机特征或复杂门控，位置变换是否保持可分解性与数值稳定，需要按实现逐项检查，不能只看配置里是否出现 `rope_theta`。

## 线性注意力方法的演进脉络

下面这张表不是完整论文清单，而是理解技术路线最有用的几个节点。

| 时间 | 工作 | 关键贡献 |
|---|---|---|
| 2018 | Efficient Attention | 用结合律避免显式构造大 attention map，较早系统讨论线性复杂度 attention |
| 2020 | Transformers are RNNs | 核化注意力、因果前缀状态与 RNN 形式 |
| 2020/2021 | Performer | FAVOR+ 正交正随机特征，近似 softmax attention |
| 2021 | Fast Weight Programmers | 把线性注意力解释为关联记忆，引入 delta rule 修正写入 |
| 2023 | RetNet、RWKV | 强化 parallel/recurrent/chunkwise 多形态与衰减记忆 |
| 2024 | GLA | 数据依赖门控与 I/O-aware chunkwise 训练 |
| 2024 | Based | 线性 attention + 局部窗口，明确 recall-state size 权衡 |
| 2024 | DeltaNet 并行化 | 让 delta rule 更新可在现代硬件上高效训练 |
| 2024 | Mamba-2 / SSD | 建立 SSM 与结构化 attention 的统一视角 |
| 2025 | Kimi Linear | KDA 细粒度门控、混合全局注意力与大规模验证 |

从这条路线可以看出，研究重点已经从“能否推导出 <span class="math-inline">\\(O(N)\\)</span>”转向三个更难的问题：**有限状态如何管理记忆、训练如何适配硬件、效果如何追上 full attention。**

## 什么时候应该考虑线性注意力

### 更适合的场景

- 流式输入很长，系统必须让推理状态不随上下文继续增长；
- 生成长度很长，decode 的 KV cache 容量或内存带宽成为主要瓶颈；
- 任务依赖长期趋势、累计统计或模糊关联，多于逐字精确复制；
- 可以从头训练或充分继续训练，让模型适应新的序列算子；
- 团队有能力使用成熟 kernel，并针对真实硬件做端到端基准。

### full attention 往往更稳妥的场景

- 上下文不长，FlashAttention 已经足够快；
- 任务高度依赖精确检索、代码变量绑定、needle recall 或逐 token 拷贝；
- 需要直接复用一个已经训练好的 softmax Transformer；
- 推理框架只对标准 attention、GQA 和 paged KV cache 有成熟优化；
- 主要瓶颈其实在 MLP、MoE、通信或模型权重带宽。

### 很多时候，混合架构是更现实的答案

纯线性状态追求固定内存，纯 full attention 追求显式检索。滑动窗口、少量全局 attention 层和线性状态可以分别承担局部精确交互、稀疏全局检索与长期压缩记忆。

Based、DeltaNet 的混合实验和 Kimi Linear 都沿着这个方向发展。它们没有把选择题做成“linear 或 softmax 二选一”，而是把不同算子放到最擅长的位置。

## 评估线性注意力时应该测什么

只报告 perplexity 或理论复杂度远远不够。更完整的评估至少应包含：

1. **质量**：语言建模 loss、下游准确率、长上下文检索、长度外推和精确复制。
2. **prefill**：不同 prompt 长度下的 tokens/s、峰值显存与首 token 延迟。
3. **decode**：不同已缓存长度、batch size 和生成长度下的 tokens/s、每请求状态大小与吞吐。
4. **硬件条件**：GPU 型号、dtype、Tensor Core 使用情况、kernel 版本和 chunk size。
5. **模型公平性**：参数量、训练 token、数据配方、局部卷积、门控和 full-attention 层比例是否一致。
6. **稳定性**：状态范数、分母分布、长序列溢出、梯度尺度和不同随机特征种子的方差。

尤其要把 prefill 和 decode 分开。某个方法可能在长 prompt 的 prefill 上不占优势，却因为固定状态在超长生成中表现很好；也可能 attention 层 benchmark 很快，但端到端模型被其他模块限制。

## 常见误解

### 线性注意力就是没有 softmax 的 `Q(K^T V)` 吗

这是最简化、未归一化的一种形式。完整核化推导还包含特征映射和分母 <span class="math-inline">\\(\phi(q)^\top z\\)</span>。很多现代架构确实删除分母并使用 identity feature map，但随后会加入门控、归一化和新的训练配方，它们已经是新的序列算子。

### Linear Attention 与 Linformer 是一回事吗

不是。Linformer 沿序列维度对 Key 和 Value 做低秩投影，把长度 <span class="math-inline">\\(N\\)</span> 压到固定秩 <span class="math-inline">\\(r\\)</span>；本文主要讨论用核特征和结合律得到矩阵递推状态的方法。两者都可能具有关于 <span class="math-inline">\\(N\\)</span> 的线性复杂度，但归纳偏置和解码方式不同。

### ALiBi 名字里有 Linear，所以是线性注意力吗

不是。ALiBi 的 Linear 指 attention bias 随相对距离线性变化。若它仍在完整 softmax attention 上计算所有 Query-Key 对，复杂度仍是平方级。

### 使用 FlashAttention 后还需要研究线性注意力吗

两者优化目标不同。FlashAttention 大幅改善精确 attention 的 I/O 和显存，通常是中短上下文的强基线；线性注意力试图改变随上下文增长的 FLOPs 与推理状态。正确比较应使用优化后的 FlashAttention，而不是朴素 PyTorch attention。

### 线性注意力完全不需要 KV cache 吗

纯递推线性层会把增长型 KV cache 换成固定矩阵状态。混合架构中的局部窗口或全局 attention 层仍然需要对应的 KV cache，所以应计算整个模型的加权缓存成本，不能只看某一层。

### O(1) decode 是否意味着生成任意长度都一样快

它表示每个线性 attention 层的单步成本不随历史长度增长。模型仍需逐 token 生成，单步仍包含投影、状态更新、MLP、采样和通信；总生成时间仍随输出 token 数线性增长。

## 总结

线性注意力最漂亮的地方，是一个非常朴素的代数变化：当相似度可以分解为 <span class="math-inline">\\(\phi(q)^\top\phi(k)\\)</span> 时，先算 <span class="math-inline">\\(K^\top V\\)</span>，就不必构造 <span class="math-inline">\\(N\times N\\)</span> 的注意力矩阵。加入因果约束后，这个汇总量自然变成可递推的矩阵状态，Transformer 也因此显露出 RNN 的一面。

但复杂度下降的代价同样明确：历史 token 不再各自保存在 KV cache 中，而是被叠加进有限状态。模型获得固定内存和稳定的单步 decode 成本，同时承担核近似、记忆干扰、精确召回和硬件实现上的新问题。

因此，判断线性注意力是否合适，不能只问“它是不是 <span class="math-inline">\\(O(N)\\)</span>”。更有用的问题是：状态有多大、如何写入和遗忘、训练 kernel 是否真正高效、任务需要怎样的召回，以及是否应该保留局部或少量全局 softmax attention。

从 Performer 到 DeltaNet、GLA、Based 和 Kimi Linear，技术演进已经给出一个清晰方向：未来高效长上下文模型的竞争，不只是减少计算，更是在有限状态中更聪明地保存、更新和检索信息。

## 参考论文与技术文章

### 论文

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017.
2. [Efficient Attention: Attention with Linear Complexities](https://arxiv.org/abs/1812.01243), Shen et al., 2018.
3. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236), Katharopoulos et al., ICML 2020.
4. [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794), Choromanski et al., ICLR 2021.
5. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174), Schlag et al., ICML 2021.
6. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al., NeurIPS 2022.
7. [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621), Sun et al., 2023.
8. [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048), Peng et al., EMNLP 2023 Findings.
9. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://proceedings.mlr.press/v235/yang24ab.html), Yang et al., ICML 2024.
10. [Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff](https://arxiv.org/abs/2402.18668), Arora et al., ICML 2024.
11. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484), Yang et al., NeurIPS 2024.
12. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060), Dao and Gu, ICML 2024.
13. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692), Kimi Team, 2025.

### 技术文章

1. [Rethinking Attention with Performers](https://research.google/blog/rethinking-attention-with-performers/), Google Research, 2020.
2. [Linear Attention Fundamentals](https://haileyschoelkopf.github.io/blog/2024/linear-attn/), Hailey Schoelkopf, 2024.
3. [Based: Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff](https://hazyresearch.stanford.edu/blog/2024-03-03-based), Hazy Research, 2024.
