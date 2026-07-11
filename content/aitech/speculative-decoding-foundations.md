---
title: "推测解码基础：从拒绝采样到无损并行验证"
date: 2026-07-10T09:20:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM 推理优化"]
topics: ["推测解码"]
tags: ["Speculative Decoding", "Speculative Sampling", "Rejection Sampling", "LLM Inference", "Draft Model"]
summary: "完整推导 draft-then-verify、接受概率和残差分布，解释为什么推测采样能保持目标模型分布，并分析接受长度、草稿成本、批处理和系统吞吐之间的真实关系。"
---

自回归语言模型有一个难以绕开的依赖：第 <span class="math-inline">\\(t+1\\)</span> 个 token 的分布依赖已经生成的前缀，因此通常必须先得到第 <span class="math-inline">\\(t\\)</span> 个 token，再执行下一次模型 forward。

推测解码（speculative decoding）的想法是：先让便宜的草稿机制猜多个未来 token，再让目标模型一次并行验证。猜对时，一次昂贵 forward 可以产出多个 token；猜错时，目标模型负责纠正。

困难不在“猜”，而在“纠正后是否仍与原目标模型完全同分布”。对 greedy decoding，只需保证输出 token 与目标模型 argmax 一致；对随机采样，则需要一套严格的接受与残差重采样规则。本文重点推导后者。

## 先给出核心结论

1. 目标模型能在一次 teacher-forcing forward 中并行给出一段草稿每个位置的条件分布，这是推测解码可行的计算基础。
2. 对随机采样，草稿 token <span class="math-inline">\\(x\\)</span> 的接受概率是 <span class="math-inline">\\(\min(1,p(x)/q(x))\\)</span>；拒绝后不能简单从 <span class="math-inline">\\(p\\)</span> 重采样，而要从归一化残差 <span class="math-inline">\\((p-q)_+\\)</span> 采样。
3. 正确实现能在数学上保持目标模型分布，因此常称 lossless speculative sampling；浮点数、top-k/top-p 处理和分布不一致的工程实现仍可能破坏这一性质。
4. 加速取决于每轮接受 token 数、草稿成本、验证成本和系统并发，不只取决于“acceptance rate”。
5. 草稿越长不一定越快。后部 token 的存活概率连乘下降，目标模型验证无望 token 会浪费 batch capacity。

## 为什么一次 target forward 能验证多个 token

设当前前缀为 <span class="math-inline">\\(y_{1:t}\\)</span>，草稿模型依次提出：

<div class="math-display">\[
x_1,x_2,\ldots,x_\gamma.
\]</div>

把整段拼到输入后：

```text
[prefix, x1, x2, ..., x_gamma]
```

因果 Transformer 在一次 forward 中会为每个位置计算对应的 next-token logits。虽然 GPU 上矩阵并行执行，但 causal mask 保证位置 <span class="math-inline">\\(i\\)</span> 只读取前缀和 <span class="math-inline">\\(x_{<i}\\)</span>。于是目标模型同时给出：

<div class="math-display">\[
p_1(\cdot)=p(\cdot\mid y_{1:t}),
\]</div>

<div class="math-display">\[
p_i(\cdot)=p(\cdot\mid y_{1:t},x_1,\ldots,x_{i-1}),
\quad i=2,\ldots,\gamma+1.
\]</div>

其中最后一个分布 <span class="math-inline">\\(p_{\gamma+1}\\)</span> 用于“全部草稿都通过时再采一个额外 token”。

验证一段 token 的 FLOPs 比验证一个 token 多，但 GPU 上的矩阵更宽、算术强度更高，墙钟时间可能只略有增加。这是推测解码的硬件前提，而不是说一次 forward 的计算量完全不变。

## Greedy 验证最容易理解

若目标解码策略是 greedy，每个位置只需比较：

<div class="math-display">\[
x_i\stackrel{?}{=}\arg\max_v p_i(v).
\]</div>

从第一个位置开始接受最长连续匹配前缀。遇到首个不匹配位置时，输出目标模型的 argmax，并丢弃后面的草稿，因为后续草稿条件在一个错误 token 上，已经不对应真实前缀。

若 <span class="math-inline">\\(\gamma\\)</span> 个草稿全部匹配，还可从 <span class="math-inline">\\(p_{\gamma+1}\\)</span> 输出一个额外 token。因此一轮最多推进 <span class="math-inline">\\(\gamma+1\\)</span> 个 token，最少也能由目标模型推进 1 个 token。

这能保持 greedy 输出完全一致，但不能直接推广到 temperature sampling。随机采样中，目标分布允许多个 token；“草稿 token 不是目标 argmax”不代表必须拒绝。

## 单 token 推测采样的精确推导

先只看一个位置。草稿分布是 <span class="math-inline">\\(q(x)\\)</span>，目标分布是 <span class="math-inline">\\(p(x)\\)</span>。流程为：

1. 从 <span class="math-inline">\\(q\\)</span> 采样候选 <span class="math-inline">\\(x\\)</span>；
2. 以概率 <span class="math-inline">\\(a(x)=\min(1,p(x)/q(x))\\)</span> 接受；
3. 若拒绝，从某个修正分布 <span class="math-inline">\\(r\\)</span> 重新采样。

### 被接受的概率质量

最终通过“草稿并接受”输出 token <span class="math-inline">\\(x\\)</span> 的概率为：

<div class="math-display">\[
q(x)a(x)
=q(x)\min\left(1,\frac{p(x)}{q(x)}\right)
=\min(q(x),p(x)).
\]</div>

因此接受路径覆盖了 <span class="math-inline">\\(p\\)</span> 和 <span class="math-inline">\\(q\\)</span> 重叠的概率质量。

总接受概率为：

<div class="math-display">\[
\beta=\sum_x\min(p(x),q(x)).
\]</div>

利用 total variation distance：

<div class="math-display">\[
\operatorname{TV}(p,q)
=\frac12\sum_x|p(x)-q(x)|,
\]</div>

可得：

<div class="math-display">\[
\beta=1-\operatorname{TV}(p,q).
\]</div>

这给出了清晰解释：草稿和目标分布越接近，总接受概率越高。

### 拒绝后为什么要采样正残差

目标输出中仍未由接受路径覆盖的质量是：

<div class="math-display">\[
p(x)-\min(p(x),q(x))=(p(x)-q(x))_+,
\]</div>

其中 <span class="math-inline">\\((z)_+=\max(z,0)\\)</span>。因此拒绝后的修正分布必须为：

<div class="math-display">\[
r(x)=\frac{(p(x)-q(x))_+}
{\sum_v(p(v)-q(v))_+}.
\]</div>

拒绝总概率为 <span class="math-inline">\\(1-\beta\\)</span>，恰好等于分母。于是最终输出 <span class="math-inline">\\(x\\)</span> 的总概率为：

<div class="math-display">\[
\min(p(x),q(x))
+(1-\beta)r(x)
=\min(p(x),q(x))+(p(x)-q(x))_+
=p(x).
\]</div>

这证明了单步输出严格服从目标分布。

### 边界情况

- 若 <span class="math-inline">\\(q(x)=0\\)</span>，草稿不会提出 <span class="math-inline">\\(x\\)</span>，比值无需计算；它仍可在残差分布中被输出。
- 若 <span class="math-inline">\\(p=q\\)</span>，所有 token 都以概率 1 接受，残差分母为 0，但拒绝事件不会发生。
- 实现时应避免直接除以极小 <span class="math-inline">\\(q(x)\\)</span> 造成数值问题，可比较 uniform 随机数的对数或显式 clamp。

## 多 token 算法

对每个位置 <span class="math-inline">\\(i\\)</span>，草稿模型在已经接受/提出的草稿前缀上给出 <span class="math-inline">\\(q_i\\)</span>，目标模型并行给出 <span class="math-inline">\\(p_i\\)</span>。随后按顺序验证：

```text
for i = 1 ... gamma:
    accept xi with min(1, p_i(xi) / q_i(xi))
    if rejected:
        sample correction from normalize(max(p_i - q_i, 0))
        stop this round

if all drafts accepted:
    sample one extra token from p_{gamma+1}
```

顺序验证不可省略。第一个拒绝位置之后，目标模型 logits 是在错误草稿前缀上算出的，不能继续作为真实序列的条件分布使用。

### 一个数值例子

词表只有 `A/B/C`：

```text
q = [0.60, 0.30, 0.10]
p = [0.30, 0.50, 0.20]
```

若草稿抽到 `A`，接受概率为 <span class="math-inline">\\(0.30/0.60=0.5\\)</span>。若抽到 `B` 或 `C`，因为 <span class="math-inline">\\(p/q>1\\)</span>，必然接受。

接受路径提供的质量：

```text
min(p, q) = [0.30, 0.30, 0.10]
```

总接受概率是 0.70。剩余目标质量：

```text
max(p - q, 0) = [0.00, 0.20, 0.10]
```

拒绝后的修正分布是 `[0, 2/3, 1/3]`。将 0.30 的拒绝概率乘回去，恰好补足 B 的 0.20 和 C 的 0.10，最终仍得到 `p=[0.30,0.50,0.20]`。

## sampling 参数必须在哪一侧处理

Temperature、top-k、top-p 和 repetition penalty 都会改变分布。为了保持目标解码策略，<span class="math-inline">\\(p_i\\)</span> 必须是**应用目标采样变换后的最终分布**，<span class="math-inline">\\(q_i\\)</span> 也必须与实际草稿采样使用的分布一致。

常见错误包括：

- 草稿从 top-k 后的 <span class="math-inline">\\(q\\)</span> 采样，却用原始 softmax <span class="math-inline">\\(q\\)</span> 算接受率；
- 对目标 logits 和草稿 logits 使用不同 token history penalty；
- 先用低精度截断概率，再计算差分导致负残差或质量不守恒；
- 目标与草稿 tokenizer 不一致，却按同一个 token ID 词表做分布比值。

“目标模型没改”并不足以证明 lossless，接受器使用的分布也必须正确。

## 接受长度如何计算

设每个草稿位置在到达该位置的条件下，接受概率近似相同为 <span class="math-inline">\\(\alpha\\)</span>，草稿长度为 <span class="math-inline">\\(\gamma\\)</span>。第 <span class="math-inline">\\(i\\)</span> 个草稿被接受，需要前 <span class="math-inline">\\(i\\)</span> 个全部通过，概率为 <span class="math-inline">\\(\alpha^i\\)</span>。

每轮输出 token 数 <span class="math-inline">\\(A\\)</span> 至少为 1，额外接受的草稿贡献为：

<div class="math-display">\[
\mathbb E[A]
=1+\sum_{i=1}^{\gamma}\alpha^i
=\frac{1-\alpha^{\gamma+1}}{1-\alpha},
\quad \alpha\ne1.
\]</div>

若 <span class="math-inline">\\(\alpha=1\\)</span>，则 <span class="math-inline">\\(\mathbb E[A]=\gamma+1\\)</span>。

这个公式展示了边际收益递减。例如 <span class="math-inline">\\(\alpha=0.7\\)</span> 时，靠后的草稿存活概率按 <span class="math-inline">\\(0.7^i\\)</span> 下降。无限延长草稿，期望推进长度也只趋近 <span class="math-inline">\\(1/(1-0.7)=3.33\\)</span>，而验证成本仍会增加。

现实中每个位置和上下文的接受率不同。更准确的形式是：

<div class="math-display">\[
\mathbb E[A]
=1+\sum_{i=1}^{\gamma}
\Pr(x_1,\ldots,x_i\text{ 全部接受}).
\]</div>

这也是动态树、置信度截断和 DSpark 一类方法关注“前缀存活概率”而非单 token 平均准确率的原因。

## 一个简化的速度模型

设：

- <span class="math-inline">\\(T_d(\gamma)\\)</span>：生成 <span class="math-inline">\\(\gamma\\)</span> 个草稿的时间；
- <span class="math-inline">\\(T_v(\gamma)\\)</span>：目标模型验证这段草稿的时间；
- <span class="math-inline">\\(T_o\\)</span>：接受、采样、KV 管理等额外开销；
- <span class="math-inline">\\(T_t(1)\\)</span>：目标模型普通单步 decode 时间。

推测解码每输出一个 token 的平均时间近似：

<div class="math-display">\[
\bar T_{spec}
=\frac{T_d(\gamma)+T_v(\gamma)+T_o}
{\mathbb E[A]}.
\]</div>

相对普通解码的理想加速为：

<div class="math-display">\[
S\approx\frac{T_t(1)}{\bar T_{spec}}
=\frac{\mathbb E[A]T_t(1)}
{T_d(\gamma)+T_v(\gamma)+T_o}.
\]</div>

推测解码只有在分子增长快于总成本时才有收益。提高接受率、降低草稿延迟、让验证 kernel 更高效、减少控制面开销，缺一不可。

### 为什么理论接受率高仍可能不加速

1. 草稿模型太大或自回归生成太慢；
2. target verification 随候选数增长太快；
3. batch 已很大，普通 decode 已充分利用 GPU，额外候选挤占吞吐；
4. tree attention、KV 复制和 logits materialization 成本过高；
5. CPU sampler、同步或动态 shape 导致大量 launch gap；
6. prompt 很短、输出很少，初始化成本来不及摊销。

## 延迟优化与吞吐优化不是同一道题

低并发时，普通 decode 常因每步矩阵太瘦而 GPU 利用不足。一次验证多个 token 能增加计算密度，因此 speculative decoding 往往有明显单用户速度收益。

高并发 serving 中，GPU 已通过 continuous batching 同时处理很多请求。每条请求的多个候选 token 会增加本轮 token budget、KV 槽位和 attention 工作。若大量候选在后部被拒绝，系统用宝贵 batch capacity 验证了不会输出的 token，aggregate throughput 可能下降。

因此需要分别报告：

- 单请求或固定并发下的 tokens/s；
- TTFT 和 inter-token latency；
- matched latency 下的 request/token throughput；
- 每轮 draft tokens、accepted tokens 和 acceptance length；
- target tokens computed / output tokens，即验证浪费；
- 不同并发下的 Pareto frontier。

只报告“加速 3x”而不说明并发和基线，信息是不完整的。

## 草稿从哪里来

### 独立小模型

使用同 tokenizer、同模型家族的较小模型。优点是简单、无需修改 target；缺点是额外权重占显存，且小模型与 target 的分布未必足够对齐。

### 同模型提前退出

只运行 target 的前若干层生成草稿，再用剩余层验证，如 LayerSkip/self-speculative decoding。它减少额外权重，并可共享部分激活；但早退层需要足够准确，执行调度也更复杂。

### 多 token prediction heads

在 target hidden state 上添加多个轻量 head，直接预测未来不同位置，如 Medusa。草稿成本低，但独立 head 难以建模候选 token 之间的依赖。

### feature-level drafter

EAGLE 类方法读取 target 中间特征，用小型网络预测后续特征或 token，通常比完全独立小模型更对齐。

### 检索式草稿

n-gram、prompt lookup 或 suffix matching 从当前 prompt、历史输出或缓存语料中查找重复片段。无需训练和额外模型，对代码、模板、重复文本特别有效，但无匹配时收益有限。

### 原生 MTP

部分模型训练时带 multi-token prediction 模块，可直接作为 drafter。它与 target 共同训练、部署方便，但支持哪些模型和接受策略取决于具体引擎。

## 线性候选与树形候选

基本算法每个位置只提出一个 token，形成一条链。若早期 token 猜错，后面整条链失效。

树形 speculation 在某些深度保留 top-k 分支，把多条候选路径压入一棵树，并用 tree attention 一次验证。它提高至少一条路径命中的机会，但候选节点数会快速增长：

<div class="math-display">\[
1+b+b^2+\cdots+b^d
=\frac{b^{d+1}-1}{b-1}.
\]</div>

实际方法会使用稀疏树、固定 node budget 或动态置信度扩展。验证时，树中每个节点只能读取其祖先，不能读取同层其他分支；因此需要特殊 position id、attention mask 和 KV 重排。DeFT 等工作专门优化这类 tree attention 的数据复用。

## 一份实现级检查清单

### 正确性

1. greedy 模式逐 token 与普通 target decode 完全一致；
2. sampling 模式用大样本统计检验输出分布，而不只比一条随机序列；
3. top-k/top-p/temperature 在 proposal 与 acceptance 中一致；
4. 首次拒绝后立即停止使用后续 target logits；
5. 全接受时使用 <span class="math-inline">\\(p_{\gamma+1}\\)</span> 采额外 token；
6. EOS、stop token、maximum length 和 bad words 边界正确；
7. 拒绝后 KV Cache 回滚或重排正确。

### 性能

1. 分开记录 draft、verify、accept/sampler、KV 管理时间；
2. 测不同 input/output length、batch 和并发；
3. 同时报告 accepted length 与端到端速度；
4. 普通 baseline 使用同样的 attention kernel、CUDA Graph、量化和并行设置；
5. 统计失败候选占用的 target token budget；
6. 检查动态候选长度是否导致频繁重新捕获 graph 或 fallback。

## 常见误解

### “无损”是否意味着浮点逐 bit 一致

论文中的 lossless 通常指理论输出分布与 target decoding 相同，或 greedy token 序列相同。不同 kernel、并行归约、低精度 logits 和随机数消费顺序可能导致非 bitwise-identical 结果。应明确讨论的是分布一致、token 一致还是位级一致。

### 草稿模型越准确越好吗

只看准确率不够。一个和 target 极接近但几乎同样大的 drafter，可能因草稿成本太高而更慢。优化目标是单位成本带来的 accepted tokens，而不是单独最大化 draft quality。

### 验证多个 token 等于一次只算一个 token 的成本吗

不是。它通常比单 token forward 贵，但可以因矩阵更宽而提高 GPU 利用率，墙钟增长小于 token 数增长。具体比例必须实测。

### 所有推测解码都保持目标分布吗

不是。典型接受、放宽阈值、近似 top-k 树和某些启发式方法可能以质量变化换取更高接受率。只有满足相应校正规则的算法才能声称精确保持目标采样分布。

## 总结

推测解码没有打破自回归模型的概率分解。它利用的是硬件执行的不对称：小模型顺序猜测多个 token，加上大模型一次并行评分，可能比大模型逐 token 多次加载权重更快。

接受概率 <span class="math-inline">\\(\min(1,p/q)\\)</span> 提取了草稿与目标重叠的概率质量，正残差分布 <span class="math-inline">\\((p-q)_+\\)</span> 补回剩余质量。两者共同保证最终样本仍来自目标分布。后续 EAGLE、DFlash 和 DSpark 等方法并没有改变这个基本验证原则，而是在优化草稿质量、草稿并行度、候选结构和系统调度。

真正评价一个方法时，应问四个问题：一轮平均输出多少 token，草稿花多久，目标验证了多少最终被丢弃的 token，以及这些数字在真实并发下如何变化。

## 参考论文

1. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), Leviathan, Kalman and Matias, ICML 2023.
2. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), Chen et al., 2023.
3. [SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781), Miao et al., 2023.
4. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774), Cai et al., 2024.
5. [Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109), Ankner et al., 2024.
6. [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710), Elhoushi et al., 2024.
7. [DeFT: Decoding with Flash Tree-Attention for Efficient Tree-Structured LLM Inference](https://arxiv.org/abs/2404.00242), 2024.
