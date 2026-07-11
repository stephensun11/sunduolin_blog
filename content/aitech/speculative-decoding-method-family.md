---
title: "推测解码方法谱系：从 Medusa、EAGLE 到 DFlash 与 DSpark"
date: 2026-07-10T09:30:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM 推理优化"]
topics: ["推测解码"]
tags: ["EAGLE", "EAGLE-3", "DFlash", "DSpark", "Medusa", "MTP", "Speculative Decoding"]
summary: "按草稿来源、依赖结构和验证策略梳理推测解码方法家族，重点解释 Medusa/Hydra、EAGLE 1-3、并行草稿 DFlash，以及面向高并发调度的 DSpark。"
---

推测解码的基本框架很稳定：draft 提出候选，target 并行验证，接受器只提交合法前缀。但 2023 年以后方法迅速增多，因为“怎样产生便宜且高质量的草稿”有很多答案。

有的方法使用独立小模型，有的复用目标模型 hidden state，有的增加多 token head，有的从 prompt 中检索重复片段，还有的方法把整段草稿改成并行块生成。到 2026 年，研究重点又从单请求延迟扩展到高并发 serving：候选 token 即使生成很便宜，验证它们仍会占用 target 的 batch capacity。

本文不把论文按年份简单罗列，而是先建立分类坐标，再重点追踪 Medusa/Hydra、EAGLE 1/2/3、DFlash 和 DSpark 的问题链。

## 先建立三个分类坐标

### 坐标一：草稿信息从哪里来

| 类型 | 代表方法 | 优点 | 主要代价 |
|---|---|---|---|
| 独立小模型 | vanilla speculative sampling | 通用、target 无需训练 | 额外权重与顺序 drafting |
| target 多头 | Medusa、Hydra | head 很轻，共享 target 特征 | 需训练附加模块 |
| feature drafter | EAGLE 系列 | 与 target 高度对齐 | 依赖 target 中间特征和专用训练 |
| 原生 MTP | DeepSeek MTP 等 | 训练时已集成，部署路径清晰 | 只适用于带相应模块的模型 |
| 检索/匹配 | n-gram、prompt lookup、suffix | 无模型、几乎无训练成本 | 对重复文本依赖强 |
| 块并行生成 | DFlash、DSpark | 显著减少 drafting 串行深度 | 块内依赖和后缀准确率困难 |

### 坐标二：候选之间是否有因果依赖

- **独立并行**：不同未来位置同时预测，如原始 Medusa head；延迟低，但后部位置不知道前面实际猜了什么。
- **自回归**：每个草稿依赖前一个草稿，如小模型和 EAGLE；质量高，但 drafting 本身仍串行。
- **半自回归/块并行**：大部分计算并行，增加轻量局部因果模块，如 DSpark；在延迟和后缀质量之间折中。

### 坐标三：验证候选是链还是树

- **链**：每个深度只有一个候选，验证便宜，但早期错误让整个后缀失效。
- **静态树**：预先规定每层分支和 node budget，如 Medusa/EAGLE-1 常见设置。
- **动态树**：根据当前上下文置信度扩展高价值节点，如 EAGLE-2。
- **自适应长度**：不一定增加分支，而是决定每条请求验证多长，如 DSpark 的 confidence-scheduled verification。

这些维度彼此独立。EAGLE-3 描述的是 drafter 训练，EAGLE-2 描述的是动态 draft tree；EAGLE-3 可以继续使用 EAGLE-2 的树策略。

## 第一代基线：独立小模型 draft-then-verify

2022-2023 年的 [speculative decoding](https://arxiv.org/abs/2211.17192) 与 [speculative sampling](https://arxiv.org/abs/2302.01318) 奠定了现代框架：小模型 <span class="math-inline">\\(q\\)</span> 自回归提出 <span class="math-inline">\\(\gamma\\)</span> 个 token，大模型 <span class="math-inline">\\(p\\)</span> 一次验证，并使用接受-拒绝校正保持目标分布。

它解决了 target 串行调用次数，却留下三个问题：

1. drafter 仍要顺序运行 <span class="math-inline">\\(\gamma\\)</span> 次；
2. 独立小模型可能与 target 分布不对齐，尤其跨模型家族或专业领域；
3. drafter 权重和 KV Cache 额外占显存。

后续方法大多在解决其中一个。

## 无模型草稿：Prompt Lookup、n-gram 与 suffix

语言输出中存在大量重复。代码会复用变量名和模板，摘要会复制原文片段，文档问答会引用 prompt。Prompt lookup decoding 在已知上下文中查找与当前后缀匹配的 n-gram，把匹配位置之后的 token 作为草稿。

例如当前输出后缀是：

```text
for i in range
```

若 prompt 早先出现相同片段，后续 `(...):` 可能直接成为候选。目标模型仍负责验证，所以 greedy 模式可以保持结果一致。

这类方法的特点是：

- proposal 几乎不使用 GPU 模型计算；
- 没有额外权重和训练；
- 命中时收益高，无重复时退化为普通 decode；
- 哈希表、suffix array/trie 和跨请求缓存决定查找成本。

它适合作为 serving 引擎中的低成本第一选择，也可以与模型 drafter 组合。vLLM 当前文档将 `ngram` 和 `suffix` 与 `mtp`、`eagle3`、`dflash` 并列为不同 proposal method。

## Medusa：把未来预测头挂在 target 上

[Medusa](https://arxiv.org/abs/2401.10774) 不再维护完整小模型，而是在目标模型最后 hidden state 上添加多个轻量 decoding head。第 <span class="math-inline">\\(k\\)</span> 个 head 预测未来第 <span class="math-inline">\\(k\\)</span> 个 token：

<div class="math-display">\[
q_k(x_{t+k}\mid h_t).
\]</div>

所有 head 可一次并行执行，再从各 head 的 top-k 组合出候选树。目标模型使用 tree attention 并行验证多条路径。

### 为什么它快

- 所有 head 只读取一次 target hidden state；
- 不需要一个完整 drafter 逐 token forward；
- 附加参数相对 target 很小；
- 候选树提高至少一条路径命中的概率。

### 独立 head 的根本限制

第 3 个 head 在预测 <span class="math-inline">\\(x_{t+3}\\)</span> 时，并不知道第 1、2 个 head 实际提出了什么。它学习的是从同一个 <span class="math-inline">\\(h_t\\)</span> 直接预测不同未来距离的边缘条件，而不是严格的：

<div class="math-display">\[
p(x_{t+3}\mid x_{\le t},x_{t+1},x_{t+2}).
\]</div>

预测距离越远，多模态不确定性越大，候选质量通常快速下降。

Medusa-1 冻结 backbone、只训练 heads，易于给现有模型增加加速；Medusa-2 联合微调 backbone 与 heads，可提高预测质量，但需要防止改变原模型能力。论文报告的 2.2x、2.3-3.6x 等结果来自其模型和实验设置，不能脱离 target、树大小和硬件使用。

## Hydra：让 draft heads 顺序依赖

[Hydra](https://arxiv.org/abs/2402.05109) 直接针对 Medusa 的独立性问题。后一个 head 不只读取 target hidden state，也条件于前面 draft head 选择的 token embedding：

<div class="math-display">\[
q_k(x_{t+k}\mid h_t,x_{t+1},\ldots,x_{t+k-1}).
\]</div>

这样更接近真实自回归条件分布，提高深层候选准确率。代价是 head 之间出现顺序依赖，不能像 Medusa 那样完全一次并行。

Hydra++ 进一步调整 architecture 和训练目标。论文在其设置中报告相对 Medusa 最高约 1.31 倍 throughput、相对普通自回归最高约 2.70 倍。它体现了一条贯穿后续研究的张力：**并行越彻底，块内依赖越弱；依赖建模越完整，draft latency 越高。**

## EAGLE-1：为什么改为预测 feature

[EAGLE-1](https://arxiv.org/abs/2401.15077) 的出发点是：token 是从一个分布中采样的离散结果。只看 token 会丢失 target 对其他候选的概率信息，而 target 顶层 feature（LM head 之前的 hidden state）包含更丰富的上下文表示。

EAGLE 训练一个轻量自回归 drafter，复用 target 的顶层 feature，并预测下一时刻 feature。预测 feature 经过 target 自己的 LM head 得到 token 分布，再采样 token，并把 token embedding 与 feature 信息用于下一步 drafting。

可以把流程简化为：

```text
target top-layer feature f_t
          + sampled token embedding e(x_{t+1})
          |
          v
small autoregressive feature drafter
          |
          v
predicted feature -> target LM head -> draft token distribution
```

### “feature uncertainty”指什么

从前缀到 next-token distribution 是确定的，但从分布中采样的 token 是随机的。若只预测未来 feature 而不把实际采样 token 反馈给 drafter，就无法知道下一步应该沿哪个条件分支继续。EAGLE 同时使用 feature 和已采样 token，显式处理这种不确定性。

### EAGLE 为什么通常比独立小模型更对齐

- 它直接读取 target 已计算的语义表示；
- 输出复用 target LM head，词表几何保持一致；
- drafter 只需学习 target feature 的局部演进，而不是从 token 重新完成全部语言建模。

EAGLE 仍然是自回归 drafting：为了生成深度为 <span class="math-inline">\\(d\\)</span> 的候选路径，小模型要执行多步。它可以在每步保留多个高概率 token，形成静态 draft tree，再由 target 验证。

## EAGLE-2：从静态树到上下文动态树

静态树通常按“越靠前越容易接受”设置固定分支数，例如第一层 4 个候选、第二层每个保留 2 个。问题是接受率不仅取决于深度，也取决于上下文：

- `1 + 1 =` 的下一 token 高度确定；
- 开放式写作下一 token 分布可能很平；
- 代码模板某些位置确定，变量命名位置不确定。

[EAGLE-2](https://arxiv.org/abs/2406.16858) 发现 EAGLE drafter 的 confidence 能较好近似候选接受概率，因此在给定 node budget 下动态扩展高价值节点。

### 动态扩树的直觉

候选节点的路径存活概率近似为沿路径条件概率的乘积：

<div class="math-display">\[
C(x_{1:i})\approx\prod_{j=1}^{i}q_j(x_j\mid x_{<j}).
\]</div>

维护一个候选池，每次扩展当前置信度最高的叶节点，把有限 node budget 用在最可能被 target 接受的路径上。draft 完成后再剪枝和组织 tree attention。

这类似 best-first search，而不是每个深度平均撒候选。EAGLE-2 论文在其三个模型系列、六项任务上报告 3.05x-4.26x 相对普通解码加速，并比 EAGLE-1 快 20%-40%。收益既来自候选质量，也依赖动态树管理和验证 kernel。

## EAGLE-3：从 feature 回归转向直接 token 预测

EAGLE-1 的 feature prediction loss 帮助 drafter学习多步能力，却也施加了额外约束：最终目标是预测可接受 token，不一定需要精确回归 target 的某个 hidden vector。

[EAGLE-3](https://arxiv.org/abs/2503.01840) 做了两项关键改变。

### 1. 去掉 feature prediction constraint

EAGLE-3 不再要求输出贴合下一时刻顶层 feature，而是直接用 token prediction 目标训练。drafter 的中间向量可以自由形成最有利于未来 token 的表示，减少 feature regression 与 token accuracy 之间的冲突。

### 2. Training-Time Test

只用 ground-truth feature/token 训练一步，会产生 exposure bias：推理第二步看到的是 drafter 自己第一步的输出，分布已偏离训练数据。

EAGLE-3 在训练阶段显式模拟多步 test-time drafting，把前一步预测带入后续步骤，让模型学习纠正自己的分布偏移。它还融合 target 的低层、中层和高层 feature，而不是只使用直接服务 next-token logits 的顶层 feature。

论文观察到原 EAGLE 增加训练数据后收益趋于饱和，而 EAGLE-3 更能从扩大 drafter 数据中获益。其测试报告最高 6.5x 相对普通解码、约 1.4x 相对 EAGLE-2；在 SGLang 的一个 batch size 64 设置中 throughput 提升约 38%。这些数字跨任务差异很大，HumanEval 等模板化代码任务通常更容易 draft。

### EAGLE-1/2/3 到底分别改了什么

| 版本 | 主要问题 | 核心改动 |
|---|---|---|
| EAGLE-1 | 独立 drafter 对 target 对齐不足 | 复用 target 顶层 feature，自回归预测 feature 与 token |
| EAGLE-2 | 静态候选树不适应上下文 | 用 drafter confidence 构造动态 draft tree |
| EAGLE-3 | feature 回归约束与多步 exposure bias | 直接 token 预测、training-time test、多层 feature 融合 |

EAGLE-3 不是简单替换 EAGLE-2。前者改 drafter 的表示与训练，后者改候选树搜索；官方实现让 EAGLE-3 继续使用 EAGLE-2 的动态树。

## MTP：把多 token 预测放进主模型训练

Multi-Token Prediction（MTP）在训练时让模型除 next token 外，还预测更远的未来 token。推理时，这些附加模块可以成为低成本 speculator。

它与 Medusa 的表面相似点是“多个未来预测”，但具体模型可能使用顺序 MTP module、共享层或更复杂依赖，不应一概视为独立 heads。原生 MTP 的工程优势是：

- target 与 drafter 同步训练，词表和表示天然一致；
- checkpoint 明确携带 speculator；
- serving 引擎可针对模型结构实现专用 kernel。

代价是它不是任意现有 checkpoint 都能即插即用。不同模型的 MTP 层数、训练目标和公开权重也不同。

## EAGLE 路线的瓶颈：drafting 仍然串行

EAGLE-3 提高了每一步质量，但生成深度为 6 的链仍需多次轻量 drafter forward。随着 target kernel 和硬件变快，drafter 的串行时间会占越来越大比例。

这催生了 P-EAGLE、DFlash 等并行 drafting 方向：宁可一次提出整块，再依赖 target 验证修正，也不让 drafter 成为新的串行瓶颈。

并行草稿的问题同样明显。若所有位置独立预测，第 <span class="math-inline">\\(i\\)</span> 个位置不知道前 <span class="math-inline">\\(1:i-1\\)</span> 个候选，错误会沿后缀累积，越靠后越难接受。

## DFlash：用轻量 block diffusion 一次生成草稿块

[DFlash](https://arxiv.org/abs/2602.06036) 将一个轻量 block diffusion language model 用作 drafter。它从 target 提取上下文 feature，并对一组 mask position 使用非因果/块级交互，在一次 forward 中并行预测整段草稿，而不是像 EAGLE-3 那样逐 token 自回归。

可以把核心差异写成：

```text
EAGLE-3:  draft step 1 -> step 2 -> ... -> step gamma
DFlash:   [MASK_1, MASK_2, ..., MASK_gamma] -> one block forward
```

### target feature conditioning

纯 diffusion drafter 若只看 token prefix，轻量模型可能质量不足。DFlash 复用 target 的隐藏上下文特征，把大模型已经提取的语义传给 drafter，再由小型 block model 预测未来位置。

### 非因果块内注意力

草稿块中的 mask embedding 可以互相注意，并共同读取 target context feature。这让每个位置在同一层交换信息，同时保持所有位置并行。它不是标准自回归因果条件，因此后部 token 仍可能缺少“前面已经确定的离散 token”这一强条件。

### anchor point 与候选验证

官方 speculators 文档将其流程概括为从 anchor 预测一个或多个 token block，再由 target 验证并接受最长合法前缀。anchor 可以理解为已确认前缀上的出发位置；实现还要处理 position、mask、target hidden feature 和 KV 对齐。

DFlash 论文（2026 年 5 月 v2，ICML 2026 camera-ready）在其模型和任务上报告超过 6x 的 lossless acceleration、最高比 EAGLE-3 的 speedup 高 2.5 倍。这里“lossless”来自后续 target 验证，不代表 DFlash 自己生成的块等于 target；倍数也必须结合论文 baseline 和硬件理解。

### DFlash 的关键收益和风险

收益：

- drafting 串行深度从 <span class="math-inline">\\(O(\gamma)\\)</span> 降到近似常数次 forward；
- 一次较宽矩阵比多次瘦矩阵更适合 GPU；
- target feature 提升轻量并行 drafter 的对齐度。

风险：

- 并行位置缺少完整因果依赖，接受率可能随位置快速衰减；
- 固定验证长块会计算大量低存活概率后缀；
- 不同 block size、模型、领域和并发下的最优点差异很大；
- 2026 年仍是快速演进中的新实现，框架支持矩阵需逐版本确认。

## DSpark：并行 backbone 加轻量顺序模块

[DSpark](https://arxiv.org/abs/2607.05147) 于 2026 年 7 月发布。它直接针对 DFlash 一类 block-parallel drafter 的两个系统问题：

1. **suffix decay**：块内缺少 token 间因果条件，越靠后的 token 越容易错；
2. **verification waste**：即使长块生成很便宜，把低存活概率后缀送给 target 仍会浪费高并发 batch capacity。

### Semi-autoregressive drafter

DSpark 组合：

- 一个并行 backbone，用一次宽 forward 提取整块表示；
- 一个轻量 sequential module，在块内注入前序 token 依赖。

这不是回到完整自回归 drafter。目标是让昂贵表示计算保持并行，只把必要的局部因果建模放入很轻的顺序路径。概念上，它位于 DFlash 的全块并行与 EAGLE 的逐步自回归之间。

### 从 token confidence 到 prefix survival probability

验证只接受连续前缀。即使第 7 个 token 本身置信度高，只要第 3 个 token 被拒绝，第 7 个就没有输出价值。因此调度应关注：

<div class="math-display">\[
s_i=\Pr(x_1,\ldots,x_i\text{ 全部存活}),
\]</div>

而不是孤立的 <span class="math-inline">\\(\Pr(x_i\text{ 正确})\\)</span>。

在简化独立近似下，<span class="math-inline">\\(s_i\\)</span> 是逐位置条件接受概率的乘积，所以会随深度下降。DSpark 用置信度估计 prefix survival，并据此决定每条请求值得验证多长。

### Confidence-scheduled verification

最佳长度不只由模型 confidence 决定，也由引擎负载决定：

- 低并发时，GPU 有空闲计算，验证更长候选可能提高单用户速度；
- 高并发时，每个候选 token 都会挤占 batch token budget，应只验证高收益前缀；
- 不同硬件和 engine 对验证长度的吞吐曲线不同。

可将每个请求长度选择抽象为：

<div class="math-display">\[
k^*=\arg\max_k
\frac{\mathbb E[\text{accepted tokens}\mid k]}
{\text{engine cost}(k,\text{load})}.
\]</div>

DSpark 将 confidence schedule 与 engine-specific throughput profile 结合，为不同请求动态裁剪 verification length。它优化的不只是离线 accepted length，而是服务系统的 latency-throughput Pareto frontier。

### 论文结果该怎样解读

DSpark v1 论文报告：在 DeepSeek-V4 serving system 的真实流量中，相对生产 baseline MTP-1，在 matched throughput 下 per-user generation speed 提升 60%-85%，并缓解严格交互延迟下的吞吐崩塌。

这类结果比单 batch 离线 speedup 更贴近服务价值，但仍需要注意：

- 论文于 2026 年 7 月 6 日首次提交，距离本文写作很近；
- production baseline、DeepSeek-V4 硬件和内部 engine profile 不等于通用开源部署；
- 60%-85% 是相对 MTP-1 且 matched throughput，不是“所有模型加速 6-8 倍”；
- 开源 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) 提供 EAGLE-3、DFlash、DSpark 的统一训练评估代码和 Qwen3/Gemma4 checkpoints，复现时仍需对齐数据、block size 和 target 配置。

## 从 EAGLE 到 DSpark，研究问题如何变化

| 阶段 | 主要瓶颈 | 代表答案 |
|---|---|---|
| vanilla | drafter 与 target 不对齐 | 使用 target feature |
| EAGLE-1 | 离散 token 丢失分布信息 | feature-level autoregressive drafting |
| EAGLE-2 | 固定树浪费 node budget | context-aware dynamic tree |
| EAGLE-3 | feature constraint 与 exposure bias | direct token prediction + training-time test |
| DFlash | EAGLE drafting 串行 | block diffusion parallel drafting |
| DSpark | 并行块 suffix decay、验证浪费 | semi-autoregressive drafter + load-aware verification |

这不是简单的“后一篇完全替代前一篇”。例如：

- 重复型代码可优先使用 n-gram lookup，根本不需要训练 drafter；
- batch 1 低延迟可能适合更长 EAGLE tree；
- 高并发 serving 可能更看重 DSpark 的验证调度；
- 已有 EAGLE-3 checkpoint 和成熟 SGLang/vLLM kernel 时，稳定性可能比新方法的论文峰值更重要。

## 还应关注哪些分支

### Lookahead decoding

Lookahead 使用 Jacobi-style 并行迭代和 n-gram pool 产生候选，不一定需要独立 drafter。它适合无训练修改场景，但候选命中和内存开销依赖任务。

### Self-speculative / LayerSkip

LayerSkip 让早层通过共享 LM head 产生草稿，再由后层验证。它减少额外模型内存并共享部分计算，但需要专门训练使早层可可靠退出。

### SpecInfer 与 DeFT

SpecInfer 系统化使用树形候选；DeFT 优化 tree attention 中共享前缀的 KV 读取，避免每条路径重复搬运祖先数据。这一分支提醒我们：更好的候选树若没有匹配的验证 kernel，也可能在系统上亏损。

### P-EAGLE 与并行 EAGLE

2026 年出现的 P-EAGLE 路线尝试保留 EAGLE-3 的 feature conditioning，同时并行生成多个 proposal，减少顺序 drafter step。它与 DFlash 都针对 drafting latency，但训练目标和候选分布构造不同。相关框架支持仍在推进，应以具体 release 文档为准。

### Speculative Speculative Decoding

2026 年的 SSD/Saguaro 进一步尝试重叠 drafting 与 verification：target 验证当前候选时，drafter预测可能的验证结果并提前为多个结果准备下一轮草稿。它优化的是 draft-verify 两阶段之间的串行依赖，代价是更多预计算和分支管理。

## 如何公平比较这些方法

### 统一 target 与解码设置

必须固定 target checkpoint、tokenizer、dtype、temperature、top-p、最大长度和输入数据。若一个方法用 greedy、另一个用 sampling，接受长度不可直接比较。

### 统一 serving baseline

普通解码应使用同样的 FlashAttention/FlashInfer、paged KV、CUDA Graph、quantization 和并行配置。与未优化 Hugging Face eager baseline 比出的倍数不代表在 vLLM/SGLang 中仍成立。

### 同时报告四组指标

1. **质量/一致性**：greedy exact match 或 sampling distribution test；
2. **算法指标**：acceptance rate、average accepted length、tree nodes、draft depth；
3. **阶段延迟**：draft、tree build、target verify、accept/sampler、KV commit；
4. **系统指标**：不同并发下 TTFT、ITL、tokens/s、requests/s、P99、显存。

### 报告验证浪费

定义一轮 target 计算的候选 token 数为 <span class="math-inline">\\(C\\)</span>，最终输出 token 数为 <span class="math-inline">\\(A\\)</span>。简单的验证利用率为：

<div class="math-display">\[
U_{verify}=\frac{A}{C}.
\]</div>

树方法还要考虑节点共享和 kernel 成本，不能只用这个比例，但它能揭示高并发时为什么“accepted length 变长”仍可能吞吐下降。

## 选型建议

### 无训练、快速接入

先尝试 n-gram/prompt lookup/suffix。对重复少的通用对话，再评估同 tokenizer 的小 drafter。

### 已有成熟 speculator checkpoint

EAGLE-3 是当前较成熟的 feature drafter 路线，已进入 vLLM、SGLang、TensorRT-LLM 等生态。先用框架官方支持的模型和参数做 baseline，再调整 speculative tokens/tree。

### 追求 batch 1 极低延迟

关注 drafter 串行时间和 target verification shape。EAGLE 动态树、DFlash 块并行都可能有效，但应在目标 GPU 上 profile，而不是只比较 accepted length。

### 高并发在线服务

重点测验证 token 对 batch capacity 的影响。需要动态候选长度、负载感知调度和稳定 CUDA Graph bucket。DSpark 提供了值得参考的设计，但其开源实现和论文都很新，应做独立复现和灰度压测。

## 常见概念错误

### EAGLE-2 是一个更大的 drafter 吗

不是。EAGLE-2 的核心是动态 draft tree，利用 confidence 在固定预算内选择候选结构。

### EAGLE-3 还在预测 target feature 吗

它仍复用 target 多层 feature 作为输入信息，但取消了把 drafter 输出回归为 target 顶层 feature 的约束，改为直接 token prediction。

### DFlash 是 FlashAttention 的一种吗

不是。DFlash 中的 “Flash” 指快速 speculative drafting；它使用 block diffusion drafter。FlashAttention 是精确 attention 的 I/O-aware kernel 家族。

### DSpark 是 DeepSeek 模型的新版本吗

不是。DSpark 是 speculative decoding framework/drafter 与验证调度方案，可附加到对应 target。论文在 DeepSeek-V4 服务系统中验证，但开源 DeepSpec 也提供 Qwen3 和 Gemma4 配置。

### 论文中的 lossless 是否意味着任何接受策略都无损

不是。drafter 可以任意近似，但最终是否保持 target 分布取决于验证与校正规则。若使用 typical acceptance、阈值放宽或近似树剪枝，应单独说明质量定义。

## 总结

推测解码方法的演进可以看成四个连续问题：

1. **草稿是否与 target 对齐？** EAGLE 用 target feature 缩小差距。
2. **有限候选预算放在哪里？** EAGLE-2 用动态树适应上下文。
3. **drafting 本身是否仍然串行？** DFlash 用块并行生成压缩串行深度。
4. **target 是否在验证注定失败的后缀？** DSpark 用半自回归建模和负载感知长度调度减少浪费。

真正成熟的系统不会只选一个论文名字。proposal model、candidate structure、verification kernel、KV manager、scheduler 和采样正确性必须共同设计。单请求下最快的方法，在高并发下可能因验证浪费而变慢；接受率最高的方法，也可能因 drafter 太重而输给简单 n-gram。

因此，方法选择的最终单位不是“猜中率”，而是**在目标服务等级和并发下，每单位 target 资源稳定交付多少正确 token**。

## 参考论文与官方实现

1. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), Leviathan et al., ICML 2023.
2. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), Chen et al., 2023.
3. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774), Cai et al., 2024.
4. [Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109), Ankner et al., 2024.
5. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077), Li et al., ICML 2024.
6. [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858), Li et al., EMNLP 2024.
7. [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840), Li et al., NeurIPS 2025.
8. [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE), official implementation.
9. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036), Chen et al., ICML 2026.
10. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147), Cheng et al., 2026.
11. [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec), official training and evaluation code.
12. [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/), official documentation.
13. [DFlash in vLLM Speculators](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/), official documentation.
14. [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251), Kumar, Dao and May, 2026.
