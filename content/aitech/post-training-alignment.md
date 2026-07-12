---
title: "LLM 后训练与对齐：SFT、DPO、RLHF 的目标和边界"
date: 2026-07-12T09:40:00+08:00
draft: false
summary: "串联 SFT、奖励模型、RLHF、DPO、KTO 与可验证奖励，厘清不同后训练方法的目标、数据和边界。"
categories: ["AiTech"]
subcategories: ["LLM 对齐"]
topics: ["对齐全景"]
tags: ["LLM", "Post-training", "SFT", "RLHF", "DPO", "Reward Model"]
---

预训练模型擅长续写，但“继续一段网页”与“理解用户意图并给出可靠回答”不是同一个目标。后训练（post-training）通过指令示范、偏好比较和可验证反馈，把基础模型塑造成助手。

对齐不是一个单一算法，也不是把模型变成抽象意义上的“符合所有人类价值”。它是在明确的数据分布、标注规范和约束下，让模型更接近期望行为。谁提供偏好、怎样处理冲突、哪些风险被评估，都会决定最终的对齐方向。

## 一条典型的后训练流水线

```text
预训练模型
  -> 指令数据构造与 SFT
  -> 候选回答采样
  -> 人类/AI/规则反馈
  -> 偏好优化或强化学习
  -> 安全与能力评测
  -> 数据回流和迭代
```

这不是固定配方。有些模型直接做偏好优化，有些在 SFT 后进行大规模可验证奖励 RL，有些会交替使用拒绝采样、蒸馏和在线训练。

## SFT：先教模型什么是任务

监督微调（Supervised Fine-Tuning）使用输入 <span class="math-inline">\(x\)</span> 和目标回答 <span class="math-inline">\(y\)</span>，最小化回答 token 的负对数似然：

<div class="math-display">\[
\mathcal{L}_{\text{SFT}}=-\sum_{t=1}^{|y|}\log \pi_\theta(y_t\mid x,y_{&lt;t})
\]</div>

SFT 主要让模型学习：

- 对话角色和格式；
- 指令到回答的映射；
- 工具调用、引用和结构化输出协议；
- 期望的语气、拒答边界和领域流程。

它不是简单把知识“写入”模型。少量样本可以显著改变行为格式，但可靠注入新事实仍受覆盖、冲突和模型容量限制。

### SFT 数据的三种来源

1. **人工示范**：质量可控但昂贵，标注者差异需要规范和复审；
2. **强模型蒸馏**：扩展快，但会继承教师错误、风格和使用限制；
3. **模型生成后筛选**：可通过规则、执行器或人工保留正确样本，核心是筛选器质量。

去重与多样性同样重要。大量近似模板会让训练 loss 降得很快，却使模型对真实表达变化不稳健。

## Chat template 是模型接口的一部分

对话最终要序列化成 token：

```text
<system>...</system>
<user>...</user>
<assistant>...</assistant>
```

不同模型使用不同特殊 token、换行和终止标记。训练与推理模板不一致会造成明显性能下降。多轮数据还要决定：

- 是否只对最后一轮 assistant 计算 loss；
- 是否训练所有 assistant 回合；
- system 与 user token 是否 mask；
- tool result 属于上下文还是监督目标；
- EOS 出现在每轮还是整个会话末尾。

模板、tokenizer 和权重应作为同一个版本发布。

## 为什么只有 SFT 还不够

一个提示往往存在多个合理回答。SFT 对某个参考答案做逐 token 模仿，难以直接表达“回答 A 比回答 B 更好”。偏好数据则把监督信号改为比较：

<div class="math-display">\[
(x,y_w,y_l)
\]</div>

其中 <span class="math-inline">\(y_w\)</span> 是 preferred/chosen 回答，<span class="math-inline">\(y_l\)</span> 是 rejected 回答。

比较应基于明确准则，例如正确性、完整性、相关性、安全性和风格。把多个维度压成一个“总体更好”标签会隐藏冲突，最好同时保存维度标签和标注理由。

## Reward Model 怎样学习偏好

奖励模型为回答输出标量 <span class="math-inline">\(r_\phi(x,y)\)</span>。常见 Bradley-Terry 形式假设 chosen 胜出的概率为：

<div class="math-display">\[
P(y_w\succ y_l\mid x)=
\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\]</div>

对应损失：

<div class="math-display">\[
\mathcal{L}_{\text{RM}}=-\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\]</div>

奖励的绝对值没有天然含义，核心是排序差异。必须在 held-out 提示和不同来源回答上测排序准确率、校准、长度偏差与分域性能。

奖励模型可能学会捷径：偏爱更长回答、固定格式、自信语气或特定模型风格。若策略模型持续优化这些漏洞，就会发生 reward hacking。

## 经典 RLHF：优化奖励，同时限制漂移

[InstructGPT](https://arxiv.org/abs/2203.02155) 展示的经典路线是：人工示范训练 SFT，人工比较训练奖励模型，再用 PPO 优化策略。

一个抽象目标为：

<div class="math-display">\[
\max_\theta\;
\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r_\phi(x,y)]
-\beta D_{\mathrm{KL}}\left(\pi_\theta(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x)\right)
\]</div>

KL 项限制策略偏离参考模型。它既保护语言能力，也减少模型跑到奖励模型未见区域的风险。<span class="math-inline">\(\beta\)</span> 太大，策略几乎不学习；太小，则更容易奖励过优化和能力漂移。

### PPO 训练为什么复杂

LLM RLHF 通常需要：

- policy：正在更新的模型；
- reference：计算 KL 的冻结模型；
- reward model：为完整回答打分；
- critic/value model：估计未来回报；
- rollout engine：从当前策略采样回答。

还要处理新旧策略概率比、clip、优势估计、长度 mask、生成与训练引擎一致性。因此 RLHF 的难点既是算法，也是多模型系统工程。

## DPO：不用显式奖励模型和在线 RL

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) 从带 KL 约束的最优策略与隐式奖励关系出发，把偏好学习写成直接的分类目标。

定义策略相对参考模型的对数概率差：

<div class="math-display">\[
s_\theta(x,y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)
\]</div>

DPO 损失为：

<div class="math-display">\[
\mathcal{L}_{\text{DPO}}=-\mathbb{E}
\log\sigma\left(\beta\left[
s_\theta(x,y_w)-s_\theta(x,y_l)
\right]\right)
\]</div>

它提高 chosen 相对 reference 的概率优势，同时降低 rejected 的相对优势。reference 不可省略：若只比较策略自身对两个回答的概率，目标的约束含义会改变。

### DPO 的优点

- 训练形态接近普通监督学习；
- 不需要单独训练 reward model 和 critic；
- 使用固定离线偏好数据，工程更简单；
- 复现实验和排查通常比在线 RL 容易。

### DPO 的边界

- 质量受离线 pair 覆盖限制；
- chosen/rejected 难度差太大时，学习到的只是简单区分；
- 序列概率受长度影响，模板和 EOS 处理很关键；
- 无法自动探索数据中没有的更优回答；
- 仍可能过拟合标注偏差。

DPO 不是“效果永远等于 PPO 的简化版”，而是在特定理论假设和数据分布下直接优化偏好。

## KTO 与非配对反馈

有时数据只有“这个回答好/不好”，没有同提示下的成对比较。[KTO](https://arxiv.org/abs/2402.01306) 以 prospect theory 的效用建模利用 desirable/undesirable 样本，并相对参考策略构造目标。

它降低了数据配对要求，但非配对标签的信息结构不同，不能假设只要样本量相同就与高质量 pair 等价。选择方法应从实际能可靠获得哪种反馈开始。

## 可验证奖励与推理型 RL

数学、代码和部分结构化任务可以用执行器、单元测试或答案检查器提供奖励。这类 Reinforcement Learning with Verifiable Rewards（RLVR）减少了主观奖励模型的依赖。

例如同一题采样一组回答，按正确性打分，再用组内相对优势优化策略。[DeepSeekMath](https://arxiv.org/abs/2402.03300) 提出的 GRPO 不训练单独 critic，而使用同题组内奖励作为基线。

可验证不等于没有漏洞：

- 测试可能覆盖不足；
- 字符串判分可能误杀等价答案；
- 模型可能利用执行环境或格式漏洞；
- 只奖励最终答案会让过程 credit assignment 稀疏；
- 训练分布可能只覆盖易自动判定任务。

验证器本身必须像生产代码一样做安全隔离、对抗测试和版本管理。

## Offline 与 On-policy 的关键区别

离线偏好优化使用固定数据；on-policy 训练从当前模型采样，再获得反馈。

| 维度 | 离线方法（如 DPO） | On-policy RL |
| --- | --- | --- |
| 数据 | 固定 pair/标签 | 随策略持续生成 |
| 工程复杂度 | 较低 | 高，需要 rollout 与训练闭环 |
| 探索 | 受离线数据限制 | 可探索当前策略附近的新行为 |
| 分布匹配 | 训练后可能偏离数据 | 反馈更贴近当前策略 |
| 风险 | 过拟合 pair 偏差 | 奖励利用、训练不稳定、成本高 |

实际系统常组合两者：先 SFT/DPO 获得稳定起点，再用 on-policy 数据针对推理或特定行为优化。

## 拒绝采样与迭代式自训练

一个简单而强的后训练方法是：每个 prompt 生成多个候选，用规则、验证器、奖励模型或人工选择最优，再把结果用于下一轮 SFT。

它不需要 policy gradient，但筛选会改变数据分布：

- 只保留最高分可能降低多样性；
- 奖励模型偏差会被固化；
- 高温采样提供探索，也带来更多垃圾候选；
- 同一模型自举可能放大已有错误。

应保留采样配置、全部候选、分数和选择原因，才能分析改进来自生成还是筛选。

## 安全对齐不能只靠拒答样本

安全行为至少包含：

- 识别真实风险，而非关键词触发；
- 在允许范围内提供有帮助的信息；
- 对高风险请求进行适度拒绝或安全重定向；
- 抵抗角色扮演、编码、翻译和多轮诱导；
- 不泄露 system prompt、隐私或工具权限；
- 在不确定时表达边界。

过量粗糙拒答数据会造成 over-refusal。安全集要同时包含危险正例、相似但无害的反例和边界案例，并单独评估 helpfulness 与 harmlessness。

## 数据闭环怎样避免自我欺骗

生产日志可以发现真实失败，但回流前需要：

1. 隐私脱敏与用户授权边界；
2. 按失败类型聚类，而不是只选最差分样本；
3. 区分模型错误、检索错误、工具错误和产品约束；
4. 由独立规则或人员生成修正信号；
5. 保留不可训练的长期回归集；
6. 防止测试样本进入下一轮训练。

如果每次都把失败用例加入训练，又继续在同一集合上报告成绩，得到的是记忆进步而非泛化进步。

## 后训练评测必须分层

### 能力

知识、推理、代码、长上下文、工具调用和目标领域任务。

### 行为

指令遵循、事实性、简洁性、格式遵循、多轮一致性和不确定性表达。

### 安全

危险请求、越狱、偏见、隐私、过度拒绝和工具权限边界。

### 回归

与 reference/SFT 相比，基础语言能力、多语言、长文本和原有任务是否退化。

### 系统

后训练是否改变输出长度、拒答率、工具调用次数，从而影响延迟和成本。

平均总分会掩盖取舍。报告应给出分域指标、置信区间和典型失败案例。

## 常见误区

### RLHF 代表全体人类偏好

它学习的是特定标注者、规范、数据和奖励模型所表达的偏好。应明确这个范围。

### DPO 不需要 reference

标准 DPO 目标中的 reference 是隐式奖励和 KL 约束的基准。训练时可做工程优化，但数学角色仍存在。

### 奖励越高，模型越好

在奖励模型分布外持续优化可能只是在利用缺陷。必须用独立人工/规则评测验证。

### SFT 数据越多越好

重复、冲突和低质量示范会降低模型。数量必须与覆盖和质量一起报告。

### 一个对齐 checkpoint 适合所有产品

写作助手、代码代理、医疗问答和客服的风险、工具权限和成功标准不同，通常还需要领域策略与外部控制。

## 方法选择建议

| 条件 | 合理起点 |
| --- | --- |
| 只有高质量示范 | SFT |
| 有成对偏好，训练资源有限 | DPO 类离线优化 |
| 只有好/坏标签 | KTO 类非配对方法或重构 pair |
| 有可靠奖励模型和 rollout 基础设施 | KL 约束的 RLHF |
| 有可执行验证器 | 拒绝采样 + SFT，进一步考虑 RLVR |
| 线上分布变化快 | 小步 on-policy 数据闭环 + 固定回归集 |

## 总结

后训练的核心不是选择一个最热门缩写，而是把反馈转成正确的学习信号。SFT 教模型任务形式，偏好数据表达相对质量，奖励模型把比较压缩成标量，DPO 直接优化离线偏好，RLHF 与 RLVR 则允许策略在反馈下继续探索。

每种方法都只能对其数据和目标负责。严谨的对齐系统必须同时说明：谁定义了好坏，模型怎样被优化，哪些行为被约束，以及独立评测是否真的支持改进结论。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
