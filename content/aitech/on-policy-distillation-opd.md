---
title: "OPD 蒸馏详解：让学生模型从自己的错误中学习"
date: 2026-07-11T10:00:00+08:00
draft: false
categories: ["AiTech"]
subcategories: ["LLM训练"]
tags: ["LLM", "知识蒸馏", "OPD", "On-Policy Distillation", "GKD", "Reverse KL"]
summary: "从分布偏移、Reverse KL 和在线采样出发，系统梳理 On-Policy Distillation 的原理、训练流程、论文脉络、工程实现与常见失败模式。"
---

传统知识蒸馏像是让学生反复观看老师的标准答案，OPD 则更像老师坐在学生旁边，逐步批改学生自己写出的解题过程。

这一区别看起来只是“谁来生成训练答案”，实际上改变了学生在什么状态上接受监督，也直接触及自回归模型最麻烦的问题之一：**训练时看到的前缀和推理时真正遇到的前缀并不一致**。

OPD（On-Policy Distillation，在线策略蒸馏）的核心价值可以概括成一句话：**让学生生成轨迹，再让教师对学生实际访问到的每个状态提供稠密的 token 级反馈。** 它试图同时获得 SFT 的稠密监督和在线强化学习的分布匹配。

## 从传统知识蒸馏说起

[Hinton 等人在 2015 年系统化提出知识蒸馏](https://arxiv.org/abs/1503.02531)时，核心思想不是只学习 one-hot 标签，而是学习教师完整的软概率分布。教师对错误类别分配的细小概率也包含信息，这通常被称为“暗知识”（dark knowledge）。

设教师和学生在词表上的条件分布分别为 <span class="math-inline">\(p_T(\cdot\mid h)\)</span> 和 <span class="math-inline">\(p_\theta(\cdot\mid h)\)</span>，其中 <span class="math-inline">\(h\)</span> 是当前前缀。常见的 token 级蒸馏目标是前向 KL：

<div class="math-display">\[
D_{\mathrm{KL}}(p_T\Vert p_\theta)
=\sum_{v\in V}p_T(v\mid h)
\log\frac{p_T(v\mid h)}{p_\theta(v\mid h)}.
\]</div>

忽略与学生无关的教师熵后，这就是用教师软标签训练学生的交叉熵。

对分类模型来说，一个输入通常只对应一次预测；但语言模型是自回归的，当前 token 会成为下一个 token 的输入。因此，“在哪些前缀上计算蒸馏损失”变得非常重要。

## Off-policy 蒸馏的问题：训练前缀与推理前缀错位

最常见的离线蒸馏流程是：教师先生成一批高质量回答，学生再对这批固定数据做 SFT 或 logit distillation。它的优点是简单、稳定，教师推理结果也可以缓存复用。

问题在于，学生训练时主要看到教师写出的正确前缀，推理时却必须接着自己刚刚生成的 token 往下写。一旦学生早期犯错，后续就会进入训练数据很少覆盖的状态，错误还会沿序列累积。这就是序列建模中的 exposure bias。

[Scheduled Sampling](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)尝试在训练中逐渐用模型生成 token 替换真实 token；更早的 [DAgger](https://proceedings.mlr.press/v15/ross11a.html)则从模仿学习角度给出了一条更清晰的路线：让当前策略访问环境，再请专家标注当前策略真正访问到的状态。

OPD 可以看作这个思想在语言模型蒸馏中的自然延伸：

```text
采样 prompt
    ↓
学生模型生成 response
    ↓
教师读取同一条 prompt + 学生 response
    ↓
教师为每个位置给出 token 概率或 log-prob
    ↓
最小化师生分布差异，更新学生
    ↓
用更新后的学生重新生成下一批轨迹
```

监督信号始终落在学生自己的状态分布上，因此学生会直接学习如何处理自己容易犯错的前缀。

## OPD 的目标函数

[Agarwal 等人的 GKD 论文](https://arxiv.org/abs/2306.13649)在 ICLR 2024 正式发表，它把离线蒸馏和在线蒸馏统一到一个框架中。若 <span class="math-inline">\(\lambda\)</span> 表示学生生成数据的比例，可以把目标直观地写成：

<div class="math-display">\[
\mathcal L_{\mathrm{GKD}}
=(1-\lambda)\mathcal L_{\mathrm{off}}
+\lambda\mathcal L_{\mathrm{on}}.
\]</div>

- <span class="math-inline">\(\lambda=0\)</span>：完全使用固定的教师或人工数据；
- <span class="math-inline">\(\lambda=1\)</span>：完全使用学生当前策略生成的数据，也就是纯 OPD；
- <span class="math-inline">\(0<\lambda<1\)</span>：混合训练，常用于冷启动和提高稳定性。

GKD 的另一个重要结论是：采样分布和分布距离可以分别选择。OPD 并不天然等于 Reverse KL，但在生成式模型中，Reverse KL 是非常常见的选择：

<div class="math-display">\[
D_{\mathrm{KL}}(p_\theta\Vert p_T)
=\sum_{v\in V}p_\theta(v\mid h)
\log\frac{p_\theta(v\mid h)}{p_T(v\mid h)}.
\]</div>

它的期望由学生分布定义，正好可以在学生采样到的状态和 token 上估计。对于学生采样的 token <span class="math-inline">\(y_t\)</span>，一个常见的即时反馈写法是：

<div class="math-display">\[
r_t
=\log p_T(y_t\mid h_t)
-\log p_\theta(y_t\mid h_t).
\]</div>

教师比学生更认可某个 token 时，<span class="math-inline">\(r_t\)</span> 为正；教师明显不认可时，反馈为负。在 RL 风格的实现中，可以把它作为每个 token 的 advantage，并用重要性采样目标更新学生。若能获得完整词表或可靠的 top-k logits，也可以直接计算更低方差的分布损失。

这里有一个容易混淆的细节：**“在学生轨迹上训练”描述的是数据分布，“Forward KL 或 Reverse KL”描述的是优化距离。** 两者不是同一个维度。纯 on-policy GKD 也可以使用 Forward KL，而 Reverse KL 同样可以搭配混合轨迹。

## 为什么常用 Reverse KL

Forward KL 倾向于覆盖教师分布的多个模式。如果学生容量远小于教师，它可能不得不用有限容量去照顾教师分布中大量低概率区域，生成结果容易变得过于分散。

Reverse KL 更偏向 mode-seeking：学生优先集中到教师认可、自己又能够表达的高概率模式。[MiniLLM](https://arxiv.org/abs/2306.08543)正是从这个角度论证 Reverse KL 更适合生成式大模型蒸馏，并报告了更低的 exposure bias、更好的长文本生成与校准表现。

但 mode-seeking 不是免费的午餐。若学生初始化太弱，正确推理路径在学生分布中的概率几乎为零，纯 Reverse KL 很难凭空发现它。这也是为什么实践中经常先做一段 SFT 或 off-policy distillation，再切换到 OPD。

## OPD、SFT 与 RL 到底有什么不同

| 方法 | 轨迹由谁生成 | 监督密度 | 主要优势 | 典型短板 |
|---|---|---|---|---|
| SFT / 离线蒸馏 | 教师或固定数据集 | token 级、稠密 | 简单稳定，可离线缓存 | 训练与推理分布错位 |
| 在线强化学习 | 当前学生 | 通常是序列级、稀疏 | 直接优化任务奖励，可探索新策略 | 奖励设计难，方差与成本高 |
| OPD | 当前学生 | token 级、稠密 | 在学生状态上获得教师细粒度反馈 | 需要白盒 logits，教师前向成本高 |

Thinking Machines Lab 的[技术文章](https://thinkingmachines.ai/blog/on-policy-distillation/)用一句很直观的话总结了这种组合：SFT 是 off-policy + dense，RL 是 on-policy + sparse，而 OPD 是 on-policy + dense。

这不意味着 OPD 可以替代 RL。RL 的价值在于搜索教师尚未直接给出的新策略；OPD 更擅长把已经存在于教师分布中的能力快速迁移给学生。一个常见的组合是：先用 RL 得到强教师或专家策略，再用 OPD 把最终策略压缩进更小的模型。

## 一个最小训练流程

下面的伪代码省略了分布式推理、padding mask 和重要性采样细节，但保留了 OPD 的核心数据流：

```python
teacher.eval()

for prompts in dataloader:
    # 1. 必须由当前学生生成，构造 on-policy 轨迹
    student.eval()
    with torch.no_grad():
        responses = student.generate(
            prompts,
            temperature=1.0,
            top_p=0.95,
        )

    sequences = concat(prompts, responses)

    # 2. 教师和学生在完全相同的前缀上计算分布
    with torch.no_grad():
        teacher_logits = teacher(sequences).logits
    student.train()
    student_logits = student(sequences).logits

    # 3. 只在 response token 上计算蒸馏损失
    loss = reverse_kl(
        student_logits,
        teacher_logits,
        mask=response_mask,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

真正的大规模实现通常会把 rollout、教师 log-prob 计算和学生训练拆成不同 worker。若保存完整词表 logits 太贵，可以只传输 top-k logits，或者只查询学生实际采样 token 的教师 log-prob。后者最省通信，但信号方差更大，也更容易在长序列上失真。

## 工程实现中最值得盯住的细节

### 1. 教师与学生必须看到同一个前缀

教师不是重新生成一条答案，而是对学生已经生成的轨迹做 teacher forcing 评分。prompt 模板、system prompt、特殊 token 和 response mask 只要有一处没有对齐，log-prob 就失去可比性。

### 2. 同词表最简单，跨 tokenizer 更麻烦

标准 logit distillation 默认师生共享 tokenizer，因为两个分布必须定义在同一组 token 上。Hugging Face 的 [GOLD 技术文章与实现](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)专门讨论了跨 tokenizer 的序列与词表对齐；如果是第一次实现，优先选择同一模型家族的教师和学生。

### 3. 不要一开始就追求 100% on-policy

如果学生还不会产生基本正确的解题结构，教师只能在大量糟糕前缀上打分，训练信号会非常嘈杂。更稳妥的路线是先用教师轨迹做冷启动，再逐步增大 <span class="math-inline">\(\lambda\)</span>。这与 DAgger 中逐步把状态分布交给学生的思路一致。

### 4. 记录“分布是否真的重叠”

只看训练 KL 下降并不够。2026 年的 [Rethinking OPD](https://arxiv.org/abs/2604.13016)进一步指出，OPD 成功通常要求师生思考模式相容，同时教师确实提供学生尚未掌握的新能力。该工作观察到，有效学习主要集中在学生访问状态上的一小组高概率 token，并建议监控 top-k overlap 与相应 token advantage。

这篇工作是很有价值的近期机制分析，但它晚于 GKD 和 MiniLLM，结论仍需要更多模型家族和长程任务验证。工程上可以把它当作诊断工具，而不是已经封闭的理论答案。

### 5. 长序列会放大脆弱性

只对采样 token 计算 <span class="math-inline">\(\log p_T-\log p_\theta\)</span> 很便宜，但每个位置只有一个 Monte Carlo 样本。序列越长，学生越可能漂移到教师很少访问的前缀，单 token 估计也越不稳定。可尝试的改进包括：

- 缩短 rollout，先学习局部行为；
- 使用 top-k 分布而非单个采样 token；
- 增加同一 prompt 的 rollout 数量；
- 混入 off-policy 冷启动数据；
- 对异常大的 token reward、KL 或 importance ratio 做裁剪；
- 单独评估不同长度区间，避免平均指标掩盖长答案退化。

## 一套更稳妥的实践配方

如果要从零搭一条 OPD 流水线，我更推荐下面的顺序：

1. 选择同 tokenizer、能力差距适中的师生模型，先排除词表对齐问题。
2. 用少量高质量教师轨迹做 SFT 或 off-policy distillation，让学生进入正确行为的支持集。
3. 从混合 GKD 开始，例如保留一部分教师轨迹，再逐渐提高学生轨迹比例。
4. 先在短 response 上验证完整词表或 top-k KL，再扩大上下文长度。
5. 同时记录任务准确率、response 长度、师生 KL、top-k overlap 和训练前后通用能力。
6. 用固定 prompt 集定期离线评估，不能只相信在线 batch 上持续下降的蒸馏损失。

目前可直接参考的工程入口包括 [Hugging Face TRL 的 DistillationTrainer](https://huggingface.co/docs/trl/distillation_trainer)、[Thinking Machines 的 Tinker recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)以及 [THUNLP 的 OPD/verl 实现](https://github.com/thunlp/OPD)。三者的抽象层次不同，适合先用 Trainer 验证算法，再转向 verl 一类系统做大规模训练。

## 如何理解 Qwen3 的结果

[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)让 OPD 在大模型推理训练中获得了更多关注。报告对 Qwen3-8B 的对比显示，on-policy distillation 在 AIME'24 和 GPQA-Diamond 上优于其 RL 对照，并使用约十分之一的 GPU 小时。

这个结果说明 OPD 在“强教师已经具备目标推理能力”的场景中非常有竞争力，但不能简单外推为 OPD 总是比 RL 强。实验起点、教师质量、训练栈和任务可验证性都会影响结论，而且技术报告没有公开所有稳定训练细节。

Thinking Machines 后续给出的复现实验和成本分析很有参考价值，但同样应把具体倍数视为特定模型、数据与基础设施下的经验结果，而不是算法常数。

## 总结

OPD 真正重要的地方，不只是把教师 logits 换成了逐 token reward，而是把监督搬到了学生自己的状态分布上。

它解决的是一个非常现实的问题：学生最终必须接着自己的输出继续生成，因此最有价值的老师，不只是展示完美答案，而是能够指出学生在自己走过的路径上究竟从哪里开始偏离。

从方法谱系看，OPD 连接了知识蒸馏、模仿学习和在线强化学习；从工程角度看，它用教师前向计算换取稠密、低方差得多的监督。它最适合做已有能力的高效迁移、压缩与恢复，而不擅长凭空教会一个支持集中完全不存在的新策略。

如果只记住一条实践原则，那就是：**先让学生有机会走到正确答案附近，再用 OPD 精细纠正它在真实生成路径上的每一步。**

## 参考资料

1. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Hinton et al., 2015.
2. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), Ross et al., 2011.
3. [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html), Bengio et al., 2015.
4. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes / GKD](https://arxiv.org/abs/2306.13649), Agarwal et al., ICLR 2024.
5. [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), Gu et al., ICLR 2024.
6. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), Qwen Team, 2025.
7. [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), Thinking Machines Lab, 2025.
8. [General On-Policy Logit Distillation（GOLD）](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation), Hugging Face, 2025.
9. [Rethinking On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2604.13016), Li et al., 2026；[配套代码](https://github.com/thunlp/OPD)。
