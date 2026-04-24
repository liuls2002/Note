# DPO（Direct Preference Optimization）详解

DPO（Direct Preference Optimization，直接偏好优化）是 RLHF 之后非常重要的一类对齐方法。它的目标仍然是：让语言模型更偏好人类认为更好的回答；但它不再显式训练奖励模型、也不再用 PPO 做在线强化学习，而是把带 KL 约束的 RLHF 目标推导成一个可以直接在偏好数据上优化的监督学习损失。

如果说 PPO-RLHF 的流程是：

$$
\text{SFT 模型}
\rightarrow
\text{奖励模型 } R_\psi
\rightarrow
\text{PPO 优化策略 } \pi_\theta,
$$

那么 DPO 的流程更短：

$$
\text{SFT/reference model } \pi_{\mathrm{ref}}
\rightarrow
\text{偏好数据 } (x,y_w,y_l)
\rightarrow
\text{直接优化 } \pi_\theta.
$$

其中 $y_w$ 表示 chosen / winner，即人类更偏好的回答；$y_l$ 表示 rejected / loser，即人类较不偏好的回答。DPO 的关键在于：它不是简单地对 $y_w$ 做 SFT，而是同时利用 $y_w$ 与 $y_l$ 的相对偏好，并通过 reference model 隐式保留 KL 正则。



## 动机：为什么需要 DPO

在 PPO-RLHF 中，通常有三步：

1. 用人工指令数据训练 SFT 模型 $\pi_{\mathrm{SFT}}$；
2. 用偏好数据训练奖励模型 $R_\psi(x,y)$；
3. 用 PPO 最大化奖励，同时用 KL 惩罚限制策略偏离参考模型。

这种流程有效，但工程复杂：

* 奖励模型训练质量会直接影响最终策略；
* PPO 需要采样、价值函数、优势估计、KL 控制、奖励归一化等一系列技巧；
* 训练不稳定时很难判断问题来自 reward model、value model、采样分布还是 PPO 超参数；
* 语言模型的动作空间是 token 级巨大离散空间，完整 RL 训练成本较高。

DPO 的出发点是：既然 RLHF 的最终目标可以写成带 KL 正则的策略优化，而人类偏好又常用 Bradley-Terry 模型建模，那么能否把奖励函数消掉，直接得到一个关于 $\pi_\theta$ 的偏好损失？

答案是可以。DPO 的核心贡献正是下面这个闭式变换：

$$
\text{KL-regularized RLHF}
\quad \Longrightarrow \quad
\text{binary classification loss over preference pairs}.
$$



## 偏好数据与 Bradley-Terry 模型

DPO 的训练数据通常是三元组：

$$
(x, y_w, y_l),
$$

其中 $x$ 是 prompt，$y_w$ 是更优回答，$y_l$ 是较差回答。偏好学习中常用 Bradley-Terry 模型描述「回答 $y_w$ 优于 $y_l$」的概率：

$$
P(y_w \succ y_l \mid x)
=
\sigma
\left(
r(x,y_w) - r(x,y_l)
\right),
$$

其中 $r(x,y)$ 是隐含奖励函数，$\sigma(z)=1/(1+\exp(-z))$ 是 sigmoid 函数。

**Bradley-Terry 模型** 的基本假设是：每个候选回答都有一个潜在分数（在 RLHF / DPO 中可理解为 reward），人类对两个回答的偏好只由二者分数差决定，而不是由绝对分数决定。若 $r(x,y_w)-r(x,y_l)$ 很大，则：

$$
P(y_w \succ y_l \mid x) \approx 1;
$$

若二者分数相同，则：

$$
P(y_w \succ y_l \mid x) = \sigma(0)=\frac{1}{2};
$$

若 $r(x,y_w)-r(x,y_l)$ 为负，则模型认为 $y_l$ 更可能被偏好。

这也说明 Bradley-Terry 模型具有 **平移不变性**：如果对同一个 prompt 下所有回答的奖励都加上常数 $c(x)$，偏好概率不变，因为

$$
\bigl(r(x,y_w)+c(x)\bigr)-\bigl(r(x,y_l)+c(x)\bigr)
=
r(x,y_w)-r(x,y_l).
$$

因此，偏好数据通常不能唯一确定每个回答的绝对奖励，只能确定回答之间的相对排序。DPO 后面能把 $\beta\log Z(x)$ 消掉，正是利用了这种「只关心奖励差」的性质。

若显式训练奖励模型 $R_\psi$，对应的 pairwise loss 是：

$$
\mathcal{L}_{\mathrm{RM}}(\psi)
=
-\mathbb{E}_{(x,y_w,y_l)}
\left[
\log\sigma
\left(
R_\psi(x,y_w) - R_\psi(x,y_l)
\right)
\right].
$$

PPO-RLHF 会先学出 $R_\psi$，再拿它指导策略优化。DPO 则走另一条路：不显式表示 $r(x,y)$，而是用策略 $\pi_\theta$ 与参考策略 $\pi_{\mathrm{ref}}$ 的 log probability ratio 表示奖励差。



## KL-regularized RLHF 目标

先看标准的 KL 正则化策略优化目标。给定 prompt $x$，希望策略 $\pi$ 生成高奖励回答，同时不要离参考模型 $\pi_{\mathrm{ref}}$ 太远：

$$
\max_{\pi}
\;
\mathbb{E}_{y \sim \pi(\cdot|x)}
\left[
r(x,y)
\right]
-
\beta
D_{\mathrm{KL}}
\left(
\pi(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)
\right),
$$

其中 $\beta>0$ 控制 KL 正则强度。展开 KL：

$$
D_{\mathrm{KL}}(\pi\,\|\,\pi_{\mathrm{ref}})
=
\mathbb{E}_{y \sim \pi}
\left[
\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}
\right].
$$

于是单个 prompt 下的目标可写为：

$$
\mathcal{J}(\pi)
=
\sum_y \pi(y|x)
\left[
r(x,y)
-
\beta
\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}
\right].
$$

直观上，第一项鼓励高奖励，第二项惩罚策略偏离 reference model。$\beta$ 越大，策略越保守；$\beta$ 越小，策略越容易为了偏好奖励大幅偏移。



## 最优策略的闭式形式

对固定的 $x$，考虑在所有分布 $\pi(\cdot|x)$ 上最大化 $\mathcal{J}(\pi)$，并加入归一化约束：

$$
\sum_y \pi(y|x)=1.
$$

构造拉格朗日函数：

$$
\mathcal{F}(\pi,\lambda)
=
\sum_y \pi(y|x)
\left[
r(x,y)
-
\beta
\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}
\right]
+
\lambda
\left(
\sum_y \pi(y|x)-1
\right).
$$

对 $\pi(y|x)$ 求偏导并令其为零：

$$
\frac{\partial \mathcal{F}}{\partial \pi(y|x)}
=
r(x,y)
-
\beta
\left(
\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)} + 1
\right)
+
\lambda
=0.
$$

整理得：

$$
\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}
=
\frac{1}{\beta}r(x,y)
+
\frac{\lambda-\beta}{\beta}.
$$

指数化：

$$
\pi(y|x)
=
\pi_{\mathrm{ref}}(y|x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right)
\exp
\left(
\frac{\lambda-\beta}{\beta}
\right).
$$

最后一个指数项与 $y$ 无关，只负责归一化。记归一化常数为：

$$
Z(x)
=
\sum_y
\pi_{\mathrm{ref}}(y|x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right),
$$

则最优策略满足：

$$
\pi^*(y|x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y|x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right).
$$

这一步很关键：在 KL 正则化的 RLHF 目标下，最优策略是 reference model 按奖励指数加权后的分布。高奖励回答概率升高，但升高幅度受到 $\pi_{\mathrm{ref}}$ 和 $\beta$ 的共同约束。



## 从最优策略反推出隐式奖励

由上式可反解奖励：

$$
\pi^*(y|x)Z(x)
=
\pi_{\mathrm{ref}}(y|x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right).
$$

取对数并整理：

$$
r(x,y)
=
\beta
\log
\frac{\pi^*(y|x)}
{\pi_{\mathrm{ref}}(y|x)}
+
\beta\log Z(x).
$$

DPO 用参数化策略 $\pi_\theta$ 近似最优策略 $\pi^*$，于是得到隐式奖励：

$$
r_\theta(x,y)
=
\beta
\log
\frac{\pi_\theta(y|x)}
{\pi_{\mathrm{ref}}(y|x)}
+
\beta\log Z(x).
$$

注意 $\log Z(x)$ 只与 prompt $x$ 有关，与具体回答 $y$ 无关。因此在比较同一个 prompt 下两个回答 $y_w,y_l$ 时，它会相消：

$$
\begin{aligned}
r_\theta(x,y_w)-r_\theta(x,y_l)
&=
\beta
\log
\frac{\pi_\theta(y_w|x)}
{\pi_{\mathrm{ref}}(y_w|x)}
-
\beta
\log
\frac{\pi_\theta(y_l|x)}
{\pi_{\mathrm{ref}}(y_l|x)} \\
&=
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right].
\end{aligned}
$$

这就是 DPO 能绕开奖励模型的原因：偏好概率只依赖奖励差，而奖励差可以由策略和 reference model 的 log ratio 表示。



## DPO 损失函数

将上面的隐式奖励差代入 Bradley-Terry 模型：

$$
P_\theta(y_w \succ y_l|x)
=
\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right]
\right).
$$

因此 DPO 的训练损失为：

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-
\mathbb{E}_{(x,y_w,y_l)}
\left[
\log\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right]
\right)
\right].
$$

也常写成更紧凑的形式。定义：

$$
\Delta_\theta
=
\log\pi_\theta(y_w|x)
-
\log\pi_\theta(y_l|x),
$$

$$
\Delta_{\mathrm{ref}}
=
\log\pi_{\mathrm{ref}}(y_w|x)
-
\log\pi_{\mathrm{ref}}(y_l|x).
$$

则：

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-
\mathbb{E}
\left[
\log\sigma
\left(
\beta(\Delta_\theta-\Delta_{\mathrm{ref}})
\right)
\right].
$$

这说明 DPO 不只是让 $\pi_\theta(y_w|x)$ 大于 $\pi_\theta(y_l|x)$，而是要求当前策略相对 reference model 更偏向 chosen：

$$
\Delta_\theta > \Delta_{\mathrm{ref}}.
$$

如果 reference model 本来就强烈偏好 $y_w$，DPO 不需要把差距继续拉得很大；如果 reference model 错误地偏好 $y_l$，DPO 会推动策略纠正这个相对偏好。



## 序列概率与 token 级实现

对语言模型而言，回答 $y=(y_1,\ldots,y_T)$ 的条件概率为：

$$
\pi_\theta(y|x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t|x,y_{<t}).
$$

因此：

$$
\log\pi_\theta(y|x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t|x,y_{<t}).
$$

DPO 中的 $\log\pi_\theta(y_w|x)$ 与 $\log\pi_\theta(y_l|x)$ 通常就是对 answer tokens 的 log probability 求和。实现时要注意：

* prompt token 不应计入回答的 log probability；
* padding token 需要 mask 掉；
* chosen 与 rejected 长度不同，直接求和会天然把完整序列概率纳入比较；
* 有些变体会做长度归一化，但原始 DPO 通常使用序列 log probability。

于是单个样本的 DPO logit 可写为：

$$
z
=
\beta
\left[
\left(
\log\pi_\theta(y_w|x)-\log\pi_{\mathrm{ref}}(y_w|x)
\right)
-
\left(
\log\pi_\theta(y_l|x)-\log\pi_{\mathrm{ref}}(y_l|x)
\right)
\right].
$$

损失就是：

$$
\ell_{\mathrm{DPO}}
=
-\log\sigma(z).
$$



## 梯度直觉

令：

$$
\Delta
=
\Delta_\theta-\Delta_{\mathrm{ref}},
\qquad
\ell(\theta)
=
-\log\sigma(\beta\Delta).
$$

由于：

$$
\frac{\partial \ell}{\partial \Delta}
=
-\beta\sigma(-\beta\Delta),
$$

且 $\Delta_{\mathrm{ref}}$ 与 $\theta$ 无关，有：

$$
\nabla_\theta \ell
=
-\beta\sigma(-\beta\Delta)
\left[
\nabla_\theta\log\pi_\theta(y_w|x)
-
\nabla_\theta\log\pi_\theta(y_l|x)
\right].
$$

用梯度下降更新参数时，方向相当于：

$$
\theta
\leftarrow
\theta
+
\eta\beta\sigma(-\beta\Delta)
\left[
\nabla_\theta\log\pi_\theta(y_w|x)
-
\nabla_\theta\log\pi_\theta(y_l|x)
\right].
$$

因此 DPO 的行为很直观：

* 增大 chosen 回答 $y_w$ 的 log probability；
* 减小 rejected 回答 $y_l$ 的 log probability；
* 当模型已经足够偏好 $y_w$ 时，$\Delta$ 较大，$\sigma(-\beta\Delta)$ 变小，更新自动减弱；
* 当模型仍偏好 $y_l$ 或偏好差距不足时，更新信号更强。

这和 PPO 中「优势为正时提高动作概率，优势为负时降低动作概率」的精神一致，但 DPO 的比较单位是同一个 prompt 下的两个完整回答，而不是 rollout 中逐 token 的 advantage。



## $\beta$ 的作用

DPO 中的 $\beta$ 来自 KL-regularized RLHF 目标，它控制策略相对 reference model 的偏移强度。

从 RL 目标看：

$$
\max_\pi
\mathbb{E}_{y\sim\pi}[r(x,y)]
-
\beta D_{\mathrm{KL}}(\pi\,\|\,\pi_{\mathrm{ref}}),
$$

$\beta$ 越大，KL 惩罚越强，理论上的最优策略越接近 reference model。

从 DPO loss 看：

$$
\mathcal{L}_{\mathrm{DPO}}
=
-\log\sigma(\beta(\Delta_\theta-\Delta_{\mathrm{ref}})).
$$

$\beta$ 同时影响分类 logit 的尺度：

* $\beta$ 较大：对偏好差距更敏感，但梯度也可能更快饱和；
* $\beta$ 较小：更新更温和，允许模型以较小步幅改变相对偏好；
* 实践中 $\beta$ 是非常关键的超参数，需要结合数据质量、模型大小和训练轮数调节。

这里容易产生一个表面矛盾：RL 目标中 $\beta$ 越大代表 KL 惩罚越强，但 DPO loss 中 $\beta$ 越大又会放大偏好 logit。理解时要回到推导：DPO 的 $\beta$ 是从隐式奖励与策略 log ratio 的比例关系中来的，它同时决定「奖励尺度」与「离 reference 的偏移尺度」。实际训练时，应把它视为控制偏好优化强度和 reference 约束之间平衡的温度参数。



## 与 SFT 的区别

SFT 在偏好数据上最朴素的做法是只训练 chosen：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=
-
\mathbb{E}_{(x,y_w)}
\left[
\log\pi_\theta(y_w|x)
\right].
$$

这会提高好回答概率，但没有显式告诉模型「坏回答为什么不好」。DPO 的损失则是 pairwise 的：

$$
\mathcal{L}_{\mathrm{DPO}}
=
-
\log\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right]
\right).
$$

因此 DPO 同时利用正例和负例，并且优化的是相对偏好：

$$
\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)
$$

相对 reference model 的提升。

直观地说，SFT 是「模仿好答案」，DPO 是「在好答案和坏答案之间做偏好排序，并且不要无约束地偏离原模型」。



## 与 PPO-RLHF 的关系

DPO 不是抛弃 RLHF 目标，而是把一类 RLHF 目标解析化了。二者关系如下：

| 维度 | PPO-RLHF | DPO |
|---|---|---|
| 数据 | prompt + 在线采样回答 + RM 奖励 | 离线偏好对 $(x,y_w,y_l)$ |
| 奖励模型 | 显式训练 $R_\psi$ | 不显式训练奖励模型 |
| 策略优化 | PPO / Actor-Critic | 监督式 pairwise loss |
| KL 约束 | 显式 KL penalty | 通过 $\pi_\theta/\pi_{\mathrm{ref}}$ 隐式出现 |
| 价值函数 | 通常需要 $V_\phi$ | 不需要 |
| 训练复杂度 | 高 | 低 |
| 数据分布 | 可在线更新采样分布 | 依赖已有偏好数据 |

从数学上看，PPO-RLHF 优化的是：

$$
\max_\theta
\mathbb{E}_{y\sim\pi_\theta}
\left[
R_\psi(x,y)
\right]
-
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|x)
\,\|\,
\pi_{\mathrm{ref}}(\cdot|x)
\right).
$$

DPO 则利用这个目标的最优解形式，把奖励差替换为：

$$
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right],
$$

从而直接在偏好对上训练。

一句话：PPO-RLHF 是「先学奖励，再用 RL 优化策略」；DPO 是「把奖励模型和 RL 优化折叠进一个偏好分类损失」。



## DPO 算法流程

```
输入：偏好数据 D={(x, y_w, y_l)}，参考模型 π_ref，初始策略 π_θ
for 每个训练 step do
  1. 取一批偏好样本 (x, y_w, y_l)
  2. 用当前模型计算：
       logπ_θ(y_w|x), logπ_θ(y_l|x)
  3. 用冻结的参考模型计算：
       logπ_ref(y_w|x), logπ_ref(y_l|x)
  4. 计算：
       Δ_θ   = logπ_θ(y_w|x)   - logπ_θ(y_l|x)
       Δ_ref = logπ_ref(y_w|x) - logπ_ref(y_l|x)
       loss  = -log σ(β(Δ_θ - Δ_ref))
  5. 对 θ 反向传播并更新；π_ref 保持冻结
end for
```

注意：$\pi_{\mathrm{ref}}$ 通常是 SFT 模型的冻结副本；$\pi_\theta$ 可从同一个 SFT 权重初始化。训练过程中只更新 $\pi_\theta$，不更新 $\pi_{\mathrm{ref}}$。



## PyTorch 伪代码

下面只展示 DPO loss 的核心计算。真实训练中还需要处理 tokenizer、attention mask、只统计 answer token、padding mask、分布式训练等工程细节。

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (policy_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return losses.mean(), chosen_rewards, rejected_rewards
```

其中 `policy_chosen_logps` 表示当前模型对 chosen answer tokens 的 log probability 之和，`ref_chosen_logps` 表示参考模型对应的 log probability 之和。`chosen_rewards` 和 `rejected_rewards` 不是额外训练奖励模型，而是由 DPO 推导出的隐式 reward，常用于监控训练是否把 chosen 和 rejected 拉开。



## 常见细节与注意点

### （1）reference model 必须冻结

DPO 的 KL 约束通过 $\pi_{\mathrm{ref}}$ 进入损失。如果 reference model 跟着训练一起变化，那么 log ratio 的参照系会漂移，DPO 推导中的「相对 reference 的偏好提升」就不再成立。

### （2）不要把 DPO 理解成简单的二分类

DPO loss 形式像二分类，但分类 logit 不是任意打分器输出，而是：

$$
\beta(\Delta_\theta-\Delta_{\mathrm{ref}}).
$$

其中 $\Delta_{\mathrm{ref}}$ 承担了 KL 正则和 reference 校准的角色。没有 reference 项时，目标会退化得更接近朴素偏好分类，容易过度偏移。

### （3）数据质量非常关键

DPO 是离线偏好优化，训练信号完全来自偏好对。如果数据中 chosen / rejected 差异很小、标注噪声大、格式偏差强，模型会直接学习这些偏差。PPO 至少可以通过在线采样探索当前策略分布，DPO 则更依赖偏好数据覆盖面。

### （4）DPO 不需要 value model

因为 DPO 不做 rollout 上的长期回报估计，也不需要 advantage，所以没有 Critic、GAE、TD error 这些组件。这是它比 PPO 简洁很多的主要原因。



## 优缺点小结

* **优点**：不需要显式奖励模型；不需要 PPO、value function、GAE 和在线采样；目标简单稳定；可直接用偏好数据做 supervised fine-tuning 风格训练。
* **缺点**：依赖偏好数据质量；不能像在线 RL 那样主动探索新回答；对 $\beta$、训练轮数、长度处理较敏感；理论推导依赖 KL-regularized RLHF 与 Bradley-Terry 偏好模型假设。

**一句话总结**：DPO 从带 KL 正则的 RLHF 目标出发，利用最优策略与奖励之间的闭式关系，把人类偏好概率写成当前策略相对 reference model 的 log probability ratio，从而用一个简单的 pairwise loss 直接提高 chosen 回答、压低 rejected 回答，并隐式保持策略不远离 SFT 模型。



## 参考

- Rafailov et al., Direct Preference Optimization: Your Language Model is Secretly a Reward Model, 2023.
- Ouyang et al., Training language models to follow instructions with human feedback, 2022.
- Christiano et al., Deep Reinforcement Learning from Human Preferences, 2017.
