# 策略梯度算法 REINFORCE

REINFORCE（Williams, 1992）是最经典的 **蒙特卡洛策略梯度（Monte Carlo Policy Gradient）** 方法：直接参数化策略 $\pi_\theta(a|s)$，用整条轨迹上的回报估计梯度并做梯度上升。它与基于价值的方法（如 Q-Learning）互补，在连续或大离散动作空间上尤其自然。



## 背景：为什么需要策略梯度

**基于价值的方法**先学习 $Q(s,a)$ 或 $V(s)$，再间接得到策略（如 $\varepsilon$-贪心）。这类方法在离散且较小的动作空间上效果很好，但当动作空间**连续或极大**时，对「每个动作」做 argmax 往往不可行或代价很高。

**基于策略的方法**则另辟蹊径：把策略写成带参数 $\theta$ 的概率分布 $\pi_\theta(a|s)$，将「期望回报」直接当作目标函数，对 $\theta$ 求导并梯度上升。REINFORCE 正是这一思路最朴素的实现——用**完整回合（episode）**采样轨迹，用**回报作为权重**调节各步动作的对数概率梯度。

**核心直觉**：若某次交互中在状态 $s_t$ 选了动作 $a_t$，且整条轨迹回报偏高，则增大 $\pi_\theta(a_t|s_t)$；回报偏低则减小。



## 策略梯度的基本原理

### 轨迹与其概率

一步交互：$s_t \xrightarrow{a_t} r_{t+1}, s_{t+1}$。一条完整轨迹可记为：

$$
\tau = (s_1, a_1, s_2, a_2, \ldots, s_T, a_T)
$$

在马尔可夫假设下，给定策略参数 $\theta$，轨迹的概率为：

$$
p_\theta(\tau)
= p(s_1)\,\prod_{t=1}^{T} \pi_\theta(a_t \mid s_t)\,p(s_{t+1} \mid s_t, a_t)
$$

其中 $p(s_1)$ 和 $p(s_{t+1}|s_t,a_t)$ 由**环境**决定，$\pi_\theta(a_t|s_t)$ 由**智能体**决定。



### 目标函数与梯度推导

设轨迹 $\tau$ 的总回报为 $R(\tau)$（如各步奖励之和），我们要最大化期望回报：

$$
J_\theta = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]
= \sum_{\tau} R(\tau)\,p_\theta(\tau)
$$

对 $\theta$ 求梯度，关键一步是**对数导数技巧（log-derivative trick）**：$\nabla p = p\,\nabla\log p$。

$$
\begin{aligned}
\nabla_\theta J_\theta
&= \sum_{\tau} R(\tau)\,\nabla_\theta p_\theta(\tau) \\
&= \sum_{\tau} R(\tau)\,p_\theta(\tau)\,\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \\
&= \sum_{\tau} R(\tau)\,p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau) \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}\bigl[ R(\tau)\,\nabla_\theta \log p_\theta(\tau) \bigr]
\end{aligned}
$$



### 为什么梯度里只剩下策略的对数概率？

对 $\log p_\theta(\tau)$ 展开：

$$
\log p_\theta(\tau)
= \log p(s_1) + \sum_{t=1}^{T} \bigl( \log \pi_\theta(a_t \mid s_t) + \log p(s_{t+1} \mid s_t, a_t) \bigr)
$$

对 $\theta$ 求梯度时，$p(s_1)$ 和 $p(s_{t+1}|s_t,a_t)$ **不依赖 $\theta$**，导数为零，因此：

$$
\nabla_\theta \log p_\theta(\tau)
= \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

代回期望形式，得到同一轨迹上**所有时间步共用一个标量权重 $R(\tau)$** 的梯度：

$$
\nabla_\theta J_\theta
= \mathbb{E}_{\tau}\left[ R(\tau)\,\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$



### 蒙特卡洛估计与因果性改进

期望无法解析计算时，用 $N$ 条独立采样轨迹近似：

$$
\nabla_\theta J_\theta
\approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n}
R(\tau^n)\,\nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n)
$$

**但这里有一个问题**：同一条轨迹里，$t$ 时刻的动作只影响 $t$ 及之后的奖励，用**整局回报** $R(\tau)$ 加权所有步，方差大且不符合因果性。

**改进**：用从 $t$ 开始的**折扣回报**作为第 $t$ 步的权重。定义：

$$
G_t = \sum_{t'=t}^{T} \gamma^{t'-t}\, r_{t'} = r_{t+1} + \gamma\, G_{t+1}
\quad (\gamma \in [0,1] \text{ 为折扣因子})
$$

于是梯度估计变为：

$$
\nabla_\theta J_\theta
\approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n}
G_t^n\,\nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n)
$$

这正是**策略梯度定理**的标准形式：用回报 $G_t$ 作为优势信号的蒙特卡洛估计。



### 参数更新

用学习率 $\eta > 0$ 做梯度上升：

$$
\theta \leftarrow \theta + \eta\,\hat{\nabla}_\theta J_\theta
$$

实现中常等价于最小化损失 $-\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$，然后做一次反向传播。

*一个重要注意点：策略梯度用**当前** $\pi_\theta$ 生成的轨迹估计梯度；更新 $\theta$ 后，轨迹分布 $p_\theta(\tau)$ 改变，旧轨迹来自旧策略，一般**不再重复使用**（这与 off-policy 方法形成对比）。*



## REINFORCE 算法

### 算法流程

```
初始化策略参数 θ
重复每个 episode:
  1. 用 π_θ 采样完整轨迹，记录 (s_t, a_t, r_{t+1}) 及 log π_θ(a_t|s_t)
  2. 从轨迹末端向前计算每步折扣回报 G_t
  3. θ ← θ + η · Σ_t G_t · ∇_θ log π_θ(a_t|s_t)
     （实现中等价于最小化 -Σ_t log π_θ(a_t|s_t) · G_t）
```

这便是 **REINFORCE**：必须在 episode 结束后才能算齐所有 $G_t$，属于**蒙特卡洛**更新，方差较大但实现简单。



### PyTorch 实现（CartPole 任务）

下面的结构与常见教学代码一致：两层 MLP 输出动作概率，按分布采样动作，回合结束后用 $G_t$ 加权 `log_prob` 反传。注意 `gymnasium` / `gym` 的 API 可能略有差异，需按版本微调。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

GAMMA = 0.99

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=128, lr=5e-4):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)

    def act(self, state):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.forward(s)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs):
        # 从后向前算 G_t
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # 启发式方差缩减

        loss = torch.stack([-lp * Gt for lp, Gt in zip(log_probs, returns)]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

> **说明**：上式中对 `returns` 做减均值除标准差是一种**启发式方差缩减**技巧。严格来说，若 baseline 与策略无关则不应破坏无偏性；仅用 batch 内标准化是常见的工程做法，实践中常配合真正的 baseline 或 Actor-Critic 使用。



## 算法改进

朴素 REINFORCE 的核心问题是**方差大**。下面介绍两个经典改进方向。



### （1）添加 Baseline：减小方差而不引入偏差

将每步权重从 $G_t$ 改为 $G_t - b$，其中 $b$ 可取标量（如轨迹回报的移动平均）、或与状态有关的 $b(s_t)$（如状态价值估计）。梯度估计变为：

$$
\nabla_\theta J_\theta
\approx \mathbb{E}\left[ \sum_t \bigl(G_t - b\bigr)\,\nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

#### 无偏性：为什么减 baseline 不改变梯度期望？

要证减去 baseline 后梯度的期望不变，只需证明 baseline 对应项的期望为零。在固定状态 $s$ 下，对动作求期望（$b$ 与采样动作 $a$ 无关，可提到求和外）：

$$
\mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right]
= b(s) \sum_a \pi_\theta(a|s)\,\nabla_\theta \log \pi_\theta(a|s)
$$

利用 $\nabla_\theta \log \pi = \dfrac{\nabla_\theta \pi}{\pi}$：

$$
= b(s) \sum_a \pi_\theta(a|s)\,\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}
= b(s) \sum_a \nabla_\theta \pi_\theta(a|s)
$$

由于 $\sum_a \pi_\theta(a|s) = 1$（概率归一），其梯度 $\sum_a \nabla_\theta \pi_\theta(a|s) = \nabla_\theta 1 = 0$。因此：

$$
\mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = 0
$$

**结论**：只要 $b$ 不依赖本次采样的动作（可与 $s$ 有关），减去 baseline **不引入偏差**。



#### 方差分析：为什么能降方差？

记 $Y = \nabla_\theta \log \pi_\theta(a|s)$，$G$ 为回报，$b$ 为与 $Y$ 独立的常数 baseline。

无 baseline 时：

$$
\mathrm{Var}(YG) = \mathbb{E}[Y^2 G^2] - \bigl(\mathbb{E}[YG]\bigr)^2
$$

有 baseline 时，由无偏性知 $\mathbb{E}[Y(G-b)] = \mathbb{E}[YG]$，因此只需比较 $\mathbb{E}[Y^2(G-b)^2]$ 与 $\mathbb{E}[Y^2 G^2]$：

$$
\mathbb{E}\bigl[Y^2(G-b)^2\bigr] = \mathbb{E}[Y^2 G^2] - 2b\,\mathbb{E}[Y^2 G] + b^2\,\mathbb{E}[Y^2]
$$

方差**减少量**为：

$$
\Delta = 2b\,\mathbb{E}[Y^2 G] - b^2\,\mathbb{E}[Y^2]
$$

对 $b$ 求导令其为零，得到**使方差最小的最优 baseline**：

$$
\boxed{b^* = \frac{\mathbb{E}[Y^2\, G]}{\mathbb{E}[Y^2]}}
$$

代入可得最大方差缩减量非负：$\Delta_{\max} = \frac{(\mathbb{E}[Y^2 G])^2}{\mathbb{E}[Y^2]} \geq 0$。



#### 直觉理解

```
无 baseline:
  G:    |----|--------|-----------------|
        0   mean     50               100
        ▲ 绝对水平放大了梯度乘子的波动

有 baseline (b ≈ E[G]):
  G-b:  |--|----------|-----|-----------|
       -50  0        +50
        ▲ 中心化后，正负围绕零，有效信号更集中
```

- **中心化效应**：若 $G$ 整体偏大，$YG$ 的尺度被「绝对水平」放大；减去接近均值的 $b$ 后，$G-b$ 在零附近波动，数值更稳定。
- **信号保留**：$G_t > b$ 仍对应「相对更好」的动作得到正强化，$G_t < b$ 对应负强化，相对优劣信息完全保留。
- **类比**：记录身高相对全班均值的偏差，往往比记录绝对厘米数更「紧凑」，方差更小而排序信息不变。



#### 为什么常用 $V^\pi(s)$ 作为 Baseline？

最优形式 $b^* = \mathbb{E}[Y^2 G]/\mathbb{E}[Y^2]$ 计算代价高，实践中常见取舍：

| baseline 选择 | 说明 |
|---|---|
| $b = 0$ | 无 baseline，方差最大 |
| $b = \bar{G}$（移动平均） | 实现简单，但未区分状态差异 |
| $b \approx V^\pi(s)$ | 自然选择——$V^\pi(s)$ 是从 $s$ 出发的期望回报，用神经网络 $V_\phi(s)$ 逼近后，权重变为 $G_t - V_\phi(s_t)$，即优势函数 $A(s,a)$ 的蒙特卡洛估计，由此过渡到 **Actor-Critic** |



### （2）每步折扣回报 $G_t$

前文已从「整局 $R(\tau)$」推进到「从 $t$ 起的 $G_t$」：

- **因果性**：第 $t$ 步动作只应对 $t$ 之后的奖励负责。
- **折扣 $\gamma$**：越远未来的奖励对当前动作的 credit 越弱，进一步降方差，也符合无限 horizon 的收敛需求。

REINFORCE + baseline + $G_t$ 常被称为 **REINFORCE with Baseline**，再与价值网络结合便过渡到 Actor-Critic 方法。



## 小结

| 优点 | 缺点 |
|------|------|
| 直接优化随机策略，连续动作空间天然适配 | 蒙特卡洛方差大，样本效率低 |
| 推导清晰，实现简单 | 需等 episode 结束才能更新 |
| 策略梯度家族的理论基石 | 对学习率等超参数敏感 |

**一句话总结**：REINFORCE 用采样轨迹上的折扣回报 $G_t$（及可选 baseline）作为信号，通过 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 做梯度上升，增大「好动作」的概率、减小「差动作」的概率。



## 参考

- [策略梯度算法 REINFORCE 原理与代码实现（博客园）](https://www.cnblogs.com/xumaomao/p/18805908)
- Sutton & Barto《强化学习导论》第 13 章。
- 李宏毅等《Easy RL》策略梯度章节。