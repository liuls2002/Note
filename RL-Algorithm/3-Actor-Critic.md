# Actor-Critic 方法详解

Actor-Critic（AC）在策略梯度框架中同时维护两套参数：

* **Actor** $\pi_\theta(a|s)$：负责选择动作；
* **Critic** $V_w(s)$（或 $Q_w(s,a)$）：负责评价状态（或状态–动作）。

Critic 用 **时序差分（TD）** 提供低方差的更新信号，使 Actor **无需等待整条轨迹结束**就能更新策略。这正是从 REINFORCE 发展到 A2C / PPO 等实用算法的核心桥梁。



## 为什么需要 Actor-Critic

强化学习有两条经典路线，各自的优缺点如下：

| 路线       | 代表算法           | 核心思路                                | 典型痛点                           |
| -------- | -------------- | ----------------------------------- | ------------------------------ |
| **基于价值** | Q-Learning、DQN | 学 $Q(s,a)$，间接得到策略                   | 离散大动作空间上 argmax 困难；连续动作需要额外技巧  |
| **基于策略** | REINFORCE      | 直接优化 $\pi_\theta$，用蒙特卡洛回报 $G_t$ 作权重 | **高方差**；**必须等 episode 结束**才能更新 |

在 REINFORCE 中，$G_t = r_{t+1} + \gamma r_{t+2} + \cdots$ 把整条轨迹的随机性卷成一个标量，导致梯度抖动大，长回合或在线更新效率低。

**Actor-Critic 的思路**：保持「对数策略梯度 × 标量信号」的形式，但将信号从 **$G_t$** 替换为 Critic 提供的 **TD 误差 $\delta_t$** 或优势估计。用 **bootstrap** 代替全轨迹蒙特卡洛，降低方差，并支持 **每步更新**（代价是引入偏差）。

**核心直觉**：Actor「演戏」、Critic「打分」。TD 误差告诉 Critic「预测偏了多少」，也告诉 Actor「这一步动作相对当前价值估计好不好」。


## 算法原理

### 架构示意

```
┌─────────────────────────────────┐
│          Actor-Critic            │
│                                 │
│   ┌─────────┐       ┌─────────┐ │
│   │ Actor   │       │ Critic  │ │
│   │ π(a|s)  │       │ V(s)    │ │
│   └───┬─────┘       └───┬─────┘ │
│       │                 │       │
│   决定动作 a         评价状态 s  │
│                                 │
│ 更新信号 δ = r + γV(s') − V(s)  │
└─────────────────────────────────┘
```

* **Actor** 目标：最大化长期期望回报
  $$
  \nabla_\theta J(\theta) = \mathbb{E}_\pi \bigl[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \bigr]
  $$
  其中 $A(s,a)$ 是优势函数，通常用 Critic 的 TD 误差或 GAE 估计。

* **Critic** 目标：逼近价值函数 $V^\pi(s)$，常用 TD 误差更新。



### 时序差分 TD 误差

在一步交互后，观测 $r_{t+1}, s_{t+1}$，定义 TD 误差：

$$
\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)
$$

若 $s_{t+1}$ 为终止状态，通常取 $V(s_{t+1})=0$。

**期望含义**：
$$
\mathbb{E}[\delta_t \mid s_t] = (B^\pi V_w)(s_t) - V_w(s_t)
$$
即 Critic 在 $s$ 处的 Bellman 残差。当 $V_w = V^\pi$ 时，$\mathbb{E}[\delta_t] = 0$。因此，训练好的 Critic 能使 $\delta_t$ 波动在零附近，Actor 可用它代替全轨迹回报 $G_t - V(s_t)$。



### 与 REINFORCE 对比

|          | REINFORCE  | Actor-Critic         |
| -------- | ---------- | -------------------- |
| Actor 信号 | $G_t$（全回报） | $\delta_t$（TD 误差）    |
| 方差       | 高          | 较低                   |
| 偏差       | 无偏         | 有偏（bootstrap + 函数逼近） |
| 更新时机     | episode 结束 | 每步更新                 |




## 基础算法流程

```
初始化 Actor 参数 θ，Critic 参数 w
对每个时间步 t:
  1. 用 π(a|s_t; θ) 采样动作 a_t
  2. 环境返回 r_{t+1}, s_{t+1}
  3. δ ← r_{t+1} + γ V(s_{t+1}; w) − V(s_t; w)
  4. Critic: w ← w + α_w · δ · ∇_w V(s_t; w)
  5. Actor: θ ← θ + α_θ · δ · ∇_θ log π(a_t|s_t)
  6. s_t ← s_{t+1}，直到回合结束或达到步数上限
```

> TD 误差同时驱动 Critic 和 Actor：Critic 更新价值预测，Actor 更新策略。实现上常用 `.detach()` 避免 TD 误差反传到 Critic 分支。


### PyTorch 教学示例

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

GAMMA = 0.99

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.actor_head = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        logits = self.actor_head(z)
        value = self.critic_head(z).squeeze(-1)
        return logits, value

def train_step(model, optimizer, state, action, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    logits, value = model(state)
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(torch.tensor(action))

    next_value = 0.0 if done else model(next_state)[1].item()
    td_target = reward + GAMMA * next_value
    td_error = td_target - value.item()

    critic_loss = (td_target - value) ** 2
    actor_loss = -log_prob * td_error

    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return td_error
```



## 算法改进

### n-step return

介于 1-step TD（偏差大、方差小）和全回报 MC（无偏、方差大）之间：
$$
G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V_w(s_{t+n})
$$

### GAE（广义优势估计）

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l},
\quad \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

* $\lambda=0$ → 单步 TD
* $\lambda \to 1$ → 近似 MC
* 实现时可以运行$T$步，在 rollout 末尾可截断，用 $V(s_T)$ 自举

直观上，GAE 在偏差–方差之间做平滑插值。



## Actor-Critic 家族


| 维度 | 原始 AC | A2C | A3C | DDPG |
|---|---|---|---|---|
| **动作空间** | 离散/连续 | 离散/连续 | 离散/连续 | **连续** |
| **策略类型** | 随机 | 随机 | 随机 | **确定性** |
| **学习方式** | on-policy | on-policy | on-policy | **off-policy** |
| **经验回放** | 无 | 无 | 无 | **有** |
| **目标网络** | 无 | 无 | 无 | **有** |
| **并行方式** | 单线程 | 同步并行 | 异步并行 | 单线程 |
| **优势估计** | TD 误差 | GAE | GAE | Critic 直接输出 Q |
| **探索机制** | 策略本身的随机性 | 同上 | 同上 + 异步噪声 | 外加噪声 |
| **典型应用** | 教学/简单任务 | Atari、简单控制 | Atari（历史） | 机器人控制、自动驾驶 |