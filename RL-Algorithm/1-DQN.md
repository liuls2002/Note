# DQN（Deep Q-Network）详解

DQN（Deep Q-Network）是深度强化学习的经典起点：它把 Q-learning 中的动作价值函数 $Q(s,a)$ 用神经网络 $Q_\theta(s,a)$ 近似，从而能处理高维状态输入（如 Atari 游戏画面）。DQN 的核心目标是学习最优动作价值函数 $Q^*(s,a)$，再通过贪心策略选择动作：

$$
a_t=\arg\max_a Q_\theta(s_t,a).
$$

如果说后面的 REINFORCE、Actor-Critic、PPO 是「直接优化策略」的路线，那么 DQN 属于「先学习价值函数，再由价值函数导出策略」的路线。



## 从 Q-learning 说起

在强化学习基础中，最优动作价值函数满足 Bellman 最优方程：

$$
Q^*(s,a)
=
\mathbb{E}_{s'\sim P(\cdot|s,a)}
\left[
r+\gamma\max_{a'}Q^*(s',a')
\right].
$$

Q-learning 的思想是：用当前估计 $Q(s,a)$ 去逼近右边的 Bellman target。对于一次转移：

$$
(s_t,a_t,r_{t+1},s_{t+1}),
$$

定义 TD target：

$$
y_t
=
r_{t+1}
+
\gamma\max_{a'}Q(s_{t+1},a').
$$

TD 误差为：

$$
\delta_t
=
y_t-Q(s_t,a_t)
=
r_{t+1}
+
\gamma\max_{a'}Q(s_{t+1},a')
-
Q(s_t,a_t).
$$

表格型 Q-learning 更新为：

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
\left[
r_{t+1}
+
\gamma\max_{a'}Q(s_{t+1},a')
-
Q(s_t,a_t)
\right].
$$

其中 $\alpha$ 是学习率。

**直觉**：如果当前 $Q(s_t,a_t)$ 低估了「即时奖励 + 下一状态最优动作价值」，就增大它；如果高估了，就减小它。



## Q-learning 的 off-policy 特性

Q-learning 的 target 使用：

$$
\max_{a'}Q(s_{t+1},a'),
$$

也就是说，它学习的是「下一步采取当前估计下最优动作」的目标策略，而不要求采样动作一定来自这个贪心策略。因此 Q-learning 是 off-policy 算法。

常见采样策略是 $\epsilon$-greedy：

$$
a_t=
\begin{cases}
\arg\max_a Q(s_t,a), & \text{以概率 } 1-\epsilon,\\
\text{随机动作}, & \text{以概率 } \epsilon.
\end{cases}
$$

行为策略负责探索，Q-learning 仍然朝最优贪心策略学习。

这和 SARSA 不同。SARSA 的 target 是：

$$
y_t^{\mathrm{SARSA}}
=
r_{t+1}+\gamma Q(s_{t+1},a_{t+1}),
$$

其中 $a_{t+1}$ 是行为策略真实采样到的动作，因此 SARSA 是 on-policy。



## 为什么需要 DQN

表格型 Q-learning 需要为每个 $(s,a)$ 存一个值：

$$
Q:\mathcal{S}\times\mathcal{A}\rightarrow \mathbb{R}.
$$

当状态空间很大或连续时，这几乎不可行。例如 Atari 游戏中，状态是一叠像素图像；机器人控制中，状态可能是连续传感器读数。

DQN 用神经网络近似 Q 函数：

$$
Q_\theta(s,a)\approx Q^*(s,a).
$$

对于离散动作空间，常见网络结构是：

$$
s
\xrightarrow{\text{CNN/MLP}}
\left[
Q_\theta(s,a_1),
Q_\theta(s,a_2),
\ldots,
Q_\theta(s,a_{|\mathcal{A}|})
\right].
$$

也就是说，网络输入状态 $s$，一次前向输出所有离散动作的 Q 值。



## DQN 的 TD 目标与损失

DQN 用神经网络参数 $\theta$ 表示当前 Q 函数。最直接的 TD target 是：

$$
y_t
=
r_{t+1}
+
\gamma
\max_{a'}Q_\theta(s_{t+1},a').
$$

然后最小化平方 TD 误差：

$$
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
\left(
y_t-Q_\theta(s_t,a_t)
\right)^2
\right].
$$

但这样会有严重不稳定性：target 里也用了同一个网络 $Q_\theta$。参数 $\theta$ 一更新，预测值和目标值同时变化，训练目标像「追逐移动靶」。

DQN 的关键改进有两个：

1. **经验回放（experience replay）**；
2. **目标网络（target network）**。



## 经验回放：打破样本相关性

强化学习采样到的连续转移高度相关：

$$
s_t,a_t,r_{t+1},s_{t+1}
\quad \text{和} \quad
s_{t+1},a_{t+1},r_{t+2},s_{t+2}
$$

来自同一条轨迹，分布并不独立。若直接按时间顺序训练神经网络，容易导致震荡和过拟合最近经验。

DQN 使用 replay buffer $\mathcal{D}$ 存储转移：

$$
\mathcal{D}
=
\{(s_t,a_t,r_{t+1},s_{t+1},d_t)\},
$$

其中 $d_t$ 表示 episode 是否终止。训练时从 $\mathcal{D}$ 中随机采样 minibatch：

$$
(s,a,r,s',d)\sim\mathcal{D}.
$$

经验回放的作用：

* 打破时间相关性，使训练更接近 supervised learning 中的 i.i.d. minibatch；
* 提高样本利用率，一条经验可以被多次训练；
* 平滑数据分布，减少训练震荡。



## 目标网络：稳定 TD target

DQN 维护两套网络：

* **online network** $Q_\theta$：正在训练的网络；
* **target network** $Q_{\theta^-}$：用于计算 TD target 的冻结网络。

TD target 写为：

$$
y
=
r
+
\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a').
$$

其中 $d=1$ 表示终止状态，此时没有未来价值：

$$
y=r.
$$

DQN 的损失为：

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{(s,a,r,s',d)\sim\mathcal{D}}
\left[
\left(
y-Q_\theta(s,a)
\right)^2
\right],
$$

其中：

$$
y
=
r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a').
$$

每隔若干步，把 online network 参数复制给 target network：

$$
\theta^- \leftarrow \theta.
$$

这样 target 在一段时间内相对固定，训练更稳定。



## DQN 算法流程

```
初始化 online Q 网络 Q_θ
初始化 target Q 网络 Q_{θ^-}，令 θ^- ← θ
初始化 replay buffer D
for 每个环境步 do
  1. 用 ε-greedy 根据 Q_θ(s_t, ·) 选择动作 a_t
  2. 执行动作，得到 r_{t+1}, s_{t+1}, done
  3. 将 (s_t, a_t, r_{t+1}, s_{t+1}, done) 存入 D
  4. 从 D 中随机采样 minibatch
  5. 计算 TD target:
       y = r + γ(1-done) max_{a'} Q_{θ^-}(s', a')
  6. 最小化:
       (y - Q_θ(s,a))^2
  7. 每隔 C 步同步 target network:
       θ^- ← θ
end for
```

注意：DQN 虽然用 $\epsilon$-greedy 采样，但 target 使用 $\max_{a'}Q_{\theta^-}(s',a')$，因此它是 off-policy。



## DQN 的关键稳定技巧

### （1）Reward clipping

Atari DQN 中常把奖励裁剪到：

$$
r\in[-1,1].
$$

这样可以统一不同游戏的奖励尺度，避免 TD target 过大导致训练不稳定。

### （2）Huber loss

相比纯 MSE，Huber loss 对异常 TD 误差更稳：

$$
\mathcal{L}_\kappa(\delta)
=
\begin{cases}
\frac{1}{2}\delta^2, & |\delta|\le \kappa,\\
\kappa(|\delta|-\frac{1}{2}\kappa), & |\delta|>\kappa.
\end{cases}
$$

其中：

$$
\delta=y-Q_\theta(s,a).
$$

PyTorch 中常用 `smooth_l1_loss` 实现。

### （3）Frame stacking

单帧图像可能无法表示速度方向，例如球的位置变化。Atari DQN 常把最近 $4$ 帧堆叠作为状态：

$$
s_t=(x_{t-3},x_{t-2},x_{t-1},x_t).
$$

这样状态更接近马尔可夫性。

### （4）$\epsilon$ 衰减

训练早期需要更多探索，后期需要更多利用。常让 $\epsilon$ 从较大值逐渐衰减：

$$
\epsilon: 1.0 \rightarrow 0.1 \text{ 或 } 0.01.
$$



## PyTorch 核心伪代码

下面展示 DQN 中最核心的一步更新。假设 `q_net` 输出所有动作的 Q 值，`target_net` 是冻结的目标网络。

```python
import torch
import torch.nn.functional as F

def dqn_update(q_net, target_net, optimizer, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # Q_θ(s,a)
    q_values = q_net(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_q_max = next_q_values.max(dim=1).values
        targets = rewards + gamma * (1.0 - dones.float()) * next_q_max

    loss = F.smooth_l1_loss(q_sa, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()
```

动作选择使用 $\epsilon$-greedy：

```python
import random

def select_action(q_net, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)

    with torch.no_grad():
        q_values = q_net(state.unsqueeze(0))
        return q_values.argmax(dim=1).item()
```



## Double DQN：缓解过估计

DQN 的 target 使用：

$$
\max_{a'}Q_{\theta^-}(s',a').
$$

由于同一个 max 操作既选择动作又估计动作价值，带噪声的 Q 估计容易被系统性高估。这称为 **overestimation bias**。

Double DQN 将「动作选择」和「动作评估」分开：

* online network $Q_\theta$ 选择动作；
* target network $Q_{\theta^-}$ 评估该动作。

目标变为：

$$
a^*
=
\arg\max_{a'}Q_\theta(s',a'),
$$

$$
y^{\mathrm{Double}}
=
r+\gamma(1-d)Q_{\theta^-}(s',a^*).
$$

这样可以显著缓解 max 带来的过估计问题。



## Dueling DQN：拆分状态价值与优势

Dueling DQN 将 Q 值分解为状态价值和动作优势：

$$
Q(s,a)=V(s)+A(s,a).
$$

但这个分解不唯一，因为给 $V$ 加常数、给 $A$ 减常数不改变 $Q$。实践中常用：

$$
Q(s,a)
=
V(s)
+
\left(
A(s,a)
-
\frac{1}{|\mathcal{A}|}
\sum_{a'}A(s,a')
\right).
$$

这样网络可以分别学习：

* 当前状态整体好不好：$V(s)$；
* 各动作相对当前状态平均动作好多少：$A(s,a)$。

在很多状态下，不同行为的价值差异很小，Dueling 结构能更有效地学习状态价值。



## Prioritized Experience Replay

普通经验回放均匀采样 transition。但不是所有样本同样重要：TD 误差大的样本说明当前网络预测不准，可能更值得学习。

Prioritized Experience Replay 根据 TD 误差设置采样优先级：

$$
p_i \propto |\delta_i|+\epsilon.
$$

采样概率为：

$$
P(i)
=
\frac{p_i^\alpha}{\sum_k p_k^\alpha}.
$$

其中 $\alpha$ 控制优先级强度。为了修正非均匀采样带来的偏差，使用 importance sampling weight：

$$
w_i
=
\left(
\frac{1}{N}\cdot\frac{1}{P(i)}
\right)^\beta.
$$

PER 能加速学习，但实现复杂度更高，也需要调节 $\alpha,\beta$。



## DQN 家族

| 算法 | 核心改进 |
|---|---|
| DQN | 神经网络近似 Q，经验回放，目标网络 |
| Double DQN | 动作选择和动作评估分离，缓解 Q 过估计 |
| Dueling DQN | 拆分 $V(s)$ 与 $A(s,a)$ |
| Prioritized Replay | 更频繁采样 TD 误差大的样本 |
| Rainbow DQN | 集成 Double、Dueling、PER、multi-step、distributional RL 等技巧 |

这些改进大多围绕两个目标：让 Q target 更准，让训练样本更有效。



## DQN 的局限

### （1）主要适合离散动作空间

DQN 需要计算：

$$
\max_{a'}Q(s',a').
$$

如果动作空间是连续的，这个 max 通常无法枚举，因此 DQN 不直接适合连续控制。连续动作空间通常使用 DDPG、TD3、SAC 或 PPO 等方法。

### （2）对超参数和奖励尺度敏感

学习率、replay buffer 大小、target update 频率、$\epsilon$ 衰减、reward clipping 都会影响稳定性。

### （3）Q 值可能过估计或发散

函数逼近、bootstrapping、off-policy 三者结合容易不稳定，这常被称为 deadly triad：

* function approximation；
* bootstrapping；
* off-policy learning。

DQN 的 replay buffer 和 target network 正是在缓解这个问题。



## 与策略梯度方法的对比

| 维度 | DQN | 策略梯度 / PPO |
|---|---|---|
| 学习对象 | $Q_\theta(s,a)$ | $\pi_\theta(a|s)$ |
| 动作空间 | 离散更自然 | 离散/连续都自然 |
| 策略形式 | 贪心或 $\epsilon$-greedy | 随机策略 |
| 数据使用 | off-policy，可 replay | 多数 on-policy，旧数据易过期 |
| 更新信号 | TD error | advantage / return |
| 典型问题 | Q 过估计、训练不稳定 | 方差大、样本效率低 |

从算法发展看，DQN 代表深度 RL 中的价值函数路线；PPO / GRPO 代表策略优化路线。Actor-Critic 则把两者结合起来：Critic 学价值，Actor 学策略。



## 小结

| 组件 | 公式 / 作用 |
|---|---|
| Q-learning target | $r+\gamma\max_{a'}Q(s',a')$ |
| DQN target network | $y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')$ |
| DQN loss | $(y-Q_\theta(s,a))^2$ |
| Experience replay | 随机采样历史 transition，打破相关性 |
| Target network | 固定 TD target，降低训练震荡 |
| $\epsilon$-greedy | 平衡探索与利用 |

**一句话总结**：DQN 用神经网络近似最优动作价值函数 $Q^*(s,a)$，通过经验回放和目标网络稳定 Q-learning 的 TD 更新，是深度强化学习中基于价值方法的代表算法。



## 参考

- Mnih et al., Playing Atari with Deep Reinforcement Learning, 2013.
- Mnih et al., Human-level control through deep reinforcement learning, 2015.
- Van Hasselt et al., Deep Reinforcement Learning with Double Q-learning, 2016.
- Wang et al., Dueling Network Architectures for Deep Reinforcement Learning, 2016.
- Schaul et al., Prioritized Experience Replay, 2016.
