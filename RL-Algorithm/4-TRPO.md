# TRPO（Trust Region Policy Optimization）详解

TRPO（信任域策略优化）处在 **Actor-Critic / 策略梯度** 与 **PPO** 之间：它保留「用 Critic 估计优势、用策略梯度更新 Actor」的主干，但把**步长**从固定学习率换成 **KL 信任域 + 自然梯度方向**，在深度网络下显著缓解「一步更新过大 → 策略崩塌」的问题。下面按 **策略目标 → 替代目标与一阶一致 → KL 约束 → 二次近似求解 → 共轭梯度与线搜索 → 与 GAE 配合** 的顺序展开，并给出实现层面的算法流程。



## 动机：Actor-Critic 仍缺什么

在基础 Actor-Critic 中，Actor 的更新常写为「对数似然梯度 × 优势」的形式，学习率 $\alpha$ 是全局标量。深度策略网络下存在两难：

* **$\alpha$ 太小**：收敛慢，样本效率差；
* **$\alpha$ 太大**：$\pi_\theta(a|s)$ 在若干状态上可能剧烈变形，优势估计与梯度立刻「过期」，表现为回报断崖、甚至发散。

**TRPO 的核心想法**：不显式调 $\alpha$，而要求每次更新后新策略 $\pi_{\theta'}$ 与旧策略 $\pi_{\theta_{\mathrm{old}}}$ 在**分布意义下足够接近**（用平均 KL 度量），并在此 **信任域** 内最大化一个**替代目标**。这样用 **$\delta$（KL 半径）** 代替 **$\alpha$**，稳定性通常更好，代价是每步优化更重（二阶信息、线搜索）。



## 策略性能与优势分解
> 数学推导参考 https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95 和 https://zhuanlan.zhihu.com/p/26308073

记折扣回报下的策略性能为 $\eta(\pi)$（期望从初始分布出发的折扣累计回报）。对任意两策略 $\pi$、$\pi'$，有标准的 **性能差分解**（advantage 形式）：

$$
\eta(\pi') = \eta(\pi) + \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t\, A^{\pi}(s_t, a_t)\right],
$$

其中 $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$ 为优势函数。推导思路是：把每一步的即时回报写成 Bellman 形式，错位相消后只剩下优势项在 $\pi'$ 轨迹上的期望。

**困难**：右边期望是依 **新策略** $\pi'$ 采样的，更新时我们手里只有 **旧策略** 的样本。因此需要 **重要性采样**，把行为策略改为 $\pi$。

> 直观理解公式：新策略与旧策略的期望回报差值为：基于新策略下的(s, a)的分布，在旧策略下的值函数V(s,a)的期望。
新策略是我们的目标，无法事前得到。所以假设新旧策略差异很小（用KL散度约束），用旧策略来采样，并乘上比例因子（重要性因子）来近似新策略的分布。


## 替代目标与一阶一致

定义（与旧策略 $\pi_{\theta_{\mathrm{old}}}$ 对齐的）替代目标，常用状态–动作期望写法：

$$
L_{\theta_{\mathrm{old}}}(\theta)
= \mathbb{E}_{s \sim \rho_{\theta_{\mathrm{old}}},\, a \sim \pi_{\theta_{\mathrm{old}}}}
\left[
\frac{\pi_\theta(a|s)}{\pi_{\theta_{\mathrm{old}}}(a|s)}\, A^{\pi_{\theta_{\mathrm{old}}}}(s,a)
\right],
$$

其中 $\rho_{\theta_{\mathrm{old}}}$ 表示在旧策略下的（折扣）状态访问分布。比值

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
$$

即时间步 $t$ 上的 **重要性采样权重**。

**关键性质（一阶一致）**：在 $\theta = \theta_{\mathrm{old}}$ 处，$L_{\theta_{\mathrm{old}}}$ 与真实性能 $\eta(\pi_\theta)$ **同值且梯度相同**：

$$
L_{\theta_{\mathrm{old}}}(\theta_{\mathrm{old}}) = \eta(\pi_{\theta_{\mathrm{old}}}),
\qquad
\nabla_\theta L_{\theta_{\mathrm{old}}}(\theta)\big|_{\theta=\theta_{\mathrm{old}}}
= \nabla_\theta \eta(\pi_\theta)\big|_{\theta=\theta_{\mathrm{old}}}.
$$

因此，在旧参数附近沿 $\nabla_\theta L_{\theta_{\mathrm{old}}}$ 做小步更新，与直接优化 $\eta$ 的一阶方向一致；但若步长过大，$r_t(\theta)$ 会偏离 1，重要性采样方差爆炸，线性近似失效——这正是要加 **信任域** 的原因。



## 信任域：平均 KL 约束

TRPO 将每次更新约束为：

$$
\max_{\theta}\; L_{\theta_{\mathrm{old}}}(\theta)
\quad \text{s.t.}\quad
\bar{D}_{\mathrm{KL}}\!\left(\pi_{\theta_{\mathrm{old}}} \,\|\, \pi_\theta\right) \le \delta,
$$

其中 **平均 KL**（对状态分布加权）定义为：

$$
\bar{D}_{\mathrm{KL}}\!\left(\pi_{\theta_{\mathrm{old}}} \,\|\, \pi_\theta\right)
= \mathbb{E}_{s \sim \rho_{\theta_{\mathrm{old}}}}
\left[
D_{\mathrm{KL}}\!\left(\pi_{\theta_{\mathrm{old}}}(\cdot|s) \,\|\, \pi_\theta(\cdot|s)\right)
\right].
$$

**直觉**：在参数空间里画一个以 $\theta_{\mathrm{old}}$ 为中心的「橡皮筋」区域——不是欧氏半径固定，而是 **策略输出分布** 的变化量（KL）有上界；在此区域内最大化 $L_{\theta_{\mathrm{old}}}$，避免一步跳到替代模型不可信的区域。

```
                    策略流形/参数空间（示意）
        ┌─────────────────────────────────────┐
        │  π_old 附近：KL ≤ δ 的信任域         │
        │       ┌───────────────┐             │
        │       │  ● θ_old       │             │
        │       │    ╲           │             │
        │       │     ╲ 自然梯度  │             │
        │       │      ◆ θ_new   │（落在边界附近）│
        │       └───────────────┘             │
        └─────────────────────────────────────┘
```



## 局部二次近似与自然梯度方向

直接求解带 KL 的约束优化代价高。TRPO 在 $\theta_{\mathrm{old}}$ 处作局部近似：

* **替代目标**一阶展开：
  $$
  L_{\theta_{\mathrm{old}}}(\theta) \approx L_{\theta_{\mathrm{old}}}(\theta_{\mathrm{old}}) + g^\top (\theta - \theta_{\mathrm{old}}),
  \quad
  g = \nabla_\theta L_{\theta_{\mathrm{old}}}(\theta)\big|_{\theta_{\mathrm{old}}}.
  $$
* **KL 约束**二阶展开：
  $$
  \bar{D}_{\mathrm{KL}}\!\left(\pi_{\theta_{\mathrm{old}}} \,\|\, \pi_\theta\right)
  \approx \frac{1}{2}\,(\theta - \theta_{\mathrm{old}})^\top H\,(\theta - \theta_{\mathrm{old}}),
  $$
  其中 $H$ 为平均 KL 在 $\theta_{\mathrm{old}}$ 处的 **Hessian**。在常见参数化下，它与 **Fisher 信息矩阵**（在适当意义下）对应，因而更新方向常被称为 **自然梯度** 方向。

近似问题化为：

$$
\max_{\Delta\theta}\; g^\top \Delta\theta
\quad \text{s.t.}\quad
\frac{1}{2}\,\Delta\theta^\top H\,\Delta\theta \le \delta.
$$

**解析解**（拉格朗日乘子法，约束取等号）：

$$
\Delta\theta = \sqrt{\frac{2\delta}{g^\top H^{-1} g}}\; H^{-1} g.
$$

* 方向 $H^{-1}g$：在 **KL 诱导的度量** 下最陡上升方向；
* 标量 $\sqrt{2\delta/(g^\top H^{-1} g)}$：把步长顶到信任域边界上。

**与普通策略梯度的对比**：朴素法用 $\Delta\theta \propto g$（欧氏最陡）；TRPO 用 $\Delta\theta \propto H^{-1}g$，使「同样大小的 KL 变化」下目标增益更合理。



## 共轭梯度：不显式求 $H^{-1}$

神经网络参数量极大，不能构造稠密 $H$ 更不能求逆。TRPO 用 **共轭梯度（CG）** 解线性方程

$$
H x = g,
$$

得到 $x \approx H^{-1}g$，再代入

$$
\Delta\theta = \sqrt{\frac{2\delta}{g^\top x}}\, x.
$$

CG 只需 **Hessian–向量积（HVP）** $v \mapsto Hv$，可通过自动微分对 **KL 散度标量** 做反向传播得到，无需显式存储 $H$（实现上常用 Pearlmutter 技巧或双次反向）。

```
共轭梯度（概念）
  输入：向量 g，黑盒 HVP：(v) ↦ H v
  输出：近似 x ≈ H^{-1} g
  迭代中用 HVP 与内积更新共轭方向，无需显式 H 或 H^{-1}
```



## 线搜索：修正泰勒误差

泰勒展开与有限样本估计都会带来误差，**纯理论步长**可能略超真实信任域或使替代目标下降。TRPO 在得到方向 $\Delta\theta$ 后做 **回溯线搜索**：

```
α ← 1
repeat:
    θ_new ← θ_old + α · Δθ
    若  L_{θ_old}(θ_new) 未改进（相对容差） → α 缩小
    若  D̄_KL(π_old || π_new) > δ           → α 缩小
    否则接受 θ_new 并结束
```

实践中常同时检查 **KL 阈值** 与 **替代目标是否足够上升**，与单调改进理论在近似意义下对齐。



## 与 Critic、GAE 的配合

TRPO 只解决 **策略更新** 的几何与步长；优势 $A^{\pi_{\mathrm{old}}}(s,a)$ 仍需从轨迹数据估计。《动手学强化学习》等教程中与工程实现里，普遍采用 **GAE（广义优势估计）** 在偏差–方差之间折中：

$$
\hat{A}_t^{\mathrm{GAE}(\gamma,\lambda)}
= \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l},
\qquad
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t).
$$

典型流程：用当前策略 rollout 一批数据 → 用 Critic $V_\phi$ 与 GAE 得到 $\hat{A}_t$ → 可同步或交替更新 $V_\phi$（如 MSE 拟合回报或 TD 目标）→ 再用 TRPO 步更新 $\theta$。这与 `Actor-Critic.md` 中的 **Actor-Critic + GAE** 主线一致，只是把 Actor 的「怎么迈步」换成了 **KL 信任域 + 自然梯度**。



## 算法流程（与实现对应）

```
初始化 θ, φ（策略与价值）
for 每次迭代 do
  用 π_θ 采样若干条轨迹，得到 (s_t, a_t, r_{t+1}, ...)
  用 V_φ 与 GAE 计算 Â_t
  （可选）回归或 TD 更新 φ
  用样本估计：
      g ← ∇_θ E_t[ r_t(θ) Â_t ]   （在 θ_old 处，r_t=1）
  定义 KL(θ) = D̄_KL(π_old || π_θ)，用 CG + HVP 求 x 使 Hx ≈ g
  Δθ ← sqrt(2δ / (g^T x)) · x
  线搜索确定 α，令 θ ← θ_old + α Δθ
end for
```

其中期望由 minibatch 样本近似；$r_t(\theta)$ 在算 $g$ 时对 $\theta$ 求导会得到与策略梯度一致的形式。



## 与朴素策略梯度、PPO 的对比

| 维度 | 朴素策略梯度 | TRPO | PPO（常见实现） |
|------|----------------|------|------------------|
| 上升方向 | $g$ | $\propto H^{-1}g$（自然梯度） | 一阶梯度 + clip / KL penalty |
| 步长 | 学习率 $\alpha$ | 由 $\delta$ 与二次近似决定 | 学习率 + 目标截断 |
| 稳定性 | 对 $\alpha$ 敏感 | 通常更稳 | 实践中很稳且简单 |
| 计算 | 低 | 高（CG、多次前向） | 中等 |

**一句话关系**：PPO 可视为对 TRPO **信任域思想**的一阶、易实现近似；TRPO 则给出更「几何正确」的步，但工程复杂度更高。



## 优缺点小结

* **优点**：单调改进思路清晰；$\delta$ 往往比调多个学习率更直观；曾是深度连续控制上稳定训练的重要里程碑。
* **缺点**：每步 CG 与线搜索带来额外算力；分布式与部分架构（如复杂 RNN 策略）下工程负担大于 PPO。

若已理解 **Actor-Critic 的优势估计** 与 **策略梯度方向**，TRPO 多出来的部分主要是：**用 $L_{\theta_{\mathrm{old}}}$ 在局部代替 $\eta$、用 KL 画信任域、用 $H^{-1}g$ 与线搜索落实这一步更新**。掌握这三点，再读 PPO 的 clipped surrogate，会看到同一条主线的两种落地方式。


## 参考
https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95