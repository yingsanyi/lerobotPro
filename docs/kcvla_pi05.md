# KC-VLA on PI0.5 (kcvla)

这份文档对应当前仓库里已经落地的 `policy.type=kcvla` 实现。当前版本是无 `bbox` 的弱监督终版：在 **不改原始 `pi05` 代码** 的前提下，引入外部关键词和轻量级 keyword-to-vision cross-attention，使策略从“会看”变成“会找”。

---

## 1. Problem Formulation

给定：

- 视觉观测 `I`
- 自然语言指令 `L`
- 外部提供的关键词集合 `K = {k_j}_{j=1}^M`

模型预测动作序列：

```text
a_t = πθ(I, L, K)
```

与标准 VLA 不同，`K` 被视为显式语义控制变量，而不是从 instruction 中隐式抽取的提示词。

---

## 2. Model Overview

代码路径：

- 配置：[configuration_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/configuration_kcvla.py)
- 模型：[modeling_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/modeling_kcvla.py)
- 预处理：[processor_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/processor_kcvla.py)

结构如下：

```text
image I ──> PI05 vision tower ──> visual tokens V
                                     |
keyword set K ──> tokenizer ──> pooled keyword queries Q_k
                                     |
                                     v
                       Keyword-to-Vision Cross-Attention
                                     |
                                     v
                         grounded visual tokens V'

instruction L + state ──> original PI05 prompt path ──> language tokens T

(V', T) ──> original PI05 transformer + action expert ──> action chunk
```

数学上：

```text
V = f_v(I),  T = f_l(L),  q_j = f_l(k_j)
```

每个关键词作为 query 对视觉 token 做检索：

```text
A_j = Softmax((q_j W_q)(V W_k)^T / sqrt(d))
V_j = A_j (V W_v)
```

然后聚合 grounded context 并回写到视觉特征：

```text
V_K = (1 / M) * Σ_j V_j
V' = V + g(V_K)
```

其中 `g(·)` 在实现里是一个轻量线性层和 residual scale。原始 `pi05` 的视觉编码、prompt 路径、transformer 主干和 action head 保持不变。

---

## 3. Grounding as a Latent Variable

当前实现不再使用任何 `bbox`、patch target 或显式定位标注。

关键词对应的注意力分布 `A_j` 被建模为隐式 grounding 变量：

```text
A_j ≈ p(object_j | I, k_j)
```

也就是说，模型不是被告知“目标在哪”，而是必须通过任务损失和干预约束，自己学出关键词对应的视觉区域。

---

## 4. Training Objective

当前 `kcvla` 的训练目标为：

```text
L = L_policy + λ_cf * L_counterfactual + λ_con * L_contrast + λ_sp * L_sparse
```

### 4.1 `L_policy`

完全沿用 `pi05` 的 flow-matching MSE loss。

### 4.2 `L_counterfactual`

对同一观测 `(I, L)`，提供另一组关键词 `K_cf`。理论上希望：

```text
πθ(I, L, K) != πθ(I, L, K_cf)
```

实现里使用一个稳定的 margin surrogate：

```text
L_counterfactual = max(0, m_cf - d(a(K), a(K_cf)))
```

其中 `d(·)` 是两条 flow/action 轨迹的均方根距离。

### 4.3 `L_contrast`

为了避免不同关键词塌缩到同一片区域，当前实现直接对关键词 attention map 做 pairwise contrast：

```text
L_contrast = - mean_{i != j} ||A_i - A_j||_2^2
```

最小化该项会推动不同关键词关注不同区域。

### 4.4 `L_sparse`

为了获得更局部化的 attention 分布，当前实现最小化注意力熵：

```text
L_sparse = mean_j H(A_j)
```

它抑制“整张图一起看”的假 grounding。

---

## 5. Data Interface

主输入仍然沿用 `pi05`：

- `observation.images.*`
- `observation.state`
- `action`
- `task`

KC-VLA 额外使用：

- `keyword_text`
- `counterfactual_keyword_text` 可选

processor 会自动生成：

- `observation.language.keyword.tokens`
- `observation.language.keyword.attention_mask`
- `observation.language.keyword_cf.tokens`
- `observation.language.keyword_cf.attention_mask`

注意：`keyword_text` 被解释成关键词集合，推荐写成逗号分隔字符串，例如：

```text
止痛药, 维生素
```

反事实关键词分支的行为是：

- 如果人工提供了 `counterfactual_keyword_text`，优先使用人工值。
- 如果该字段为空且 `counterfactual_auto_generate=True`，processor 会按同帧关键词置换自动补一个反事实集合。
- 如果 `counterfactual_enabled=False`，则不会生成 `observation.language.keyword_cf.*`，训练中也不会计算反事实损失。

---

## 6. 和 `pi05` / `s2_pi05` 的区别

和原始 `pi05` 的区别：

- `pi05`: `language -> implicit conditioning`
- `kcvla`: `keyword -> explicit grounding -> action`

和 `s2_pi05` 的区别：

- `s2_pi05` 走重语义路线：object slots、OCR、memory、graph、target pointer
- `kcvla` 走轻量路线：只增加关键词 query、视觉 cross-attention 和弱监督 grounding 损失

如果做论文对比，这两条线正好可以形成：

- lightweight controllable grounding (`kcvla`)
- structured semantic grounding (`s2_pi05`)

---

## 7. 代码使用方式

创建配置：

```python
from lerobot.policies.factory import make_policy_config

cfg = make_policy_config(
    "kcvla",
    device="cuda",
    keyword_max_count=8,
    keyword_text_max_tokens=16,
    counterfactual_enabled=True,
    counterfactual_auto_generate=True,
    loss_contrast_w=0.1,
    loss_counterfactual_w=0.1,
    loss_sparse_w=0.01,
)
```

从 `pi05` 权重初始化：

```python
from lerobot.policies.kcvla.modeling_kcvla import KCVLAPolicy

policy = KCVLAPolicy.from_pretrained("your-pi05-checkpoint", strict=False)
```

`strict=False` 是默认行为，因为新加的 grounding 模块在加载旧 `pi05` checkpoint 时应保持随机初始化。
