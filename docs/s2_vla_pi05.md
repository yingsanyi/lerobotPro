# S2-VLA (Structured Semantic VLA) on pi0.5 (pi05)

本文档是一份可直接交给工程实现的技术设计,目标是在本仓库的 **pi0.5 (policy.type=pi05)** 基础上,通过修改源码,完整实现你的创新点:

- object parsing: detect + track (对象级解析)
- OCR + confidence gating: 置信门控融合视觉与文字
- temporal fusion: 轨迹级跨时间聚合
- semantic graph: 显式语义图(节点+关系)
- language grounding: 指令到图节点的显式对齐/匹配
- decision decomposition: 目标选择(离散)与控制(连续动作生成)解耦
- training objectives: policy loss + text align + counterfactual + temporal consistency

本设计严格对齐当前 repo 的 pi05 实现方式(Flow Matching, prefix/suffix attention mask),并给出关键张量/keys/改动点。

---

## 0. Baseline pi05: 现状约束与可插拔位置

关键文件:

- pi05 模型与推理/训练入口: `/root/songling/lerobotPro/src/lerobot/policies/pi05/modeling_pi05.py`
- pi05 预处理(把 state 离散化后拼进 prompt,再 tokenizer): `/root/songling/lerobotPro/src/lerobot/policies/pi05/processor_pi05.py`
- pi05 配置(默认 tokenizer_max_length=200, STATE/ACTION=QUANTILES): `/root/songling/lerobotPro/src/lerobot/policies/pi05/configuration_pi05.py`
- RTC(解决 chunk 生成延迟导致的动作不连续): `/root/songling/lerobotPro/docs/source/rtc.mdx`

你需要牢记的 baseline 事实:

1. **pi05 的输入主干只有两种 token 源:**
   - 图像 token: `embed_image()` 从 vision tower 得到 `[B, N_img, D_model]`
   - 语言 token: `embed_language_tokens()` 得到 `[B, N_lang, D_model]`
2. pi05 的 action chunk 是 suffix,在 attention mask 上与 prefix(图像+语言)隔离:
   - prefix 的 `att_masks=0`
   - suffix 的第一个 token `att_masks=1`,让 prefix 不能 attend 到 suffix,但 suffix 可以 attend prefix
3. 真实执行建议使用 `predict_action_chunk()` 而不是 `select_action()`:
   - `select_action` 里直接 assert RTC 不支持
   - 你的方法增加前端计算,更需要 RTC/异步

**最关键的插拔点:**

- `PI05Pytorch.embed_prefix(...)` 是把新语义信息“以 token/embedding 的形式”注入模型的最佳位置。

---

## 1. 总体实现策略

### 1.1 建议新增 policy.type = `s2_pi05` (推荐)

不要直接把创新塞进 `pi05` 原类里,否则难做 clean ablation。

建议新增:

- `lerobotPro/src/lerobot/policies/s2_pi05/configuration_s2_pi05.py`
- `lerobotPro/src/lerobot/policies/s2_pi05/modeling_s2_pi05.py`
- `lerobotPro/src/lerobot/policies/s2_pi05/processor_s2_pi05.py` (可选,但推荐)

并在 `/root/songling/lerobotPro/src/lerobot/policies/factory.py` 注册:

- `get_policy_class("s2_pi05") -> S2PI05Policy`
- `make_policy_config("s2_pi05") -> S2PI05Config`
- `make_pre_post_processors` 对 `S2PI05Config` 做 dispatch

### 1.2 为什么要用“prefix token 注入”而不是“把图序列化成文字 prompt”

- prompt 有 `tokenizer_max_length=200` 的硬预算,序列化 graph 很容易截断
- embedding 注入不受 tokenizer 长度限制,但会增加 transformer 序列长度(可控: K 很小)
- 与当前 pi05 的 prefix/suffix mask 机制天然匹配,改动最小

---

## 2. 新增配置面 (S2PI05Config)

建议从 `PI05Config` 继承,只加你需要的 knobs。

关键点: **无论你内部 node_dim 多大,最终注入 prefix 的 semantic token 必须投影到 pi05 的 D_model(即 PaliGemma text hidden size)。**

示例(伪代码):

```python
@PreTrainedConfig.register_subclass("s2_pi05")
class S2PI05Config(PI05Config):
    # semantic frontend
    semantic_enabled: bool = True
    semantic_max_objects: int = 8
    semantic_node_dim: int = 512

    # OCR + gating
    semantic_text_enabled: bool = True
    semantic_text_max_tokens: int = 12
    semantic_gate_alpha: float = 1.0
    semantic_gate_beta: float = 1.0

    # temporal fusion
    semantic_temporal_fusion: str = "gru"  # {none, ema, gru, attn}
    semantic_track_memory: int = 8

    # graph
    semantic_graph_edges: str = "spatial"  # {none, spatial, spatial+interaction}
    semantic_graph_encoder: str = "gat"   # {none, gat, graph_transformer}
    semantic_graph_layers: int = 2

    # grounding
    grounding_enabled: bool = True
    grounding_mode: str = "soft"  # {soft, hard_gumbel, hard_argmax_infer}
    grounding_temperature: float = 1.0
    target_token_count: int = 1

    # loss weights
    loss_text_align_w: float = 0.1
    loss_counterfactual_w: float = 0.1
    loss_temporal_w: float = 0.1
    loss_grounding_w: float = 0.0
```

---

## 3. Batch Schema: 新增 observation keys 与张量契约

为了让训练/推理可复现,建议把语义前端输出写进 batch。

### 3.1 常量命名

在 `/root/songling/lerobotPro/src/lerobot/utils/constants.py` 增加:

- `OBS_SEMANTIC = "observation.semantic"`
- `OBS_SEMANTIC_NODE_EMBS = "observation.semantic.node_embs"`  float32, `[B, K, D_node]`
- `OBS_SEMANTIC_NODE_MASK = "observation.semantic.node_mask"`  bool,    `[B, K]`

可选(若你要显式边):

- `OBS_SEMANTIC_EDGE_INDEX = "observation.semantic.edge_index"` int64, `[B, 2, E]` 或 ragged
- `OBS_SEMANTIC_EDGE_ATTR  = "observation.semantic.edge_attr"`  float32, `[B, E, D_edge]`

可选(调试):

- `OBS_SEMANTIC_BOXES_XYXY = "observation.semantic.boxes_xyxy"` float32, `[B, K, 4]` in [0,1]
- `OBS_SEMANTIC_TRACK_IDS  = "observation.semantic.track_ids"`  int64,   `[B, K]`
- `OBS_SEMANTIC_OCR_CONF   = "observation.semantic.ocr_conf"`   float32, `[B, K]`

### 3.2 Processor vs Policy 产出这些 keys

两种路线:

- Processor-based(推荐): `processor_s2_pi05.py` 负责 offline load 或 online compute,把结果写入 batch。
- Policy-based: `S2PI05Policy.predict_action_chunk` 运行语义前端并把结果拼进 forward。

建议:

- 训练: offline precompute + processor load
- 部署推理: online compute

---

## 4. SemanticFrontend: 模块细节 (graph/gating/temporal 全落地)

建议把前端拆成可测试模块:

- `ObjectParser` (detect+track)
- `OCRReader` (可选)
- `ObjectEmbedder` (从 pi05 vision tokens 抽对象视觉表征)
- `TextEmbedder` (把 OCR 文本 embed 成向量)
- `GateFusion` (你的 gating + 融合)
- `TemporalFuser` (轨迹级聚合)
- `GraphBuilder` + `GraphEncoder`
- `GroundingHead`

### 4.1 Object parsing: Detect + Track

输入:

- 一帧或多帧图像(多相机)

输出(每样本 Top-K):

- `boxes_xyxy` 归一化到 [0,1]
- `scores`
- `track_id`

工程落地建议:

- 首版先冻结 detect/track,保证系统闭环
- track_id 在 online 推理里保存在 `S2PI05Policy` 的状态中(TrackMemory)
- 训练尽量 offline 预计算,避免 DataLoader 卡死

### 4.2 Visual object embedding: ROI pool over pi05 vision tokens

你必须得到 `z_vis_i`。最推荐方式是 **ROI pooling**:

- pi05 的 `embed_image()` 已产出图像 token `[B, N_img, D_model]`
- 将 token reshape 成 2D grid 后,对每个 box 做 `roi_align` 得到 `[B, K, D_model]` 或 `[B, K, D_vis]`

注意: pi05 会在 `_preprocess_images` 中对输入做 `resize_with_pad_torch` 到 `config.image_resolution`。

因此 ROI 坐标必须使用同一坐标系:

- 如果 detector 在原图分辨率输出 box,你要把 box 通过 resize+pad 映射到 224x224
- 建议在 online 前端直接对“已 resize+pad 的图像”做 detect,从根源上避免坐标映射 bug

备选(更简单但慢):

- 对每个对象 crop+resize 到 224,再次调用 `embed_image`,然后 mean-pool tokens

### 4.3 OCR + text embedding

OCR 输出 `(text_i, conf_i)`。

为了与 pi05 共享语义空间,建议用 PaliGemma 的 token embedding:

- tokenizer: `google/paligemma-3b-pt-224`
- embed: `paligemma.model.language_model.embed_tokens`
- pool: mean-pool 或 attention pool 得到 `z_text_i`

### 4.4 Confidence gating (你的核心创新)

门控与融合:

- `s_i = cosine(ProjV(z_vis_i), ProjT(z_text_i))`
- `w_i = sigmoid(alpha * conf_i + beta * s_i)`
- `z_obj_i = ProjV(z_vis_i) + w_i * ProjT(z_text_i)`

落地细节:

- 训练阶段保持 soft gate,不做 hard threshold
- 推理阶段可设阈值,把 `w_i` 过低的 OCR 置空(减少噪声)

### 4.5 Temporal fusion (track-level)

目标: 把 per-frame `z_obj_i_t` 聚合成轨迹表示 `h_i`。

实现选型:

- EMA: 便宜稳定
- GRU: 能学时序
- attention: 最强但重

落地关键:

- online: `TrackMemory[track_id]` 存 `(h_i, history)`
- training: 若 batch 不连续,要么 offline precompute,要么改采样策略(连续窗口)

---

## 5. Semantic graph: BuildGraph + GraphEncoder

### 5.1 图构建

节点:

- 每个 track 一个 node,特征为 `h_i` 或 `z_obj_i`

边(首版只做 spatial):

- left_of/right_of/above/below/near

edge_attr 建议包含:

- relation type embedding
- relative geometry(bbox center delta, size ratio)

### 5.2 图编码

两条路线:

- GAT/message passing: 快,好实现
- Graph Transformer: 强,但更重

输出:

- `g_i` `[B, K, D_node]`

---

## 6. Language grounding + 决策解耦

### 6.1 Query embedding

从指令 token 得到 `q`:

- mean pool language token embeddings,或
- 引入 learnable `[Q]` token

### 6.2 Match + target distribution

- `score_i = (Wq q) dot (Wg g_i)`
- `p = softmax(score / tau)`

target embedding:

- soft: `e_target = sum_i p_i g_i`
- hard: gumbel-softmax straight-through

决策解耦:

- Stage A: 选目标(离散/软分布)
- Stage B: 连续控制(动作 chunk 生成)

---

## 7. 把 graph/target 注入 pi05: 必须改的核心源码点

### 7.1 最小侵入方案: prefix token 注入

pi05 的 prefix/suffix mask 已经能保证:

- suffix(action) 可以 attend prefix
- prefix 不会 attend suffix

所以只需要在 `embed_prefix` 里额外 concat 你的语义 token。

**关键约束:**

- 你注入的 token embedding 维度必须等于 pi05 的 `D_model` (PaliGemma hidden size)
- 因此需要 `semantic_token_proj: Linear(D_node, D_model)`

### 7.2 具体改法 (伪代码)

在 `S2PI05Pytorch` 里重写/扩展 `embed_prefix`:

```python
def embed_prefix(self, images, img_masks, tokens, masks, *, semantic_embs=None, semantic_pad_masks=None):
    # image path (same as pi05)
    img_embs = ...  # [B, N_img, D_model]

    # language path (same as pi05)
    lang_embs = ... # [B, N_lang, D_model]

    embs = [img_embs]
    pad_masks = [img_pad_masks]
    att_masks = [0] * N_img

    if semantic_embs is not None:
        sem = self.semantic_token_proj(semantic_embs)         # [B, K, D_model]
        embs.append(sem)
        pad_masks.append(semantic_pad_masks)                  # [B, K]
        att_masks += [0] * K

    # optional: add target token(s)
    # target = self.target_token_proj(e_target) -> [B, 1, D_model]

    embs.append(lang_embs)
    pad_masks.append(masks)
    att_masks += [0] * N_lang

    prefix_embs = cat(embs, dim=1)
    prefix_pad  = cat(pad_masks, dim=1)
    prefix_att  = tensor(att_masks)[None, :].expand(B, -1)
    return prefix_embs, prefix_pad, prefix_att
```

然后在 `forward(...)` / `sample_actions(...)` 调用 `embed_prefix(..., semantic_embs=..., semantic_pad_masks=...)`。

---

## 8. 训练目标: L_policy + 你的 3 个创新 loss

pi05 的 `L_policy` 保持不动(Flow Matching MSE)。

你需要在 `S2PI05Pytorch.forward` 里额外计算 aux losses,并在 `S2PI05Policy.forward` 汇总。

总损失:

```text
L_total = L_policy
        + lambda_align * L_text_align
        + lambda_cf    * L_counterfactual
        + lambda_temp  * L_temporal
        + lambda_gnd   * L_grounding (可选)
```

### 8.1 L_text_align (文字-物体对齐)

InfoNCE:

- 正样本: (object visual, its OCR)
- 负样本: batch 内其他 OCR

### 8.2 L_counterfactual (反事实文本扰动)

做两条 forward/grounding 分支:

- 原始: `p`
- 扰动: `p_cf`

扰动策略:

- distractor injection: 注入无关 OCR
- swap attribute: 替换关键属性词
- drop OCR: 删除部分 OCR

损失可选:

- invariant 扰动: `KL(p || p_cf)` 让目标选择不被干扰
- sensitive 扰动: 让目标随属性替换而变化(需要弱监督/规则)

### 8.3 L_temporal (时序一致性)

若 batch 提供连续窗口:

- `sum_t KL(p_t || p_{t-1})` 或 `||e_target_t - e_target_{t-1}||^2`

若使用 offline track 融合:

- 对齐 frame-level 与 track-level embedding

---

## 9. 推理系统: chunk 更新目标 + RTC 保动作连续

你的前端会增加延迟,所以建议:

- 每个 chunk 更新一次 target/graph
- 动作执行使用 RTC(ActionQueue + predict_action_chunk)

参考:

- `/root/songling/lerobotPro/docs/source/rtc.mdx`

---

## 10. NeurIPS 风格伪代码 (可直接写 Method)

### Algorithm 1: S2-VLA on pi05

```text
Input: frames {I_t}, robot state {s_t}, instruction L
Output: action sequence {a_t}

Initialize TrackMemory M

for t = 1..T do
  O_t = DetectObjects(I_t)
  T_t = TrackObjects(O_t, M)

  for each object i in T_t do
    z_vis_i  = VisualTokenPool(I_t, box_i)
    text_i, conf_i = OCR(I_t, box_i)
    z_text_i = TextEmbed(text_i)
    s_i = Cosine(ProjV(z_vis_i), ProjT(z_text_i))
    w_i = Sigmoid(alpha * conf_i + beta * s_i)
    z_obj_i = ProjV(z_vis_i) + w_i * ProjT(z_text_i)
  end for

  h_i = TemporalFuse(M[track_i], z_obj_i)
  Update M

  G = BuildGraph(nodes={h_i}, edges=SpatialRelations(+Interaction))
  g_i = GraphEncoder(G)

  q = QueryEmbed(L)
  p_i = Softmax(Match(q, g_i) / tau)
  e_target = Sum_i p_i * g_i

  prefix = Concat(ImageTokens(I_t), GraphTokens(g), TargetTokens(e_target), LanguageTokens(L))
  a_{t:t+H} = pi05_flow_matching(prefix)

  Execute a_t (or RTC chunk schedule)
end for
```

### Algorithm 2: Training With Counterfactual Perturbation

```text
Loss = L_policy
     + lambda_align * L_text_align
     + lambda_cf    * L_counterfactual
     + lambda_temp  * L_temporal

for each batch B do
  policy_losses, aux = ForwardS2PI05(B)

  B_cf = PerturbText(B)
  p,  e  = Grounding(B)
  p_cf, e_cf = Grounding(B_cf)

  L_cf   = KL(p || p_cf) (or task-dependent)
  L_temp = TemporalConsistency(...)
  L_align= TextAlign(...)

  L_total = mean(policy_losses) + w_align*L_align + w_cf*L_cf + w_temp*L_temp
  Backprop(L_total)
end for
```

---

## 11. 最小工程清单 (落地顺序)

1. 新增 `s2_pi05` policy/config/processor 并注册到 factory。
2. 实现 semantic frontend: detect/track -> OCR gate -> temporal fuse。
3. 实现 graph encoder + grounding head。
4. 修改 `embed_prefix` 支持 semantic/target token 注入(核心改动)。
5. 在 `forward` 中加入 aux losses 与日志。
6. 写一个示例脚本: chunk 更新目标 + RTC 执行。

---

## 12. Ablation 建议

- pi05 baseline
- + gating only
- + temporal fusion
- + graph encoder
- + decision decomposition
- + counterfactual training

指标:

- success rate
- distractor text 鲁棒性
- target selection accuracy(若有标签)
- chunk continuity(有无 RTC)
