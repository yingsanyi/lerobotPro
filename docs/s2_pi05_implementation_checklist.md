# S2-PI05 Implementation Checklist (File-by-File)

这份清单对应当前已经生成的 `s2_pi05` 骨架代码,按文件拆分到“类/函数级别”说明下一步该补什么。

---

## 1. `/root/songling/lerobotPro/src/lerobot/utils/constants.py`

### 新增常量

- `OBS_SEMANTIC`
- `OBS_SEMANTIC_NODE_EMBS`
- `OBS_SEMANTIC_NODE_MASK`
- `OBS_SEMANTIC_EDGE_INDEX`
- `OBS_SEMANTIC_EDGE_ATTR`
- `OBS_SEMANTIC_BOXES_XYXY`
- `OBS_SEMANTIC_TRACK_IDS`
- `OBS_SEMANTIC_OCR_CONF`

### 你后续要做的事

- 确定是否还需要 `OBS_SEMANTIC_OCR_TEXT` 这类只用于 debug 的 key。
- 若要离线缓存 detector/track 结果,建议再补 `OBS_SEMANTIC_LABELS` 与 `OBS_SEMANTIC_SCORES`。

---

## 2. `/root/songling/lerobotPro/src/lerobot/policies/__init__.py`

### 已做

- 导出 `S2PI05Config`

### 你后续要做的事

- 如果后续还会暴露 `S2PI05Policy`,可以考虑在 package level 也导出,但当前不是必须。

---

## 3. `/root/songling/lerobotPro/src/lerobot/policies/factory.py`

### 已做

- 导入 `S2PI05Config`
- 在 `get_policy_class` 中新增 `s2_pi05`
- 在 `make_policy_config` 中新增 `s2_pi05`
- 在 `make_pre_post_processors` 中优先处理 `S2PI05Config`

### 你后续要做的事

- 如果以后把 `s2_pi05` 做成插件式 policy,这里可以删掉显式分支,完全交给 registry fallback。
- 如果要支持 async-inference registry,记得同步更新相应模块。

---

## 4. `/root/songling/lerobotPro/src/lerobot/policies/s2_pi05/configuration_s2_pi05.py`

### 类: `S2PI05Config`

#### 已有字段

- semantic frontend 开关与维度
- gating alpha/beta
- temporal fusion 模式
- graph encoder 模式与层数
- grounding 模式
- aux loss 权重

#### 下一步要补的函数/逻辑

- `__post_init__`
  - 增加对 `semantic_graph_edges` 的合法性检查
  - 如果 `grounding_mode == "hard_gumbel"`,可额外校验 temperature 区间
- `validate_features`
  - 当前沿用 `PI05Config.validate_features`
  - 若你后续希望 semantic tensors 出现在 `input_features` 元信息里,需要决定它们映射到哪个 `FeatureType`

---

## 5. `/root/songling/lerobotPro/src/lerobot/policies/s2_pi05/processor_s2_pi05.py`

### 类: `S2Pi05SemanticInputsProcessorStep`

#### 已做

- 检查 `observation.semantic.node_embs`
- 自动补 `node_mask`
- 过滤掉非 tensor 的 edge 字段

#### 下一步要补的函数/逻辑

- `__call__`
  - 从离线缓存字段加载 detector/track/OCR 结果
  - 支持从 `complementary_data` 搬运 semantic fields 到 `observation`
  - 支持从 bbox/track/OCR 原始字段现算 node embeddings
  - 如果要做 online detector,这里可以调用轻量前端,但注意 DataLoader 性能
- `transform_features`
  - 若你最终决定把 semantic tensors 作为正式 policy feature,这里要补 feature schema

### 函数: `make_s2_pi05_pre_post_processors`

#### 已做

- 继承 pi05 流水线
- 在 tokenizer 之后插入 semantic input step

#### 下一步要补的函数/逻辑

- 如果 semantic frontend 需要访问未 tokenized 的自然语言,可在 tokenizer 之前增加一个解析 step
- 如果 counterfactual augmentation 想在 processor 层做,可以新增一个 `S2Pi05CounterfactualProcessorStep`

---

## 6. `/root/songling/lerobotPro/src/lerobot/policies/s2_pi05/modeling_s2_pi05.py`

### 数据结构: `SemanticFrontendOutput`

#### 已做

- 承载 node/edge/aux loss

#### 下一步要补的事

- 增加 `boxes_xyxy`, `track_ids`, `ocr_conf`, `ocr_text_tokens` 等字段,方便调试与损失计算

### 数据结构: `SemanticConditioningOutput`

#### 已做

- 承载 prefix token、pad mask、target_probs、aux losses

#### 下一步要补的事

- 如果需要多目标 grounding,增加 `target_indices` 或 `target_logits`

### 类: `S2SemanticFrontend`

#### 已做

- 支持读取预计算 `node_embs`
- 用 `LazyLinear` 对输入维度做 lazy 投影
- 缺省时回落为空语义输入

#### 下一步要补的函数/逻辑

- `forward`
  - 在线 detector/tracker 接入
  - OCR 模块接入
  - object crop / ROI pooling 提取 `z_vis_i`
  - 文本 embedding 提取 `z_text_i`
  - gating: `w_i = sigmoid(alpha * conf_i + beta * s_i)`
  - temporal fusion: EMA/GRU/attention
  - graph 构建前的 node 特征输出
- `_zero_aux_losses`
  - 目前全 0
  - 后续应返回真实 `L_text_align` / `L_counterfactual` / `L_temporal`
- `reset_state`
  - 当前为空
  - 后续要清空 TrackMemory / OCR cache / online tracker state

### 类: `S2GraphEncoder`

#### 已做

- 一个不依赖 edge 的 residual MLP 编码器

#### 下一步要补的函数/逻辑

- `forward`
  - 使用 `edge_index` / `edge_attr`
  - GAT message passing 或 graph transformer attention bias
  - 多相机节点间关系编码
  - interaction edge 支持(contact, in_gripper, on_top_of 等)

### 类: `S2GroundingHead`

#### 已做

- query-node dot product pointer
- masked softmax
- 返回 target embedding 与 probs

#### 下一步要补的函数/逻辑

- `forward`
  - 支持 `hard_gumbel`
  - 支持 top-k target selection
  - 支持 grounding supervision loss 输出

### 类: `S2PI05Pytorch`

#### 已做

- `semantic_frontend`
- `graph_encoder`
- `grounding_head`
- `semantic_token_proj`
- `target_token_proj`
- `build_semantic_conditioning`
- `embed_prefix` 的 semantic token 注入
- `forward` 与 `sample_actions` 接受 `semantic_batch`

#### 下一步要补的函数/逻辑

- `build_semantic_conditioning`
  - 真正把 graph encoder 与 grounding loss 对接起来
  - 若做 query parser,在这里把 instruction 转 structured query
  - 若做多目标/子任务分解,这里输出多个 target token
- `embed_prefix`
  - 决定 token 顺序是否固定为 `[image, graph, target, language]`
  - 若要区分 graph token 与 target token,可给不同的 learnable type embedding
- `forward`
  - 接入真实 aux loss
  - 接入 counterfactual branch
  - 若要 temporal loss,支持连续时间窗口输入
- `sample_actions`
  - 若 target 不是每步更新,可缓存上一 chunk 的 target embedding
  - 如果要与 RTC 更紧密配合,在这里加入 target refresh 逻辑
- `reset_semantic_state`
  - 当前只调用 frontend reset
  - 后续可清空 target cache / graph cache

### 类: `S2PI05Policy`

#### 已做

- 继承 pi05 policy 接口
- 默认 `strict=False` 加载兼容权重
- `from_pretrained` 会自动把 `PI05Config` 升级为 `S2PI05Config`
- 可以直接用 `S2PI05Policy.from_pretrained("lerobot/pi05_base")` 继承原 `pi05` 骨干权重
- 新增的 semantic / graph / grounding 模块在加载 `pi05` checkpoint 时保持随机初始化
- 抽取 `semantic_batch`
- 在 `forward` 中汇总 aux loss

#### 下一步要补的函数/逻辑

- `from_pretrained`
  - 如果以后同时支持“纯 VLM checkpoint / pi05 policy checkpoint / s2_pi05 checkpoint”,建议显式加 `init_mode` 或日志分支
  - 当前只把 `pi05` 视为结构兼容初始化源; 不相关的 base VLM checkpoint 仍需要单独 remap 逻辑
- `reset`
  - 若 online detector/tracker 有状态,这里统一 reset
- `_extract_semantic_batch`
  - 如果 semantic 输入不只 observation 下的 tensor,这里继续扩展
- `forward`
  - 若 RA-BC 等需要 per-sample aux loss,要把 aux loss 从标量改为 per-sample tensor
- `predict_action_chunk`
  - 如果要支持 chunk 间 target cache 或 chunk 级 graph 更新频率控制,这里是入口

---

## 7. `/root/songling/lerobotPro/docs/s2_vla_pi05.md`

### 你后续要做的事

- 把当前“设计文档”同步到代码实现状态
- 每完成一项,回写到文档里,避免设计和实现分叉

---

## 8. 新增建议文件 (下一步可选)

### 推荐新增

- `examples/tutorial/s2_pi05/using_s2_pi05_example.py`
  - 演示如何给 batch 塞 `observation.semantic.node_embs`
- `tests/policies/s2_pi05/test_s2_pi05_smoke.py`
  - 测 import / config / processor / forward fallback 到 vanilla pi05
- `tests/policies/s2_pi05/test_s2_pi05_prefix.py`
  - 测 semantic tokens 是否真的进入 prefix 长度

---

## 9. 推荐开发顺序

1. 先让 `s2_pi05` 在 **没有 semantic inputs** 时完全等价于 pi05。
2. 再接入 **预计算 node embeddings**，验证 prefix 注入链路通了。
3. 再接 graph encoder + grounding head。
4. 然后接 OCR gating。
5. 最后接 temporal fusion 和 counterfactual training。
