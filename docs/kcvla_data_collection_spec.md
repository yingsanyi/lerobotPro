# KC-VLA 数据采集与标注规范

本文档对应 `policy.type=kcvla` 的轻量关键词 grounding 路线。当前版本是弱监督方案，不依赖任何 `bbox` 或显式定位标注。

它回答三个问题：

1. 相比原始 `pi05`，你需要额外采什么字段。
2. 这些字段怎样标，才能和当前 `kcvla` 实现严格对齐。
3. 在无显式定位监督下，怎样构造有效的关键词干预数据。

相关实现：

- [processor_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/processor_kcvla.py)
- [modeling_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/modeling_kcvla.py)

---

## 1. 基础字段

这些字段与 `pi05` 完全一致，必须保留：

- `observation.images.<camera_name>`
- `observation.state`
- `action`
- `task`

也就是说，`kcvla` 的基座仍然是标准的 `pi05` 行为克隆训练链路。

---

## 2. 新增字段

### 2.1 必需字段：`keyword_text`

这是 KC-VLA 的外部控制变量。

推荐格式：

```text
止痛药, 维生素
```

约束：

- 关键词顺序要稳定。
- 推荐用逗号分隔的字符串；processor 会自动拆成关键词集合。
- 关键词应该对应当前帧里真实可区分的语义目标，而不是抽象任务标签。

### 2.2 可选但强烈推荐：`counterfactual_keyword_text`

这是反事实训练分支使用的字段。它与原图共享同一帧，只改变关键词条件。

例如：

```text
keyword_text = "止痛药, 维生素"
counterfactual_keyword_text = "维生素, 止痛药"
```

或者：

```text
keyword_text = "止痛药"
counterfactual_keyword_text = "维生素"
```

当前实现的优先级是：

1. 如果人工提供了非空 `counterfactual_keyword_text`，直接使用人工值。
2. 如果该字段缺失或为空，并且 `policy.counterfactual_auto_generate=true`，processor 会按同帧关键词置换自动生成。
3. 如果 `policy.counterfactual_enabled=false`，则整个反事实关键词分支关闭，不会生成 `observation.language.keyword_cf.*`。

也就是说，你可以混合使用两种来源：

- 难样本人工指定反事实关键词
- 普通样本让 processor 自动补齐

---

## 3. 弱监督数据构造规则

### 3.1 反事实关键词必须对应可执行的真实替代目标

不要只做字符串扰动。更好的做法是保证反事实关键词在当前帧里也存在对应视觉实体，否则干预约束会退化成噪声。

推荐两种采法：

- 同一帧同时出现多个可混淆对象，只切换关键词。
- 同一任务模板里交换关键词顺序或目标对象。

### 3.2 任务必须真的依赖语义识别

如果任务只靠位置就能完成，比如“拿左边那个盒子”，那么关键词分支很容易被主干忽略。

更适合的任务类型是：

- 外观近似但文字不同
- 相同容器但标签不同
- 同场景多目标，需要按关键词决定抓哪个

### 3.3 关键词集合尽量覆盖同帧竞争目标

当前实现会对不同关键词的 attention map 施加 contrast 约束，所以同一帧里有多个相近目标时，训练信号会更强。

### 3.4 自动生成 `counterfactual_keyword_text`

运行时 processor 的自动生成规则很保守，只做同帧关键词置换，例如：

- `"止痛药, 维生素" -> "维生素, 止痛药"`

这种方式最安全，因为反事实目标仍然来自同一帧。

如果你想给单关键词样本也补上更强的反事实数据，建议再配合离线脚本批量生成：

```shell
lerobot-generate-counterfactual-keywords \
    --root /path/to/your_dataset
```

离线脚本的优先级和 processor 保持一致：

- 已有非空 `counterfactual_keyword_text` 默认保留不覆盖
- 缺失样本自动生成

离线脚本的扩展规则：

- 多关键词样本优先做同帧关键词置换，例如 `"止痛药, 维生素" -> "维生素, 止痛药"`
- 单关键词样本优先从同任务关键词池里找替代词
- 同任务找不到时回退到全局关键词池
- 已有非空 `counterfactual_keyword_text` 默认保留不覆盖

如果你有更严格的替换规则，可以传一个 JSON 映射文件：

```json
{
  "止痛药": ["维生素", "感冒药"],
  "维生素": ["止痛药"],
  "饼干": ["面包"]
}
```

命令示例：

```shell
lerobot-generate-counterfactual-keywords \
    --root /path/to/your_dataset \
    --mapping-path /path/to/keyword_map.json \
    --overwrite-existing
```


例如：

- `keyword_text = "止痛药, 维生素"`
- `counterfactual_keyword_text = "维生素, 止痛药"`

这种样本同时支持：

- 行为级反事实差异
- 跨关键词 attention 解耦
- attention 稀疏化

---

## 4. 推荐最小采集模板

每帧至少：

- 图像
- 状态
- 动作
- `task`
- `keyword_text`

如果你要启用完整弱监督训练，再补：

- `counterfactual_keyword_text`

或者保持该字段为空，同时开启：

- `policy.counterfactual_enabled=true`
- `policy.counterfactual_auto_generate=true`

这样就能覆盖当前代码里的：

- `L_policy`
- `L_counterfactual`
- `L_contrast`
- `L_sparse`

---

## 5. 不再需要的字段

当前弱监督版本不再消费以下字段：

- `observation.keyword.boxes_xyxy`
- `observation.keyword.camera_ids`
- 任何 patch-level grounding target

如果数据集里已经有这些字段，可以保留作离线分析或可视化，但它们不会进入当前 `kcvla` 的训练损失。
