# S2-PI05 数据采集字段清单与标注规范

本文档面向 `s2_pi05` 的数据采集、离线标注与训练前检查。

它回答三个核心问题：

1. 相比原始 `pi05`，你需要额外采什么数据。
2. 这些字段应该如何标注，才能和当前 `s2_pi05` 实现对齐。
3. 对于一个具体任务，完整的采集和模型处理链路应该是什么样子。

---

## 1. 和原始 `pi05` 相比，你的数据需要增加什么

原始 `pi05` 的核心是：

- 图像
- 语言指令
- 机器人状态
- 专家动作

也就是学习：

- `image + language + state -> action`

`s2_pi05` 在这条链路上额外引入了：

- 对象级表示
- OCR 文本及其置信度
- 时序轨迹聚合
- 对象图结构
- 显式目标 grounding

也就是它希望学习：

- `image + language + state + object/text/graph/target -> action`

所以相比 `pi05`，你的数据不能只保证“能做动作”，还要保证：

- 任务必须真的依赖对象语义区分才能完成。
- 相同外观但不同文字的对象要频繁出现。
- 时间上要连续，方便 temporal fusion 学到稳定目标。
- 对象顺序和 `target_index` 要稳定，不然 grounding supervision 会错位。

---

## 2. 数据采集总原则

### 2.1 先保留 `pi05` 的采集规范

这些字段仍然是必需的：

- `observation.images.*`
- `observation.state`
- `action`
- `task`

并且以下约束不要变：

- 状态定义不要频繁改。
- 动作维度和物理含义不要改。
- 相机安装位姿尽量稳定，或至少记录配置版本。
- 图像、状态、动作必须严格时间对齐。
- 仍然需要 `STATE/ACTION` 的 quantiles 统计，和 `pi05` 一致。

### 2.2 新模型最关键的额外采集原则

- 要多采“看起来很像，但文字不同”的对象。
- 要多采“文字部分被遮挡、倾斜、反光、模糊”的真实情况。
- 要保证连续片段完整，不要只留下稀疏关键帧。
- 要保证任务里真的包含语义约束，而不是仅靠位置就能完成。
- 对于多步任务，要标注“当前控制阶段的目标对象”，而不是只给全局任务一句话。

---

## 3. 字段清单

下面按三层来列：

- `A` 类：必须保留的 `pi05` 基础字段
- `B` 类：强烈推荐为 `s2_pi05` 采集/离线生成的字段
- `C` 类：可选增强字段

### 3.1 A 类字段：基础必需字段

| 字段名 | 是否必需 | 类型/形状 | 说明 |
| --- | --- | --- | --- |
| `observation.images.<camera_name>` | 必需 | `float32`, `[B, C, H, W]` | 原始相机图像，训练时仍是主视觉输入 |
| `observation.state` | 必需 | `float32`, `[B, D_state]` | 机器人状态，沿用 `pi05` |
| `action` | 必需 | `float32`, `[B, T, D_action]` 或 `[B, D_action]` | 专家动作 |
| `task` | 必需 | `list[str]` | 全局任务文本 |

### 3.2 B 类字段：`s2_pi05` 强烈推荐字段

这些字段已经和当前实现对齐：

- 常量定义见 [constants.py](/root/songling/lerobotPro/src/lerobot/utils/constants.py)
- 预处理见 [processor_s2_pi05.py](/root/songling/lerobotPro/src/lerobot/policies/s2_pi05/processor_s2_pi05.py)
- 模型消费路径见 [modeling_s2_pi05.py](/root/songling/lerobotPro/src/lerobot/policies/s2_pi05/modeling_s2_pi05.py)

| 字段名 | 是否必需 | 类型/形状 | 来源 | 用途 |
| --- | --- | --- | --- | --- |
| `observation.semantic.boxes_xyxy` | 强烈推荐 | `float32`, `[B, K, 4]` | 检测器/人工标注 | 对象框，供 crop、几何关系、graph 使用 |
| `observation.semantic.ocr_text` | 强烈推荐 | `list[list[str]]` | OCR 或人工修订 | 对象文字内容 |
| `observation.semantic.ocr_conf` | 强烈推荐 | `float32`, `[B, K]` | OCR 引擎 | 文字置信度门控 |
| `observation.semantic.target_index` | 强烈推荐 | `int64`, `[B]` | 人工/规则标注 | 当前时刻 supervision target |
| `observation.semantic.track_ids` | 推荐 | `int64`, `[B, K]` | tracker | 时序聚合与 online memory |
| `observation.semantic.visual_embs` | 可替代 `boxes` | `float32`, `[B, K, D]` | 离线视觉前端 | 直接作为对象视觉 embedding |
| `observation.semantic.node_embs` | 可替代前端 | `float32`, `[B, K, D]` | 离线语义前端 | 直接作为节点输入 |

### 3.3 C 类字段：可选增强字段

| 字段名 | 是否必需 | 类型/形状 | 用途 |
| --- | --- | --- | --- |
| `observation.semantic.edge_index` | 可选 | `int64`, `[B, 2, E]` | 显式图边 |
| `observation.semantic.edge_attr` | 可选 | `float32`, `[B, E, D_edge]` | 边属性 |
| `observation.semantic.camera_ids` | 可选 | `int64`, `[B, K]` | 多相机对象来源 |
| `observation.semantic.cf_ocr_text` | 可选 | `list[list[str]]` | 反事实文字分支 |
| `observation.semantic.prev_node_embs` | 通常不作为离线数据 | `float32`, `[B, N_hist, K, D]` | 多用于 online / 调试 |

---

## 4. 标注规范

## 4.1 图像与时间同步

### 规范

- 每一个 `action[t]` 都必须能对应到同一时刻或固定延迟规则下的图像与状态。
- 如果你有多相机，所有相机帧要尽量硬同步，至少误差固定且可校正。
- 不要在 episode 中间随机丢帧，否则 temporal fusion 会学坏。

### 建议

- 保留完整连续轨迹，不要只留起点和终点。
- 每个 episode 都保留统一采样频率。
- 若设备层存在不同步，务必在离线处理阶段做重采样或对齐。

## 4.2 `boxes_xyxy` 标注规范

### 当前实现最重要的约束

当前 `s2_pi05` 的对象 crop 逻辑是：

- 先走 `pi05` 的图像预处理，把图像 resize + pad 到 `config.image_resolution`
- 再用 `boxes_xyxy` 在这张预处理后的图上做 crop

因此你要特别注意：

- `observation.semantic.boxes_xyxy` 最好直接使用“经过同样 resize + pad 后”的坐标系
- 并且归一化到 `[0, 1]`

### 强烈建议

训练输入使用的 `boxes_xyxy` 应满足：

- 格式：`[x1, y1, x2, y2]`
- 归一化范围：`[0, 1]`
- 坐标基准：和进入 policy 的图像一致
- 无效框：宽高为 0，或直接通过 `node_mask` 标成无效

### 不推荐做法

- 直接把原图归一化框喂给当前实现，而不做 resize+pad 坐标变换

这会导致：

- crop 错位
- OCR text 与视觉区域不对应
- graph 几何关系偏移

## 4.3 `ocr_text` 标注规范

### 规范

- 每个对象对应一个字符串。
- 看不清时用空字符串 `""`，不要乱猜。
- 不同帧同一轨迹的文字尽量统一。
- 如果 OCR 引擎经常错，把人工修订版作为训练字段更稳。

### 建议格式

- 中文直接保留中文文本
- 去除多余换行和控制字符
- 保留和任务相关的关键词，不需要过度清洗成“类别名”

例如：

- `"止痛药"`
- `"维生素C"`
- `"cold medicine"`
- `""`（不可读）

## 4.4 `ocr_conf` 标注规范

### 规范

- 范围建议归一化到 `[0, 1]`
- OCR 引擎没有置信度时，可人工规则映射
- 没有 OCR 时不要随便填高分

### 简单规则示例

- 清晰正视图：`0.9 ~ 1.0`
- 部分遮挡/轻微模糊：`0.5 ~ 0.8`
- 基本不可读：`0.0 ~ 0.2`

## 4.5 `track_ids` 标注规范

### 规范

- 同一 episode 内，同一物体在连续帧中必须保持同一个 `track_id`
- 不同 episode 间不需要全局唯一
- 没有可靠 tracking 时可以不提供，但会损失 temporal fusion 效果

### 建议

- 优先按真实物体轨迹排序，而不是按检测器置信度排序
- 如果对象列表顺序会变化，必须有 `track_id` 来稳定 supervision

## 4.6 `target_index` 标注规范

这是和原始 `pi05` 最大的新增监督之一。

### 含义

`target_index` 表示：

- 在当前这一帧或这一 transition 上，**当前控制阶段最应该被显式选中的对象节点索引**

它不是“全局任务唯一目标”。

对于多步任务，它应该随阶段变化。

### 关键要求

- `target_index` 必须指向当前对象列表中的稳定顺序
- 对象顺序必须有固定规则

### 推荐对象排序规则

按优先级推荐：

1. 按 `track_id` 升序排序
2. 如果没有 `track_id`，按 `x_center` 从左到右，再按 `y_center` 从上到下
3. 不要直接使用 detector 原始输出顺序

### 标注原则

- 抓取阶段：标操作对象
- 放置阶段：标目标容器/目标位置对象
- 搜索/靠近阶段：标当前最需要对齐的对象

这套规则最符合你现在“离散目标选择 + 连续动作控制”的解耦设计。

## 4.7 多目标/复合任务的标注原则

对于一句话中有多个子目标的任务，例如：

- “将止痛药放在左边的盒子，将维生素放在右边的盒子”

不要只打一条静态全局标签。

应该这样做：

- `task` 保持完整全局指令不变
- `target_index` 随时间切换，表示当前阶段的目标

这样模型才能学到：

- 全局任务不变
- 当前控制目标在变
- 图结构和语言约束共同决定当前应该操作谁

---

## 5. 推荐的数据组织方式

## 5.1 最实用的工程方案

### 在线采集时保存

- 原始图像
- 状态
- 动作
- 任务文本
- episode 元信息

### 离线处理时生成

- `boxes_xyxy`
- `ocr_text`
- `ocr_conf`
- `track_ids`
- `target_index`
- 可选 `edge_index/edge_attr`
- 可选 `cf_ocr_text`

这是最稳的，因为：

- detector / OCR / tracker 经常迭代
- 你可以反复重跑前端，而不用重新录机器人数据

## 5.2 训练前检查清单

每个样本至少检查：

- 图像、状态、动作是否齐全
- `boxes_xyxy` 和图像坐标系是否一致
- `ocr_text` 与 box 是否一一对应
- `ocr_conf` 长度是否与对象数一致
- `target_index` 是否落在有效对象范围内
- 复合任务中 `target_index` 是否随阶段正确切换

---

## 6. 具体示例

下面用任务：

- `将止痛药放在左边的盒子，将维生素放在右边的盒子`

来说明如何采集和模型如何处理输入。

## 6.1 场景设计

场景中至少放这些物体：

- 一盒止痛药
- 一瓶维生素
- 一个左侧盒子
- 一个右侧盒子
- 至少一个干扰物，例如感冒药

### 为什么要有干扰物

如果场景里只有止痛药和维生素，而且外观差别很大，模型可能根本不需要 OCR 就能完成任务。

你这个模型的价值在于：

- 让模型学会利用文字和结构化对象关系，而不是只看颜色和位置 shortcut

所以建议：

- 止痛药和感冒药包装外观相近，但文字不同
- 左右两个盒子外观一致，仅靠空间关系区分

## 6.2 采集时怎么录

### 每个 episode 采什么

每个时刻都采：

- `observation.images.base_0_rgb`
- 可选更多视角图像
- `observation.state`
- `action`
- `task = "将止痛药放在左边的盒子，将维生素放在右边的盒子"`

### 采集动作时的建议

- 录完整连续演示，不要只录抓取瞬间
- 包含四个自然阶段：
  - 接近并抓取止痛药
  - 移动并放入左边盒子
  - 接近并抓取维生素
  - 移动并放入右边盒子

### 场景多样化建议

- 止痛药和维生素初始位置互换
- 左右盒子的位置有轻微扰动
- 药盒朝向变化，文字有时正向有时斜放
- 加入遮挡和手部遮挡
- 不同光照和反光条件

## 6.3 一条 episode 的实际采集流程

建议按下面顺序执行：

### 步骤 1：布置场景

- 把止痛药、维生素、感冒药摆在桌面不同位置
- 把两个盒子放在桌面左侧和右侧，外观尽量相同
- 检查主要相机里这几个物体都能看到，至少文字区域大部分可见

### 步骤 2：开始录制

开始录制后连续保存：

- 全部相机图像
- 机器人状态
- 专家动作
- 全局任务文本
- episode id、时间戳、相机配置版本

### 步骤 3：人类示教完整完成任务

操作员连续完成：

1. 靠近止痛药
2. 抓起止痛药
3. 移动到左边盒子
4. 放下止痛药
5. 靠近维生素
6. 抓起维生素
7. 移动到右边盒子
8. 放下维生素

中间不要暂停录制，也不要只截关键帧。

### 步骤 4：离线跑检测、OCR、tracking

对每一帧或每个采样点生成：

- `boxes_xyxy`
- `ocr_text`
- `ocr_conf`
- `track_ids`

### 步骤 5：离线标 `target_index`

按当前控制阶段，为每个 transition 打上当前目标对象索引。

### 步骤 6：训练前质检

至少抽检下面几项：

- 框是否落在正确物体上
- OCR 文本是否和框对应
- `track_id` 是否跨帧稳定
- `target_index` 是否真的对应当前阶段目标
- 左右盒子是否在坐标和语义上没有标反

---

## 7. 这个任务应该怎么标注

假设某一帧经过检测和排序后，对象列表如下：

| 索引 | 对象 | OCR 文本 | OCR 置信度 | 备注 |
| --- | --- | --- | --- | --- |
| `0` | 止痛药 | `止痛药` | `0.97` | 目标药品 |
| `1` | 维生素 | `维生素` | `0.95` | 第二个目标药品 |
| `2` | 左边盒子 | `""` | `0.00` | 放置容器 |
| `3` | 右边盒子 | `""` | `0.00` | 放置容器 |
| `4` | 感冒药 | `感冒药` | `0.92` | 干扰物 |

对应语义字段可写成：

```python
observation.semantic.boxes_xyxy = [
    [0.10, 0.20, 0.24, 0.42],  # 止痛药
    [0.29, 0.18, 0.42, 0.40],  # 维生素
    [0.05, 0.62, 0.35, 0.95],  # 左边盒子
    [0.62, 0.61, 0.93, 0.95],  # 右边盒子
    [0.46, 0.21, 0.58, 0.43],  # 感冒药
]

observation.semantic.ocr_text = [[
    "止痛药",
    "维生素",
    "",
    "",
    "感冒药",
]]

observation.semantic.ocr_conf = [[0.97, 0.95, 0.0, 0.0, 0.92]]
observation.semantic.track_ids = [[11, 12, 21, 22, 13]]
```

## 7.1 `target_index` 如何随阶段变化

### 阶段 1：接近并抓取止痛药

此时建议：

- `target_index = 0`

原因：

- 当前最该被显式选中的对象是“止痛药”本身

### 阶段 2：把止痛药放入左边盒子

此时建议：

- `target_index = 2`

原因：

- 已经抓住止痛药后，当前控制目标变成“左边盒子”这个放置容器

### 阶段 3：接近并抓取维生素

此时建议：

- `target_index = 1`

### 阶段 4：把维生素放入右边盒子

此时建议：

- `target_index = 3`

所以同一个 episode 中，`target_index` 不是一个固定值，而是按阶段变化：

```text
止痛药抓取阶段: 0
止痛药放置阶段: 2
维生素抓取阶段: 1
维生素放置阶段: 3
```

这正是你模型里“decision decomposition”的监督来源。

## 7.2 一个更具体的时间线示例

下面给一个示意性的 episode 时间线。帧号只是示例，核心是阶段切换逻辑。

| 时间段 | 机器人行为 | 当前主要对象列表顺序 | `target_index` | 说明 |
| --- | --- | --- | --- | --- |
| `t=0~20` | 观察并靠近止痛药 | `[止痛药, 维生素, 左盒子, 右盒子, 感冒药]` | `0` | 先锁定要抓的药品 |
| `t=21~40` | 抓起止痛药 | `[止痛药, 维生素, 左盒子, 右盒子, 感冒药]` | `0` | 仍然以被操纵物体为目标 |
| `t=41~70` | 携带止痛药移动到左盒子 | `[止痛药, 维生素, 左盒子, 右盒子, 感冒药]` | `2` | 控制重点切到放置容器 |
| `t=71~85` | 放下止痛药 | `[止痛药, 维生素, 左盒子, 右盒子, 感冒药]` | `2` | 直到放置稳定完成 |
| `t=86~110` | 靠近并抓取维生素 | `[止痛药, 维生素, 左盒子, 右盒子, 感冒药]` | `1` | 第二阶段抓取目标 |
| `t=111~145` | 携带维生素移动到右盒子 | `[止痛药, 维生素, 左盒子, 右盒子, 感冒药]` | `3` | 放置阶段切到右盒子 |
| `t=146~160` | 放下维生素 | `[止痛药, 维生素, 左盒子, 右盒子, 感冒药]` | `3` | 直到放置完成 |

如果某些帧里 detector 顺序变化，但 `track_id` 稳定，那么应先按固定规则重排对象列表，再写入 `target_index`。不要直接沿用 detector 原始顺序。

### 一个 transition 样本示例

以“把止痛药移动到左边盒子”的中间一帧为例，这一条训练样本可以抽象成：

```python
sample = {
    "task": "将止痛药放在左边的盒子，将维生素放在右边的盒子",
    "observation.images.base_0_rgb": image_t,
    "observation.state": state_t,
    "action": action_t,
    "observation.semantic.boxes_xyxy": boxes_t,
    "observation.semantic.ocr_text": [["止痛药", "维生素", "", "", "感冒药"]],
    "observation.semantic.ocr_conf": [[0.97, 0.95, 0.0, 0.0, 0.92]],
    "observation.semantic.track_ids": [[11, 12, 21, 22, 13]],
    "observation.semantic.target_index": [2],
}
```

这里 `target_index = 2` 的原因不是“全局任务只关心左盒子”，而是“当前这一时刻的控制目标是左盒子”。

---

## 8. 模型如何处理这类输入

下面按当前代码路径描述。

## 8.1 预处理阶段

相关实现：

- [processor_s2_pi05.py](/root/songling/lerobotPro/src/lerobot/policies/s2_pi05/processor_s2_pi05.py)
- [processor_pi05.py](/root/songling/lerobotPro/src/lerobot/policies/pi05/processor_pi05.py)

### 步骤 1：状态和动作处理

- `observation.state` 先按 `pi05` 规则归一化
- state 会被离散成 256 bins
- 全局任务文本会被拼成类似：

```text
Task: 将止痛药放在左边的盒子，将维生素放在右边的盒子, State: ...;
Action:
```

### 步骤 2：任务文本 tokenization

- 全局任务通过 PaliGemma tokenizer 变成语言 tokens

### 步骤 3：OCR 文本 tokenization

- `ocr_text` 中每个对象的文本被单独 tokenizer
- 形成：
  - `observation.semantic.text.tokens`
  - `observation.semantic.text.attention_mask`

### 步骤 4：语义输入整理

- `boxes_xyxy`、`ocr_conf`、`track_ids`、`target_index` 被 batch 化
- `node_mask` 会自动推断或显式提供

---

## 8.2 语义前端阶段

相关实现：

- [modeling_s2_pi05.py](/root/songling/lerobotPro/src/lerobot/policies/s2_pi05/modeling_s2_pi05.py)

### 步骤 1：对象视觉表征

如果你提供的是 `boxes_xyxy`，模型会：

- 在 policy 预处理后的图像上按 box 做 crop
- 送入与 `pi05` 相同的视觉 backbone
- 得到每个对象的视觉 embedding `z_vis_i`

### 步骤 2：对象文字表征

模型把 `ocr_text` 对应的 token embedding 做池化，得到 `z_text_i`

例如：

- 止痛药节点得到 `z_text(止痛药)`
- 维生素节点得到 `z_text(维生素)`
- 盒子节点因为没有文字，文本分支接近空

### 步骤 3：OCR confidence gating

模型计算：

- 视觉和文字的一致性 `s_i`
- OCR 置信度 `conf_i`
- 门控权重 `w_i = sigmoid(alpha * conf_i + beta * s_i)`

因此在这个例子里：

- 止痛药节点：高 `conf` + 高图文一致性，文字会强烈注入节点表征
- 感冒药节点：虽然也有高文字置信度，但语言任务与它不匹配，后续 grounding 会压低它
- 左右盒子节点：没有 OCR 文本，主要依靠视觉与几何关系表征

### 步骤 4：temporal fusion

如果 `track_ids` 存在，模型会把连续帧中同一个对象的历史节点做融合：

- 解决 OCR 抖动
- 解决短时遮挡
- 让“止痛药”这个节点在多帧中更稳定

这也是为什么采集时必须保留连续帧和稳定轨迹。

---

## 8.3 Graph 与 Grounding 阶段

### 图结构

模型会根据 `boxes_xyxy` 自动形成几何关系，例如：

- 止痛药在左盒子上方
- 右盒子在左盒子右侧
- 感冒药位于中间区域

如果你还提供 `edge_index/edge_attr`，就会用显式边。

### Grounding

模型读取完整任务：

- `将止痛药放在左边的盒子，将维生素放在右边的盒子`

并结合当前语义节点，输出当前对象选择概率 `target_probs`。

在不同阶段，它应当学会：

- 阶段 1：把概率集中到“止痛药”节点
- 阶段 2：把概率集中到“左边盒子”节点
- 阶段 3：把概率集中到“维生素”节点
- 阶段 4：把概率集中到“右边盒子”节点

训练时，这由 `target_index` 监督。

---

## 8.4 Prefix 注入与动作生成

当前 `s2_pi05` 不是把对象图全部序列化成 prompt 文本，而是把语义 embedding 直接注入 prefix。

可以理解成：

- 图像 token
- 语义节点 token
- 当前 target token
- 任务语言 token

共同组成 prefix

然后交给原始 `pi05` 的 action backbone 去生成动作 chunk。

也就是说：

- 原来的 `pi05` backbone 仍负责连续控制
- 你新增的模块负责让“该关注哪个对象、该依据什么文字和关系”变得显式可控

---

## 9. 对这个任务最容易出问题的地方

### 9.1 只标药品，不标盒子

如果你只在全程都把 `target_index` 标成药品：

- 抓取阶段可能没问题
- 放置阶段模型会缺少明确的容器目标监督

### 9.2 对象顺序不稳定

如果一帧里止痛药是索引 0，下一帧因为检测器排序变化变成索引 3，而你没有稳定 `track_id` 和排序规则：

- `target_index` 会语义漂移
- grounding head 会被错误监督

### 9.3 盒子没有明显视觉或几何区分

如果左右盒子太像，而且画面里左/右关系不稳定：

- 模型很难学会“左边盒子”和“右边盒子”

建议：

- 左右盒子在场景中保持明显空间分离
- 训练时覆盖不同相机视角下的左右关系

### 9.4 文本太容易、场景太干净

如果止痛药和维生素外观差异巨大，OCR 分支会被弱化。

建议：

- 加入外观相似的干扰药盒
- 增加文字依赖性

---

## 10. 最后建议

如果你要让 `s2_pi05` 真正比 `pi05` 有优势，数据集至少要满足下面这句话：

- 不看对象文字和关系，任务就容易做错。

对于本文这个例子，最理想的数据分布应该包含：

- 多种药品同时出现
- 至少两个相同外观的容器
- 指令中同时包含“对象语义”和“空间语义”
- 多阶段目标切换监督

这样你的新增模块：

- OCR gating
- temporal fusion
- semantic graph
- explicit grounding

才会真的学到东西，而不是只在论文里存在。
