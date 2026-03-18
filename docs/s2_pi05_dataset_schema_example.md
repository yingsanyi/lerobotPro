# S2-PI05 数据集 JSON/Parquet 字段 Schema 示例

本文档给出一份面向 `s2_pi05` 的 LeRobot v3 数据集 schema 示例，重点覆盖：

- `meta/info.json` 应该如何写；
- `meta/tasks.parquet`、`meta/episodes/**/*.parquet`、`data/**/*.parquet` 分别存什么；
- `s2_pi05` 新增的语义字段如何定 shape、dtype 和 padding 规则；
- 用“将止痛药放在左边的盒子，将维生素放在右边的盒子”这个任务举一个可直接参考的样例。

配套阅读：

- [s2_pi05_data_collection_spec.md](/root/songling/lerobotPro/docs/s2_pi05_data_collection_spec.md)
- [s2_pi05_implementation_checklist.md](/root/songling/lerobotPro/docs/s2_pi05_implementation_checklist.md)

## 1. 版本与兼容说明

当前仓库的数据集实现以 **LeRobot v3** 为主，推荐格式是：

- `meta/info.json`
- `meta/stats.json`
- `meta/tasks.parquet`
- `meta/episodes/chunk-xxx/file-xxx.parquet`
- `data/chunk-xxx/file-xxx.parquet`
- `videos/<camera>/chunk-xxx/file-xxx.mp4`

注意：

- 一些文档或旧数据可能还会提到 `meta/tasks.jsonl`。
- 但这个仓库当前的读取/写入实现实际使用的是 `meta/tasks.parquet`。
- 所以你如果要新建数据集，应该优先按 `tasks.parquet` 写。
- 如果你确实想保留 `tasks.jsonl` 作为中间产物，也可以，但需要在导入阶段转成 `tasks.parquet`。

## 2. 推荐目录结构

```text
your_s2_dataset/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── front_rgb/
        └── chunk-000/
            └── file-000.mp4
```

如果你有多相机，可以继续增加：

```text
videos/
├── front_rgb/chunk-000/file-000.mp4
├── wrist_rgb/chunk-000/file-000.mp4
└── side_rgb/chunk-000/file-000.mp4
```

## 3. S2-PI05 的字段设计原则

相对原始 `pi05`，`s2_pi05` 需要额外携带对象级语义信息。推荐采用 **固定长度 schema**，不要在同一个字段里混用可变长度对象列表。

建议固定这些超参数：

- `K = semantic_max_objects = 8`
- `E_max = 16` 或 `24`
- `D_node = semantic_node_dim = 512`
- `T_mem = semantic_track_memory = 8`

这样每一帧都能写成固定 shape，便于：

- Parquet/Arrow 存储；
- `LeRobotDataset` 加载；
- `s2_pi05` processor 自动 batch 化；
- 模型训练时稳定 padding。

### 3.1 空槽位 padding 规则

当某一帧只有 4 个对象，而 schema 允许 `K=8` 时，剩余 4 个槽位统一按下面规则补齐：

- `observation.semantic.node_mask`: `false`
- `observation.semantic.boxes_xyxy`: `[0, 0, 0, 0]`
- `observation.semantic.track_ids`: `-1`
- `observation.semantic.camera_ids`: `-1`
- `observation.semantic.ocr_text`: `""`
- `observation.semantic.ocr_conf`: `0.0`
- `observation.semantic.visual_embs`: 全 0
- `observation.semantic.node_embs`: 全 0

如果没有可用目标，或者该帧不打 grounding 标注：

- `observation.semantic.target_index = -1`

如果边数量不足 `E_max`：

- `observation.semantic.edge_index` 中未使用边位置填 `-1`
- `observation.semantic.edge_attr` 中未使用边位置填 `0`

### 3.2 坐标与排序规则

强烈建议：

- `observation.semantic.boxes_xyxy` 使用 **归一化坐标** `[x1, y1, x2, y2]`，范围 `[0, 1]`
- 同一帧内所有语义字段按 **同一个对象顺序** 排列
- 这个顺序优先按 `track_id` 稳定，其次按检测得分/目标相关性

也就是说，第 `i` 个槽位在下面这些字段里必须始终指向同一个对象：

- `boxes_xyxy[i]`
- `track_ids[i]`
- `ocr_text[i]`
- `ocr_conf[i]`
- `camera_ids[i]`
- `target_index == i`

## 4. `meta/info.json` 示例

下面给的是一份简化但结构正确的例子。为了聚焦 `s2_pi05`，省略了部分非核心字段和视频编码细节。

```json
{
  "codebase_version": "0.4.0",
  "robot_type": "so101_follower",
  "total_episodes": 128,
  "total_frames": 18432,
  "total_tasks": 6,
  "chunks_size": 1000,
  "data_files_size_in_mb": 100,
  "video_files_size_in_mb": 200,
  "fps": 10,
  "splits": {
    "train": "0:110",
    "val": "110:119",
    "test": "119:128"
  },
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": {
    "timestamp": {
      "dtype": "float32",
      "shape": [1],
      "names": null
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "episode_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "task_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [14],
      "names": {
        "axes": ["joint_or_pose_dim"]
      }
    },
    "action": {
      "dtype": "float32",
      "shape": [14],
      "names": {
        "axes": ["action_dim"]
      }
    },
    "observation.images.front_rgb": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channel"]
    },
    "observation.semantic.node_mask": {
      "dtype": "bool",
      "shape": [8],
      "names": ["object_slot"]
    },
    "observation.semantic.boxes_xyxy": {
      "dtype": "float32",
      "shape": [8, 4],
      "names": ["object_slot", "xyxy"]
    },
    "observation.semantic.track_ids": {
      "dtype": "int64",
      "shape": [8],
      "names": ["object_slot"]
    },
    "observation.semantic.camera_ids": {
      "dtype": "int64",
      "shape": [8],
      "names": ["object_slot"]
    },
    "observation.semantic.ocr_text": {
      "dtype": "string",
      "shape": [8],
      "names": ["object_slot"]
    },
    "observation.semantic.cf_ocr_text": {
      "dtype": "string",
      "shape": [8],
      "names": ["object_slot"]
    },
    "observation.semantic.ocr_conf": {
      "dtype": "float32",
      "shape": [8],
      "names": ["object_slot"]
    },
    "observation.semantic.target_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    "observation.semantic.edge_index": {
      "dtype": "int64",
      "shape": [2, 16],
      "names": ["src_or_dst", "edge_slot"]
    },
    "observation.semantic.edge_attr": {
      "dtype": "float32",
      "shape": [16, 4],
      "names": ["edge_slot", "edge_attr_dim"]
    }
  }
}
```

### 4.1 建议保留的可选预计算字段

如果你计划做离线预计算，也可以在 `features` 里再加：

```json
{
  "observation.semantic.visual_embs": {
    "dtype": "float32",
    "shape": [8, 512],
    "names": ["object_slot", "node_dim"]
  },
  "observation.semantic.node_embs": {
    "dtype": "float32",
    "shape": [8, 512],
    "names": ["object_slot", "node_dim"]
  },
  "observation.semantic.prev_node_embs": {
    "dtype": "float32",
    "shape": [8, 8, 512],
    "names": ["history_step", "object_slot", "node_dim"]
  }
}
```

是否真的要把这些 embedding 存进数据集，建议按下面原则判断：

- 第一版数据集：优先只存原始语义字段，如 `boxes`、`ocr_text`、`ocr_conf`、`target_index`
- 感知前端已经稳定：再考虑离线存 `visual_embs` 或 `node_embs`
- `prev_node_embs` 只在你明确想做离线 temporal cache 时再存

## 5. `meta/stats.json` 示例

`pi05` 和 `s2_pi05` 的主干仍然依赖 `state`/`action` 归一化统计，尤其是 quantiles。语义字段通常不做归一化，或者只做模型内部处理，不必强行写进 `stats.json`。

下面给一个简化例子：

```json
{
  "observation.state": {
    "mean": [0.01, -0.05, 0.12, 0.00, 0.31, -0.22, 0.44, 0.09, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "std": [0.19, 0.22, 0.17, 0.14, 0.28, 0.16, 0.21, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    "min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "q01": [-0.92, -0.88, -0.79, -0.70, -0.85, -0.77, -0.82, -0.50, -0.80, -0.80, -0.80, -0.80, -0.80, -0.80],
    "q99": [0.91, 0.86, 0.81, 0.73, 0.88, 0.76, 0.84, 0.48, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]
  },
  "action": {
    "mean": [0.00, 0.01, -0.01, 0.00, 0.03, -0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "std": [0.08, 0.09, 0.07, 0.06, 0.10, 0.09, 0.08, 0.04, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    "min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "q01": [-0.45, -0.42, -0.39, -0.35, -0.48, -0.46, -0.44, -0.20, -0.80, -0.80, -0.80, -0.80, -0.80, -0.80],
    "q99": [0.44, 0.41, 0.38, 0.36, 0.49, 0.45, 0.43, 0.21, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80]
  }
}
```

## 6. `meta/tasks.parquet` schema 示例

当前仓库推荐使用 `meta/tasks.parquet`。它本质上是一张很简单的表：

| 列名 | dtype | 含义 |
| --- | --- | --- |
| `task` | string | 自然语言任务描述，通常作为 DataFrame index |
| `task_index` | int64 | 任务 ID |

示例内容：

| task | task_index |
| --- | --- |
| 将止痛药放在左边的盒子，将维生素放在右边的盒子 | 0 |
| 将红色杯子放到托盘中央 | 1 |
| 拿起写着茶的盒子并放回原位 | 2 |

### 6.1 兼容旧格式的 `tasks.jsonl` 示例

如果你前处理脚本里已经有 JSON Lines，也可以先产生下面这种文件，然后在导入 LeRobot v3 时转成 `tasks.parquet`：

```jsonl
{"task": "将止痛药放在左边的盒子，将维生素放在右边的盒子", "task_index": 0}
{"task": "将红色杯子放到托盘中央", "task_index": 1}
{"task": "拿起写着茶的盒子并放回原位", "task_index": 2}
```

但注意：

- `s2_pi05` 在当前仓库里不会直接优先读取这个 JSONL
- 正式训练前最好仍然转成 `meta/tasks.parquet`

## 7. `meta/episodes/chunk-000/file-000.parquet` schema 示例

这是 episode 级元数据，不是逐帧数据。最核心的列如下：

| 列名 | dtype | 示例 | 含义 |
| --- | --- | --- | --- |
| `episode_index` | int64 | `12` | episode ID |
| `tasks` | list<string> | `["将止痛药放在左边的盒子，将维生素放在右边的盒子"]` | 本 episode 涉及的任务列表 |
| `length` | int64 | `120` | 该 episode 总帧数 |
| `dataset_from_index` | int64 | `1560` | 这条 episode 在 `data/*.parquet` 里的起始全局 index |
| `dataset_to_index` | int64 | `1680` | 终止全局 index，通常为开区间右边界 |
| `meta/episodes/chunk_index` | int64 | `0` | 该 episode 对应的数据 chunk |
| `meta/episodes/file_index` | int64 | `0` | 该 episode 对应的数据 file |

实际文件中还可能出现很多按列展开的 `stats/...` 字段。这些通常由 LeRobot 写入流程自动维护，不需要你手工逐列设计第一版 schema。

一个简化行示例如下：

```json
{
  "episode_index": 12,
  "tasks": ["将止痛药放在左边的盒子，将维生素放在右边的盒子"],
  "length": 120,
  "dataset_from_index": 1560,
  "dataset_to_index": 1680,
  "meta/episodes/chunk_index": 0,
  "meta/episodes/file_index": 0
}
```

## 8. `data/chunk-000/file-000.parquet` schema 示例

这是逐帧表，也是 `s2_pi05` 训练时最关键的地方。

### 8.1 推荐字段表

| 列名 | dtype | shape | 必需性 | 说明 |
| --- | --- | --- | --- | --- |
| `timestamp` | float32 | `[1]` | 必需 | 当前帧时间戳 |
| `frame_index` | int64 | `[1]` | 必需 | 当前 episode 内帧号 |
| `episode_index` | int64 | `[1]` | 必需 | 所属 episode |
| `index` | int64 | `[1]` | 必需 | 全局帧索引 |
| `task_index` | int64 | `[1]` | 必需 | 任务 ID |
| `observation.state` | float32 | `[14]` | 必需 | 原始机器人状态 |
| `action` | float32 | `[14]` | 必需 | 当前监督动作 |
| `observation.semantic.node_mask` | bool | `[8]` | 强烈推荐 | 对象槽位是否有效 |
| `observation.semantic.boxes_xyxy` | float32 | `[8, 4]` | 强烈推荐 | 归一化 bbox |
| `observation.semantic.track_ids` | int64 | `[8]` | 强烈推荐 | 跨帧稳定对象 ID |
| `observation.semantic.camera_ids` | int64 | `[8]` | 推荐 | 对象来自哪个相机 |
| `observation.semantic.ocr_text` | string | `[8]` | 强烈推荐 | 每个对象的 OCR 文本 |
| `observation.semantic.cf_ocr_text` | string | `[8]` | 推荐 | 反事实文本版本 |
| `observation.semantic.ocr_conf` | float32 | `[8]` | 强烈推荐 | OCR 置信度 |
| `observation.semantic.target_index` | int64 | `[1]` | 强烈推荐 | 当前阶段目标对象槽位 |
| `observation.semantic.edge_index` | int64 | `[2, 16]` | 可选 | 图边，`[src, dst]` |
| `observation.semantic.edge_attr` | float32 | `[16, 4]` | 可选 | 边属性 |
| `observation.semantic.visual_embs` | float32 | `[8, 512]` | 可选 | 离线视觉特征 |
| `observation.semantic.node_embs` | float32 | `[8, 512]` | 可选 | 离线融合后对象特征 |
| `observation.semantic.prev_node_embs` | float32 | `[8, 8, 512]` | 可选 | 历史对象特征缓存 |

### 8.2 `edge_index` / `edge_attr` 的具体约定

`s2_pi05` 当前模型里，`edge_index` 的约定是：

- 单帧 shape: `[2, E_max]`
- 第 0 行是所有 `src`
- 第 1 行是所有 `dst`

例如：

```json
{
  "observation.semantic.edge_index": [
    [2, 2, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  ]
}
```

可以表示：

- `止痛药 -> 左盒子`
- `止痛药 -> 右盒子`
- `维生素 -> 左盒子`
- `维生素 -> 右盒子`

`edge_attr` 你可以自己定义，但建议第一版固定成 4 维，例如：

- 第 0 维：`left_of`
- 第 1 维：`right_of`
- 第 2 维：`near`
- 第 3 维：`overlap_or_inside`

## 9. 面向药品分拣任务的一条逐帧样例

任务：

```text
将止痛药放在左边的盒子，将维生素放在右边的盒子
```

假设这一帧的场景里有 4 个有效对象，槽位定义如下：

| 槽位 | 对象 | track_id |
| --- | --- | --- |
| `0` | 左边盒子 | `101` |
| `1` | 右边盒子 | `102` |
| `2` | 止痛药盒 | `201` |
| `3` | 维生素盒 | `202` |

其余槽位 `4..7` 全部为空。

### 9.1 一条 frame 的 JSON 风格示例

注意：真正落盘是在 Parquet 里，这里只是用 JSON 风格把一行展示出来。

```json
{
  "timestamp": 2.4,
  "frame_index": 24,
  "episode_index": 12,
  "index": 1584,
  "task_index": 0,
  "observation.state": [0.11, -0.22, 0.35, 0.04, 0.62, -0.18, 0.09, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
  "action": [0.02, -0.01, 0.03, 0.00, 0.05, -0.03, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
  "observation.semantic.node_mask": [true, true, true, true, false, false, false, false],
  "observation.semantic.boxes_xyxy": [
    [0.08, 0.51, 0.28, 0.88],
    [0.61, 0.50, 0.84, 0.88],
    [0.32, 0.34, 0.47, 0.63],
    [0.52, 0.33, 0.66, 0.61],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00]
  ],
  "observation.semantic.track_ids": [101, 102, 201, 202, -1, -1, -1, -1],
  "observation.semantic.camera_ids": [0, 0, 0, 0, -1, -1, -1, -1],
  "observation.semantic.ocr_text": [
    "左边",
    "右边",
    "止痛药",
    "维生素",
    "",
    "",
    "",
    ""
  ],
  "observation.semantic.cf_ocr_text": [
    "左边",
    "右边",
    "维生素",
    "止痛药",
    "",
    "",
    "",
    ""
  ],
  "observation.semantic.ocr_conf": [0.98, 0.97, 0.95, 0.94, 0.00, 0.00, 0.00, 0.00],
  "observation.semantic.target_index": 2,
  "observation.semantic.edge_index": [
    [2, 2, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  ],
  "observation.semantic.edge_attr": [
    [1.0, 0.0, 0.7, 0.0],
    [0.0, 1.0, 0.3, 0.0],
    [1.0, 0.0, 0.2, 0.0],
    [0.0, 1.0, 0.8, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
  ]
}
```

这一帧的语义含义是：

- 当前任务仍然是整句复合指令，不拆成两条 task
- 但当前阶段目标已经切到 `target_index = 2`，表示模型此时应优先处理“止痛药”
- `ocr_text[2] = "止痛药"`，`ocr_text[3] = "维生素"` 用来帮助 grounding 区分两个外观相似对象
- `cf_ocr_text` 则给反事实训练使用

### 9.2 同一个 episode 中 `target_index` 的阶段变化

对于这个复合任务，`target_index` 不应该整条轨迹固定不变，而是应该随子阶段切换。

一个合理示例：

| 帧范围 | 当前语义阶段 | `target_index` |
| --- | --- | --- |
| `0-25` | 接近并抓取止痛药 | `2` |
| `26-60` | 搬运止痛药去左边盒子 | `2` |
| `61-70` | 已完成第一个子任务，重新观察场景 | `-1` 或 `3` |
| `71-95` | 接近并抓取维生素 | `3` |
| `96-120` | 搬运维生素去右边盒子 | `3` |

如果你还维护子任务表，也可以额外增加：

- `subtask_index`
- `meta/subtasks.parquet`

但这不是 `s2_pi05` 最小闭环的必需项。

## 10. 字段落盘时最容易出错的地方

### 10.1 `ocr_text` 不要做成可变长乱序 list

错误方式：

- 某一帧 3 个对象，就写 3 个字符串
- 下一帧 5 个对象，就写 5 个字符串
- 且对象顺序跟检测器置信度随时变化

这样会导致：

- `target_index` 语义漂移
- temporal fusion 失效
- `track_ids` 和 `ocr_text` 对不上

正确方式：

- 固定 `K`
- 固定对象槽位顺序
- 对空槽位写 `""`

### 10.2 `edge_index` 必须遵循 `[2, E_max]`

当前 `s2_pi05` 模型读入 `edge_index` 时，默认第 0 行是 `src`，第 1 行是 `dst`。不要改成 `[E_max, 2]`。

### 10.3 `target_index` 必须对齐当前槽位，而不是对齐 `track_id`

例如这一帧中：

- `track_id=201` 的对象被放在槽位 `2`

那么：

- `target_index` 应该写 `2`

而不是：

- `201`

### 10.4 预计算 embedding 不要和原始字段失配

如果你同时存：

- `boxes_xyxy`
- `ocr_text`
- `node_embs`

那这三者必须共享同一对象顺序。否则训练时你会得到“框是 A，对应 embedding 却是 B”的脏数据。

## 11. 最小可用版本和增强版本

### 11.1 最小可用 raw schema

如果你只想先跑通数据链路，推荐只保留这些字段：

- `observation.state`
- `action`
- `observation.images.front_rgb`
- `observation.semantic.node_mask`
- `observation.semantic.boxes_xyxy`
- `observation.semantic.track_ids`
- `observation.semantic.ocr_text`
- `observation.semantic.ocr_conf`
- `observation.semantic.target_index`

这已经足够支持：

- OCR gating
- grounding
- 基于框的对象裁剪
- 基于 track 的时序记忆

### 11.2 增强版 schema

当你前端稳定后，再加：

- `observation.semantic.camera_ids`
- `observation.semantic.cf_ocr_text`
- `observation.semantic.edge_index`
- `observation.semantic.edge_attr`
- `observation.semantic.visual_embs`
- `observation.semantic.node_embs`
- `observation.semantic.prev_node_embs`

这会让你更方便做：

- 反事实训练
- 稀疏图推理
- 离线缓存加速
- 多相机对象关联

## 12. 结论

对 `s2_pi05` 来说，最重要的不是“字段越多越好”，而是：

- 对象槽位顺序稳定
- `target_index` 真正反映当前阶段目标
- `ocr_text` 与 `boxes`、`track_ids` 严格对齐
- 所有语义字段尽量使用固定长度 Parquet schema

如果你后面要把这份 schema 真正落成一个数据集导出脚本，建议直接按本文中的：

- `K=8`
- `E_max=16`
- `string[8]`
- `float32[8,4]`
- `int64[2,16]`

这一版先做，不要一上来追求完全动态 schema。
