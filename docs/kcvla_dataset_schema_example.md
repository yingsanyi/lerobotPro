# KC-VLA 数据集 Schema 示例

这份文档给出一份面向 `kcvla` 的 LeRobot v3 字段示例，重点是把 `keyword_text` 和反事实关键词接到现有 `pi05` 数据格式上。当前版本不要求 `bbox` 或相机级 grounding 标注。

配套阅读：

- [kcvla_pi05.md](/root/songling/lerobotPro/docs/kcvla_pi05.md)
- [kcvla_data_collection_spec.md](/root/songling/lerobotPro/docs/kcvla_data_collection_spec.md)

---

## 1. 推荐目录结构

```text
your_kcvla_dataset/
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

---

## 2. `meta/info.json` 里的关键新增字段

下面是一个缩减版示例，只保留与 KC-VLA 直接相关的 feature：

```json
{
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [14]
    },
    "action": {
      "dtype": "float32",
      "shape": [14]
    },
    "observation.images.front_rgb": {
      "dtype": "video",
      "shape": [480, 640, 3]
    },
    "keyword_text": {
      "dtype": "string",
      "shape": [1]
    },
    "counterfactual_keyword_text": {
      "dtype": "string",
      "shape": [1]
    }
  }
}
```

当前实现默认：

- `keyword_max_count = 8`
- `keyword_text_max_tokens = 16`

processor 会把原始字符串拆成关键词集合，并自动生成固定槽位的关键词 token 张量。

---

## 3. 单帧样例

```json
{
  "timestamp": 2.4,
  "frame_index": 24,
  "episode_index": 12,
  "index": 1584,
  "task_index": 0,
  "task": "将止痛药放在左边的盒子，将维生素放在右边的盒子",
  "keyword_text": "止痛药, 维生素",
  "counterfactual_keyword_text": "维生素, 止痛药",
  "observation.state": [0.11, -0.22, 0.35, 0.04, 0.62, -0.18, 0.09, 0.01],
  "action": [0.02, -0.01, 0.03, 0.00, 0.05, -0.03, 0.01, 0.00],
  "observation.images.front_rgb": "video frame"
}
```

---

## 4. 训练时 processor 生成的中间张量

在进入模型前，`processor_kcvla.py` 会把补充字段转成：

- `observation.language.keyword.tokens` -> `[B, K, T_kw]`
- `observation.language.keyword.attention_mask` -> `[B, K, T_kw]`
- `observation.language.keyword_cf.tokens` -> `[B, K, T_kw]`
- `observation.language.keyword_cf.attention_mask` -> `[B, K, T_kw]`

这些中间字段不一定非要预先写入原始 parquet；只要原始样本能提供上面的源字段，processor 就能自动构造。

---

## 5. 兼容说明

如果你的旧数据里已经包含：

- `observation.keyword.boxes_xyxy`
- `observation.keyword.camera_ids`

可以继续保留在 parquet 中，但当前弱监督版 `kcvla` 不会读取它们。
