# KC-VLA Implementation Checklist

这份清单对应当前已经落地的无 `bbox` 弱监督版 `kcvla` 实现，方便你后续继续扩展或写论文时对照代码。

---

## 1. 共享常量

文件：[constants.py](/root/songling/lerobotPro/src/lerobot/utils/constants.py)

当前 `kcvla` 直接依赖的新增常量主要有：

- `COUNTERFACTUAL_KEYWORD_TEXT`
- `OBS_LANGUAGE_KEYWORD_CF_TOKENS`
- `OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK`

作用：

- 把关键词集和反事实关键词接到现有 LeRobot 数据流上。

---

## 2. 配置

文件：[configuration_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/configuration_kcvla.py)

已实现：

- `KCVLAConfig` 继承 `PI05Config`
- `from_pi05_config(...)`
- 关键词分支、counterfactual margin、contrast/sparse loss 权重等超参
- 对旧版 `bbox-align` 配置字段的向后兼容

后续可扩展：

- 关键词 query 的聚合方式
- contrast / sparsity 的 schedule
- 更强的 counterfactual curriculum

---

## 3. Processor

文件：[processor_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/processor_kcvla.py)

已实现：

- `KCVLAKeywordSetTokenizerProcessorStep`
- `make_kcvla_pre_post_processors(...)`

当前行为：

- 保持 `pi05` 原始 task/state prompt 路径不变
- 单独把 `keyword_text` 切成关键词集合并 token 化
- 可选生成反事实关键词张量
- 不再规范化 per-keyword box / camera id

后续可扩展：

- dataset 级在线关键词增强
- 自动 hard negative / counterfactual 采样

---

## 4. 模型

文件：[modeling_kcvla.py](/root/songling/lerobotPro/src/lerobot/policies/kcvla/modeling_kcvla.py)

已实现：

- `KCVLAKeywordVisionCrossAttention`
- `KCVLAPytorch`
- `KCVLAPolicy`
- 从 `PI05Config` 升级加载 checkpoint

当前结构：

1. 复用 `PI05Pytorch` 的 vision tower、language embedding、transformer 和 action head。
2. 把关键词集编码成 query。
3. 用 query 对视觉 token 做 cross-attention。
4. 将 grounded residual 回写到视觉 token。
5. 再走原始 `pi05` action 生成链路。

当前 loss：

- `loss_counterfactual`
- `loss_contrast`
- `loss_sparse`

后续可扩展：

- 每个关键词单独输出 target token 而不是共享 residual
- patch-level top-k hard grounding 分析
- attention 可视化与定量评价工具

---

## 5. Factory / Registry

文件：

- [factory.py](/root/songling/lerobotPro/src/lerobot/policies/factory.py)
- [__init__.py](/root/songling/lerobotPro/src/lerobot/policies/__init__.py)
- [constants.py](/root/songling/lerobotPro/src/lerobot/async_inference/constants.py)

已实现：

- `make_policy_config("kcvla")`
- `get_policy_class("kcvla")`
- `make_kcvla_pre_post_processors(...)`
- async inference 常量里加入 `kcvla`

---

## 6. 单测

文件：[test_kcvla.py](/root/songling/lerobotPro/tests/policies/kcvla/test_kcvla.py)

已覆盖：

- factory 注册
- `PI05Config -> KCVLAConfig` 升级
- counterfactual complementary data 透传
- 关键词 tokenizer
- 弱监督配置默认项
- 关键词 cross-attention 模块

如果后续继续迭代，优先补：

- `loss_contrast` 的数值测试
- `loss_counterfactual` 的数值测试
- 从 `pi05` checkpoint 加载 `kcvla` 的 smoke test
- attention 可视化回归测试
