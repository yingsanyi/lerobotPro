# KC-VLA (`kcvla`)

`kcvla` 是一个建立在 `pi05` 上的轻量关键词条件化策略。

它保留原始 `pi05` 的：

- vision tower
- task/state prompt path
- transformer backbone
- flow-matching action head

并额外加入：

- externally provided keyword set
- keyword-to-vision cross-attention grounding
- counterfactual / cross-keyword contrast / sparsity losses

## Required inputs

Baseline `pi05` fields:

- `observation.images.*`
- `observation.state`
- `action`
- `task`

KC-VLA fields:

- `keyword_text`
- `counterfactual_keyword_text` optional

Counterfactual behavior:

- manual `counterfactual_keyword_text` overrides auto generation
- missing values can be auto-generated from same-frame keyword permutations
- `counterfactual_enabled=False` disables the branch entirely

## Notes

- The original `pi05` implementation is left untouched.
- `KCVLAPolicy.from_pretrained(...)` can upgrade a `PI05Config` checkpoint into `KCVLAConfig`.
- New grounding modules are initialized randomly when loading old `pi05` weights.
- The weakly supervised variant no longer consumes per-keyword bbox or camera-id annotations.
