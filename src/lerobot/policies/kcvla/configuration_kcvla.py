#!/usr/bin/env python

from dataclasses import dataclass, fields

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("kcvla")
@dataclass
class KCVLAConfig(PI05Config):
    """Keyword-Conditioned VLA configuration.

    KC-VLA keeps the original PI05 action backbone and adds:
    - externally provided keyword queries,
    - lightweight keyword-to-vision cross-attention grounding,
    - weakly supervised counterfactual / contrastive / sparsity objectives.
    """

    keyword_conditioning_enabled: bool = True
    keyword_max_count: int = 8
    keyword_text_max_tokens: int = 16

    keyword_cross_attention_heads: int = 4
    keyword_cross_attention_dropout: float = 0.1
    keyword_cross_attention_init_scale: float = 0.01

    counterfactual_enabled: bool = True
    counterfactual_auto_generate: bool = True
    counterfactual_action_margin: float = 0.05

    loss_contrast_w: float = 0.1
    loss_counterfactual_w: float = 0.1
    loss_sparse_w: float = 0.01

    # Backward-compatibility knobs kept so older configs still load cleanly.
    keyword_require_boxes: bool = False
    counterfactual_attention_margin: float = 0.0
    loss_align_w: float = 0.0

    @classmethod
    def from_pi05_config(cls, config: PI05Config, **overrides) -> "KCVLAConfig":
        init_kwargs = {field.name: getattr(config, field.name) for field in fields(config)}
        init_kwargs.update(overrides)
        return cls(**init_kwargs)

    def __post_init__(self):
        super().__post_init__()

        if self.keyword_max_count <= 0:
            raise ValueError("keyword_max_count must be > 0")
        if self.keyword_text_max_tokens <= 0:
            raise ValueError("keyword_text_max_tokens must be > 0")
        if self.keyword_cross_attention_heads <= 0:
            raise ValueError("keyword_cross_attention_heads must be > 0")
        if not 0.0 <= self.keyword_cross_attention_dropout < 1.0:
            raise ValueError("keyword_cross_attention_dropout must be in [0, 1)")
        if self.keyword_cross_attention_init_scale < 0:
            raise ValueError("keyword_cross_attention_init_scale must be >= 0")
        if self.counterfactual_action_margin < 0:
            raise ValueError("counterfactual_action_margin must be >= 0")
        if self.counterfactual_attention_margin < 0:
            raise ValueError("counterfactual_attention_margin must be >= 0")
        for loss_name in ("loss_contrast_w", "loss_counterfactual_w", "loss_sparse_w", "loss_align_w"):
            if getattr(self, loss_name) < 0:
                raise ValueError(f"{loss_name} must be >= 0")
