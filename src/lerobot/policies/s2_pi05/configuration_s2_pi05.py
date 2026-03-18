#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, fields

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("s2_pi05")
@dataclass
class S2PI05Config(PI05Config):
    """Configuration for the structured-semantic PI05 variant.

    S2-PI05 keeps the original PI05 action backbone and augments it with:
    - object-centric semantic inputs,
    - OCR-aware gating,
    - temporal object fusion,
    - graph reasoning,
    - explicit target grounding.
    """

    semantic_enabled: bool = True
    semantic_require_inputs: bool = False
    semantic_use_precomputed_nodes: bool = True
    semantic_use_boxes_geometry: bool = True
    semantic_use_online_track_memory: bool = True

    semantic_max_objects: int = 8
    semantic_node_dim: int = 512
    semantic_text_enabled: bool = True
    semantic_text_max_tokens: int = 12
    semantic_gate_alpha: float = 1.0
    semantic_gate_beta: float = 1.0
    semantic_text_align_temperature: float = 0.07

    semantic_temporal_fusion: str = "gru"  # {none, ema, gru, attn}
    semantic_track_memory: int = 8
    semantic_ema_decay: float = 0.9
    semantic_attn_heads: int = 4

    semantic_graph_edges: str = "spatial+interaction"  # {none, spatial, spatial+interaction}
    semantic_graph_encoder: str = "gat"  # {none, gat, graph_transformer}
    semantic_graph_layers: int = 2
    semantic_graph_dropout: float = 0.1

    grounding_enabled: bool = True
    grounding_mode: str = "soft"  # {soft, hard_gumbel, hard_argmax_infer}
    grounding_temperature: float = 1.0
    target_token_count: int = 1

    keyword_conditioning_enabled: bool = True
    keyword_text_max_tokens: int = 16
    keyword_cross_attention_heads: int = 4
    keyword_cross_attention_dropout: float = 0.1
    keyword_cross_attention_init_scale: float = 0.01

    loss_text_align_w: float = 0.1
    loss_counterfactual_w: float = 0.1
    loss_temporal_w: float = 0.1
    loss_grounding_w: float = 0.1

    @classmethod
    def from_pi05_config(cls, config: PI05Config, **overrides) -> "S2PI05Config":
        """Upgrade a PI05 config into S2-PI05 while preserving compatible backbone settings."""
        init_kwargs = {field.name: getattr(config, field.name) for field in fields(config)}
        init_kwargs.update(overrides)
        return cls(**init_kwargs)

    def __post_init__(self):
        super().__post_init__()

        if self.semantic_max_objects <= 0:
            raise ValueError("semantic_max_objects must be > 0")
        if self.semantic_node_dim <= 0:
            raise ValueError("semantic_node_dim must be > 0")
        if self.semantic_text_max_tokens <= 0:
            raise ValueError("semantic_text_max_tokens must be > 0")
        if self.semantic_track_memory <= 0:
            raise ValueError("semantic_track_memory must be > 0")
        if self.semantic_graph_layers < 0:
            raise ValueError("semantic_graph_layers must be >= 0")
        if self.target_token_count <= 0:
            raise ValueError("target_token_count must be > 0")
        if self.semantic_temporal_fusion not in {"none", "ema", "gru", "attn"}:
            raise ValueError(
                "semantic_temporal_fusion must be one of {'none', 'ema', 'gru', 'attn'}"
            )
        if self.semantic_graph_edges not in {"none", "spatial", "spatial+interaction"}:
            raise ValueError(
                "semantic_graph_edges must be one of {'none', 'spatial', 'spatial+interaction'}"
            )
        if self.semantic_graph_encoder not in {"none", "gat", "graph_transformer"}:
            raise ValueError(
                "semantic_graph_encoder must be one of {'none', 'gat', 'graph_transformer'}"
            )
        if self.grounding_mode not in {"soft", "hard_gumbel", "hard_argmax_infer"}:
            raise ValueError(
                "grounding_mode must be one of {'soft', 'hard_gumbel', 'hard_argmax_infer'}"
            )
        if not 0.0 <= self.semantic_ema_decay < 1.0:
            raise ValueError("semantic_ema_decay must be in [0, 1)")
        if self.semantic_attn_heads <= 0:
            raise ValueError("semantic_attn_heads must be > 0")
        if self.semantic_node_dim % self.semantic_attn_heads != 0:
            raise ValueError("semantic_node_dim must be divisible by semantic_attn_heads")
        if self.semantic_text_align_temperature <= 0:
            raise ValueError("semantic_text_align_temperature must be > 0")
        if self.grounding_temperature <= 0:
            raise ValueError("grounding_temperature must be > 0")
        if self.keyword_text_max_tokens <= 0:
            raise ValueError("keyword_text_max_tokens must be > 0")
        if self.keyword_cross_attention_heads <= 0:
            raise ValueError("keyword_cross_attention_heads must be > 0")
        if not 0.0 <= self.keyword_cross_attention_dropout < 1.0:
            raise ValueError("keyword_cross_attention_dropout must be in [0, 1)")
        if self.keyword_cross_attention_init_scale < 0:
            raise ValueError("keyword_cross_attention_init_scale must be >= 0")
        for loss_name in (
            "loss_text_align_w",
            "loss_counterfactual_w",
            "loss_temporal_w",
            "loss_grounding_w",
        ):
            if getattr(self, loss_name) < 0:
                raise ValueError(f"{loss_name} must be >= 0")
