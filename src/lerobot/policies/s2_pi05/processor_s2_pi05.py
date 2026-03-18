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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition
from lerobot.utils.constants import (
    KEYWORD_TEXT,
    OBS_LANGUAGE_KEYWORD_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_TOKENS,
    OBS_SEMANTIC_BOXES_XYXY,
    OBS_SEMANTIC_CAMERA_IDS,
    OBS_SEMANTIC_CF_OCR_TEXT,
    OBS_SEMANTIC_CF_TEXT_ATTENTION_MASK,
    OBS_SEMANTIC_CF_TEXT_TOKENS,
    OBS_SEMANTIC_EDGE_ATTR,
    OBS_SEMANTIC_EDGE_INDEX,
    OBS_SEMANTIC_NODE_EMBS,
    OBS_SEMANTIC_NODE_MASK,
    OBS_SEMANTIC_OCR_CONF,
    OBS_SEMANTIC_OCR_TEXT,
    OBS_SEMANTIC_PREV_NODE_EMBS,
    OBS_SEMANTIC_TARGET_INDEX,
    OBS_SEMANTIC_TEXT_ATTENTION_MASK,
    OBS_SEMANTIC_TEXT_TOKENS,
    OBS_SEMANTIC_TRACK_IDS,
    OBS_SEMANTIC_VISUAL_EMBS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_s2_pi05 import S2PI05Config

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


def _ensure_min_rank(tensor: torch.Tensor, expected_rank: int) -> torch.Tensor:
    while tensor.dim() < expected_rank:
        tensor = tensor.unsqueeze(0)
    return tensor


def _convert_to_tensor(value: Any) -> torch.Tensor | None:
    if value is None or isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return torch.empty(0)
        try:
            return torch.as_tensor(value)
        except Exception:
            return None
    if hasattr(value, "tolist"):
        try:
            return torch.as_tensor(value.tolist())
        except Exception:
            return None
    return None


def _normalize_text_value(raw_text: Any) -> Any:
    if raw_text is None:
        return None
    if isinstance(raw_text, (list, tuple)):
        return raw_text
    if hasattr(raw_text, "tolist") and not isinstance(raw_text, str):
        return raw_text.tolist()
    return raw_text


def _clean_text(item: Any) -> str:
    if item is None:
        return ""
    return str(item).replace("\n", " ").replace("\t", " ").strip()


def _standardize_text_batch(
    raw_text: Any,
    *,
    expected_len: int | None = None,
    repeat_singleton: bool = True,
) -> list[str] | None:
    raw_text = _normalize_text_value(raw_text)
    if raw_text is None:
        return None

    if isinstance(raw_text, str):
        texts = [_clean_text(raw_text)]
    elif isinstance(raw_text, (list, tuple)) and all(isinstance(x, str) or x is None for x in raw_text):
        texts = [_clean_text(x) for x in raw_text]
    else:
        return None

    if expected_len is None:
        return texts
    if len(texts) == expected_len:
        return texts
    if len(texts) == 1 and repeat_singleton and expected_len > 1:
        return texts * expected_len
    if len(texts) == 0:
        return ["" for _ in range(expected_len)]
    raise ValueError(f"Expected {expected_len} text entries but received {len(texts)}")


@ProcessorStepRegistry.register(name="s2_pi05_prepare_state_keyword_tokenizer_processor_step")
@dataclass
class S2Pi05PrepareStateKeywordTokenizerProcessorStep(Pi05PrepareStateTokenizerProcessorStep):
    """Prepare the PI05 prompt and splice keyword_text into the task prompt when available."""

    keyword_key: str = KEYWORD_TEXT
    keyword_label: str = "Key text"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get("observation.state")
        if state is None:
            raise ValueError("State is required for S2-PI05")

        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        tasks = _standardize_text_batch(complementary_data.get(self.task_key))
        if tasks is None:
            raise ValueError("No task found in complementary data")

        keywords = _standardize_text_batch(complementary_data.get(self.keyword_key), expected_len=len(tasks))
        if keywords is None:
            keywords = ["" for _ in tasks]

        state = deepcopy(state)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_task = _clean_text(task).replace("_", " ")
            cleaned_keyword = _clean_text(keywords[i]).replace("_", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            if cleaned_keyword:
                full_prompt = (
                    f"Task: {cleaned_task}, {self.keyword_label}: {cleaned_keyword}, State: {state_str};\nAction: "
                )
            else:
                full_prompt = f"Task: {cleaned_task}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        complementary_data[self.task_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return transition


@ProcessorStepRegistry.register(name="s2_pi05_keyword_tokenizer")
@dataclass
class S2Pi05KeywordTokenizerProcessorStep(ProcessorStep):
    """Tokenize the lightweight keyword_text field into a dedicated keyword branch."""

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    tokenizer: Any | None = None
    max_length: int = 16
    keyword_key: str = KEYWORD_TEXT
    tokens_key: str = OBS_LANGUAGE_KEYWORD_TOKENS
    attention_mask_key: str = OBS_LANGUAGE_KEYWORD_ATTENTION_MASK
    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.tokenizer is not None:
            self.input_tokenizer = self.tokenizer
            return
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'`."
            )
        if AutoTokenizer is None:
            raise ImportError("AutoTokenizer is not available")
        self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        keywords = _standardize_text_batch(complementary_data.get(self.keyword_key))
        if keywords is None:
            return new_transition

        has_keyword = torch.tensor([bool(text) for text in keywords], dtype=torch.bool)
        tokenized = self.input_tokenizer(
            [text if text else " " for text in keywords],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        observation = dict(new_transition.get(TransitionKey.OBSERVATION, {}) or {})
        observation[self.tokens_key] = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"].to(dtype=torch.bool)
        if (~has_keyword).any():
            attention_mask[~has_keyword] = False
        observation[self.attention_mask_key] = attention_mask
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="s2_pi05_semantic_text_tokenizer")
@dataclass
class S2Pi05SemanticTextTokenizerProcessorStep(ProcessorStep):
    """Tokenize object-level OCR strings into per-object token tensors."""

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    max_length: int = 12
    max_objects: int = 8
    text_key: str = OBS_SEMANTIC_OCR_TEXT
    tokens_key: str = OBS_SEMANTIC_TEXT_TOKENS
    attention_mask_key: str = OBS_SEMANTIC_TEXT_ATTENTION_MASK
    tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'`."
            )
        if AutoTokenizer is None:
            raise ImportError("AutoTokenizer is not available")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def _normalize_raw_text(self, raw_text: Any) -> Any:
        if raw_text is None:
            return None
        if isinstance(raw_text, (tuple, list)):
            return raw_text
        if hasattr(raw_text, "tolist") and not isinstance(raw_text, str):
            return raw_text.tolist()
        return raw_text

    def _clean_text(self, item: Any) -> str:
        if item is None:
            return ""
        return str(item).replace("\n", " ").replace("\t", " ").strip()

    def _standardize_text_rows(self, raw_text: Any) -> list[list[str]] | None:
        raw_text = self._normalize_raw_text(raw_text)
        if raw_text is None:
            return None

        rows: list[list[str]]
        if isinstance(raw_text, str):
            rows = [[self._clean_text(raw_text)]]
        elif isinstance(raw_text, (list, tuple)) and all(
            isinstance(x, str) or x is None for x in raw_text
        ):
            rows = [[self._clean_text(x) for x in raw_text]]
        elif isinstance(raw_text, (list, tuple)) and all(isinstance(x, (list, tuple)) for x in raw_text):
            rows = [[self._clean_text(item) for item in row] for row in raw_text]
        else:
            return None

        if not rows:
            return None

        max_objects = min(max(len(row) for row in rows), self.max_objects)
        if max_objects == 0:
            return None

        standardized: list[list[str]] = []
        for row in rows:
            clipped = [item for item in row[:max_objects]]
            if len(clipped) < max_objects:
                clipped.extend([""] * (max_objects - len(clipped)))
            standardized.append(clipped)
        return standardized

    def _tokenize_rows(self, rows: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(rows)
        num_objects = len(rows[0]) if rows else 0
        flat = [item if item else " " for row in rows for item in row]
        tokenized = self.tokenizer(
            flat,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"].view(batch_size, num_objects, -1)
        masks = tokenized["attention_mask"].view(batch_size, num_objects, -1).to(dtype=torch.bool)
        return tokens, masks

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = dict(new_transition.get(TransitionKey.OBSERVATION, {}) or {})
        rows = self._standardize_text_rows(observation.get(self.text_key))
        if rows is None:
            return new_transition

        tokens, masks = self._tokenize_rows(rows)
        observation[self.tokens_key] = tokens
        observation[self.attention_mask_key] = masks

        if OBS_SEMANTIC_NODE_MASK not in observation:
            observation[OBS_SEMANTIC_NODE_MASK] = torch.tensor(
                [[bool(item) for item in row] for row in rows],
                dtype=torch.bool,
            )

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="s2_pi05_semantic_inputs")
@dataclass
class S2Pi05SemanticInputsProcessorStep(ProcessorStep):
    """Validate, batchify, and lightly normalize semantic inputs for S2-PI05."""

    require_inputs: bool = False
    max_objects: int = 8

    def _ensure_prev_history_rank(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.dim() == 3:
            return tensor.unsqueeze(1)
        return tensor

    def _truncate(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        object_axis_map = {
            OBS_SEMANTIC_NODE_EMBS: 1,
            OBS_SEMANTIC_NODE_MASK: 1,
            OBS_SEMANTIC_VISUAL_EMBS: 1,
            OBS_SEMANTIC_TEXT_TOKENS: 1,
            OBS_SEMANTIC_TEXT_ATTENTION_MASK: 1,
            OBS_SEMANTIC_CF_TEXT_TOKENS: 1,
            OBS_SEMANTIC_CF_TEXT_ATTENTION_MASK: 1,
            OBS_SEMANTIC_BOXES_XYXY: 1,
            OBS_SEMANTIC_TRACK_IDS: 1,
            OBS_SEMANTIC_CAMERA_IDS: 1,
            OBS_SEMANTIC_OCR_CONF: 1,
            OBS_SEMANTIC_PREV_NODE_EMBS: 2,
        }
        object_axis = object_axis_map.get(key)
        if object_axis is None or tensor.dim() <= object_axis or tensor.shape[object_axis] <= self.max_objects:
            return tensor

        index = [slice(None)] * tensor.dim()
        index[object_axis] = slice(0, self.max_objects)
        return tensor[tuple(index)]

    def _infer_num_objects(self, observation: dict[str, Any]) -> int | None:
        candidates = [
            observation.get(OBS_SEMANTIC_NODE_EMBS),
            observation.get(OBS_SEMANTIC_VISUAL_EMBS),
            observation.get(OBS_SEMANTIC_TEXT_TOKENS),
            observation.get(OBS_SEMANTIC_TEXT_ATTENTION_MASK),
            observation.get(OBS_SEMANTIC_CF_TEXT_TOKENS),
            observation.get(OBS_SEMANTIC_BOXES_XYXY),
            observation.get(OBS_SEMANTIC_TRACK_IDS),
            observation.get(OBS_SEMANTIC_CAMERA_IDS),
            observation.get(OBS_SEMANTIC_OCR_CONF),
            observation.get(OBS_SEMANTIC_NODE_MASK),
            observation.get(OBS_SEMANTIC_PREV_NODE_EMBS),
        ]
        for value in candidates:
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() >= 3 and value is observation.get(OBS_SEMANTIC_PREV_NODE_EMBS):
                return int(value.shape[2]) if value.dim() == 4 else int(value.shape[1])
            if value.dim() >= 2:
                return int(value.shape[1])
        return None

    def _infer_node_mask(self, observation: dict[str, Any], num_objects: int) -> torch.Tensor | None:
        node_embs = observation.get(OBS_SEMANTIC_NODE_EMBS)
        visual_embs = observation.get(OBS_SEMANTIC_VISUAL_EMBS)
        boxes = observation.get(OBS_SEMANTIC_BOXES_XYXY)
        text_masks = observation.get(OBS_SEMANTIC_TEXT_ATTENTION_MASK)
        prev_node_embs = observation.get(OBS_SEMANTIC_PREV_NODE_EMBS)

        if isinstance(node_embs, torch.Tensor):
            return torch.ones(node_embs.shape[:2], dtype=torch.bool, device=node_embs.device)
        if isinstance(visual_embs, torch.Tensor):
            return torch.ones(visual_embs.shape[:2], dtype=torch.bool, device=visual_embs.device)
        if isinstance(boxes, torch.Tensor):
            widths = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
            heights = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
            return (widths > 0) & (heights > 0)
        if isinstance(text_masks, torch.Tensor):
            return text_masks.any(dim=-1)
        if isinstance(prev_node_embs, torch.Tensor):
            if prev_node_embs.dim() == 4:
                return prev_node_embs.abs().sum(dim=(-1, -3)) > 0
            if prev_node_embs.dim() == 3:
                return prev_node_embs.abs().sum(dim=-1) > 0

        if num_objects <= 0:
            return None
        return None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = dict(new_transition.get(TransitionKey.OBSERVATION, {}) or {})

        rank_specs = {
            OBS_SEMANTIC_NODE_EMBS: 3,
            OBS_SEMANTIC_NODE_MASK: 2,
            OBS_SEMANTIC_VISUAL_EMBS: 3,
            OBS_SEMANTIC_TEXT_TOKENS: 3,
            OBS_SEMANTIC_TEXT_ATTENTION_MASK: 3,
            OBS_SEMANTIC_CF_TEXT_TOKENS: 3,
            OBS_SEMANTIC_CF_TEXT_ATTENTION_MASK: 3,
            OBS_SEMANTIC_EDGE_INDEX: 3,
            OBS_SEMANTIC_EDGE_ATTR: 3,
            OBS_SEMANTIC_BOXES_XYXY: 3,
            OBS_SEMANTIC_TRACK_IDS: 2,
            OBS_SEMANTIC_CAMERA_IDS: 2,
            OBS_SEMANTIC_PREV_NODE_EMBS: 4,
            OBS_SEMANTIC_TARGET_INDEX: 1,
            OBS_SEMANTIC_OCR_CONF: 2,
        }

        for key, expected_rank in rank_specs.items():
            value = _convert_to_tensor(observation.get(key))
            if not isinstance(value, torch.Tensor):
                continue

            if key == OBS_SEMANTIC_PREV_NODE_EMBS:
                value = self._ensure_prev_history_rank(value)
            else:
                value = _ensure_min_rank(value, expected_rank)

            value = self._truncate(key, value)
            if key in {OBS_SEMANTIC_NODE_MASK, OBS_SEMANTIC_TEXT_ATTENTION_MASK, OBS_SEMANTIC_CF_TEXT_ATTENTION_MASK}:
                value = value.to(dtype=torch.bool)
            if key == OBS_SEMANTIC_TARGET_INDEX:
                value = value.to(dtype=torch.long)
            observation[key] = value

        num_objects = self._infer_num_objects(observation)
        semantic_present = any(
            observation.get(key) is not None
            for key in (
                OBS_SEMANTIC_NODE_EMBS,
                OBS_SEMANTIC_VISUAL_EMBS,
                OBS_SEMANTIC_TEXT_TOKENS,
                OBS_SEMANTIC_TEXT_ATTENTION_MASK,
                OBS_SEMANTIC_BOXES_XYXY,
                OBS_SEMANTIC_PREV_NODE_EMBS,
            )
        )
        if self.require_inputs and not semantic_present:
            raise ValueError(
                "S2-PI05 requires semantic inputs, but none of the expected semantic tensors were found."
            )

        node_mask = observation.get(OBS_SEMANTIC_NODE_MASK)
        if isinstance(node_mask, torch.Tensor):
            node_mask = node_mask.to(dtype=torch.bool)
        elif num_objects is not None:
            node_mask = self._infer_node_mask(observation, num_objects)

        if isinstance(node_mask, torch.Tensor):
            observation[OBS_SEMANTIC_NODE_MASK] = self._truncate(OBS_SEMANTIC_NODE_MASK, node_mask)

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_s2_pi05_pre_post_processors(
    config: S2PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build S2-PI05 processors.

    The pipeline mirrors PI05 and adds:
    - object-level OCR tokenization,
    - semantic tensor batch-shaping,
    - semantic validation.
    """

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        S2Pi05PrepareStateKeywordTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
    ]

    if config.keyword_conditioning_enabled:
        input_steps.append(
            S2Pi05KeywordTokenizerProcessorStep(
                max_length=config.keyword_text_max_tokens,
                keyword_key=KEYWORD_TEXT,
                tokens_key=OBS_LANGUAGE_KEYWORD_TOKENS,
                attention_mask_key=OBS_LANGUAGE_KEYWORD_ATTENTION_MASK,
            )
        )

    if config.semantic_text_enabled:
        input_steps.extend(
            [
                S2Pi05SemanticTextTokenizerProcessorStep(
                    max_length=config.semantic_text_max_tokens,
                    max_objects=config.semantic_max_objects,
                    text_key=OBS_SEMANTIC_OCR_TEXT,
                    tokens_key=OBS_SEMANTIC_TEXT_TOKENS,
                    attention_mask_key=OBS_SEMANTIC_TEXT_ATTENTION_MASK,
                ),
                S2Pi05SemanticTextTokenizerProcessorStep(
                    max_length=config.semantic_text_max_tokens,
                    max_objects=config.semantic_max_objects,
                    text_key=OBS_SEMANTIC_CF_OCR_TEXT,
                    tokens_key=OBS_SEMANTIC_CF_TEXT_TOKENS,
                    attention_mask_key=OBS_SEMANTIC_CF_TEXT_ATTENTION_MASK,
                ),
            ]
        )

    input_steps.extend(
        [
            S2Pi05SemanticInputsProcessorStep(
                require_inputs=config.semantic_require_inputs,
                max_objects=config.semantic_max_objects,
            ),
            DeviceProcessorStep(device=config.device),
        ]
    )

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
