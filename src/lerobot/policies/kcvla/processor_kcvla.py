#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    COUNTERFACTUAL_KEYWORD_TEXT,
    KEYWORD_TEXT,
    OBS_LANGUAGE_KEYWORD_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_CF_TOKENS,
    OBS_LANGUAGE_KEYWORD_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_kcvla import KCVLAConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


def _normalize_text_value(raw_text: Any) -> Any:
    if raw_text is None:
        return None
    if hasattr(raw_text, "tolist") and not isinstance(raw_text, str):
        return raw_text.tolist()
    return raw_text


def _clean_text(item: Any) -> str:
    if item is None:
        return ""
    return str(item).replace("\n", " ").replace("\t", " ").strip()


def _split_keyword_items(value: Any) -> list[str]:
    value = _clean_text(value)
    if not value:
        return []
    for delimiter in ("，", ";", "；", "|", "\n"):
        value = value.replace(delimiter, ",")
    return [item for item in (_clean_text(part) for part in value.split(",")) if item]


def _normalize_keyword_items(raw_keywords: Any) -> list[str]:
    raw_keywords = _normalize_text_value(raw_keywords)
    if raw_keywords is None:
        return []
    if isinstance(raw_keywords, str):
        return _split_keyword_items(raw_keywords)
    if isinstance(raw_keywords, (list, tuple)):
        items: list[str] = []
        for item in raw_keywords:
            if isinstance(item, (list, tuple)):
                items.extend(_normalize_keyword_items(item))
                continue
            cleaned = _clean_text(item)
            if not cleaned:
                continue
            if any(delimiter in cleaned for delimiter in (",", "，", ";", "；", "|", "\n")):
                items.extend(_split_keyword_items(cleaned))
            else:
                items.append(cleaned)
        return items
    cleaned = _clean_text(raw_keywords)
    return [cleaned] if cleaned else []


def _standardize_keyword_batch(raw_keywords: Any) -> list[list[str]] | None:
    raw_keywords = _normalize_text_value(raw_keywords)
    if raw_keywords is None:
        return None
    if isinstance(raw_keywords, str):
        return [_normalize_keyword_items(raw_keywords)]
    if isinstance(raw_keywords, (list, tuple)):
        if len(raw_keywords) == 0:
            return [[]]
        return [_normalize_keyword_items(sample) for sample in raw_keywords]
    return None


def _permute_counterfactual_keywords(keywords: list[str]) -> list[str]:
    if len(keywords) < 2 or len(set(keywords)) < 2:
        return []
    rotated_keywords = keywords[1:] + keywords[:1]
    if rotated_keywords != keywords:
        return rotated_keywords
    reversed_keywords = list(reversed(keywords))
    if reversed_keywords != keywords:
        return reversed_keywords
    return []


def _validate_keyword_batch_size(
    batch_keywords: list[list[str]] | None,
    *,
    expected_batch_size: int,
    field_name: str,
):
    if batch_keywords is None:
        return
    if len(batch_keywords) != expected_batch_size:
        raise ValueError(
            f"Expected '{field_name}' to contain {expected_batch_size} samples, got {len(batch_keywords)}"
        )


@ProcessorStepRegistry.register(name="kcvla_keyword_set_tokenizer")
@dataclass
class KCVLAKeywordSetTokenizerProcessorStep(ProcessorStep):
    """Tokenize keyword sets into a dedicated per-keyword tensor branch."""

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    tokenizer: Any | None = None
    max_length: int = 16
    max_keywords: int = 8
    keyword_key: str = KEYWORD_TEXT
    counterfactual_keyword_key: str = COUNTERFACTUAL_KEYWORD_TEXT
    counterfactual_enabled: bool = True
    counterfactual_auto_generate: bool = True
    tokens_key: str = OBS_LANGUAGE_KEYWORD_TOKENS
    attention_mask_key: str = OBS_LANGUAGE_KEYWORD_ATTENTION_MASK
    counterfactual_tokens_key: str = OBS_LANGUAGE_KEYWORD_CF_TOKENS
    counterfactual_attention_mask_key: str = OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK
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

    def _tokenize_keyword_batch(self, batch_keywords: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch_keywords)
        flattened_keywords: list[str] = []
        presence = torch.zeros(batch_size, self.max_keywords, dtype=torch.bool)
        for batch_index, sample_keywords in enumerate(batch_keywords):
            clipped_keywords = sample_keywords[: self.max_keywords]
            padded_keywords = clipped_keywords + [""] * (self.max_keywords - len(clipped_keywords))
            for keyword_index, keyword in enumerate(padded_keywords):
                if keyword:
                    presence[batch_index, keyword_index] = True
                flattened_keywords.append(keyword if keyword else " ")

        tokenized = self.input_tokenizer(
            flattened_keywords,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].view(batch_size, self.max_keywords, self.max_length)
        attention_mask = tokenized["attention_mask"].view(batch_size, self.max_keywords, self.max_length)
        attention_mask = attention_mask.to(dtype=torch.bool) & presence.unsqueeze(-1)
        return input_ids, attention_mask

    def _resolve_counterfactual_keywords(
        self,
        batch_keywords: list[list[str]],
        manual_counterfactual_keywords: list[list[str]] | None,
    ) -> list[list[str]] | None:
        if not self.counterfactual_enabled:
            return None

        resolved_keywords: list[list[str]] = []
        has_counterfactual_keywords = False
        for batch_index, sample_keywords in enumerate(batch_keywords):
            manual_keywords = []
            if manual_counterfactual_keywords is not None:
                manual_keywords = manual_counterfactual_keywords[batch_index]

            if manual_keywords:
                resolved_keywords.append(manual_keywords)
                has_counterfactual_keywords = True
                continue

            if self.counterfactual_auto_generate:
                auto_keywords = _permute_counterfactual_keywords(sample_keywords)
                resolved_keywords.append(auto_keywords)
                has_counterfactual_keywords = has_counterfactual_keywords or bool(auto_keywords)
                continue

            resolved_keywords.append([])

        if not has_counterfactual_keywords:
            return None
        return resolved_keywords

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        observation = dict(new_transition.get(TransitionKey.OBSERVATION, {}) or {})

        batch_keywords = _standardize_keyword_batch(complementary_data.get(self.keyword_key))
        if batch_keywords is not None:
            input_ids, attention_mask = self._tokenize_keyword_batch(batch_keywords)
            observation[self.tokens_key] = input_ids
            observation[self.attention_mask_key] = attention_mask

        manual_counterfactual_keywords = _standardize_keyword_batch(
            complementary_data.get(self.counterfactual_keyword_key)
        )
        if batch_keywords is not None:
            _validate_keyword_batch_size(
                manual_counterfactual_keywords,
                expected_batch_size=len(batch_keywords),
                field_name=self.counterfactual_keyword_key,
            )
            resolved_counterfactual_keywords = self._resolve_counterfactual_keywords(
                batch_keywords,
                manual_counterfactual_keywords,
            )
        else:
            resolved_counterfactual_keywords = None

        if resolved_counterfactual_keywords is not None:
            cf_input_ids, cf_attention_mask = self._tokenize_keyword_batch(resolved_counterfactual_keywords)
            observation[self.counterfactual_tokens_key] = cf_input_ids
            observation[self.counterfactual_attention_mask_key] = cf_attention_mask
        else:
            observation.pop(self.counterfactual_tokens_key, None)
            observation.pop(self.counterfactual_attention_mask_key, None)

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_kcvla_pre_post_processors(
    config: KCVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build KC-VLA processors without modifying the base PI05 prompt path."""

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
    ]

    if config.keyword_conditioning_enabled:
        input_steps.append(
            KCVLAKeywordSetTokenizerProcessorStep(
                max_length=config.keyword_text_max_tokens,
                max_keywords=config.keyword_max_count,
                keyword_key=KEYWORD_TEXT,
                counterfactual_keyword_key=COUNTERFACTUAL_KEYWORD_TEXT,
                counterfactual_enabled=config.counterfactual_enabled,
                counterfactual_auto_generate=config.counterfactual_auto_generate,
                tokens_key=OBS_LANGUAGE_KEYWORD_TOKENS,
                attention_mask_key=OBS_LANGUAGE_KEYWORD_ATTENTION_MASK,
                counterfactual_tokens_key=OBS_LANGUAGE_KEYWORD_CF_TOKENS,
                counterfactual_attention_mask_key=OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK,
            )
        )

    input_steps.append(DeviceProcessorStep(device=config.device))

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
