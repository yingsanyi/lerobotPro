#!/usr/bin/env python

import torch

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.kcvla.configuration_kcvla import KCVLAConfig
from lerobot.policies.kcvla.modeling_kcvla import KCVLAKeywordVisionCrossAttention, KCVLAPolicy
from lerobot.policies.kcvla.processor_kcvla import KCVLAKeywordSetTokenizerProcessorStep
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor.converters import batch_to_transition
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import (
    COUNTERFACTUAL_KEYWORD_TEXT,
    KEYWORD_TEXT,
    OBS_LANGUAGE_KEYWORD_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_CF_TOKENS,
    OBS_LANGUAGE_KEYWORD_TOKENS,
)


class _DummyTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        batch_size = len(texts)
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            token_count = min(len(text.replace(",", " ").split()), max_length)
            if token_count > 0:
                input_ids[i, :token_count] = torch.arange(1, token_count + 1)
                attention_mask[i, :token_count] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_factory_registration_for_kcvla():
    cfg = make_policy_config("kcvla", device="cpu")
    assert isinstance(cfg, KCVLAConfig)
    assert cfg.type == "kcvla"
    assert get_policy_class("kcvla").__name__ == "KCVLAPolicy"


def test_kcvla_config_can_upgrade_from_pi05_config():
    pi05_cfg = PI05Config(device="cpu", chunk_size=12, n_action_steps=6, max_action_dim=9, max_state_dim=18)
    kcvla_cfg = KCVLAConfig.from_pi05_config(pi05_cfg)
    assert isinstance(kcvla_cfg, KCVLAConfig)
    assert kcvla_cfg.chunk_size == 12
    assert kcvla_cfg.n_action_steps == 6
    assert kcvla_cfg.max_action_dim == 9
    assert kcvla_cfg.max_state_dim == 18
    assert kcvla_cfg.keyword_conditioning_enabled is True
    assert kcvla_cfg.counterfactual_auto_generate is True
    assert kcvla_cfg.loss_contrast_w > 0
    assert kcvla_cfg.loss_align_w == 0.0


def test_kcvla_policy_coerces_pi05_pretrained_config():
    pi05_cfg = PI05Config(device="cpu", chunk_size=10, n_action_steps=5)
    coerced = KCVLAPolicy._coerce_pretrained_config(pi05_cfg)
    assert isinstance(coerced, KCVLAConfig)
    assert coerced.chunk_size == 10
    assert coerced.n_action_steps == 5


def test_batch_to_transition_keeps_counterfactual_keyword_text():
    transition = batch_to_transition(
        {
            "task": "sort medicine by label",
            KEYWORD_TEXT: "painkiller, vitamin",
            COUNTERFACTUAL_KEYWORD_TEXT: "vitamin, painkiller",
            "observation.state": torch.zeros(8),
        }
    )
    complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary_data[KEYWORD_TEXT] == "painkiller, vitamin"
    assert complementary_data[COUNTERFACTUAL_KEYWORD_TEXT] == "vitamin, painkiller"


def test_keyword_tokenizer_produces_keyword_set_and_counterfactual_branches():
    step = KCVLAKeywordSetTokenizerProcessorStep(tokenizer=_DummyTokenizer(), max_length=4, max_keywords=3)
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            KEYWORD_TEXT: ["painkiller, vitamin", ""],
            COUNTERFACTUAL_KEYWORD_TEXT: ["vitamin, painkiller", "aspirin"],
        },
    }
    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    assert obs[OBS_LANGUAGE_KEYWORD_TOKENS].shape == (2, 3, 4)
    assert obs[OBS_LANGUAGE_KEYWORD_ATTENTION_MASK].shape == (2, 3, 4)
    assert obs[OBS_LANGUAGE_KEYWORD_ATTENTION_MASK][0, 0].any()
    assert obs[OBS_LANGUAGE_KEYWORD_ATTENTION_MASK][0, 1].any()
    assert not obs[OBS_LANGUAGE_KEYWORD_ATTENTION_MASK][1].any()
    assert obs[OBS_LANGUAGE_KEYWORD_CF_TOKENS].shape == (2, 3, 4)
    assert obs[OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK][1, 0].any()


def test_keyword_tokenizer_prefers_manual_counterfactual_keywords():
    step = KCVLAKeywordSetTokenizerProcessorStep(
        tokenizer=_DummyTokenizer(),
        max_length=4,
        max_keywords=3,
        counterfactual_enabled=True,
        counterfactual_auto_generate=True,
    )
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            KEYWORD_TEXT: ["painkiller, vitamin"],
            COUNTERFACTUAL_KEYWORD_TEXT: ["manual override"],
        },
    }

    out = step(transition)
    cf_mask = out[TransitionKey.OBSERVATION][OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK]
    assert cf_mask[0, 0].sum().item() == 2
    assert not cf_mask[0, 1].any()


def test_keyword_tokenizer_auto_generates_counterfactual_keywords_when_missing():
    step = KCVLAKeywordSetTokenizerProcessorStep(
        tokenizer=_DummyTokenizer(),
        max_length=4,
        max_keywords=3,
        counterfactual_enabled=True,
        counterfactual_auto_generate=True,
    )
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            KEYWORD_TEXT: ["painkiller, vitamin"],
        },
    }

    out = step(transition)
    cf_mask = out[TransitionKey.OBSERVATION][OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK]
    assert cf_mask[0, 0].any()
    assert cf_mask[0, 1].any()


def test_keyword_tokenizer_skips_counterfactual_branch_when_disabled():
    step = KCVLAKeywordSetTokenizerProcessorStep(
        tokenizer=_DummyTokenizer(),
        max_length=4,
        max_keywords=3,
        counterfactual_enabled=False,
        counterfactual_auto_generate=True,
    )
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            KEYWORD_TEXT: ["painkiller, vitamin"],
            COUNTERFACTUAL_KEYWORD_TEXT: ["manual override"],
        },
    }

    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    assert OBS_LANGUAGE_KEYWORD_CF_TOKENS not in obs
    assert OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK not in obs


def test_kcvla_uses_weak_supervision_objectives_by_default():
    cfg = KCVLAConfig(device="cpu")
    assert cfg.counterfactual_enabled is True
    assert cfg.counterfactual_auto_generate is True
    assert cfg.loss_contrast_w > 0
    assert cfg.loss_counterfactual_w > 0
    assert cfg.loss_sparse_w > 0
    assert cfg.loss_align_w == 0.0
    assert cfg.keyword_require_boxes is False


def test_keyword_vision_cross_attention_returns_attention_maps():
    module = KCVLAKeywordVisionCrossAttention(dim=32, num_heads=4, dropout=0.0, init_scale=0.5)
    vision = torch.randn(2, 9, 32)
    vision_pad_masks = torch.ones(2, 9, dtype=torch.bool)
    keyword_queries = torch.randn(2, 3, 32)
    keyword_mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)

    out, _, attn = module(vision, vision_pad_masks, keyword_queries, keyword_mask)
    assert out.shape == vision.shape
    assert attn.shape == (2, 3, 9)
    assert not torch.allclose(out[0], vision[0])
    assert torch.allclose(out[1], vision[1])
    assert torch.allclose(attn[0, 0].sum(), torch.tensor(1.0), atol=1e-5)
    assert not attn[1].any()
