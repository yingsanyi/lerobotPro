#!/usr/bin/env python

import torch

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.s2_pi05.configuration_s2_pi05 import S2PI05Config
from lerobot.policies.s2_pi05.modeling_s2_pi05 import (
    S2GraphEncoder,
    S2GroundingHead,
    S2KeywordVisionCrossAttention,
    S2PI05Policy,
    S2SemanticFrontend,
)
from lerobot.policies.s2_pi05.processor_s2_pi05 import (
    S2Pi05KeywordTokenizerProcessorStep,
    S2Pi05PrepareStateKeywordTokenizerProcessorStep,
    S2Pi05SemanticInputsProcessorStep,
)
from lerobot.processor.converters import batch_to_transition
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import (
    KEYWORD_TEXT,
    OBS_LANGUAGE_KEYWORD_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_TOKENS,
    OBS_SEMANTIC_BOXES_XYXY,
    OBS_SEMANTIC_NODE_EMBS,
    OBS_SEMANTIC_NODE_MASK,
    OBS_SEMANTIC_OCR_CONF,
    OBS_SEMANTIC_PREV_NODE_EMBS,
    OBS_SEMANTIC_TEXT_ATTENTION_MASK,
    OBS_SEMANTIC_TEXT_TOKENS,
    OBS_SEMANTIC_VISUAL_EMBS,
)


def test_factory_registration_for_s2_pi05():
    cfg = make_policy_config("s2_pi05", device="cpu")
    assert isinstance(cfg, S2PI05Config)
    assert cfg.type == "s2_pi05"
    assert get_policy_class("s2_pi05").__name__ == "S2PI05Policy"


def test_s2_config_can_upgrade_from_pi05_config():
    pi05_cfg = PI05Config(device="cpu", chunk_size=12, n_action_steps=6, max_action_dim=9, max_state_dim=18)
    s2_cfg = S2PI05Config.from_pi05_config(pi05_cfg)
    assert isinstance(s2_cfg, S2PI05Config)
    assert s2_cfg.chunk_size == 12
    assert s2_cfg.n_action_steps == 6
    assert s2_cfg.max_action_dim == 9
    assert s2_cfg.max_state_dim == 18
    assert s2_cfg.semantic_enabled is True


def test_s2_policy_coerces_pi05_pretrained_config():
    pi05_cfg = PI05Config(device="cpu", chunk_size=10, n_action_steps=5)
    coerced = S2PI05Policy._coerce_pretrained_config(pi05_cfg)
    assert isinstance(coerced, S2PI05Config)
    assert coerced.chunk_size == 10
    assert coerced.n_action_steps == 5


def test_semantic_inputs_processor_adds_batch_and_mask_from_nested_lists():
    step = S2Pi05SemanticInputsProcessorStep(max_objects=4)
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_SEMANTIC_BOXES_XYXY: [[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]],
            OBS_SEMANTIC_OCR_CONF: [0.9, 0.1],
        }
    }
    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    assert isinstance(obs[OBS_SEMANTIC_BOXES_XYXY], torch.Tensor)
    assert obs[OBS_SEMANTIC_BOXES_XYXY].shape == (1, 2, 4)
    assert obs[OBS_SEMANTIC_OCR_CONF].shape == (1, 2)
    assert obs[OBS_SEMANTIC_NODE_MASK].shape == (1, 2)
    assert obs[OBS_SEMANTIC_NODE_MASK].dtype == torch.bool
    assert obs[OBS_SEMANTIC_NODE_MASK].all()


def test_semantic_frontend_accepts_raw_visual_and_text_inputs():
    config = S2PI05Config(device="cpu", semantic_node_dim=32, semantic_attn_heads=4)
    frontend = S2SemanticFrontend(config)

    semantic_batch = {
        OBS_SEMANTIC_VISUAL_EMBS: torch.randn(2, 3, 12),
        OBS_SEMANTIC_TEXT_TOKENS: torch.randint(0, 32, (2, 3, 5)),
        OBS_SEMANTIC_TEXT_ATTENTION_MASK: torch.ones(2, 3, 5, dtype=torch.bool),
        OBS_SEMANTIC_OCR_CONF: torch.rand(2, 3),
        OBS_SEMANTIC_BOXES_XYXY: torch.rand(2, 3, 4),
        OBS_SEMANTIC_NODE_MASK: torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
    }

    def embed_language_tokens(tokens):
        eye = torch.eye(16, dtype=torch.float32, device=tokens.device)
        return eye[(tokens % 16).long()]

    output = frontend(
        images=[],
        semantic_batch=semantic_batch,
        embed_image=lambda image: image,
        embed_language_tokens=embed_language_tokens,
    )
    assert output.node_embs is not None
    assert output.node_embs.shape == (2, 3, config.semantic_node_dim)
    assert output.node_mask is not None
    assert output.aux_losses["loss_text_align"].ndim == 0


def test_semantic_frontend_supports_temporal_history():
    config = S2PI05Config(
        device="cpu",
        semantic_node_dim=32,
        semantic_attn_heads=4,
        semantic_temporal_fusion="attn",
    )
    frontend = S2SemanticFrontend(config)

    semantic_batch = {
        OBS_SEMANTIC_NODE_EMBS: torch.randn(2, 3, 10),
        OBS_SEMANTIC_PREV_NODE_EMBS: torch.randn(2, 2, 3, 32),
        OBS_SEMANTIC_NODE_MASK: torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
    }

    output = frontend(
        images=[],
        semantic_batch=semantic_batch,
        embed_image=lambda image: image,
        embed_language_tokens=lambda tokens: tokens.float(),
    )
    assert output.node_embs is not None
    assert output.node_embs.shape == (2, 3, 32)
    assert output.aux_losses["loss_temporal"].ndim == 0


def test_graph_encoder_and_grounding_head_shapes():
    config = S2PI05Config(device="cpu", semantic_node_dim=32, semantic_attn_heads=4)
    encoder = S2GraphEncoder(config)
    grounding = S2GroundingHead(query_dim=32, node_dim=32)

    node_embs = torch.randn(2, 4, 32)
    node_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    boxes = torch.rand(2, 4, 4)
    encoded = encoder(node_embs, node_mask, boxes=boxes)
    assert encoded.shape == node_embs.shape

    query = torch.randn(2, 32)
    target_embs, logits, probs = grounding(
        query,
        encoded,
        node_mask,
        temperature=1.0,
        mode="soft",
        training=True,
    )
    assert target_embs.shape == (2, 32)
    assert logits.shape == (2, 4)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


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


def test_batch_to_transition_keeps_keyword_text():
    transition = batch_to_transition(
        {
            "task": "将止痛药放在左边的盒子，将维生素放在右边的盒子",
            KEYWORD_TEXT: "止痛药, 维生素",
            "observation.state": torch.zeros(8),
        }
    )
    assert transition[TransitionKey.COMPLEMENTARY_DATA][KEYWORD_TEXT] == "止痛药, 维生素"


def test_prepare_state_prompt_includes_keyword_text():
    step = S2Pi05PrepareStateKeywordTokenizerProcessorStep(max_state_dim=8)
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": torch.zeros(1, 8)},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["将止痛药放在左边的盒子，将维生素放在右边的盒子"],
            KEYWORD_TEXT: ["止痛药, 维生素"],
        },
    }
    out = step(transition)
    prompt = out[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    assert "Key text: 止痛药, 维生素" in prompt
    assert prompt.endswith("\nAction: ")


def test_keyword_tokenizer_populates_keyword_observation_keys():
    step = S2Pi05KeywordTokenizerProcessorStep(tokenizer=_DummyTokenizer(), max_length=4)
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {
            KEYWORD_TEXT: ["止痛药, 维生素", ""],
        },
    }
    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    assert obs[OBS_LANGUAGE_KEYWORD_TOKENS].shape == (2, 4)
    assert obs[OBS_LANGUAGE_KEYWORD_ATTENTION_MASK].shape == (2, 4)
    assert obs[OBS_LANGUAGE_KEYWORD_ATTENTION_MASK][0].any()
    assert not obs[OBS_LANGUAGE_KEYWORD_ATTENTION_MASK][1].any()


def test_keyword_vision_cross_attention_is_conditional():
    module = S2KeywordVisionCrossAttention(dim=32, num_heads=4, dropout=0.0, init_scale=0.5)
    vision = torch.randn(2, 6, 32)
    vision_pad_masks = torch.ones(2, 6, dtype=torch.bool)
    keyword_tokens = torch.randn(2, 3, 32)
    keyword_pad_masks = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)

    out = module(vision, vision_pad_masks, keyword_tokens, keyword_pad_masks)
    assert out.shape == vision.shape
    assert not torch.allclose(out[0], vision[0])
    assert torch.allclose(out[1], vision[1])
