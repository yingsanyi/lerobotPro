#!/usr/bin/env python

from __future__ import annotations

import builtins
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Unpack, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    PI05Pytorch,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK,
    OBS_LANGUAGE_KEYWORD_CF_TOKENS,
    OBS_LANGUAGE_KEYWORD_TOKENS,
    OBS_LANGUAGE_TOKENS,
)

from .configuration_kcvla import KCVLAConfig


def _zeros_like_scalar(reference: Tensor | None, *, device: torch.device | None = None) -> Tensor:
    if reference is not None:
        device = reference.device
    if device is None:
        device = torch.device("cpu")
    return torch.zeros((), device=device, dtype=torch.float32)


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp(min=1.0)
    return (values * weights).sum(dim=dim) / denom.squeeze(dim)


@dataclass
class KeywordConditioningOutput:
    vision_embs: Tensor
    vision_pad_masks: Tensor
    token_counts: list[int]
    keyword_mask: Tensor | None = None
    attn_probs: Tensor | None = None


class KCVLAKeywordVisionCrossAttention(nn.Module):
    """Ground keywords onto visual tokens and inject a lightweight residual."""

    def __init__(self, dim: int, num_heads: int, dropout: float, init_scale: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.delta_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(
        self,
        vision_tokens: Tensor,
        vision_pad_masks: Tensor,
        keyword_queries: Tensor | None,
        keyword_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if keyword_queries is None or keyword_mask is None or not keyword_mask.any():
            return vision_tokens, None, None

        keyword_queries = keyword_queries.to(device=vision_tokens.device, dtype=vision_tokens.dtype)
        keyword_mask = keyword_mask.to(device=vision_tokens.device, dtype=torch.bool)
        vision_pad_masks = vision_pad_masks.to(device=vision_tokens.device, dtype=torch.bool)

        attn_out, attn_weights = self.cross_attn(
            query=keyword_queries,
            key=vision_tokens,
            value=vision_tokens,
            key_padding_mask=~vision_pad_masks,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_probs = attn_weights.mean(dim=1)
        attn_probs = torch.where(keyword_mask.unsqueeze(-1), attn_probs, torch.zeros_like(attn_probs))
        attn_probs = torch.where(vision_pad_masks.unsqueeze(1), attn_probs, torch.zeros_like(attn_probs))
        denom = attn_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn_probs = torch.where(keyword_mask.unsqueeze(-1), attn_probs / denom, torch.zeros_like(attn_probs))

        keyword_context = torch.where(keyword_mask.unsqueeze(-1), attn_out, torch.zeros_like(attn_out))
        active = keyword_mask.any(dim=-1)
        if not active.any():
            return vision_tokens, keyword_context, attn_probs

        aggregated = _masked_mean(keyword_context, keyword_mask.unsqueeze(-1), dim=1)
        delta = self.delta_proj(aggregated).unsqueeze(1)
        scale = torch.tanh(self.residual_scale).to(dtype=delta.dtype)
        updated = vision_tokens.clone()
        active_updated = self.norm(vision_tokens[active] + scale * delta[active])
        active_updated = torch.where(
            vision_pad_masks[active].unsqueeze(-1),
            active_updated,
            vision_tokens[active],
        )
        updated[active] = active_updated
        return updated, keyword_context, attn_probs


class KCVLAPytorch(PI05Pytorch):
    """Keyword-conditioned PI05 backbone."""

    def __init__(self, config: KCVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config = config
        self.model_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        if self.model_dim % config.keyword_cross_attention_heads != 0:
            raise ValueError("model_dim must be divisible by keyword_cross_attention_heads")
        self.keyword_vision_cross_attention = KCVLAKeywordVisionCrossAttention(
            self.model_dim,
            num_heads=config.keyword_cross_attention_heads,
            dropout=config.keyword_cross_attention_dropout,
            init_scale=config.keyword_cross_attention_init_scale,
        )

    def _zero_aux_losses(self, reference: Tensor | None) -> dict[str, Tensor]:
        return {
            "loss_counterfactual": _zeros_like_scalar(reference),
            "loss_contrast": _zeros_like_scalar(reference),
            "loss_sparse": _zeros_like_scalar(reference),
        }

    def _normalize_keyword_inputs(
        self,
        keyword_tokens: Tensor | None,
        keyword_masks: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if keyword_tokens is None or keyword_masks is None:
            return None, None
        if keyword_tokens.dim() == 2:
            keyword_tokens = keyword_tokens.unsqueeze(1)
        if keyword_masks.dim() == 2:
            keyword_masks = keyword_masks.unsqueeze(1)
        return keyword_tokens, keyword_masks.to(dtype=torch.bool)

    def _embed_keyword_set(self, keyword_tokens: Tensor, keyword_masks: Tensor) -> tuple[Tensor, Tensor]:
        keyword_tokens, keyword_masks = self._normalize_keyword_inputs(keyword_tokens, keyword_masks)
        if keyword_tokens is None or keyword_masks is None:
            raise ValueError("keyword_tokens and keyword_masks must both be provided")
        batch_size, keyword_count, seq_len = keyword_tokens.shape
        flat_tokens = keyword_tokens.view(batch_size * keyword_count, seq_len)
        flat_masks = keyword_masks.view(batch_size * keyword_count, seq_len)

        def embed_func(input_tokens: Tensor) -> Tensor:
            keyword_embs = self.paligemma_with_expert.embed_language_tokens(input_tokens)
            return keyword_embs * math.sqrt(keyword_embs.shape[-1])

        flat_embs = self._apply_checkpoint(embed_func, flat_tokens)
        pooled = _masked_mean(flat_embs, flat_masks.unsqueeze(-1), dim=1)
        pooled = pooled.view(batch_size, keyword_count, -1)
        keyword_mask = keyword_masks.any(dim=-1)
        return pooled, keyword_mask

    def _encode_vision(self, images, img_masks) -> tuple[Tensor, Tensor, list[int]]:
        vision_embs = []
        vision_pad_masks = []
        token_counts: list[int] = []
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(image: Tensor) -> Tensor:
                return self.paligemma_with_expert.embed_image(image)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            batch_size, num_tokens = img_emb.shape[:2]
            vision_embs.append(img_emb)
            vision_pad_masks.append(img_mask[:, None].expand(batch_size, num_tokens))
            token_counts.append(num_tokens)

        if not vision_embs:
            raise ValueError("KC-VLA requires at least one image stream")
        return torch.cat(vision_embs, dim=1), torch.cat(vision_pad_masks, dim=1), token_counts

    def build_keyword_conditioning(
        self,
        vision_embs: Tensor,
        vision_pad_masks: Tensor,
        token_counts: list[int],
        *,
        keyword_tokens: Tensor | None = None,
        keyword_masks: Tensor | None = None,
    ) -> KeywordConditioningOutput:
        keyword_tokens, keyword_masks = self._normalize_keyword_inputs(keyword_tokens, keyword_masks)
        if not self.config.keyword_conditioning_enabled or keyword_tokens is None or keyword_masks is None:
            return KeywordConditioningOutput(
                vision_embs=vision_embs,
                vision_pad_masks=vision_pad_masks,
                token_counts=token_counts,
            )

        keyword_queries, keyword_mask = self._embed_keyword_set(keyword_tokens, keyword_masks)
        conditioned_vision, _, attn_probs = self.keyword_vision_cross_attention(
            vision_embs,
            vision_pad_masks,
            keyword_queries,
            keyword_mask,
        )
        return KeywordConditioningOutput(
            vision_embs=conditioned_vision,
            vision_pad_masks=vision_pad_masks,
            token_counts=token_counts,
            keyword_mask=keyword_mask,
            attn_probs=attn_probs,
        )

    def _compose_prefix(
        self,
        vision_embs: Tensor,
        vision_pad_masks: Tensor,
        tokens: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        def lang_embed_func(input_tokens: Tensor) -> Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(input_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_embs = self._apply_checkpoint(lang_embed_func, tokens)
        pad_masks = torch.cat([vision_pad_masks, masks.to(dtype=torch.bool)], dim=1)
        embs = torch.cat([vision_embs, lang_embs], dim=1)
        att_masks = torch.zeros(embs.shape[:2], dtype=torch.bool, device=embs.device)
        return embs, pad_masks, att_masks

    def _forward_with_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        x_t: Tensor,
        time: Tensor,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, attn_masks, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)

        def action_out_proj_func(hidden: Tensor) -> Tensor:
            return self.action_out_proj(hidden)

        return self._apply_checkpoint(action_out_proj_func, suffix_out)

    def _compute_sparse_loss(
        self,
        attn_probs: Tensor | None,
        keyword_mask: Tensor | None,
        *,
        reference: Tensor,
    ) -> Tensor:
        if attn_probs is None or keyword_mask is None:
            return _zeros_like_scalar(reference)
        if keyword_mask.sum() == 0:
            return _zeros_like_scalar(reference)
        entropy = -(attn_probs.clamp(min=1e-6).log() * attn_probs).sum(dim=-1)
        return entropy[keyword_mask].mean()

    def _compute_keyword_contrast_loss(
        self,
        attn_probs: Tensor | None,
        keyword_mask: Tensor | None,
        *,
        reference: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        stats = {"keyword_pairwise_l2": _zeros_like_scalar(reference)}
        if attn_probs is None or keyword_mask is None:
            return _zeros_like_scalar(reference), stats
        if keyword_mask.sum() < 2:
            return _zeros_like_scalar(reference), stats

        pairwise_valid = keyword_mask.unsqueeze(2) & keyword_mask.unsqueeze(1)
        if pairwise_valid.shape[1] == 0:
            return _zeros_like_scalar(reference), stats
        diagonal = torch.eye(pairwise_valid.shape[1], device=pairwise_valid.device, dtype=torch.bool).unsqueeze(0)
        pairwise_valid = pairwise_valid & ~diagonal
        if pairwise_valid.sum() == 0:
            return _zeros_like_scalar(reference), stats

        pairwise_dist_sq = (attn_probs.unsqueeze(2) - attn_probs.unsqueeze(1)).square().sum(dim=-1)
        stats["keyword_pairwise_l2"] = pairwise_dist_sq[pairwise_valid].sqrt().mean()
        return -pairwise_dist_sq[pairwise_valid].mean(), stats

    def _compute_counterfactual_loss(
        self,
        primary_mask: Tensor | None,
        counterfactual_mask: Tensor | None,
        primary_flow: Tensor,
        counterfactual_flow: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        stats = {"counterfactual_action_delta": _zeros_like_scalar(primary_flow)}
        if primary_mask is None or counterfactual_mask is None:
            return _zeros_like_scalar(primary_flow), stats

        valid = primary_mask.any(dim=-1) & counterfactual_mask.any(dim=-1)
        if valid.sum() == 0:
            return _zeros_like_scalar(primary_flow), stats

        action_delta = (primary_flow[valid] - counterfactual_flow[valid]).square().mean(dim=(1, 2)).sqrt()
        stats["counterfactual_action_delta"] = action_delta.mean()
        return F.relu(self.config.counterfactual_action_margin - action_delta).mean(), stats

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise=None,
        time=None,
        keyword_tokens: Tensor | None = None,
        keyword_masks: Tensor | None = None,
        counterfactual_keyword_tokens: Tensor | None = None,
        counterfactual_keyword_masks: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor] | None]:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        vision_embs, vision_pad_masks, token_counts = self._encode_vision(images, img_masks)
        conditioning = self.build_keyword_conditioning(
            vision_embs,
            vision_pad_masks,
            token_counts,
            keyword_tokens=keyword_tokens,
            keyword_masks=keyword_masks,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._compose_prefix(
            conditioning.vision_embs,
            conditioning.vision_pad_masks,
            tokens,
            masks,
        )
        primary_flow = self._forward_with_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks, x_t, time)
        losses = F.mse_loss(u_t, primary_flow, reduction="none")

        aux_losses = self._zero_aux_losses(losses)
        aux_losses["loss_sparse"] = self._compute_sparse_loss(
            conditioning.attn_probs,
            conditioning.keyword_mask,
            reference=losses,
        )
        aux_losses["loss_contrast"], contrast_metrics = self._compute_keyword_contrast_loss(
            conditioning.attn_probs,
            conditioning.keyword_mask,
            reference=losses,
        )

        metrics = {
            "keyword_entropy": aux_losses["loss_sparse"].detach(),
            "keyword_pairwise_l2": contrast_metrics["keyword_pairwise_l2"].detach(),
            "counterfactual_action_delta": _zeros_like_scalar(losses),
        }

        if (
            self.config.counterfactual_enabled
            and counterfactual_keyword_tokens is not None
            and counterfactual_keyword_masks is not None
        ):
            counterfactual = self.build_keyword_conditioning(
                vision_embs,
                vision_pad_masks,
                token_counts,
                keyword_tokens=counterfactual_keyword_tokens,
                keyword_masks=counterfactual_keyword_masks,
            )
            cf_prefix_embs, cf_prefix_pad_masks, cf_prefix_att_masks = self._compose_prefix(
                counterfactual.vision_embs,
                counterfactual.vision_pad_masks,
                tokens,
                masks,
            )
            counterfactual_flow = self._forward_with_prefix(
                cf_prefix_embs,
                cf_prefix_pad_masks,
                cf_prefix_att_masks,
                x_t,
                time,
            )
            aux_losses["loss_counterfactual"], cf_metrics = self._compute_counterfactual_loss(
                conditioning.keyword_mask,
                counterfactual.keyword_mask,
                primary_flow,
                counterfactual_flow,
            )
            metrics.update(cf_metrics)

        return {
            "policy_losses": losses,
            "aux_losses": aux_losses,
            "metrics": metrics,
            "keyword_attention": conditioning.attn_probs,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        keyword_tokens: Tensor | None = None,
        keyword_masks: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        batch_size = tokens.shape[0]
        device = tokens.device
        if noise is None:
            actions_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        vision_embs, vision_pad_masks, token_counts = self._encode_vision(images, img_masks)
        conditioning = self.build_keyword_conditioning(
            vision_embs,
            vision_pad_masks,
            token_counts,
            keyword_tokens=keyword_tokens,
            keyword_masks=keyword_masks,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._compose_prefix(
            conditioning.vision_embs,
            conditioning.vision_pad_masks,
            tokens,
            masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            current_time = 1.0 + step * dt
            time_tensor = torch.tensor(current_time, dtype=torch.float32, device=device).expand(batch_size)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=current_time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=current_time, x_t=x_t, v_t=v_t)
        return x_t


class KCVLAPolicy(PI05Policy):
    """Keyword-conditioned PI05 policy."""

    config_class = KCVLAConfig
    name = "kcvla"

    def __init__(self, config: KCVLAConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = KCVLAPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @staticmethod
    def _coerce_pretrained_config(config: PreTrainedConfig) -> KCVLAConfig:
        if isinstance(config, KCVLAConfig):
            return config
        if isinstance(config, PI05Config):
            print(
                "Upgrading PI05Config to KCVLAConfig to load compatible backbone weights. "
                "New grounding modules will remain randomly initialized."
            )
            return KCVLAConfig.from_pi05_config(config)
        raise TypeError(
            "KCVLAPolicy.from_pretrained supports KCVLAConfig checkpoints and PI05-compatible checkpoints."
        )

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        config = cls._coerce_pretrained_config(config)
        return cast(
            T,
            super().from_pretrained(
                pretrained_name_or_path,
                config=config,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=strict,
                **kwargs,
            ),
        )

    def _extract_keyword_batch(
        self,
        batch: dict[str, Any],
        *,
        counterfactual: bool = False,
    ) -> tuple[Tensor | None, Tensor | None]:
        tokens_key = OBS_LANGUAGE_KEYWORD_CF_TOKENS if counterfactual else OBS_LANGUAGE_KEYWORD_TOKENS
        masks_key = (
            OBS_LANGUAGE_KEYWORD_CF_ATTENTION_MASK
            if counterfactual
            else OBS_LANGUAGE_KEYWORD_ATTENTION_MASK
        )
        keyword_tokens = batch.get(tokens_key)
        keyword_masks = batch.get(masks_key)
        if not isinstance(keyword_tokens, Tensor) or not isinstance(keyword_masks, Tensor):
            return None, None
        return keyword_tokens, keyword_masks

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        keyword_tokens, keyword_masks = self._extract_keyword_batch(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            keyword_tokens=keyword_tokens,
            keyword_masks=keyword_masks,
            **kwargs,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        keyword_tokens, keyword_masks = self._extract_keyword_batch(batch)
        counterfactual_keyword_tokens, counterfactual_keyword_masks = self._extract_keyword_batch(
            batch,
            counterfactual=True,
        )

        model_output = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            keyword_tokens=keyword_tokens,
            keyword_masks=keyword_masks,
            counterfactual_keyword_tokens=counterfactual_keyword_tokens,
            counterfactual_keyword_masks=counterfactual_keyword_masks,
        )
        losses = model_output["policy_losses"]
        aux_losses = model_output["aux_losses"]
        metrics = model_output.get("metrics", {})
        keyword_attention = model_output.get("keyword_attention")

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        policy_loss = losses.mean()

        aux_total = torch.zeros((), device=policy_loss.device)
        aux_weights = {
            "loss_counterfactual": self.config.loss_counterfactual_w,
            "loss_contrast": self.config.loss_contrast_w,
            "loss_sparse": self.config.loss_sparse_w,
        }
        for key, weight in aux_weights.items():
            value = aux_losses.get(key)
            if value is None or weight <= 0:
                continue
            aux_total = aux_total + weight * value.to(device=policy_loss.device)

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "policy_loss": policy_loss.item(),
            "aux_loss": aux_total.item(),
        }
        if keyword_attention is not None:
            keyword_mask = batch.get(OBS_LANGUAGE_KEYWORD_ATTENTION_MASK)
            if isinstance(keyword_mask, Tensor):
                keyword_mask = keyword_mask.any(dim=-1)
                if keyword_mask.any():
                    entropy = -(keyword_attention.clamp(min=1e-6).log() * keyword_attention).sum(dim=-1)
                    loss_dict["keyword_entropy"] = entropy[keyword_mask].mean().detach().cpu().item()
        for key, value in aux_losses.items():
            if isinstance(value, Tensor):
                loss_dict[key] = value.detach().cpu().item()
        for key, value in metrics.items():
            if isinstance(value, Tensor):
                loss_dict[key] = value.detach().cpu().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = (per_sample_loss.mean() + aux_total).item()
            return per_sample_loss, loss_dict

        total_loss = policy_loss + aux_total
        loss_dict["loss"] = total_loss.item()
        return total_loss, loss_dict
