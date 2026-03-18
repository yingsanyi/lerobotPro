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

import builtins
import math
from typing import cast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Unpack

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
    OBS_LANGUAGE_KEYWORD_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_SEMANTIC,
    OBS_SEMANTIC_BOXES_XYXY,
    OBS_SEMANTIC_CAMERA_IDS,
    OBS_SEMANTIC_CF_TEXT_ATTENTION_MASK,
    OBS_SEMANTIC_CF_TEXT_TOKENS,
    OBS_SEMANTIC_EDGE_ATTR,
    OBS_SEMANTIC_EDGE_INDEX,
    OBS_SEMANTIC_NODE_EMBS,
    OBS_SEMANTIC_NODE_MASK,
    OBS_SEMANTIC_OCR_CONF,
    OBS_SEMANTIC_PREV_NODE_EMBS,
    OBS_SEMANTIC_TARGET_INDEX,
    OBS_SEMANTIC_TEXT_ATTENTION_MASK,
    OBS_SEMANTIC_TEXT_TOKENS,
    OBS_SEMANTIC_TRACK_IDS,
    OBS_SEMANTIC_VISUAL_EMBS,
)

from .configuration_s2_pi05 import S2PI05Config


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


def _pairwise_box_features(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    area = w * h

    dx = cx.unsqueeze(2) - cx.unsqueeze(1)
    dy = cy.unsqueeze(2) - cy.unsqueeze(1)
    log_w = torch.log(w.unsqueeze(2) / w.unsqueeze(1).clamp(min=1e-6))
    log_h = torch.log(h.unsqueeze(2) / h.unsqueeze(1).clamp(min=1e-6))
    dist = torch.sqrt(dx.square() + dy.square() + 1e-6)

    inter_x1 = torch.maximum(x1.unsqueeze(2), x1.unsqueeze(1))
    inter_y1 = torch.maximum(y1.unsqueeze(2), y1.unsqueeze(1))
    inter_x2 = torch.minimum(x2.unsqueeze(2), x2.unsqueeze(1))
    inter_y2 = torch.minimum(y2.unsqueeze(2), y2.unsqueeze(1))
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    union_area = area.unsqueeze(2) + area.unsqueeze(1) - inter_area + 1e-6
    iou = inter_area / union_area
    area_ratio = area.unsqueeze(2) / area.unsqueeze(1).clamp(min=1e-6)

    return torch.stack([dx, dy, log_w, log_h, dist, iou, area_ratio], dim=-1)


@dataclass
class SemanticFrontendOutput:
    node_embs: Tensor | None = None
    node_mask: Tensor | None = None
    boxes: Tensor | None = None
    track_ids: Tensor | None = None
    edge_index: Tensor | None = None
    edge_attr: Tensor | None = None
    aux_losses: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class SemanticConditioningOutput:
    prefix_embs: Tensor | None = None
    prefix_pad_masks: Tensor | None = None
    target_probs: Tensor | None = None
    target_logits: Tensor | None = None
    aux_losses: dict[str, Tensor] = field(default_factory=dict)


class S2KeywordVisionCrossAttention(nn.Module):
    """A lightweight keyword-conditioned visual refinement block."""

    def __init__(self, dim: int, num_heads: int, dropout: float, init_scale: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(
        self,
        vision_tokens: Tensor,
        vision_pad_masks: Tensor,
        keyword_tokens: Tensor | None,
        keyword_pad_masks: Tensor | None,
    ) -> Tensor:
        if keyword_tokens is None or keyword_pad_masks is None:
            return vision_tokens

        keyword_tokens = keyword_tokens.to(device=vision_tokens.device, dtype=vision_tokens.dtype)
        keyword_pad_masks = keyword_pad_masks.to(device=vision_tokens.device, dtype=torch.bool)
        vision_pad_masks = vision_pad_masks.to(device=vision_tokens.device, dtype=torch.bool)

        active = vision_pad_masks.any(dim=-1) & keyword_pad_masks.any(dim=-1)
        if not active.any():
            return vision_tokens

        updated = vision_tokens.clone()
        active_vision = vision_tokens[active]
        active_vision_pad_masks = vision_pad_masks[active]
        active_keyword = keyword_tokens[active]
        active_keyword_pad_masks = keyword_pad_masks[active]

        attn_out, _ = self.cross_attn(
            query=active_keyword,
            key=active_vision,
            value=active_vision,
            key_padding_mask=~active_vision_pad_masks,
            need_weights=False,
        )
        pooled = _masked_mean(attn_out, active_keyword_pad_masks.unsqueeze(-1), dim=1)
        delta = self.out_proj(pooled).unsqueeze(1)
        scale = torch.tanh(self.residual_scale).to(dtype=delta.dtype)
        active_updated = self.norm(active_vision + scale * delta)
        active_updated = torch.where(active_vision_pad_masks.unsqueeze(-1), active_updated, active_vision)
        updated[active] = active_updated
        return updated


class S2SemanticFrontend(nn.Module):
    """Build object-centric semantic representations for S2-PI05.

    Supported input modes:
    - precomputed node embeddings,
    - raw visual embeddings,
    - boxes with image crops,
    - OCR token embeddings,
    - optional temporal history via previous node embeddings or online track memory.
    """

    def __init__(self, config: S2PI05Config):
        super().__init__()
        self.config = config

        self.node_input_proj = nn.LazyLinear(config.semantic_node_dim)
        self.visual_proj = nn.LazyLinear(config.semantic_node_dim)
        self.text_proj = nn.LazyLinear(config.semantic_node_dim)
        self.history_proj = nn.LazyLinear(config.semantic_node_dim)
        self.box_proj = nn.Linear(4, config.semantic_node_dim)

        self.fusion_norm = nn.LayerNorm(config.semantic_node_dim)
        self.fusion_dropout = nn.Dropout(config.semantic_graph_dropout)
        self.gate_alpha = nn.Parameter(torch.tensor(config.semantic_gate_alpha, dtype=torch.float32))
        self.gate_beta = nn.Parameter(torch.tensor(config.semantic_gate_beta, dtype=torch.float32))

        self.temporal_gru = None
        self.temporal_attn = None
        if config.semantic_temporal_fusion == "gru":
            self.temporal_gru = nn.GRUCell(config.semantic_node_dim, config.semantic_node_dim)
        elif config.semantic_temporal_fusion == "attn":
            self.temporal_attn = nn.MultiheadAttention(
                config.semantic_node_dim,
                num_heads=config.semantic_attn_heads,
                dropout=config.semantic_graph_dropout,
                batch_first=True,
            )

        self.track_memory: dict[int, Tensor] = {}
        self.track_history: dict[int, deque[Tensor]] = defaultdict(
            lambda: deque(maxlen=config.semantic_track_memory)
        )

    def reset_state(self):
        self.track_memory.clear()
        self.track_history.clear()

    def _zero_aux_losses(self, reference: Tensor | None) -> dict[str, Tensor]:
        return {
            "loss_text_align": _zeros_like_scalar(reference),
            "loss_counterfactual": _zeros_like_scalar(reference),
            "loss_temporal": _zeros_like_scalar(reference),
            "loss_grounding": _zeros_like_scalar(reference),
        }

    def _get_node_mask(self, semantic_batch: dict[str, Any], candidate_shape: tuple[int, int], device) -> Tensor:
        node_mask = semantic_batch.get(OBS_SEMANTIC_NODE_MASK)
        if isinstance(node_mask, Tensor):
            return node_mask.to(device=device, dtype=torch.bool)

        boxes = semantic_batch.get(OBS_SEMANTIC_BOXES_XYXY)
        if isinstance(boxes, Tensor):
            widths = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
            heights = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
            return ((widths > 0) & (heights > 0)).to(device=device)

        text_masks = semantic_batch.get(OBS_SEMANTIC_TEXT_ATTENTION_MASK)
        if isinstance(text_masks, Tensor):
            return text_masks.any(dim=-1).to(device=device)

        prev_node_embs = semantic_batch.get(OBS_SEMANTIC_PREV_NODE_EMBS)
        if isinstance(prev_node_embs, Tensor):
            if prev_node_embs.dim() == 4:
                if prev_node_embs.shape[2] == candidate_shape[1]:
                    return (prev_node_embs.abs().sum(dim=(1, 3)) > 0).to(device=device)
                if prev_node_embs.shape[1] == candidate_shape[1]:
                    return (prev_node_embs.abs().sum(dim=(2, 3)) > 0).to(device=device)
            elif prev_node_embs.dim() == 3:
                return (prev_node_embs.abs().sum(dim=-1) > 0).to(device=device)

        return torch.ones(candidate_shape, dtype=torch.bool, device=device)

    def _extract_text_embeddings(
        self,
        semantic_batch: dict[str, Any],
        embed_language_tokens,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None]:
        text_tokens = semantic_batch.get(OBS_SEMANTIC_TEXT_TOKENS)
        text_masks = semantic_batch.get(OBS_SEMANTIC_TEXT_ATTENTION_MASK)
        if not isinstance(text_tokens, Tensor) or not isinstance(text_masks, Tensor):
            return None, None

        text_tokens = text_tokens.to(device=device)
        text_masks = text_masks.to(device=device, dtype=torch.bool)
        batch_size, num_objects, seq_len = text_tokens.shape
        flat_tokens = text_tokens.reshape(batch_size * num_objects, seq_len)
        flat_masks = text_masks.reshape(batch_size * num_objects, seq_len)

        text_embs = embed_language_tokens(flat_tokens)
        pooled = _masked_mean(text_embs, flat_masks.unsqueeze(-1), dim=1)
        pooled = pooled.view(batch_size, num_objects, -1)
        valid_text = text_masks.any(dim=-1)
        return pooled, valid_text

    def _crop_object_embeddings(
        self,
        images: list[Tensor],
        boxes: Tensor,
        node_mask: Tensor,
        camera_ids: Tensor | None,
        embed_image,
    ) -> Tensor | None:
        if not images:
            return None

        batch_size, num_objects = boxes.shape[:2]
        crop_tensors: list[Tensor] = []
        crop_indices: list[tuple[int, int]] = []
        target_h, target_w = self.config.image_resolution

        for b in range(batch_size):
            for k in range(num_objects):
                if not bool(node_mask[b, k]):
                    continue
                camera_index = 0
                if isinstance(camera_ids, Tensor):
                    camera_index = int(camera_ids[b, k].item())
                camera_index = max(0, min(camera_index, len(images) - 1))
                image = images[camera_index][b : b + 1]
                _, _, height, width = image.shape
                box = boxes[b, k].clamp(0, 1)
                x1 = int(torch.floor(box[0] * width).item())
                y1 = int(torch.floor(box[1] * height).item())
                x2 = int(torch.ceil(box[2] * width).item())
                y2 = int(torch.ceil(box[3] * height).item())
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                crop = image[:, :, y1:y2, x1:x2]
                crop = F.interpolate(crop, size=(target_h, target_w), mode="bilinear", align_corners=False)
                crop_tensors.append(crop)
                crop_indices.append((b, k))

        if not crop_tensors:
            return None

        crop_batch = torch.cat(crop_tensors, dim=0)
        crop_tokens = embed_image(crop_batch)
        crop_embs = crop_tokens.mean(dim=1)
        out = torch.zeros(batch_size, num_objects, crop_embs.shape[-1], device=crop_embs.device, dtype=crop_embs.dtype)
        for idx, (b, k) in enumerate(crop_indices):
            out[b, k] = crop_embs[idx]
        return out

    def _extract_visual_embeddings(
        self,
        images: list[Tensor],
        semantic_batch: dict[str, Any],
        node_mask: Tensor,
        embed_image,
        device: torch.device,
    ) -> Tensor | None:
        visual_embs = semantic_batch.get(OBS_SEMANTIC_VISUAL_EMBS)
        if isinstance(visual_embs, Tensor):
            return visual_embs.to(device=device)

        boxes = semantic_batch.get(OBS_SEMANTIC_BOXES_XYXY)
        if not isinstance(boxes, Tensor):
            return None
        boxes = boxes.to(device=device, dtype=torch.float32)
        camera_ids = semantic_batch.get(OBS_SEMANTIC_CAMERA_IDS)
        if isinstance(camera_ids, Tensor):
            camera_ids = camera_ids.to(device=device)
        return self._crop_object_embeddings(images, boxes, node_mask, camera_ids, embed_image)

    def _compute_text_align_loss(self, anchor: Tensor, text: Tensor, mask: Tensor) -> Tensor:
        flat_mask = mask.reshape(-1)
        if flat_mask.sum() <= 1:
            return _zeros_like_scalar(anchor)
        anchor = F.normalize(anchor.reshape(-1, anchor.shape[-1])[flat_mask], dim=-1)
        text = F.normalize(text.reshape(-1, text.shape[-1])[flat_mask], dim=-1)
        logits = (anchor @ text.T) / self.config.semantic_text_align_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_a2t + loss_t2a)

    def _history_from_prev_tensor(self, prev_node_embs: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        prev_node_embs = prev_node_embs.to(device=reference.device, dtype=reference.dtype)
        if prev_node_embs.dim() == 2:
            prev_node_embs = prev_node_embs.unsqueeze(0).unsqueeze(0)
        elif prev_node_embs.dim() == 3:
            prev_node_embs = prev_node_embs.unsqueeze(1)

        num_nodes = reference.shape[1]
        if prev_node_embs.dim() != 4:
            raise ValueError("Expected prev_node_embs to have rank 4 after normalization")

        if prev_node_embs.shape[2] == num_nodes:
            history = prev_node_embs.permute(0, 2, 1, 3)
        elif prev_node_embs.shape[1] == num_nodes:
            history = prev_node_embs
        else:
            history = prev_node_embs.permute(0, 2, 1, 3)

        if history.shape[-1] != self.config.semantic_node_dim:
            history = self.history_proj(history)

        history_valid = history.abs().sum(dim=-1) > 0
        return history, history_valid

    def _history_from_memory(
        self,
        track_ids: Tensor,
        node_mask: Tensor,
        reference: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        max_len = 0
        for b in range(track_ids.shape[0]):
            for k in range(track_ids.shape[1]):
                if not bool(node_mask[b, k]):
                    continue
                hist = self.track_history.get(int(track_ids[b, k].item()))
                if hist:
                    max_len = max(max_len, len(hist))
        if max_len == 0:
            return None, None

        history = torch.zeros(
            track_ids.shape[0],
            track_ids.shape[1],
            max_len,
            reference.shape[-1],
            device=reference.device,
            dtype=reference.dtype,
        )
        history_valid = torch.zeros(
            track_ids.shape[0],
            track_ids.shape[1],
            max_len,
            device=reference.device,
            dtype=torch.bool,
        )
        for b in range(track_ids.shape[0]):
            for k in range(track_ids.shape[1]):
                if not bool(node_mask[b, k]):
                    continue
                hist = self.track_history.get(int(track_ids[b, k].item()))
                if not hist:
                    continue
                start = max_len - len(hist)
                for t, value in enumerate(hist):
                    history[b, k, start + t] = value.to(device=reference.device, dtype=reference.dtype)
                    history_valid[b, k, start + t] = True
        return history, history_valid

    def _get_history(
        self,
        semantic_batch: dict[str, Any],
        track_ids: Tensor | None,
        node_mask: Tensor,
        reference: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        prev_node_embs = semantic_batch.get(OBS_SEMANTIC_PREV_NODE_EMBS)
        if isinstance(prev_node_embs, Tensor):
            return self._history_from_prev_tensor(prev_node_embs, reference)

        if self.training or not self.config.semantic_use_online_track_memory or track_ids is None:
            return None, None
        return self._history_from_memory(track_ids, node_mask, reference)

    def _update_online_memory(
        self,
        track_ids: Tensor | None,
        node_mask: Tensor,
        fused_nodes: Tensor,
        *,
        update_memory: bool,
    ):
        if self.training or not update_memory or not self.config.semantic_use_online_track_memory or track_ids is None:
            return
        for b in range(track_ids.shape[0]):
            for k in range(track_ids.shape[1]):
                if not bool(node_mask[b, k]):
                    continue
                track_id = int(track_ids[b, k].item())
                value = fused_nodes[b, k].detach().cpu()
                self.track_memory[track_id] = value
                self.track_history[track_id].append(value)

    def _apply_temporal_fusion(
        self,
        node_embs: Tensor,
        node_mask: Tensor,
        track_ids: Tensor | None,
        semantic_batch: dict[str, Any],
        *,
        update_memory: bool,
    ) -> tuple[Tensor, Tensor]:
        mode = self.config.semantic_temporal_fusion
        if mode == "none":
            fused = self.fusion_dropout(self.fusion_norm(node_embs))
            fused = torch.where(node_mask.unsqueeze(-1), fused, torch.zeros_like(fused))
            self._update_online_memory(track_ids, node_mask, fused, update_memory=update_memory)
            return fused, _zeros_like_scalar(node_embs)

        history, history_valid = self._get_history(semantic_batch, track_ids, node_mask, node_embs)
        if history is None or history_valid is None or not history_valid.any():
            fused = self.fusion_dropout(self.fusion_norm(node_embs))
            fused = torch.where(node_mask.unsqueeze(-1), fused, torch.zeros_like(fused))
            self._update_online_memory(track_ids, node_mask, fused, update_memory=update_memory)
            return fused, _zeros_like_scalar(node_embs)

        batch_size, num_nodes, node_dim = node_embs.shape
        history_len = history.shape[2]
        flat_nodes = node_embs.reshape(batch_size * num_nodes, node_dim)
        flat_mask = node_mask.reshape(batch_size * num_nodes)
        flat_history = history.reshape(batch_size * num_nodes, history_len, node_dim)
        flat_history_valid = history_valid.reshape(batch_size * num_nodes, history_len)
        active = flat_mask & flat_history_valid.any(dim=-1)

        fused_flat = flat_nodes.clone()
        temporal_loss = _zeros_like_scalar(node_embs)
        if active.any():
            current = flat_nodes[active]
            hist = flat_history[active]
            hist_valid = flat_history_valid[active]

            hist_weights = hist_valid.unsqueeze(-1).to(dtype=hist.dtype)
            hist_target = (hist * hist_weights).sum(dim=1) / hist_weights.sum(dim=1).clamp(min=1.0)

            if mode == "ema":
                latest = hist_target.clone()
                for t in range(hist.shape[1]):
                    step_mask = hist_valid[:, t]
                    if step_mask.any():
                        latest[step_mask] = hist[step_mask, t]
                fused_active = self.config.semantic_ema_decay * latest + (1.0 - self.config.semantic_ema_decay) * current
            elif mode == "gru":
                if self.temporal_gru is None:
                    raise RuntimeError("GRU temporal fusion requested but temporal_gru is not initialized")
                hidden = torch.zeros_like(current)
                initialized = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
                for t in range(hist.shape[1]):
                    step_mask = hist_valid[:, t]
                    if step_mask.any():
                        hidden[step_mask] = self.temporal_gru(hist[step_mask, t], hidden[step_mask])
                        initialized[step_mask] = True
                fused_active = current.clone()
                if initialized.any():
                    fused_active[initialized] = self.temporal_gru(current[initialized], hidden[initialized])
            elif mode == "attn":
                if self.temporal_attn is None:
                    raise RuntimeError("Attention temporal fusion requested but temporal_attn is not initialized")
                sequence = torch.cat([hist, current.unsqueeze(1)], dim=1)
                sequence_mask = torch.cat(
                    [hist_valid, torch.ones(hist_valid.shape[0], 1, dtype=torch.bool, device=hist_valid.device)],
                    dim=1,
                )
                attn_out, _ = self.temporal_attn(
                    current.unsqueeze(1),
                    sequence,
                    sequence,
                    key_padding_mask=~sequence_mask,
                    need_weights=False,
                )
                fused_active = attn_out.squeeze(1)
            else:
                raise ValueError(f"Unsupported temporal fusion mode: {mode}")

            temporal_loss = F.mse_loss(
                F.normalize(current, dim=-1),
                F.normalize(hist_target.detach(), dim=-1),
            )
            fused_flat[active] = fused_active

        fused = fused_flat.view_as(node_embs)
        fused = self.fusion_dropout(self.fusion_norm(fused))
        fused = torch.where(node_mask.unsqueeze(-1), fused, torch.zeros_like(fused))
        self._update_online_memory(track_ids, node_mask, fused, update_memory=update_memory)
        return fused, temporal_loss

    def forward(
        self,
        images: list[Tensor],
        semantic_batch: dict[str, Any] | None,
        *,
        embed_image,
        embed_language_tokens,
        update_memory: bool = True,
    ) -> SemanticFrontendOutput:
        if not semantic_batch:
            return SemanticFrontendOutput(aux_losses=self._zero_aux_losses(None))

        direct_nodes = semantic_batch.get(OBS_SEMANTIC_NODE_EMBS)
        visual_embs = semantic_batch.get(OBS_SEMANTIC_VISUAL_EMBS)
        boxes = semantic_batch.get(OBS_SEMANTIC_BOXES_XYXY)
        text_tokens = semantic_batch.get(OBS_SEMANTIC_TEXT_TOKENS)
        prev_node_embs = semantic_batch.get(OBS_SEMANTIC_PREV_NODE_EMBS)

        batch_size = None
        num_objects = None
        device = None
        for candidate in (direct_nodes, visual_embs, boxes, text_tokens, prev_node_embs):
            if not isinstance(candidate, Tensor):
                continue
            if candidate.dim() == 4 and candidate is prev_node_embs:
                batch_size = candidate.shape[0]
                num_objects = candidate.shape[2] if candidate.shape[2] <= self.config.semantic_max_objects else candidate.shape[1]
            else:
                batch_size = candidate.shape[0]
                num_objects = candidate.shape[1] if candidate.dim() > 1 else 1
            device = candidate.device
            break

        if batch_size is None or num_objects is None or device is None:
            return SemanticFrontendOutput(aux_losses=self._zero_aux_losses(None))

        node_mask = self._get_node_mask(semantic_batch, (batch_size, num_objects), device=device)
        boxes_tensor = None
        if isinstance(boxes, Tensor):
            boxes_tensor = boxes.to(device=device, dtype=torch.float32).clamp(0, 1)

        track_ids = semantic_batch.get(OBS_SEMANTIC_TRACK_IDS)
        if isinstance(track_ids, Tensor):
            track_ids = track_ids.to(device=device, dtype=torch.long)

        edge_index = semantic_batch.get(OBS_SEMANTIC_EDGE_INDEX)
        if isinstance(edge_index, Tensor):
            edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_attr = semantic_batch.get(OBS_SEMANTIC_EDGE_ATTR)
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr.to(device=device)

        aux_losses = self._zero_aux_losses(direct_nodes if isinstance(direct_nodes, Tensor) else None)

        components: list[Tensor] = []
        anchor_for_text = None
        if isinstance(direct_nodes, Tensor) and self.config.semantic_use_precomputed_nodes:
            direct_proj = self.node_input_proj(direct_nodes.to(device=device))
            components.append(direct_proj)
            anchor_for_text = direct_proj

        visual_proj = None
        visual_embs = self._extract_visual_embeddings(images, semantic_batch, node_mask, embed_image, device)
        if visual_embs is not None:
            visual_proj = self.visual_proj(visual_embs)
            components.append(visual_proj)
            anchor_for_text = visual_proj if anchor_for_text is None else 0.5 * (anchor_for_text + visual_proj)

        if boxes_tensor is not None and self.config.semantic_use_boxes_geometry:
            components.append(self.box_proj(boxes_tensor))

        text_embs, valid_text = self._extract_text_embeddings(semantic_batch, embed_language_tokens, device)
        if text_embs is not None and valid_text is not None and self.config.semantic_text_enabled:
            text_proj = self.text_proj(text_embs)
            has_non_text_context = len(components) > 0
            if has_non_text_context:
                ocr_conf = semantic_batch.get(OBS_SEMANTIC_OCR_CONF)
                if not isinstance(ocr_conf, Tensor):
                    ocr_conf = torch.ones(node_mask.shape, device=device, dtype=text_proj.dtype)
                else:
                    ocr_conf = ocr_conf.to(device=device, dtype=text_proj.dtype)

                if anchor_for_text is None:
                    sim = torch.zeros(node_mask.shape, device=device, dtype=text_proj.dtype)
                else:
                    sim = F.cosine_similarity(
                        F.normalize(anchor_for_text, dim=-1),
                        F.normalize(text_proj, dim=-1),
                        dim=-1,
                    )
                gate = torch.sigmoid(self.gate_alpha.to(text_proj.dtype) * ocr_conf + self.gate_beta.to(text_proj.dtype) * sim)
                gate = gate * (node_mask & valid_text).to(dtype=text_proj.dtype)
                text_component = gate.unsqueeze(-1) * text_proj
                if anchor_for_text is not None:
                    aux_losses["loss_text_align"] = self._compute_text_align_loss(
                        anchor_for_text,
                        text_proj,
                        node_mask & valid_text,
                    )
            else:
                text_component = torch.where(node_mask.unsqueeze(-1), text_proj, torch.zeros_like(text_proj))
            components.append(text_component)

        if not components:
            return SemanticFrontendOutput(aux_losses=aux_losses)

        node_embs = torch.stack(components, dim=0).sum(dim=0)
        node_embs, temporal_loss = self._apply_temporal_fusion(
            node_embs,
            node_mask,
            track_ids,
            semantic_batch,
            update_memory=update_memory,
        )
        aux_losses["loss_temporal"] = temporal_loss

        return SemanticFrontendOutput(
            node_embs=node_embs,
            node_mask=node_mask,
            boxes=boxes_tensor,
            track_ids=track_ids,
            edge_index=edge_index,
            edge_attr=edge_attr,
            aux_losses=aux_losses,
        )


class S2GraphBlock(nn.Module):
    """A relation-aware self-attention block over object nodes."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.geo_bias = nn.Linear(7, 1)
        self.edge_bias = nn.LazyLinear(1)

    def _edge_bias_from_sparse(
        self,
        edge_index: Tensor | None,
        edge_attr: Tensor | None,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None]:
        if edge_index is None:
            return None, None

        if edge_index.dim() == 2:
            edge_index = edge_index.unsqueeze(0)
        batch_size = edge_index.shape[0]
        adjacency = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=device)
        bias = None
        if edge_attr is not None:
            if edge_attr.dim() == 2:
                edge_attr = edge_attr.unsqueeze(0)
            projected = self.edge_bias(edge_attr.to(device=device, dtype=dtype)).squeeze(-1)
            bias = torch.zeros(batch_size, num_nodes, num_nodes, dtype=dtype, device=device)
        for b in range(batch_size):
            src = edge_index[b, 0].to(device=device, dtype=torch.long)
            dst = edge_index[b, 1].to(device=device, dtype=torch.long)
            valid = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
            src = src[valid]
            dst = dst[valid]
            adjacency[b, src, dst] = True
            if bias is not None and edge_attr is not None:
                bias[b, src, dst] = projected[b][valid]
        return adjacency, bias

    def forward(
        self,
        nodes: Tensor,
        node_mask: Tensor,
        boxes: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        use_sparse_edges: bool = True,
    ) -> Tensor:
        batch_size, num_nodes, dim = nodes.shape
        q = self.query(nodes)
        k = self.key(nodes)
        v = self.value(nodes)

        logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(dim)
        adjacency = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)

        if boxes is not None:
            logits = logits + self.geo_bias(_pairwise_box_features(boxes)).squeeze(-1)

        if use_sparse_edges and edge_index is not None:
            sparse_adjacency, sparse_bias = self._edge_bias_from_sparse(
                edge_index,
                edge_attr,
                num_nodes=num_nodes,
                device=nodes.device,
                dtype=nodes.dtype,
            )
            if sparse_adjacency is not None:
                eye = torch.eye(num_nodes, dtype=torch.bool, device=nodes.device).unsqueeze(0)
                adjacency = adjacency & (sparse_adjacency | eye)
            if sparse_bias is not None:
                logits = logits + sparse_bias

        all_invalid = ~adjacency.any(dim=-1)
        logits = logits.masked_fill(~adjacency, torch.finfo(logits.dtype).min)
        logits = logits.masked_fill(all_invalid.unsqueeze(-1), 0.0)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.where(adjacency, attn, torch.zeros_like(attn))

        out = torch.einsum("bij,bjd->bid", attn, v)
        out = self.out(out)
        x = self.norm1(nodes + self.dropout(out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        x = torch.where(node_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x


class S2GraphEncoder(nn.Module):
    """Graph encoder used for semantic reasoning before grounding."""

    def __init__(self, config: S2PI05Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [S2GraphBlock(config.semantic_node_dim, config.semantic_graph_dropout) for _ in range(config.semantic_graph_layers)]
        )

    def forward(
        self,
        node_embs: Tensor,
        node_mask: Tensor,
        boxes: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        if self.config.semantic_graph_encoder == "none" or self.config.semantic_graph_layers == 0:
            return node_embs

        graph_boxes = boxes if self.config.semantic_graph_edges in {"spatial", "spatial+interaction"} else None
        graph_edge_index = edge_index if self.config.semantic_graph_edges == "spatial+interaction" else None
        graph_edge_attr = edge_attr if self.config.semantic_graph_edges == "spatial+interaction" else None
        use_sparse_edges = self.config.semantic_graph_encoder == "gat"

        x = node_embs
        for layer in self.layers:
            x = layer(
                x,
                node_mask,
                boxes=graph_boxes,
                edge_index=graph_edge_index,
                edge_attr=graph_edge_attr,
                use_sparse_edges=use_sparse_edges,
            )
        return x


class S2GroundingHead(nn.Module):
    """Ground a language query onto object nodes and produce a target embedding."""

    def __init__(self, query_dim: int, node_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, node_dim)
        self.node_proj = nn.Linear(node_dim, node_dim)
        self.scale = math.sqrt(node_dim)

    def forward(
        self,
        query_embs: Tensor,
        node_embs: Tensor,
        node_mask: Tensor,
        *,
        temperature: float,
        mode: str,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = self.query_proj(query_embs)
        nodes = self.node_proj(node_embs)
        logits = torch.einsum("bd,bkd->bk", query, nodes) / self.scale
        masked_logits = logits.masked_fill(~node_mask, torch.finfo(logits.dtype).min)
        all_invalid = ~node_mask.any(dim=-1)
        masked_logits = masked_logits.masked_fill(all_invalid.unsqueeze(-1), 0.0)

        if mode == "hard_gumbel" and training:
            probs = F.gumbel_softmax(masked_logits, tau=max(temperature, 1e-6), hard=True, dim=-1)
        elif mode == "hard_argmax_infer" and not training:
            argmax = masked_logits.argmax(dim=-1)
            probs = F.one_hot(argmax, num_classes=node_embs.shape[1]).to(dtype=node_embs.dtype)
        else:
            probs = torch.softmax(masked_logits / max(temperature, 1e-6), dim=-1)

        probs = torch.where(node_mask, probs, torch.zeros_like(probs))
        probs = torch.where(
            all_invalid.unsqueeze(-1),
            torch.zeros_like(probs),
            probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6),
        )
        target_embs = torch.einsum("bk,bkd->bd", probs, node_embs)
        return target_embs, masked_logits, probs


class S2PI05Pytorch(PI05Pytorch):
    """Structured-semantic PI05 backbone."""

    def __init__(self, config: S2PI05Config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config = config
        self.model_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        if self.model_dim % config.keyword_cross_attention_heads != 0:
            raise ValueError("model_dim must be divisible by keyword_cross_attention_heads")

        self.semantic_frontend = S2SemanticFrontend(config)
        self.graph_encoder = S2GraphEncoder(config)
        self.grounding_head = S2GroundingHead(query_dim=self.model_dim, node_dim=config.semantic_node_dim)

        self.semantic_token_proj = nn.Linear(config.semantic_node_dim, self.model_dim)
        self.target_token_proj = nn.Linear(config.semantic_node_dim, self.model_dim)
        self.semantic_prefix_norm = nn.LayerNorm(self.model_dim)
        self.target_prefix_norm = nn.LayerNorm(self.model_dim)
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.target_type_embedding = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.target_slot_embedding = nn.Parameter(torch.zeros(1, config.target_token_count, self.model_dim))
        self.semantic_slot_embedding = nn.Embedding(config.semantic_max_objects, self.model_dim)
        self.keyword_vision_cross_attention = S2KeywordVisionCrossAttention(
            self.model_dim,
            num_heads=config.keyword_cross_attention_heads,
            dropout=config.keyword_cross_attention_dropout,
            init_scale=config.keyword_cross_attention_init_scale,
        )
        nn.init.normal_(self.semantic_type_embedding, std=0.02)
        nn.init.normal_(self.target_type_embedding, std=0.02)
        nn.init.normal_(self.target_slot_embedding, std=0.02)
        nn.init.normal_(self.semantic_slot_embedding.weight, std=0.02)

    def reset_semantic_state(self):
        self.semantic_frontend.reset_state()

    def _zero_aux_losses(self, reference: Tensor) -> dict[str, Tensor]:
        return {
            "loss_text_align": _zeros_like_scalar(reference),
            "loss_counterfactual": _zeros_like_scalar(reference),
            "loss_temporal": _zeros_like_scalar(reference),
            "loss_grounding": _zeros_like_scalar(reference),
        }

    def _pool_language_query(self, tokens: Tensor, masks: Tensor) -> Tensor:
        lang_embs = self.paligemma_with_expert.embed_language_tokens(tokens)
        return _masked_mean(lang_embs, masks.unsqueeze(-1), dim=1)

    def _embed_keyword_tokens(self, keyword_tokens: Tensor) -> Tensor:
        def keyword_embed_func(input_tokens: Tensor) -> Tensor:
            keyword_embs = self.paligemma_with_expert.embed_language_tokens(input_tokens)
            return keyword_embs * math.sqrt(keyword_embs.shape[-1])

        return self._apply_checkpoint(keyword_embed_func, keyword_tokens)

    def _compute_grounding_loss(
        self,
        target_logits: Tensor | None,
        semantic_batch: dict[str, Any] | None,
    ) -> Tensor | None:
        if target_logits is None or not semantic_batch:
            return None
        target_index = semantic_batch.get(OBS_SEMANTIC_TARGET_INDEX)
        if not isinstance(target_index, Tensor):
            return None
        target_index = target_index.to(device=target_logits.device, dtype=torch.long).view(-1)
        valid = target_index >= 0
        if valid.sum() == 0:
            return _zeros_like_scalar(target_logits)
        return F.cross_entropy(target_logits[valid], target_index[valid])

    def _compute_counterfactual_loss(
        self,
        images: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        semantic_batch: dict[str, Any] | None,
        primary_probs: Tensor | None,
    ) -> Tensor | None:
        if semantic_batch is None or primary_probs is None:
            return None
        cf_tokens = semantic_batch.get(OBS_SEMANTIC_CF_TEXT_TOKENS)
        cf_masks = semantic_batch.get(OBS_SEMANTIC_CF_TEXT_ATTENTION_MASK)
        if not isinstance(cf_tokens, Tensor) or not isinstance(cf_masks, Tensor):
            return None

        cf_batch = dict(semantic_batch)
        cf_batch[OBS_SEMANTIC_TEXT_TOKENS] = cf_tokens
        cf_batch[OBS_SEMANTIC_TEXT_ATTENTION_MASK] = cf_masks
        cf_frontend = self.semantic_frontend(
            images,
            cf_batch,
            embed_image=self.paligemma_with_expert.embed_image,
            embed_language_tokens=self.paligemma_with_expert.embed_language_tokens,
            update_memory=False,
        )
        if cf_frontend.node_embs is None or cf_frontend.node_mask is None:
            return None
        cf_nodes = self.graph_encoder(
            cf_frontend.node_embs,
            cf_frontend.node_mask,
            boxes=cf_frontend.boxes,
            edge_index=cf_frontend.edge_index,
            edge_attr=cf_frontend.edge_attr,
        )
        query_embs = self._pool_language_query(tokens, masks)
        _, _, cf_probs = self.grounding_head(
            query_embs,
            cf_nodes,
            cf_frontend.node_mask,
            temperature=self.config.grounding_temperature,
            mode="soft",
            training=self.training,
        )
        valid = (primary_probs.sum(dim=-1) > 0) & (cf_probs.sum(dim=-1) > 0)
        if valid.sum() == 0:
            return _zeros_like_scalar(primary_probs)

        eps = 1e-6
        primary = primary_probs[valid].clamp(min=eps)
        counterfactual = cf_probs[valid].clamp(min=eps)
        loss_pq = F.kl_div(counterfactual.log(), primary, reduction="batchmean")
        loss_qp = F.kl_div(primary.log(), counterfactual, reduction="batchmean")
        return 0.5 * (loss_pq + loss_qp)

    def build_semantic_conditioning(
        self,
        images: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        semantic_batch: dict[str, Any] | None,
    ) -> SemanticConditioningOutput:
        if not self.config.semantic_enabled:
            return SemanticConditioningOutput(aux_losses=self._zero_aux_losses(tokens))

        frontend = self.semantic_frontend(
            images,
            semantic_batch,
            embed_image=self.paligemma_with_expert.embed_image,
            embed_language_tokens=self.paligemma_with_expert.embed_language_tokens,
            update_memory=True,
        )
        if frontend.node_embs is None or frontend.node_mask is None:
            return SemanticConditioningOutput(aux_losses=self._zero_aux_losses(tokens))

        graph_nodes = self.graph_encoder(
            frontend.node_embs,
            frontend.node_mask,
            boxes=frontend.boxes,
            edge_index=frontend.edge_index,
            edge_attr=frontend.edge_attr,
        )

        semantic_prefix = self.semantic_token_proj(graph_nodes) * math.sqrt(self.model_dim)
        slot_ids = torch.arange(graph_nodes.shape[1], device=graph_nodes.device)
        semantic_prefix = semantic_prefix + self.semantic_slot_embedding(slot_ids).unsqueeze(0)
        semantic_prefix = semantic_prefix + self.semantic_type_embedding
        semantic_prefix = self.semantic_prefix_norm(semantic_prefix)
        semantic_prefix = torch.where(
            frontend.node_mask.unsqueeze(-1),
            semantic_prefix,
            torch.zeros_like(semantic_prefix),
        )
        semantic_pad_masks = frontend.node_mask

        aux_losses = dict(frontend.aux_losses)
        for key, value in self._zero_aux_losses(tokens).items():
            aux_losses.setdefault(key, value)

        target_probs = None
        target_logits = None
        if self.config.grounding_enabled:
            query_embs = self._pool_language_query(tokens, masks)
            target_embs, target_logits, target_probs = self.grounding_head(
                query_embs,
                graph_nodes,
                frontend.node_mask,
                temperature=self.config.grounding_temperature,
                mode=self.config.grounding_mode,
                training=self.training,
            )
            target_tokens = self.target_token_proj(target_embs) * math.sqrt(self.model_dim)
            target_tokens = target_tokens.unsqueeze(1).expand(-1, self.config.target_token_count, -1)
            target_tokens = target_tokens + self.target_type_embedding + self.target_slot_embedding[:, : self.config.target_token_count]
            target_tokens = self.target_prefix_norm(target_tokens)
            valid_targets = frontend.node_mask.any(dim=-1, keepdim=True)
            target_masks = valid_targets.expand(-1, self.config.target_token_count)
            semantic_prefix = torch.cat([semantic_prefix, target_tokens], dim=1)
            semantic_pad_masks = torch.cat([semantic_pad_masks, target_masks], dim=1)

            grounding_loss = self._compute_grounding_loss(target_logits, semantic_batch)
            if grounding_loss is not None:
                aux_losses["loss_grounding"] = grounding_loss

            counterfactual_loss = self._compute_counterfactual_loss(images, tokens, masks, semantic_batch, target_probs)
            if counterfactual_loss is not None:
                aux_losses["loss_counterfactual"] = counterfactual_loss

        return SemanticConditioningOutput(
            prefix_embs=semantic_prefix,
            prefix_pad_masks=semantic_pad_masks,
            target_probs=target_probs,
            target_logits=target_logits,
            aux_losses=aux_losses,
        )

    def embed_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        *,
        semantic_embs: Tensor | None = None,
        semantic_pad_masks: Tensor | None = None,
        keyword_tokens: Tensor | None = None,
        keyword_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images and language, then splice semantic prefix tokens in between."""
        embs = []
        pad_masks = []
        att_masks = []

        keyword_embs = None
        keyword_pad_masks = None
        if self.config.keyword_conditioning_enabled and keyword_tokens is not None and keyword_masks is not None:
            keyword_tokens = keyword_tokens.to(device=tokens.device)
            keyword_pad_masks = keyword_masks.to(device=tokens.device, dtype=torch.bool)
            keyword_embs = self._embed_keyword_tokens(keyword_tokens)

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(image: Tensor) -> Tensor:
                return self.paligemma_with_expert.embed_image(image)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            batch_size, num_img_embs = img_emb.shape[:2]
            img_pad_masks = img_mask[:, None].expand(batch_size, num_img_embs)
            if keyword_embs is not None and keyword_pad_masks is not None:
                img_emb = self.keyword_vision_cross_attention(
                    img_emb,
                    img_pad_masks,
                    keyword_embs,
                    keyword_pad_masks,
                )
            embs.append(img_emb)
            pad_masks.append(img_pad_masks)
            att_masks += [0] * num_img_embs

        def lang_embed_func(input_tokens: Tensor) -> Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(input_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)

        if semantic_embs is not None:
            if semantic_pad_masks is None:
                raise ValueError("semantic_pad_masks must be provided when semantic_embs are provided")
            semantic_embs = semantic_embs.to(device=lang_emb.device, dtype=lang_emb.dtype)
            semantic_pad_masks = semantic_pad_masks.to(device=lang_emb.device, dtype=torch.bool)
            embs.append(semantic_embs)
            pad_masks.append(semantic_pad_masks)
            att_masks += [0] * semantic_embs.shape[1]

        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks.unsqueeze(0).expand(pad_masks.shape[0], -1)
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise=None,
        time=None,
        semantic_batch: dict[str, Any] | None = None,
        keyword_tokens: Tensor | None = None,
        keyword_masks: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor] | None]:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        semantic = self.build_semantic_conditioning(images, tokens, masks, semantic_batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            semantic_embs=semantic.prefix_embs,
            semantic_pad_masks=semantic.prefix_pad_masks,
            keyword_tokens=keyword_tokens,
            keyword_masks=keyword_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
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

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return {
            "policy_losses": losses,
            "aux_losses": semantic.aux_losses,
            "target_probs": semantic.target_probs,
            "target_logits": semantic.target_logits,
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
        semantic_batch: dict[str, Any] | None = None,
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

        semantic = self.build_semantic_conditioning(images, tokens, masks, semantic_batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            semantic_embs=semantic.prefix_embs,
            semantic_pad_masks=semantic.prefix_pad_masks,
            keyword_tokens=keyword_tokens,
            keyword_masks=keyword_masks,
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
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)

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
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t


class S2PI05Policy(PI05Policy):
    """Structured-semantic PI05 policy."""

    config_class = S2PI05Config
    name = "s2_pi05"

    def __init__(self, config: S2PI05Config, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = S2PI05Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @staticmethod
    def _coerce_pretrained_config(config: PreTrainedConfig) -> S2PI05Config:
        if isinstance(config, S2PI05Config):
            return config
        if isinstance(config, PI05Config):
            print(
                "Upgrading PI05Config to S2PI05Config to load compatible backbone weights. "
                "New semantic modules will remain randomly initialized."
            )
            return S2PI05Config.from_pi05_config(config)
        raise TypeError(
            "S2PI05Policy.from_pretrained supports S2PI05Config checkpoints and PI05-compatible checkpoints. "
            "Loading unrelated base VLM checkpoints requires a dedicated remapping path."
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

    def reset(self):
        super().reset()
        self.model.reset_semantic_state()

    def _extract_semantic_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        semantic_batch = {key: value for key, value in batch.items() if key.startswith(f"{OBS_SEMANTIC}.")}
        return semantic_batch or None

    def _extract_keyword_batch(self, batch: dict[str, Any]) -> tuple[Tensor | None, Tensor | None]:
        keyword_tokens = batch.get(OBS_LANGUAGE_KEYWORD_TOKENS)
        keyword_masks = batch.get(OBS_LANGUAGE_KEYWORD_ATTENTION_MASK)
        if not isinstance(keyword_tokens, Tensor) or not isinstance(keyword_masks, Tensor):
            return None, None
        return keyword_tokens, keyword_masks

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        semantic_batch = self._extract_semantic_batch(batch)
        keyword_tokens, keyword_masks = self._extract_keyword_batch(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            semantic_batch=semantic_batch,
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
        semantic_batch = self._extract_semantic_batch(batch)
        keyword_tokens, keyword_masks = self._extract_keyword_batch(batch)

        model_output = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            semantic_batch=semantic_batch,
            keyword_tokens=keyword_tokens,
            keyword_masks=keyword_masks,
        )
        losses = model_output["policy_losses"]
        aux_losses = model_output["aux_losses"]
        target_probs = model_output.get("target_probs")

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        policy_loss = losses.mean()

        aux_total = torch.zeros((), device=policy_loss.device)
        aux_weights = {
            "loss_text_align": self.config.loss_text_align_w,
            "loss_counterfactual": self.config.loss_counterfactual_w,
            "loss_temporal": self.config.loss_temporal_w,
            "loss_grounding": self.config.loss_grounding_w,
        }
        for key, weight in aux_weights.items():
            value = aux_losses.get(key)
            if value is None or weight <= 0:
                continue
            aux_total = aux_total + weight * value

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "policy_loss": policy_loss.item(),
            "aux_loss": aux_total.item(),
        }
        if target_probs is not None:
            loss_dict["target_entropy"] = (
                -(target_probs.clamp(min=1e-6).log() * target_probs).sum(dim=-1).mean().detach().cpu().item()
            )
        for key, value in aux_losses.items():
            if isinstance(value, Tensor):
                loss_dict[key] = value.detach().cpu().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = (per_sample_loss.mean() + aux_total).item()
            return per_sample_loss, loss_dict

        total_loss = policy_loss + aux_total
        loss_dict["loss"] = total_loss.item()
        return total_loss, loss_dict
