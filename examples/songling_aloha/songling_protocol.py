#!/usr/bin/env python

"""Heuristic protocol helpers for Songling integrated-chain CAN frames.

This module intentionally stays conservative:
- it only decodes observed RX frames into candidate joint positions
- it does not attempt to transmit commands yet

The current mapping is based on empirical CAN captures from this repo's
Songling setup:
- per side, 7 motion-related IDs appear at ~200 Hz
- `0x251..0x256` carry most of their useful signal in bytes [2:4]
- `0x2A8` carries its useful signal in bytes [4:6] and is a strong gripper
  candidate

This should be treated as an evolving best-effort decoder, not a confirmed
vendor protocol specification.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SONGLING_STATE_IDS = [0x251, 0x252, 0x253, 0x254, 0x255, 0x256, 0x2A8]
DEFAULT_SONGLING_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
]


@dataclass(frozen=True)
class SonglingFieldSpec:
    arbitration_id: int
    byte_offset: int
    byte_length: int
    byteorder: str = "big"
    signed: bool = True
    scale: float = 1.0
    bias: float = 0.0

    def decode(self, payload: bytes) -> float | None:
        end = self.byte_offset + self.byte_length
        if len(payload) < end:
            return None
        raw = int.from_bytes(
            payload[self.byte_offset:end],
            byteorder=self.byteorder,
            signed=self.signed,
        )
        return float(raw) * self.scale + self.bias


SONGLING_STATE_SPECS = {
    0x251: SonglingFieldSpec(0x251, byte_offset=2, byte_length=2, byteorder="big", signed=False),
    0x252: SonglingFieldSpec(0x252, byte_offset=2, byte_length=2, byteorder="big", signed=False),
    0x253: SonglingFieldSpec(0x253, byte_offset=2, byte_length=2, byteorder="big", signed=False),
    0x254: SonglingFieldSpec(0x254, byte_offset=2, byte_length=2, byteorder="big", signed=False),
    0x255: SonglingFieldSpec(0x255, byte_offset=2, byte_length=2, byteorder="big", signed=False),
    0x256: SonglingFieldSpec(0x256, byte_offset=2, byte_length=2, byteorder="big", signed=False),
    0x2A8: SonglingFieldSpec(0x2A8, byte_offset=4, byte_length=2, byteorder="big", signed=True),
}

SONGLING_ACTION_CANDIDATE_SPECS = {
    0x251: SonglingFieldSpec(0x251, byte_offset=0, byte_length=2, byteorder="big", signed=True),
    0x252: SonglingFieldSpec(0x252, byte_offset=0, byte_length=2, byteorder="big", signed=True),
    0x253: SonglingFieldSpec(0x253, byte_offset=0, byte_length=2, byteorder="big", signed=True),
    0x254: SonglingFieldSpec(0x254, byte_offset=0, byte_length=2, byteorder="big", signed=True),
    0x255: SonglingFieldSpec(0x255, byte_offset=0, byte_length=2, byteorder="big", signed=True),
    0x256: SonglingFieldSpec(0x256, byte_offset=0, byte_length=2, byteorder="big", signed=True),
    # Provisional: no clear second channel identified for gripper yet.
    # Reuse the currently best-known varying field so downstream tooling can stay aligned in 14-D.
    0x2A8: SonglingFieldSpec(0x2A8, byte_offset=4, byte_length=2, byteorder="big", signed=True),
}


def is_known_state_id(arbitration_id: int) -> bool:
    return arbitration_id in SONGLING_STATE_SPECS


def decode_known_state(arbitration_id: int, payload: bytes) -> float | None:
    spec = SONGLING_STATE_SPECS.get(arbitration_id)
    if spec is None:
        return None
    return spec.decode(payload)


def decode_known_action_candidate(arbitration_id: int, payload: bytes) -> float | None:
    spec = SONGLING_ACTION_CANDIDATE_SPECS.get(arbitration_id)
    if spec is None:
        return None
    return spec.decode(payload)


def default_state_ids_csv() -> str:
    return ",".join(hex(v) for v in DEFAULT_SONGLING_STATE_IDS)
