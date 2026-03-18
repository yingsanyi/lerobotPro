#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import logging

import pytest
import torch

from lerobot.datasets import video_utils


def test_get_safe_default_codec_falls_back_when_torchcodec_runtime_is_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(
        video_utils,
        "_probe_torchcodec_runtime",
        lambda: (False, "'torchcodec' is installed but unavailable at runtime (missing ffmpeg libs)"),
    )

    with caplog.at_level(logging.WARNING):
        backend = video_utils.get_safe_default_codec()

    assert backend == "pyav"
    assert "falling back to 'pyav'" in caplog.text


def test_decode_video_frames_falls_back_to_pyav_for_torchcodec_runtime_failures(monkeypatch, caplog):
    expected = torch.rand(1, 3, 4, 4)

    def raise_runtime_failure(*args, **kwargs):
        raise RuntimeError("Could not load libtorchcodec")

    def decode_with_pyav(video_path, timestamps, tolerance_s, backend):
        assert video_path == "video.mp4"
        assert timestamps == [0.1]
        assert tolerance_s == 0.05
        assert backend == "pyav"
        return expected

    monkeypatch.setattr(video_utils, "decode_video_frames_torchcodec", raise_runtime_failure)
    monkeypatch.setattr(video_utils, "decode_video_frames_torchvision", decode_with_pyav)

    with caplog.at_level(logging.WARNING):
        result = video_utils.decode_video_frames("video.mp4", [0.1], 0.05, backend="torchcodec")

    assert torch.equal(result, expected)
    assert "Falling back to 'pyav'" in caplog.text


def test_decode_video_frames_does_not_hide_non_runtime_torchcodec_errors(monkeypatch):
    def raise_other_runtime_error(*args, **kwargs):
        raise RuntimeError("unexpected decode failure")

    monkeypatch.setattr(video_utils, "decode_video_frames_torchcodec", raise_other_runtime_error)

    with pytest.raises(RuntimeError, match="unexpected decode failure"):
        video_utils.decode_video_frames("video.mp4", [0.1], 0.05, backend="torchcodec")
