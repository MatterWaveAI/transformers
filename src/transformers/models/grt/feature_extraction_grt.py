# Copyright 2026 MatterWave AI and the HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for GRT Audio, GRT-ASR, GRT-TTS, and GRT-Omni."""

from typing import Any

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class GRTAudioFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a GRT audio feature extractor for `GRTAudioModel`, `GRTASRForConditionalGeneration`,
    `GRTTTSForConditionalGeneration`, and `GRTOmniForConditionalGeneration`.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted waveform features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        n_fft (`int`, *optional*, defaults to 512):
            FFT window size for polar STFT framing.
        hop_length (`int`, *optional*, defaults to 160):
            STFT hop length in samples (10 ms at 16 kHz).
        win_length (`int`, *optional*, defaults to 400):
            STFT Hann window length in samples (25 ms at 16 kHz).
        max_audio_frames (`int`, *optional*, defaults to 512):
            Maximum number of STFT frames processed per audio window.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~GRTAudioFeatureExtractor.__call__`] should return `attention_mask`.
    """

    model_input_names = ["audio_values"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        max_audio_frames: int = 512,
        return_attention_mask: bool = False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_audio_frames = max_audio_frames

    @staticmethod
    def _to_1d_float32_array(item: Any) -> np.ndarray:
        if is_torch_available() and isinstance(item, torch.Tensor):
            arr = item.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(item, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = arr.mean(axis=0)
        elif arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]] | Any,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        return_attention_mask: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a "
                f"sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input was "
                f"sampled with {self.sampling_rate} and not {sampling_rate}."
            )

        if is_torch_available() and isinstance(raw_speech, torch.Tensor):
            raw_speech = raw_speech.detach().cpu().numpy()

        if isinstance(raw_speech, np.ndarray):
            if raw_speech.ndim == 1:
                batch = [raw_speech.astype(np.float32)]
            elif raw_speech.ndim == 2:
                batch = [row.astype(np.float32) for row in raw_speech]
            elif raw_speech.ndim == 3:
                return BatchFeature(
                    data={"audio_values": raw_speech.astype(np.float32)},
                    tensor_type=return_tensors,
                )
            else:
                raise ValueError(f"Unsupported `raw_speech` rank {raw_speech.ndim}; expected 1D, 2D, or 3D.")
        elif isinstance(raw_speech, (list, tuple)):
            if len(raw_speech) == 0:
                raise ValueError("`raw_speech` cannot be an empty list.")
            first = raw_speech[0]
            is_batched = (
                isinstance(first, (np.ndarray, list, tuple))
                or (is_torch_available() and isinstance(first, torch.Tensor) and first.ndim >= 1)
            )
            if is_batched:
                batch = [self._to_1d_float32_array(item) for item in raw_speech]
            else:
                batch = [self._to_1d_float32_array(raw_speech)]
        else:
            raise TypeError(f"Unsupported `raw_speech` type: {type(raw_speech)}.")

        if truncation and max_length is not None:
            batch = [seq[:max_length] for seq in batch]

        if padding == "max_length" and max_length is not None:
            target_len = max_length
        elif padding is not False and padding != PaddingStrategy.DO_NOT_PAD:
            target_len = max(len(seq) for seq in batch)
            if max_length is not None and not truncation:
                target_len = max(target_len, max_length)
        else:
            target_len = max(len(seq) for seq in batch)

        padded_batch = []
        attention_masks = []
        for seq in batch:
            seq_len = len(seq)
            if seq_len < target_len:
                pad_width = target_len - seq_len
                padded_seq = np.pad(seq, (0, pad_width), mode="constant", constant_values=self.padding_value)
                mask = np.concatenate([np.ones(seq_len, dtype=np.int64), np.zeros(pad_width, dtype=np.int64)])
            else:
                padded_seq = seq[:target_len]
                mask = np.ones(target_len, dtype=np.int64)
            padded_batch.append(padded_seq.astype(np.float32))
            attention_masks.append(mask)

        audio_values = np.stack(padded_batch, axis=0)
        data: dict[str, Any] = {"audio_values": audio_values}

        return_attention_mask = (
            self.return_attention_mask if return_attention_mask is None else return_attention_mask
        )
        if return_attention_mask:
            data["attention_mask"] = np.stack(attention_masks, axis=0)

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["GRTAudioFeatureExtractor"]
