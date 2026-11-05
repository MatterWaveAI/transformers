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
"""Processor classes for GRT-VL, GRT-ASR, and GRT-Omni."""

from typing import Any

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images, make_nested_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType, auto_docstring
from .feature_extraction_grt import GRTAudioFeatureExtractor
from .image_processing_grt import GRTImageProcessor


class GRTVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
    }


class GRTASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class GRTOmniProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
    }


def _normalize_text_list(text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput]) -> list[str]:
    if isinstance(text, str):
        return [text]
    if isinstance(text, list) and len(text) > 0 and all(isinstance(t, str) for t in text):
        return list(text)
    raise TypeError("Invalid input text. Please provide a string, or a list of strings.")


@auto_docstring
class GRTVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "GRTImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_processor_kwargs = GRTVLProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_seq_length: int = 64,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = GRTImageProcessor()
        self.image_seq_length = image_seq_length
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.image_token)
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids")
            else 151655
        )
        image_tokens_expanded = self.image_token * self.image_seq_length
        self.full_image_sequence = f"{self.vision_start_token}{image_tokens_expanded}{self.vision_end_token}"

        if image_processor is not None and tokenizer is not None:
            try:
                super().__init__(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    **kwargs,
                )
            except TypeError:
                self.image_processor = image_processor
                self.tokenizer = tokenizer
                self.chat_template = chat_template
        else:
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.chat_template = chat_template

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs: Unpack[GRTVLProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError(f"You need to provide at least one of `images` or `text` to call {self.__class__.__name__}.")

        data: dict[str, Any] = {}
        images_kwargs = dict(kwargs.pop("images_kwargs", {}) or {})
        text_kwargs = dict(kwargs.pop("text_kwargs", {}) or {})
        for k in ("do_resize", "size", "image_size", "resample", "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb"):
            if k in kwargs:
                images_kwargs[k] = kwargs.pop(k)
        text_kwargs.update(kwargs)
        if return_tensors is not None:
            images_kwargs.setdefault("return_tensors", return_tensors)
            text_kwargs.setdefault("return_tensors", return_tensors)

        if images is not None and self.image_processor is not None:
            image_inputs = self.image_processor(images, **images_kwargs)
            data.update(image_inputs)

        if text is None and images is not None and self.tokenizer is not None:
            nested = make_nested_list_of_images(images)
            text = [" ".join([self.image_token] * len(img_list)) for img_list in nested]

        if text is not None:
            text_list = _normalize_text_list(text)
            expanded_text = [self.replace_image_token(t) for t in text_list]
            if self.tokenizer is not None:
                text_inputs = self.tokenizer(expanded_text, **text_kwargs)
                text_inputs.pop("mm_token_type_ids", None)
                data.update(text_inputs)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def prepare_inputs_layout(self, images=None, text=None, **kwargs):
        images, text, *_ = super().prepare_inputs_layout(images=images, text=text, **kwargs)
        if images is not None:
            images = make_nested_list_of_images(images)
            if not text:
                text = [" ".join([self.image_token] * len(image_list)) for image_list in images]
        return images, text, None, None

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)

    @property
    def image_token_ids(self) -> list[int]:
        return [self.image_token_id]

    def replace_image_token(self, image_inputs_or_prompt=None, image_idx: int = 0, **kwargs) -> str:
        if isinstance(image_inputs_or_prompt, str):
            return image_inputs_or_prompt.replace(self.image_token, self.full_image_sequence)
        return self.full_image_sequence


@auto_docstring
class GRTASRProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "GRTAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    valid_processor_kwargs = GRTASRProcessorKwargs

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_seq_length: int = 64,
        **kwargs,
    ):
        r"""
        audio_seq_length (`int`, *optional*, defaults to 64):
            Number of `<|video_pad|>` (or `<|audio_pad|>`) placeholder tokens expanded per audio segment.
        """
        if feature_extractor is None:
            feature_extractor = GRTAudioFeatureExtractor()
        self.audio_seq_length = audio_seq_length
        self.audio_token = "<|video_pad|>"
        self.audio_pad_token = "<|audio_pad|>"
        self.audio_token_id = (
            tokenizer.convert_tokens_to_ids(self.audio_token)
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids")
            else 151656
        )
        self.full_audio_sequence = self.audio_token * self.audio_seq_length

        if feature_extractor is not None and tokenizer is not None:
            try:
                super().__init__(
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    **kwargs,
                )
            except TypeError:
                self.feature_extractor = feature_extractor
                self.tokenizer = tokenizer
                self.chat_template = chat_template
        else:
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer
            self.chat_template = chat_template

    def replace_audio_token(self, prompt: str | dict | None = None, audio_idx: int = 0, **kwargs) -> str:
        if isinstance(prompt, str):
            expanded = prompt.replace(self.audio_token, self.full_audio_sequence)
            if self.audio_pad_token in expanded:
                vocab = self.tokenizer.get_vocab() if self.tokenizer is not None and hasattr(self.tokenizer, "get_vocab") else {}
                if self.audio_pad_token in vocab and self.audio_token not in vocab:
                    expanded = expanded.replace(self.audio_pad_token, self.audio_pad_token * self.audio_seq_length)
                else:
                    expanded = expanded.replace(self.audio_pad_token, self.full_audio_sequence)
            return expanded
        return self.full_audio_sequence

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs: Unpack[GRTASRProcessorKwargs],
    ) -> BatchFeature:
        raw_speech = kwargs.pop("raw_speech", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        if audio is None and raw_speech is not None:
            audio = raw_speech
        if audio is None and text is None:
            raise ValueError(f"You need to provide at least one of `audio` or `text` to call {self.__class__.__name__}.")

        data: dict[str, Any] = {}
        audio_kwargs = dict(kwargs.pop("audio_kwargs", {}) or {})
        text_kwargs = dict(kwargs.pop("text_kwargs", {}) or {})
        if sampling_rate is not None:
            audio_kwargs["sampling_rate"] = sampling_rate
        text_kwargs.update(kwargs)
        if return_tensors is not None:
            audio_kwargs.setdefault("return_tensors", return_tensors)
            text_kwargs.setdefault("return_tensors", return_tensors)

        batch_size = 1
        if audio is not None and self.feature_extractor is not None:
            audio_inputs = self.feature_extractor(audio, **audio_kwargs)
            data["audio_values"] = audio_inputs["audio_values"]
            batch_size = len(audio_inputs["audio_values"])

        if text is None and audio is not None and self.tokenizer is not None:
            text = [self.audio_token] * batch_size

        if text is not None:
            text_list = _normalize_text_list(text)
            expanded_text = [self.replace_audio_token(t) for t in text_list]
            if self.tokenizer is not None:
                text_inputs = self.tokenizer(expanded_text, **text_kwargs)
                text_inputs.pop("mm_token_type_ids", None)
                data.update(text_inputs)

        return BatchFeature(data=data, tensor_type=return_tensors)


@auto_docstring
class GRTOmniProcessor(GRTVLProcessor):
    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "GRTImageProcessor"
    feature_extractor_class = "GRTAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    valid_processor_kwargs = GRTOmniProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        image_seq_length: int = 64,
        audio_seq_length: int = 64,
        **kwargs,
    ):
        r"""
        image_seq_length (`int`, *optional*, defaults to 64):
            Number of `<|image_pad|>` tokens expanded per image.
        audio_seq_length (`int`, *optional*, defaults to 64):
            Number of `<|video_pad|>` audio placeholder tokens expanded per audio segment.
        """
        if image_processor is None:
            image_processor = GRTImageProcessor()
        if feature_extractor is None:
            feature_extractor = GRTAudioFeatureExtractor()

        self.image_seq_length = image_seq_length
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.image_token)
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids")
            else 151655
        )
        image_tokens_expanded = self.image_token * self.image_seq_length
        self.full_image_sequence = f"{self.vision_start_token}{image_tokens_expanded}{self.vision_end_token}"

        self.audio_seq_length = audio_seq_length
        self.audio_token = "<|video_pad|>"
        self.audio_pad_token = "<|audio_pad|>"
        self.audio_token_id = (
            tokenizer.convert_tokens_to_ids(self.audio_token)
            if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids")
            else 151656
        )
        self.full_audio_sequence = self.audio_token * self.audio_seq_length

        if image_processor is not None and feature_extractor is not None and tokenizer is not None:
            try:
                ProcessorMixin.__init__(
                    self,
                    image_processor=image_processor,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    **kwargs,
                )
            except TypeError:
                self.image_processor = image_processor
                self.feature_extractor = feature_extractor
                self.tokenizer = tokenizer
                self.chat_template = chat_template
        else:
            self.image_processor = image_processor
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer
            self.chat_template = chat_template

    def replace_audio_token(self, prompt: str | dict | None = None, audio_idx: int = 0, **kwargs) -> str:
        if isinstance(prompt, str):
            expanded = prompt.replace(self.audio_token, self.full_audio_sequence)
            if self.audio_pad_token in expanded:
                vocab = self.tokenizer.get_vocab() if self.tokenizer is not None and hasattr(self.tokenizer, "get_vocab") else {}
                if self.audio_pad_token in vocab and self.audio_token not in vocab:
                    expanded = expanded.replace(self.audio_pad_token, self.audio_pad_token * self.audio_seq_length)
                else:
                    expanded = expanded.replace(self.audio_pad_token, self.full_audio_sequence)
            return expanded
        return self.full_audio_sequence

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs: Unpack[GRTOmniProcessorKwargs],
    ) -> BatchFeature:
        raw_speech = kwargs.pop("raw_speech", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        if audio is None and raw_speech is not None:
            audio = raw_speech
        if images is None and text is None and audio is None:
            raise ValueError(
                f"You need to provide at least one of `images`, `audio`, or `text` to call {self.__class__.__name__}."
            )

        data: dict[str, Any] = {}
        images_kwargs = dict(kwargs.pop("images_kwargs", {}) or {})
        audio_kwargs = dict(kwargs.pop("audio_kwargs", {}) or {})
        text_kwargs = dict(kwargs.pop("text_kwargs", {}) or {})
        for k in ("do_resize", "size", "image_size", "resample", "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb"):
            if k in kwargs:
                images_kwargs[k] = kwargs.pop(k)
        if sampling_rate is not None:
            audio_kwargs["sampling_rate"] = sampling_rate
        text_kwargs.update(kwargs)
        if return_tensors is not None:
            images_kwargs.setdefault("return_tensors", return_tensors)
            audio_kwargs.setdefault("return_tensors", return_tensors)
            text_kwargs.setdefault("return_tensors", return_tensors)

        batch_size = 1
        if images is not None and self.image_processor is not None:
            image_inputs = self.image_processor(images, **images_kwargs)
            data.update(image_inputs)
            batch_size = len(make_flat_list_of_images(images))

        if audio is not None and self.feature_extractor is not None:
            audio_inputs = self.feature_extractor(audio, **audio_kwargs)
            data["audio_values"] = audio_inputs["audio_values"]
            batch_size = max(batch_size, len(audio_inputs["audio_values"]))

        if text is None and self.tokenizer is not None:
            parts = []
            if images is not None:
                parts.append(self.image_token)
            if audio is not None:
                parts.append(self.audio_token)
            if parts:
                text = [" ".join(parts)] * batch_size

        if text is not None:
            text_list = _normalize_text_list(text)
            expanded_text = [self.replace_audio_token(self.replace_image_token(t)) for t in text_list]
            if self.tokenizer is not None:
                text_inputs = self.tokenizer(expanded_text, **text_kwargs)
                text_inputs.pop("mm_token_type_ids", None)
                data.update(text_inputs)

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["GRTASRProcessor", "GRTOmniProcessor", "GRTVLProcessor"]
