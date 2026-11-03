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
"""Processor class for GRT-VL."""

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class GRTVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
    }


@auto_docstring
class GRTVLProcessor(ProcessorMixin):
    valid_processor_kwargs = GRTVLProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_seq_length: int = 64,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.image_token) if tokenizer is not None else 151655
        )
        image_tokens_expanded = self.image_token * self.image_seq_length
        self.full_image_sequence = f"{self.vision_start_token}{image_tokens_expanded}{self.vision_end_token}"

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[GRTVLProcessorKwargs],
    ) -> BatchFeature:
        if text is not None and not isinstance(text, str):
            if not isinstance(text, list) or not isinstance(text[0], str):
                raise TypeError("Invalid input text. Please provide a string, or a list of strings.")

        model_inputs = super().__call__(images=images, text=text, **kwargs)
        model_inputs.pop("mm_token_type_ids", None)
        return model_inputs

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


__all__ = ["GRTVLProcessor"]
