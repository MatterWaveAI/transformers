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
"""Image processor class for GRT-VL and GRT-Omni."""

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class GRTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a GRT image processor for `GRTVisionModel`, `GRTVLForConditionalGeneration`, and
    `GRTOmniForConditionalGeneration`.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's `(height, width)` dimensions to `size`.
        size (`dict[str, int]` or `int`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Target output image size `(height, width)`.
        image_size (`int`, *optional*):
            Square target image resolution override; if provided when `size` is `None`, sets
            `size = {"height": image_size, "width": image_size}`.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1 / 255.0`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image with `image_mean` and `image_std`.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Per-channel mean used when normalizing the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Per-channel standard deviation used when normalizing the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB format.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | int | tuple[int, int] | None = None,
        image_size: int | None = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: int | float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if size is None:
            resolved_size = int(image_size) if image_size is not None else 224
            size = {"height": resolved_size, "width": resolved_size}
        else:
            size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.image_size = int(image_size) if image_size is not None else size.get("height", 224)
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = list(image_mean) if isinstance(image_mean, (list, tuple)) else (
            [float(image_mean)] * 3 if image_mean is not None else [0.5, 0.5, 0.5]
        )
        self.image_std = list(image_std) if isinstance(image_std, (list, tuple)) else (
            [float(image_std)] * 3 if image_std is not None else [0.5, 0.5, 0.5]
        )
        self.do_convert_rgb = do_convert_rgb

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = None,
        size: dict[str, int] | int | tuple[int, int] | None = None,
        image_size: int | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | str = ChannelDimension.FIRST,
        input_data_format: ChannelDimension | str | None = None,
        **kwargs,
    ) -> BatchFeature:
        do_resize = self.do_resize if do_resize is None else do_resize
        if size is not None:
            size = get_size_dict(size, default_to_square=True)
        elif image_size is not None:
            size = {"height": int(image_size), "width": int(image_size)}
        else:
            size = self.size
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor.")

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        processed_images = []
        for image in images:
            img_input_format = (
                infer_channel_dimension_format(image) if input_data_format is None else input_data_format
            )
            if do_resize:
                target_h = size.get("height", size.get("shortest_edge", self.image_size))
                target_w = size.get("width", size.get("shortest_edge", self.image_size))
                image = resize(
                    image,
                    size=(target_h, target_w),
                    resample=resample,
                    input_data_format=img_input_format,
                )

            if do_rescale:
                if is_scaled_image(image):
                    logger.warning_once(
                        "It looks like you are trying to rescale already rescaled images. If the input"
                        " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
                    )
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=img_input_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=img_input_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=img_input_format)
            processed_images.append(np.asarray(image, dtype=np.float32))

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["GRTImageProcessor"]
