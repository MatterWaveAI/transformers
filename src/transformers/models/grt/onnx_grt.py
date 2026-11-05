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
"""ONNX configuration and `torch.export` / `torch.onnx` export utilities for GRT models."""

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .configuration_grt import GRTConfig


try:
    from ...onnx import OnnxConfigWithPast, PatchingSpec  # type: ignore[import-not-found]

    _HAS_TRANSFORMERS_ONNX = True
except ImportError:
    OnnxConfigWithPast = object  # type: ignore[misc,assignment]
    PatchingSpec = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS_ONNX = False


class GRTOnnxConfig(OnnxConfigWithPast):  # type: ignore[misc]
    """ONNX configuration for Geometric Rotation Transformer (GRT) models.

    Provides input/output symbolic axis specifications, dynamic axes mappings, and dummy input generation for
    exporting `GRTModel`, `GRTForCausalLM`, and `GRTForSequenceClassification` via `torch.onnx.export` or
    `torch.export.export`.
    """

    default_onnx_opset: int = 17

    def __init__(
        self,
        config: GRTConfig | None = None,
        task: str = "causal-lm",
        patching_specs: list[Any] | None = None,
        use_past: bool = False,
    ):
        if _HAS_TRANSFORMERS_ONNX:
            super().__init__(config=config, task=task, patching_specs=patching_specs, use_past=use_past)
        else:
            self._config = config
            self.task = task
            self.use_past = use_past
        self.values_override: dict[str, Any] = {"use_cache": False}

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task in ("default", "feature-extraction"):
            return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"})])
        if self.task == "sequence-classification":
            return OrderedDict([("logits", {0: "batch"})])
        return OrderedDict([("logits", {0: "batch", 1: "sequence"})])

    @property
    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        axes: dict[str, dict[int, str]] = {}
        for name, spec in self.inputs.items():
            axes[name] = dict(spec)
        for name, spec in self.outputs.items():
            axes[name] = dict(spec)
        return axes

    def generate_dummy_inputs(
        self,
        batch_size: int = 2,
        seq_length: int = 8,
        vocab_size: int | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Generate dummy `input_ids` and `attention_mask` tensors for tracing or exporting."""
        if vocab_size is None:
            vocab_size = getattr(self._config, "vocab_size", 256) if self._config is not None else 256
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long, device=device)
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class GRTExportWrapper(nn.Module):
    """Thin export wrapper around a GRT model (`GRTForCausalLM`, `GRTModel`, or `GRTForSequenceClassification`).

    Forces `use_cache=False` and unpacks the model's `ModelOutput` into plain tensors so `torch.onnx.export`,
    `torch.jit.trace`, and `torch.export.export` trace without dictionary/dataclass graph breaks:
    - `GRTForCausalLM` / `GRTForSequenceClassification`: returns `logits` (`torch.Tensor`).
    - `GRTModel`: returns `last_hidden_state` (`torch.Tensor`), or `(last_hidden_state, circular_pooled_embedding)`
      when `return_pooled=True` or when the underlying model output includes `circular_pooled_embedding`.
    """

    def __init__(self, model: nn.Module, return_pooled: bool = False):
        super().__init__()
        self.model = model
        self.return_pooled = return_pooled

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "logits") and outputs.logits is not None:
            return outputs.logits
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden_state = outputs.last_hidden_state
            pooled = getattr(outputs, "circular_pooled_embedding", None)
            if pooled is not None:
                return last_hidden_state, pooled
            if self.return_pooled:
                pairs = last_hidden_state.unflatten(-1, (-1, 2))
                radii = torch.norm(pairs, dim=-1, keepdim=True).clamp(min=1e-8)
                unit_pairs = pairs / radii
                if attention_mask is not None:
                    mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1).unsqueeze(-1)
                    mean_pairs = (unit_pairs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                else:
                    mean_pairs = unit_pairs.mean(dim=1)
                return last_hidden_state, mean_pairs.flatten(-2)
            return last_hidden_state
        if isinstance(outputs, (tuple, list)):
            return outputs[0] if len(outputs) == 1 else tuple(outputs)
        raise TypeError(f"Unsupported model output type for GRTExportWrapper: {type(outputs)}")


def export_grt_program(
    model: nn.Module,
    seq_length: int = 16,
    batch_size: int = 2,
) -> torch.export.ExportedProgram:
    """Export a GRT model (`GRTForCausalLM`, `GRTModel`, or `GRTForSequenceClassification`) to a
    `torch.export.ExportedProgram`.
    """
    wrapper = model if isinstance(model, GRTExportWrapper) else GRTExportWrapper(model)
    wrapper.eval()
    base_model = wrapper.model
    config = getattr(base_model, "config", None)
    try:
        device = next(wrapper.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    onnx_cfg = GRTOnnxConfig(config=config)
    dummy = onnx_cfg.generate_dummy_inputs(batch_size=batch_size, seq_length=seq_length, device=device)
    example_args = (dummy["input_ids"], dummy["attention_mask"])

    with torch.no_grad():
        return torch.export.export(wrapper, example_args, strict=False)


def export_grt_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    opset_version: int = 17,
    dynamic_axes: bool = True,
    seq_length: int = 16,
    batch_size: int = 2,
) -> Path:
    """Export a GRT model (`GRTForCausalLM`, `GRTModel`, or `GRTForSequenceClassification`) to an ONNX file.

    Uses `torch.onnx.export` when `onnx` / `onnxscript` is installed, and falls back to saving a traced TorchScript
    / `torch.export` archive at `output_path` if the optional ONNX dependencies are unavailable in the current Python
    environment.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = model if isinstance(model, GRTExportWrapper) else GRTExportWrapper(model)
    wrapper.eval()
    base_model = wrapper.model
    config = getattr(base_model, "config", None)
    try:
        device = next(wrapper.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    task = "causal-lm" if hasattr(base_model, "lm_head") else "default"
    onnx_cfg = GRTOnnxConfig(config=config, task=task)
    dummy = onnx_cfg.generate_dummy_inputs(batch_size=batch_size, seq_length=seq_length, device=device)
    example_args = (dummy["input_ids"], dummy["attention_mask"])
    input_names = list(onnx_cfg.inputs.keys())
    output_names = list(onnx_cfg.outputs.keys())
    dyn_axes = onnx_cfg.dynamic_axes if dynamic_axes else None

    try:
        import onnx  # noqa: F401

        has_onnx = True
    except ImportError:
        has_onnx = False

    if has_onnx:
        with torch.no_grad():
            export_kwargs: dict[str, Any] = {
                "input_names": input_names,
                "output_names": output_names,
                "opset_version": opset_version,
                "do_constant_folding": True,
            }
            if dyn_axes is not None:
                export_kwargs["dynamic_axes"] = dyn_axes
            try:
                torch.onnx.export(wrapper, example_args, str(output_path), dynamo=False, **export_kwargs)
            except TypeError:
                torch.onnx.export(wrapper, example_args, str(output_path), **export_kwargs)
        return output_path

    # Fallback when the `onnx` package is not installed: verify clean tracing with `torch.jit.trace` and save the
    # traced module to `output_path` so callers still receive a serialized graph artifact.
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_args, strict=False)
        traced.save(str(output_path))
    return output_path


__all__ = ["GRTOnnxConfig", "GRTExportWrapper", "export_grt_to_onnx", "export_grt_program"]
