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
"""GRT model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring
class GRTConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GRTModel`]. It is used to instantiate a GRT
    model according to the specified arguments, defining the model architecture.

    GRT (Geometric Rotational Transformer) replaces the dense weight matrices of a standard transformer with chains
    of Givens rotations. Every projection (Q/K/V/O, FFN expand/contract, cross-head mixer) is a sequence of
    `rotate -> scale` stages applied to dimension pairs, and the token embedding stores phase angles rather than
    vectors. There are no `nn.Linear` layers in the model.

    Args:
        vocab_size (`int`, *optional*, defaults to 50000):
            Vocabulary size of the GRT model.
        d_model (`int`, *optional*, defaults to 256):
            Dimensionality of the residual stream. Must be even and divisible by `n_heads`.
        n_heads (`int`, *optional*, defaults to 4):
            Number of attention heads.
        n_blocks (`int`, *optional*, defaults to 12):
            Number of stacked GRT blocks.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied inside attention.
        max_seq_len (`int`, *optional*, defaults to 1024):
            Maximum sequence length the rotational positional encoding supports.
        output_scale_init (`float`, *optional*, defaults to 30.0):
            Initial value of the learnable logit temperature. Stored internally through a softplus
            parameterization, so the effective scale is always positive.
        ffn_mult (`int`, *optional*, defaults to 4):
            Expansion multiplier for the feed-forward width when `d_ff` is not given.
        d_ff (`int`, *optional*):
            Explicit feed-forward width. Defaults to `ffn_mult * d_model`.
        ffn_n_rotations (`int`, *optional*, defaults to 21):
            Number of rotation stages in each of the FFN expand and contract chains.
        n_qkv_rotations (`int`, *optional*, defaults to 11):
            Number of rotation stages in each Q/K/V/O projection chain.
        n_mixer_rotations (`int`, *optional*, defaults to 1):
            Number of rotation stages in the cross-head mixer. `0` disables the mixer.
        coupling_init_scale (`float`, *optional*, defaults to 0.01):
            Initialization scale of the Clifford coupling tensors.
        angle_init_scale (`float`, *optional*, defaults to 0.1):
            Initialization scale of the rotation angles.
        phase_init_scale (`float`, *optional*, defaults to 0.1):
            Initialization scale of the embedding phase angles.
        log_magnitude_init (`float`, *optional*, defaults to 0.0):
            Initial per-token log magnitude of the embedding.
        pos_freq_base (`float`, *optional*, defaults to 10000.0):
            Base of the geometric frequency ladder used by the rotational positional encoding.
        learnable_rope (`bool`, *optional*, defaults to `True`):
            Whether the positional frequencies are learnable parameters.
        coupling_init_mode (`str`, *optional*, defaults to `"diagonal"`):
            Coupling initialization strategy, either `"diagonal"` or `"random"`.
        coupling_features (`str`, *optional*, defaults to `"clifford"`):
            Feature map used for content-dependent angle modulation. One of `"clifford"` (S0, V1, V2, B12),
            `"stokes"` (S0..S3) or `"energy"` (S0 only).
        learnable_residual_gates (`bool`, *optional*, defaults to `True`):
            Whether each residual branch has a learnable gate.
        residual_gate_mode (`str`, *optional*, defaults to `"bounded_depth"`):
            Gate parameterization: `"free"`, `"bounded"`, `"bounded_depth"` or `"none"`.
        residual_norm_mode (`str`, *optional*, defaults to `"none"`):
            Whether to normalize the residual stream after each block. `"none"` or `"pearlnorm"`.
        peri_ln (`bool`, *optional*, defaults to `False`):
            Whether to apply an additional norm after each sublayer.
        ouroboros_passes (`int`, *optional*, defaults to 1):
            Number of times the full block stack is applied to the residual stream.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation used for the (rarely exercised) truncated-normal fallback initializer.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 3):
            End of stream token id.

    ```python
    >>> from transformers import GRTModel, GRTConfig

    >>> # Initializing a GRT configuration
    >>> configuration = GRTConfig()

    >>> # Initializing a model from the configuration
    >>> model = GRTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "grt"
    keys_to_ignore_at_inference = ["past_key_values"]
    # `d_model` / `n_blocks` / `n_heads` are the names used by the released checkpoints; map the
    # generic transformers names onto them so utilities such as `generate` keep working.
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_blocks",
        "max_position_embeddings": "max_seq_len",
    }

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 256,
        n_heads: int = 4,
        n_blocks: int = 12,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        output_scale_init: float = 30.0,
        ffn_mult: int = 4,
        d_ff: int | None = None,
        ffn_n_rotations: int = 21,
        n_qkv_rotations: int = 11,
        n_mixer_rotations: int = 1,
        coupling_init_scale: float = 0.01,
        angle_init_scale: float = 0.1,
        phase_init_scale: float = 0.1,
        log_magnitude_init: float = 0.0,
        pos_freq_base: float = 10000.0,
        learnable_rope: bool = True,
        coupling_init_mode: str = "diagonal",
        coupling_features: str = "clifford",
        learnable_residual_gates: bool = True,
        residual_gate_mode: str = "bounded_depth",
        residual_norm_mode: str = "none",
        peri_ln: bool = False,
        ouroboros_passes: int = 1,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs,
    ):
        if d_model % 2 != 0:
            raise ValueError(f"`d_model` must be even, got {d_model}.")
        if d_model % n_heads != 0:
            raise ValueError(f"`d_model` ({d_model}) must be divisible by `n_heads` ({n_heads}).")
        if coupling_features not in ("clifford", "stokes", "energy"):
            raise ValueError(
                f"`coupling_features` must be one of 'clifford', 'stokes', 'energy', got {coupling_features!r}."
            )
        if coupling_init_mode not in ("diagonal", "random"):
            raise ValueError(f"`coupling_init_mode` must be 'diagonal' or 'random', got {coupling_init_mode!r}.")
        if residual_gate_mode not in ("free", "bounded", "bounded_depth", "none"):
            raise ValueError(f"Unknown `residual_gate_mode`: {residual_gate_mode!r}.")
        if residual_norm_mode not in ("none", "pearlnorm"):
            raise ValueError(f"Unknown `residual_norm_mode`: {residual_norm_mode!r}.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.output_scale_init = output_scale_init
        self.ffn_mult = ffn_mult
        self.d_ff = d_ff
        self.ffn_n_rotations = ffn_n_rotations
        self.n_qkv_rotations = n_qkv_rotations
        self.n_mixer_rotations = n_mixer_rotations
        self.coupling_init_scale = coupling_init_scale
        self.angle_init_scale = angle_init_scale
        self.phase_init_scale = phase_init_scale
        self.log_magnitude_init = log_magnitude_init
        self.pos_freq_base = pos_freq_base
        self.learnable_rope = learnable_rope
        self.coupling_init_mode = coupling_init_mode
        self.coupling_features = coupling_features
        self.learnable_residual_gates = learnable_residual_gates
        self.residual_gate_mode = residual_gate_mode
        self.residual_norm_mode = residual_norm_mode
        self.peri_ln = peri_ln
        self.ouroboros_passes = ouroboros_passes
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def intermediate_size(self) -> int:
        """Feed-forward width, resolved from `d_ff` or `ffn_mult * d_model`."""
        return int(self.d_ff) if self.d_ff is not None else self.ffn_mult * self.d_model


__all__ = ["GRTConfig"]
