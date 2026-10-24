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
"""PyTorch GRT model."""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_grt import GRTConfig


# ---------------------------------------------------------------------------
# Rotational primitives
#
# Every projection in GRT is a chain of `K` stages. One stage takes the input,
# splits it into adjacent dimension pairs `(x0, x1)`, rotates each pair by an
# angle, and applies a per-dimension scale:
#
#     [x0, x1] -> [x0*cos(theta) - x1*sin(theta), x0*sin(theta) + x1*cos(theta)]
#
# A butterfly permutation before each stage changes *which* dimensions are
# paired, so a chain of K stages mixes information across the whole vector the
# way a dense matrix would, but with O(D) parameters per stage instead of O(D^2).
# ---------------------------------------------------------------------------


def _softplus_scale(raw_scale: torch.Tensor) -> torch.Tensor:
    """Map a raw scale parameter to a positive scale.

    Scales are stored unconstrained and passed through softplus so they stay positive
    without the optimizer being able to drive them through zero.
    """
    return F.softplus(raw_scale)


def _raw_scale_init(target_scale: float) -> float:
    """Inverse of `_softplus_scale`, so that `softplus(raw) ~= target_scale`."""
    if target_scale > 20:
        # softplus is numerically the identity this far out.
        return target_scale
    return math.log(math.exp(target_scale) - 1) if target_scale > 0 else 0.0


def _build_butterfly_perms(n_stages: int, dim: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Precompute the butterfly permutation and its inverse for each stage.

    Stage `k` pairs dimensions at stride `2 ** (k % n_levels)` where `n_levels = log2(dim / 2)`.
    Cycling through the strides gives every dimension a path to every other dimension
    after `n_levels` stages.
    """
    half = dim // 2
    n_levels = max(1, int(math.log2(half))) if half > 1 else 1
    perms = []
    for k in range(n_stages):
        stride = 1 << (k % n_levels)
        order = []
        for block_start in range(0, dim, 2 * stride):
            for i in range(stride):
                order.append(block_start + i)
                order.append(block_start + i + stride)
        perm = torch.tensor(order, dtype=torch.long)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(dim)
        perms.append((perm, inv_perm))
    return perms


def _clifford_features(
    x0: torch.Tensor, x1: torch.Tensor, n_features: int, mixer_bivector: bool = False
) -> torch.Tensor:
    """Build the content features that modulate rotation angles.

    The pair `(x0, x1)` is read as a Cl(2,0) multivector. `clifford` (4 features) keeps the
    scalar pair energy `S0`, both vector components `V1`/`V2`, and the cross-pair bivector `B12`;
    `stokes` uses the optical Stokes 4-vector; `energy` keeps `S0` alone.

    The projection and mixer paths use different bivector contractions, and the released
    checkpoints were trained that way, so `mixer_bivector` selects the mixer's variant.
    """
    if n_features == 1:
        return x0 * x0 + x1 * x1
    if n_features == 3:
        return torch.cat([x0 * x0 + x1 * x1, x0 * x0 - x1 * x1, 2.0 * x0 * x1], dim=-1)
    # Clifford: roll by one pair to reach the neighbouring pair for the bivector term.
    x0_next = torch.cat([x0[..., 1:], x0[..., :1]], dim=-1)
    x1_next = torch.cat([x1[..., 1:], x1[..., :1]], dim=-1)
    bivector = x0 * x0_next - x1 * x1_next if mixer_bivector else x0 * x1_next - x1 * x0_next
    return torch.cat(
        [
            x0 * x0 + x1 * x1,  # S0 — pair energy
            x0,  # V1
            x1,  # V2
            bivector,  # B12 — cross-pair bivector
        ],
        dim=-1,
    )


def _coupled_rotate(
    hidden_states: torch.Tensor,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    scales: torch.Tensor,
    perms: torch.Tensor | None,
    inv_perms: torch.Tensor | None,
    n_features: int,
) -> torch.Tensor:
    """Apply `K` content-modulated rotation + scale stages.

    Args:
        hidden_states: `(num_tokens, num_heads, head_dim)`.
        angles: `(K, num_heads, head_dim // 2)` base rotation angles.
        coupling: `(K, num_heads, n_features * head_dim // 2, head_dim // 2)`.
        scales: `(K, num_heads, head_dim)`, already positive.
        perms / inv_perms: `(K, head_dim)` butterfly permutations, or `None` for a single stage.

    Each stage derives a per-pair angle offset from the *content* of the vector, so the
    rotation a token receives depends on the token itself. This is what replaces the
    input-dependence a dense matrix would provide.
    """
    num_stages = angles.shape[0]
    head_dim = hidden_states.shape[-1]
    n_pairs = head_dim // 2
    out = hidden_states

    for k in range(num_stages):
        if perms is not None:
            out = out[..., perms[k]]

        pairs = out.view(*out.shape[:-1], n_pairs, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]

        features = _clifford_features(x0, x1, n_features)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)

        modulation = torch.einsum("nhj,hji->nhi", features, coupling[k])
        theta = angles[k] + torch.tanh(modulation)

        cos, sin = torch.cos(theta), torch.sin(theta)
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        out = rotated.view(*out.shape) * scales[k]

        if inv_perms is not None:
            out = out[..., inv_perms[k]]

    return out


def _plain_rotate(
    hidden_states: torch.Tensor,
    angles: torch.Tensor,
    raw_scales: torch.Tensor,
    perms: torch.Tensor | None,
    inv_perms: torch.Tensor | None,
) -> torch.Tensor:
    """Apply `K` fixed rotation + scale stages (no content modulation).

    Used by the feed-forward chains, where the non-linearity rather than the coupling
    provides content dependence.
    """
    scales = _softplus_scale(raw_scales)
    orig_shape = hidden_states.shape
    out = hidden_states.reshape(-1, orig_shape[-1])
    num_tokens, dim = out.shape
    num_stages, n_pairs = angles.shape

    for k in range(num_stages):
        if perms is not None:
            out = out[:, perms[k]]

        pairs = out.view(num_tokens, n_pairs, 2)
        x0, x1 = pairs[:, :, 0], pairs[:, :, 1]

        cos, sin = torch.cos(angles[k]), torch.sin(angles[k])
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        out = rotated.view(num_tokens, dim) * scales[k]

        if inv_perms is not None:
            out = out[:, inv_perms[k]]

    return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class GRTPearlNorm(nn.Module):
    """Pair-aware RMSNorm.

    Standard RMSNorm divides by the RMS over all dimensions. PearlNorm instead computes
    the radius `r_i = sqrt(x_{2i}^2 + x_{2i+1}^2)` of each pair and divides by the RMS of
    those radii, which rescales pair magnitudes while leaving every phase angle untouched.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d_model = d_model

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pairs = hidden_states.view(*hidden_states.shape[:-1], self.d_model // 2, 2)
        radius_sq = (pairs**2).sum(dim=-1)
        rms = torch.sqrt(radius_sq.mean(dim=-1, keepdim=True) + self.eps)
        return hidden_states / rms * self.scale


class GRTRadialGELU(nn.Module):
    """GELU applied to pair magnitudes, preserving phase.

    For each pair, the radius `r` is gated by `gelu(r) / r`. The direction of the pair is
    untouched, so the activation is non-linear in magnitude but exactly phase-preserving.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pairs = hidden_states.view(*hidden_states.shape[:-1], self.dim // 2, 2)
        radius = torch.norm(pairs, dim=-1, keepdim=True)
        gate = F.gelu(radius) / (radius + self.eps)
        return (pairs * gate).view(*hidden_states.shape)


class GRTEmbedding(nn.Module):
    """Phase-angle token embedding.

    Each token is stored as `d_model // 2` angles rather than `d_model` free values. The
    forward pass emits `(cos, sin)` per angle, placing every token on the unit hypersphere,
    and a per-token log magnitude lets the model move tokens off the sphere when useful.
    """

    def __init__(self, vocab_size: int, d_model: int, phase_init_scale: float = 0.1, log_magnitude_init: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.phase_init_scale = phase_init_scale
        self.phase_angles = nn.Parameter(torch.empty(vocab_size, d_model // 2))
        self.log_magnitude = nn.Parameter(torch.empty(vocab_size))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        angles = self.phase_angles[token_ids]
        pairs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        embeddings = pairs.reshape(*token_ids.shape, self.d_model)
        magnitude = torch.exp(self.log_magnitude[token_ids].clamp(-5.0, 5.0)).unsqueeze(-1)
        return embeddings * magnitude


class GRTRotaryEmbedding(nn.Module):
    """Rotational positional encoding.

    Position is applied as a rotation of each dimension pair by `position * freq_i`, where
    the frequencies follow the usual geometric ladder. Nothing is added to the hidden state;
    position is carried entirely in the phase.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.learnable = config.learnable_rope
        n_pairs = config.d_model // 2
        log_freqs = -torch.arange(0, n_pairs).float() / n_pairs * math.log(config.pos_freq_base)

        if self.learnable:
            self.log_freqs = nn.Parameter(log_freqs)
        else:
            self.register_buffer("log_freqs", log_freqs, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = hidden_states.shape
        pairs = hidden_states.view(batch, seq_len, d_model // 2, 2)

        freqs = torch.exp(self.log_freqs.to(hidden_states.dtype))
        angles = position_ids.to(hidden_states.dtype).unsqueeze(-1) * freqs
        cos, sin = torch.cos(angles), torch.sin(angles)

        x0, x1 = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return rotated.view(batch, seq_len, d_model)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GRTAttention(nn.Module):
    """Multi-head attention whose Q/K/V/O projections are rotation chains.

    Each projection is a chain of `n_qkv_rotations` stages of
    `butterfly shuffle -> content-modulated Givens rotation -> scale`. Attention itself is
    ordinary scaled dot-product attention; only the projections differ from a standard block.
    """

    def __init__(self, config: GRTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_stages = config.n_qkv_rotations
        self.dropout_p = config.dropout
        self.config = config
        self.is_causal = True
        # GRT never rescaled the attention logits beyond the usual 1/sqrt(head_dim).
        self.scaling = self.head_dim**-0.5
        # No grouped-query attention: every query head has its own key/value head.
        self.num_key_value_groups = 1
        self.n_features = {"clifford": 4, "stokes": 4, "energy": 1}[config.coupling_features]

        n_pairs = self.head_dim // 2
        angle_shape = (self.n_stages, self.n_heads, n_pairs)
        scale_shape = (self.n_stages, self.n_heads, self.head_dim)
        coupling_shape = (self.n_stages, self.n_heads, self.n_features * n_pairs, n_pairs)

        for name in ("q", "k", "v", "o"):
            self.register_parameter(f"{name}_angles", nn.Parameter(torch.empty(angle_shape)))
            self.register_parameter(f"{name}_scales", nn.Parameter(torch.empty(scale_shape)))
            self.register_parameter(f"{name}_coupling", nn.Parameter(torch.empty(coupling_shape)))

        if self.n_stages > 1:
            perms = _build_butterfly_perms(self.n_stages, self.head_dim)
            self.register_buffer("_attn_perms", torch.stack([p for p, _ in perms]), persistent=True)
            self.register_buffer("_attn_inv_perms", torch.stack([ip for _, ip in perms]), persistent=True)

    def _project(self, hidden_states, angles, raw_scales, coupling):
        return _coupled_rotate(
            hidden_states,
            angles,
            coupling,
            _softplus_scale(raw_scales),
            getattr(self, "_attn_perms", None),
            getattr(self, "_attn_inv_perms", None),
            self.n_features,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, _ = hidden_states.shape
        num_tokens = batch * seq_len

        # Projections act on each position independently, so a KV cache is exact:
        # re-projecting an earlier token yields the value already stored.
        flat = hidden_states.reshape(num_tokens, self.n_heads, self.head_dim)
        query = self._project(flat, self.q_angles, self.q_scales, self.q_coupling)
        key = self._project(flat, self.k_angles, self.k_scales, self.k_coupling)
        value = self._project(flat, self.v_angles, self.v_scales, self.v_coupling)

        query = query.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx, {"cache_position": cache_position})

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=self.dropout_p if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        out_flat = attn_output.reshape(num_tokens, self.n_heads, self.head_dim)
        out = self._project(out_flat, self.o_angles, self.o_scales, self.o_coupling)
        return out.view(batch, seq_len, self.d_model), attn_weights


class GRTMixer(nn.Module):
    """Cross-head mixing at full `d_model` width.

    Attention mixes within heads; this mixes across them, using the same
    `shuffle -> modulated rotation -> scale` primitive but with a block-diagonal coupling
    so the parameter count stays linear in `d_model`.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_rotations = config.n_mixer_rotations
        self.n_pairs = config.d_model // 2
        self.n_features = {"clifford": 4, "stokes": 4, "energy": 1}[config.coupling_features]
        self.group_size = next((g for g in (16, 8, 4, 2) if self.n_pairs % g == 0), 1)
        n_groups = self.n_pairs // self.group_size

        self.angles = nn.Parameter(torch.empty(self.n_rotations, self.n_pairs))
        self.scales = nn.Parameter(torch.empty(self.n_rotations, config.d_model))
        self.coupling = nn.Parameter(
            torch.empty(self.n_rotations, n_groups, self.n_features * self.group_size, self.group_size)
        )

        perms = _build_butterfly_perms(self.n_rotations, config.d_model)
        self.register_buffer("_perms_stack", torch.stack([p for p, _ in perms]), persistent=True)
        self.register_buffer("_inv_perms_stack", torch.stack([ip for _, ip in perms]), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        group_size = self.group_size
        n_groups = self.n_pairs // group_size
        orig_shape = hidden_states.shape

        for k in range(self.n_rotations):
            hidden_states = hidden_states[..., self._perms_stack[k]]

            flat = hidden_states.reshape(-1, self.d_model)
            num_tokens = flat.shape[0]
            # (tokens, groups, 2 * group_size) -> rotate each group with its own coupling.
            grouped = flat.view(num_tokens, n_groups, 2 * group_size)
            pairs = grouped.view(num_tokens, n_groups, group_size, 2)
            x0, x1 = pairs[..., 0], pairs[..., 1]

            features = _clifford_features(x0, x1, self.n_features, mixer_bivector=True)
            # eps=1e-2 here (not 1e-6) to match the released checkpoints' mixer.
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-2)

            modulation = torch.einsum("ngj,gji->ngi", features, self.coupling[k])
            theta = self.angles[k].view(n_groups, group_size) + torch.tanh(modulation)

            cos, sin = torch.cos(theta), torch.sin(theta)
            rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
            rotated = rotated.view(num_tokens, n_groups, 2 * group_size)
            rotated = rotated * _softplus_scale(self.scales[k]).view(n_groups, 2 * group_size)

            hidden_states = rotated.view(orig_shape)
            hidden_states = hidden_states[..., self._inv_perms_stack[k]]

        return hidden_states


class GRTFeedForward(nn.Module):
    """Feed-forward network built from rotation chains.

    The hidden state is zero-padded up to `intermediate_size`, passed through a chain of
    rotations, gated by `GRTRadialGELU`, passed through a second chain, and truncated back
    to `d_model`. The padding is what makes the expansion possible without a weight matrix:
    the rotations mix signal into the padded dimensions.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.intermediate_size
        self.n_rotations = config.ffn_n_rotations
        n_pairs = self.d_ff // 2
        self.activation = GRTRadialGELU(self.d_ff)

        self.expand_angles = nn.Parameter(torch.empty(self.n_rotations, n_pairs))
        self.expand_scales = nn.Parameter(torch.empty(self.n_rotations, self.d_ff))
        self.contract_angles = nn.Parameter(torch.empty(self.n_rotations, n_pairs))
        self.contract_scales = nn.Parameter(torch.empty(self.n_rotations, self.d_ff))

        if self.n_rotations > 1:
            perms = _build_butterfly_perms(self.n_rotations, self.d_ff)
            self.register_buffer("_ffn_perms", torch.stack([p for p, _ in perms]), persistent=True)
            self.register_buffer("_ffn_inv_perms", torch.stack([ip for _, ip in perms]), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        perms = getattr(self, "_ffn_perms", None)
        inv_perms = getattr(self, "_ffn_inv_perms", None)

        hidden_states = F.pad(hidden_states, (0, self.d_ff - self.d_model))
        hidden_states = _plain_rotate(hidden_states, self.expand_angles, self.expand_scales, perms, inv_perms)
        hidden_states = self.activation(hidden_states)
        hidden_states = _plain_rotate(hidden_states, self.contract_angles, self.contract_scales, perms, inv_perms)
        return hidden_states[..., : self.d_model]


class GRTDecoderLayer(GradientCheckpointingLayer):
    """One GRT block: norm -> attention -> norm -> mixer -> norm -> feed-forward.

    Each sublayer writes into the residual stream through a learnable gate.
    """

    def __init__(self, config: GRTConfig, layer_idx: int):
        super().__init__()
        self.gate_mode = config.residual_gate_mode if config.learnable_residual_gates else "none"

        if self.gate_mode == "free":
            gate_init = 1.0
        elif self.gate_mode == "bounded":
            gate_init = 0.0
        elif self.gate_mode == "bounded_depth":
            # Initialize the effective gate at 1/sqrt(n_blocks) so that the total residual
            # contribution stays O(sqrt(n_blocks)) rather than growing linearly with depth.
            target = 1.0 / max(1, config.n_blocks) ** 0.5
            t = min(0.9999, max(-0.9999, (target - 0.1) / 0.45 - 1.0))
            gate_init = 0.5 * math.log((1 + t) / (1 - t))  # arctanh
        else:
            gate_init = None

        self.gate_init = gate_init
        if gate_init is None:
            self.res_gate_attn = self.res_gate_ffn = self.res_gate_mixer = None
        else:
            self.res_gate_attn = nn.Parameter(torch.empty(()))
            self.res_gate_ffn = nn.Parameter(torch.empty(()))
            self.res_gate_mixer = nn.Parameter(torch.empty(()))

        self.norm1 = GRTPearlNorm(config.d_model)
        self.attention = GRTAttention(config, layer_idx)

        self.mixer = None
        if config.n_mixer_rotations > 0:
            self.norm_mixer = GRTPearlNorm(config.d_model)
            self.mixer = GRTMixer(config)

        self.norm_ffn = GRTPearlNorm(config.d_model)
        self.ffn = GRTFeedForward(config)

        self.post_norm_attn = GRTPearlNorm(config.d_model) if config.peri_ln else None
        self.post_norm_ffn = GRTPearlNorm(config.d_model) if config.peri_ln else None
        self.res_norm_block = GRTPearlNorm(config.d_model) if config.residual_norm_mode == "pearlnorm" else None

    def _gate(self, raw):
        if raw is None:
            return 1.0
        if self.gate_mode == "bounded":
            return 1.0 + torch.tanh(raw) * 0.5
        if self.gate_mode == "bounded_depth":
            return 0.55 + torch.tanh(raw) * 0.45
        return raw

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_out, _ = self.attention(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if self.post_norm_attn is not None:
            attn_out = self.post_norm_attn(attn_out)
        hidden_states = hidden_states + self._gate(self.res_gate_attn) * attn_out

        if self.mixer is not None:
            hidden_states = hidden_states + self._gate(self.res_gate_mixer) * self.mixer(
                self.norm_mixer(hidden_states)
            )

        ffn_out = self.ffn(self.norm_ffn(hidden_states))
        if self.post_norm_ffn is not None:
            ffn_out = self.post_norm_ffn(ffn_out)
        hidden_states = hidden_states + self._gate(self.res_gate_ffn) * ffn_out

        if self.res_norm_block is not None:
            hidden_states = self.res_norm_block(hidden_states)
        return hidden_states


@auto_docstring
class GRTPreTrainedModel(PreTrainedModel):
    config: GRTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GRTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    # `decay_raw` is only consumed by the optional fused FlashGivens kernel, which this
    # implementation does not use; released checkpoints still carry it.
    _keys_to_ignore_on_load_unexpected = [r"blocks\.\d+\.attention\.decay_raw"]
    _can_record_outputs = {
        "hidden_states": GRTDecoderLayer,
        "attentions": GRTAttention,
    }

    def _init_weights(self, module):
        """Angular initialization.

        GRT parameters are angles, scales and couplings rather than weight matrices, so the
        usual normal/Xavier initializers do not apply. Every tensor is initialized in the
        geometry it lives in; anything unrecognized falls back to the base implementation.

        The helpers from `transformers.initialization` are used throughout because they skip
        tensors already marked `_is_hf_initialized`, which is what keeps `from_pretrained`
        from overwriting weights that were just loaded from a checkpoint.
        """
        config = self.config
        if isinstance(module, GRTEmbedding):
            init.normal_(module.phase_angles, mean=0.0, std=config.phase_init_scale)
            init.constant_(module.log_magnitude, config.log_magnitude_init)
        elif isinstance(module, GRTPearlNorm):
            init.ones_(module.scale)
        elif isinstance(module, GRTRotaryEmbedding):
            _init_log_freqs(module.log_freqs, config)
        elif isinstance(module, GRTAttention):
            for name in ("q", "k", "v", "o"):
                init.normal_(getattr(module, f"{name}_angles"), mean=0.0, std=config.angle_init_scale)
                _init_scale_tensor(getattr(module, f"{name}_scales"), 1.0)
                _init_coupling_tensor(
                    getattr(module, f"{name}_coupling"), config.coupling_init_scale, config.coupling_init_mode
                )
            _init_butterfly_buffers(module, "_attn_perms", "_attn_inv_perms", module.n_stages, module.head_dim)
        elif isinstance(module, GRTMixer):
            init.normal_(module.angles, mean=0.0, std=config.angle_init_scale)
            _init_scale_tensor(module.scales, 1.0)
            _init_coupling_tensor(module.coupling, config.coupling_init_scale, config.coupling_init_mode)
            _init_butterfly_buffers(module, "_perms_stack", "_inv_perms_stack", module.n_rotations, module.d_model)
        elif isinstance(module, GRTFeedForward):
            init.normal_(module.expand_angles, mean=0.0, std=config.angle_init_scale)
            init.normal_(module.contract_angles, mean=0.0, std=config.angle_init_scale)
            _init_scale_tensor(module.expand_scales, 1.0)
            _init_scale_tensor(module.contract_scales, 1.0)
            _init_butterfly_buffers(module, "_ffn_perms", "_ffn_inv_perms", module.n_rotations, module.d_ff)
        elif isinstance(module, GRTModel):
            init.constant_(module.output_scale, _raw_scale_init(config.output_scale_init))
        elif isinstance(module, GRTDecoderLayer):
            for gate in (module.res_gate_attn, module.res_gate_ffn, module.res_gate_mixer):
                if gate is not None:
                    init.constant_(gate, module.gate_init)
        else:
            super()._init_weights(module)


def _init_log_freqs(tensor: torch.Tensor, config: GRTConfig) -> None:
    """Initialize the positional frequency ladder."""
    if getattr(tensor, "_is_hf_initialized", False):
        return
    n_pairs = config.d_model // 2
    log_freqs = -torch.arange(0, n_pairs, dtype=tensor.dtype, device=tensor.device) / n_pairs
    init.copy_(tensor, log_freqs * math.log(config.pos_freq_base))


def _init_butterfly_buffers(module: nn.Module, perm_name: str, inv_name: str, n_stages: int, dim: int) -> None:
    """Rebuild a module's butterfly permutation buffers.

    The permutations are a deterministic function of `(n_stages, dim)`, but they still have
    to be reconstructed here so that a model materialized from the meta device ends up with
    real values rather than empty tensors.
    """
    perm_buffer = getattr(module, perm_name, None)
    if perm_buffer is None:
        return
    perms = _build_butterfly_perms(n_stages, dim)
    init.copy_(perm_buffer, torch.stack([p for p, _ in perms]).to(perm_buffer.device))
    init.copy_(getattr(module, inv_name), torch.stack([ip for _, ip in perms]).to(perm_buffer.device))


def _init_scale_tensor(tensor: torch.Tensor, target_scale: float, spread: float = 0.01) -> None:
    """Initialize a scale tensor with a small per-element spread.

    Without the spread every scale would start identical and receive near-identical
    gradients, leaving them stuck at their initial value.
    """
    if getattr(tensor, "_is_hf_initialized", False):
        return
    raw_center = _raw_scale_init(target_scale)
    # Compensate for the softplus gradient so the spread is ~`spread` in output space.
    sigmoid_center = 1.0 / (1.0 + math.exp(-raw_center))
    raw_spread = spread / max(sigmoid_center, 0.1)
    values = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    init.copy_(tensor, values * raw_spread + raw_center)


def _init_coupling_tensor(tensor: torch.Tensor, scale: float, mode: str) -> None:
    """Initialize a coupling tensor.

    `diagonal` adds a strong self-coupling term on top of the random off-diagonal, so each
    pair is initially modulated mostly by its own features.
    """
    if getattr(tensor, "_is_hf_initialized", False):
        return
    values = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device) * scale
    if mode == "diagonal" and tensor.dim() >= 2:
        dim_out, dim_in = tensor.shape[-1], tensor.shape[-2]
        if dim_in == dim_out:
            values = values + torch.eye(dim_out, device=tensor.device, dtype=tensor.dtype) * (scale * 10)
        elif dim_in > dim_out and dim_in % dim_out == 0:
            eye = torch.eye(dim_out, device=tensor.device, dtype=tensor.dtype) * (scale * 10)
            values[..., :dim_out, :] = values[..., :dim_out, :] + eye
    init.copy_(tensor, values)


@auto_docstring
class GRTModel(GRTPreTrainedModel):
    def __init__(self, config: GRTConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        self.embedding = GRTEmbedding(
            config.vocab_size, config.d_model, config.phase_init_scale, config.log_magnitude_init
        )
        self.pos_encoding = GRTRotaryEmbedding(config)
        self.blocks = nn.ModuleList([GRTDecoderLayer(config, i) for i in range(config.n_blocks)])
        self.final_norm = GRTPearlNorm(config.d_model)
        self.output_scale = nn.Parameter(torch.empty(()))
        self.ouroboros_passes = max(1, config.ouroboros_passes)

        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
            cache in the correct position and to infer the complete sequence length.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = self.pos_encoding(inputs_embeds, position_ids)

        # `ouroboros_passes > 1` runs the whole stack repeatedly over the same residual
        # stream. Caching is only meaningful on the final pass, since earlier passes would
        # otherwise append duplicate keys for the same positions.
        for pass_idx in range(self.ouroboros_passes):
            is_last_pass = pass_idx == self.ouroboros_passes - 1
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_values=past_key_values if is_last_pass else None,
                    cache_position=cache_position,
                    **kwargs,
                )

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class GRTForCausalLM(GRTPreTrainedModel, GenerationMixin):
    """GRT with a language modeling head.

    There is no output projection. Logits are cosine similarities between the final hidden
    state and every token's embedding, multiplied by a single learnable temperature, which
    ties the head to the input embedding by construction.
    """

    def __init__(self, config: GRTConfig):
        super().__init__(config)
        self.model = GRTModel(config)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
            cache in the correct position and to infer the complete sequence length.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GRTForCausalLM

        >>> model = GRTForCausalLM.from_pretrained("matterwaveai/grt-10m")
        >>> tokenizer = AutoTokenizer.from_pretrained("matterwaveai/grt-10m")

        >>> prompt = "The capital of France is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> generate_ids = model.generate(inputs.input_ids, max_new_tokens=8)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        # Cosine-similarity head against the full (tied) embedding table.
        token_embeddings = self.model.embedding(torch.arange(self.vocab_size, device=hidden_states.device))
        hidden_norm = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-6)
        emb_norm = token_embeddings / (token_embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        logits = torch.matmul(hidden_norm, emb_norm.T) * _softplus_scale(self.model.output_scale)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["GRTForCausalLM", "GRTModel", "GRTPreTrainedModel"]
