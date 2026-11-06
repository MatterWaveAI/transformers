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
"""
Riemannian and Second-Order Toroidal Optimizers (`PairAdam` and `HessianPairAdam` / `Sophia-Torus`)
for Geometric Rotation Transformers (GRT).
"""

from __future__ import annotations

import math
import weakref
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


def _clamp_rotor3_update_(update: torch.Tensor, limit: float) -> None:
    """Isotropic geodesic-ball step clamp `||Delta omega||_2 <= max_angular_step`."""
    u_norm = update.norm(dim=-1, keepdim=True)
    scale = torch.clamp(limit / (u_norm + 1e-12), max=1.0)
    update.mul_(scale)


def _wrap_rotor3_(p_data: torch.Tensor) -> None:
    """Radial `2 * pi` ball wrapping on `so(3)` rotation vectors `[..., 3]`."""
    norm = p_data.norm(dim=-1, keepdim=True)
    wrapped_norm = (norm + math.pi) % (2.0 * math.pi) - math.pi
    scale = torch.where(norm > 1e-12, wrapped_norm / norm.clamp(min=1e-12), torch.ones_like(norm))
    p_data.mul_(scale)


def pair_model_angles_and_scales(model: nn.Module) -> dict[int, weakref.ReferenceType]:
    """Pair each rotation angle parameter in `model` with its sibling scale parameter.

    Uses Theorem 2.1 (Cauchy-Riemann Polar Scale-Angle Curvature Duality): for any scaled
    Givens rotation stage `y = softplus(s_raw) * R(theta) x`, the exact intrinsic torus
    curvature `c_theta = -<grad_y L, y>` equals:

        `c_{theta_p} = -sum_{i in {0,1}} (softplus(s_{i,p}) / sigmoid(s_{i,p})) * grad_{s_{i,p}}`

    Attaches a weakref `p._scale_partner_ref` (and `p._is_log_magnitude_partner`) on
    each matched angle parameter and returns `{id(angle_param): weakref.ref(scale_param)}`.
    """
    named_params = dict(model.named_parameters())
    scale_partners: dict[int, weakref.ReferenceType] = {}

    for name, p in named_params.items():
        partner = None
        is_log_mag = False

        if "embedding.phase_angles" in name or name.endswith("phase_angles"):
            cand_log = name.replace("phase_angles", "log_magnitude")
            partner = named_params.get(cand_log)
            if partner is not None:
                is_log_mag = True
            else:
                cand_pair = name.replace("phase_angles", "pair_scales")
                partner = named_params.get(cand_pair)
                if partner is None:
                    cand_out = name.replace("phase_angles", "output_scale")
                    partner = named_params.get(cand_out)
        elif name.endswith("_angles"):
            cand = name[: -len("_angles")] + "_scales"
            partner = named_params.get(cand)
            if partner is None and ("_base_angles" in name or "_delta_angles" in name):
                cand_expert = (
                    name.replace("expand_base_angles", "expert_expand_scales")
                    .replace("contract_base_angles", "expert_contract_scales")
                    .replace("expand_delta_angles", "expert_expand_scales")
                    .replace("contract_delta_angles", "expert_contract_scales")
                )
                partner = named_params.get(cand_expert)
        elif name.endswith(".angles") or name == "angles":
            cand = name[: -len("angles")] + "scales"
            partner = named_params.get(cand)

        if partner is not None:
            ref = weakref.ref(partner)
            scale_partners[id(p)] = ref
            p._scale_partner_ref = ref
            p._is_log_magnitude_partner = is_log_mag

    return scale_partners


def _resolve_scale_partner(
    angle_param: torch.Tensor,
    group_scale_partners: dict | None = None,
) -> tuple[torch.Tensor | None, bool]:
    """Resolve the paired raw scale parameter and `is_log_magnitude` flag for `angle_param`."""
    is_log_mag = bool(getattr(angle_param, "_is_log_magnitude_partner", False))
    partner = None

    if group_scale_partners:
        entry = group_scale_partners.get(id(angle_param))
        if entry is None:
            try:
                entry = group_scale_partners.get(angle_param)
            except TypeError:
                entry = None
        if isinstance(entry, tuple):
            entry, is_log_mag = entry[0], bool(entry[1])
        if isinstance(entry, weakref.ReferenceType):
            partner = entry()
        elif isinstance(entry, torch.Tensor):
            partner = entry

    if partner is None:
        ref = getattr(angle_param, "_scale_partner_ref", None)
        if isinstance(ref, weakref.ReferenceType):
            partner = ref()
        elif isinstance(ref, torch.Tensor):
            partner = ref

    if partner is None:
        direct = getattr(angle_param, "scale_partner", None)
        if isinstance(direct, weakref.ReferenceType):
            partner = direct()
        elif isinstance(direct, torch.Tensor):
            partner = direct

    return partner, is_log_mag


@torch.no_grad()
def compute_intrinsic_angle_hessian(
    angle_param: torch.Tensor,
    paired_raw_scale: torch.Tensor | None = None,
    is_log_magnitude: bool = False,
    group_scale_partners: dict | None = None,
    scale_partner: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exact intrinsic toroidal curvature `c_theta = -<grad_y L, y>`.

    Priority:
      1. Explicit `angle_param.hessian_diag` (or `angle_param.hess_intr`) populated by
         a fused CUDA kernel or explicit analytical hook.
      2. Theorem 2.1 Cauchy-Riemann Polar Scale-Angle Curvature Duality from the paired
         raw scale parameter `s_raw` and `s_raw.grad`:
         `c_{theta_p} = -sum_{i in {0,1}} (softplus(s_{i,p}) / sigmoid(s_{i,p})) * dL/ds_{i,p}`
         (supports `s_raw` shaped `[..., 2*P]` or `[E, ..., 2*P]` vs `angle_param` `[..., P]`,
         `GRTCosineClassificationHead.pair_scales` `[L, P]`, and `GRTEmbedding.log_magnitude`
         `[V]` vs `phase_angles` `[V, P]`).
      3. Fallback to zeros `torch.zeros_like(angle_param)`.
    """
    hess_diag = getattr(angle_param, "hessian_diag", None)
    if hess_diag is None:
        hess_diag = getattr(angle_param, "hess_intr", None)
    if hess_diag is not None:
        return hess_diag.to(device=angle_param.device, dtype=angle_param.dtype).view_as(angle_param)

    if paired_raw_scale is None and scale_partner is not None:
        paired_raw_scale = scale_partner
    if paired_raw_scale is None:
        paired_raw_scale, is_log_magnitude = _resolve_scale_partner(angle_param, group_scale_partners)

    if paired_raw_scale is None or paired_raw_scale.grad is None:
        return torch.zeros_like(angle_param)

    s_grad = paired_raw_scale.grad
    if is_log_magnitude or (
        paired_raw_scale.dim() == 1
        and angle_param.dim() == 2
        and paired_raw_scale.shape[0] == angle_param.shape[0]
    ):
        # GRTEmbedding: log_magnitude.grad[v] = sum_p <grad_{e_{v,p}} L, e_{v,p}> = -sum_p c_{phi_{v,p}}
        n_pairs = max(1, angle_param.shape[-1])
        return (-s_grad / float(n_pairs)).unsqueeze(-1).expand_as(angle_param).to(dtype=angle_param.dtype)

    work_dtype = torch.float64 if paired_raw_scale.dtype == torch.float64 else torch.float32
    if paired_raw_scale.numel() == 2 * angle_param.numel():
        s_raw_f = paired_raw_scale.detach().to(dtype=work_dtype)
        s_grad_f = s_grad.detach().to(dtype=work_dtype)
        s_act = F.softplus(s_raw_f)
        s_sig = torch.sigmoid(s_raw_f).clamp(min=1e-12 if work_dtype == torch.float64 else 1e-6)
        radial_dot = (s_act / s_sig) * s_grad_f
        radial_pairs = radial_dot.reshape(*angle_param.shape, 2)
        c_theta = -(radial_pairs[..., 0] + radial_pairs[..., 1])
        return c_theta.to(dtype=angle_param.dtype)

    if (
        paired_raw_scale.dim() == angle_param.dim() + 1
        and paired_raw_scale.shape[1:-1] == angle_param.shape[:-1]
        and paired_raw_scale.shape[-1] == 2 * angle_param.shape[-1]
    ):
        # Residual MoR base angles [n_rotations, n_pairs] paired with expert_*_scales [n_experts, n_rotations, 2*n_pairs]
        s_raw_f = paired_raw_scale.detach().to(dtype=work_dtype)
        s_grad_f = s_grad.detach().to(dtype=work_dtype)
        s_act = F.softplus(s_raw_f)
        s_sig = torch.sigmoid(s_raw_f).clamp(min=1e-12 if work_dtype == torch.float64 else 1e-6)
        radial_dot = (s_act / s_sig) * s_grad_f
        radial_pairs = radial_dot.reshape(paired_raw_scale.shape[0], *angle_param.shape, 2)
        c_theta = -(radial_pairs[..., 0] + radial_pairs[..., 1]).sum(dim=0)
        return c_theta.to(dtype=angle_param.dtype)

    if paired_raw_scale.shape == angle_param.shape:
        s_raw_f = paired_raw_scale.detach().to(dtype=work_dtype)
        s_grad_f = s_grad.detach().to(dtype=work_dtype)
        s_act = F.softplus(s_raw_f)
        s_sig = torch.sigmoid(s_raw_f).clamp(min=1e-12 if work_dtype == torch.float64 else 1e-6)
        return (-(s_act / s_sig) * s_grad_f).to(dtype=angle_param.dtype)

    if paired_raw_scale.numel() == 1:
        # Scalar output_scale paired with phase_angles [E, P] (e.g. GRTRotationalRouter or untied GRTCosineHead)
        s_raw_f = paired_raw_scale.detach().to(dtype=work_dtype)
        s_grad_f = s_grad.detach().to(dtype=work_dtype)
        s_act = F.softplus(s_raw_f)
        s_sig = torch.sigmoid(s_raw_f).clamp(min=1e-12 if work_dtype == torch.float64 else 1e-6)
        c_scalar = -(s_act / s_sig) * s_grad_f / float(max(1, angle_param.numel()))
        return c_scalar.expand_as(angle_param).to(dtype=angle_param.dtype)

    return torch.zeros_like(angle_param)


class PairAdam(Optimizer):
    """AdamW with factored second moments for matrix parameters and toroidal wrapping for angle parameters.

    Args:
        params: Iterable of parameters or param groups.
        lr (`float`, *optional*, defaults to `1e-3`):
            Learning rate.
        betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.999)`):
            Coefficients for computing running averages of gradient and its square.
        eps (`float`, *optional*, defaults to `1e-8`):
            Term added to the denominator to improve numerical stability.
        weight_decay (`float`, *optional*, defaults to `0.01`):
            Decoupled weight decay coefficient.
        symmetrize_2d (`bool`, *optional*, defaults to `False`):
            If `True`, enforce `W = 0.5 * (W + W.T)` on square 2D parameters after each step.
        max_angular_step (`float`, *optional*):
            If set, clip the per-element update so that `|Delta theta| <= max_angular_step` radians per step.
        force_factored (`bool`, *optional*, defaults to `False`):
            If `True`, reshape ND parameters (`dim >= 2`) to 2D for factored second-moment estimation.
        nesterov (`bool`, *optional*, defaults to `False`):
            If `True`, apply Dozat Nadam first-moment lookahead before angular clamping and wrapping.
        is_rotor3 (`bool`, *optional*, defaults to `False`):
            If `True`, treat parameters as `so(3)` rotation vectors `[..., 3]` with geodesic-ball clamping.
        hessian_mode (`bool`, *optional*, defaults to `False`):
            If `True`, enable second-order Riemannian (`Sophia-Torus`) curvature preconditioning and antipodal
            saddle-point escape on angle parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        symmetrize_2d: bool = False,
        max_angular_step: float | None = None,
        force_factored: bool = False,
        nesterov: bool = False,
        is_rotor3: bool = False,
        hessian_mode: bool = False,
        gamma_gn: float = 1.0,
        curv_weight: float = 0.25,
        tau_saddle: float = 1e-4,
        eps_kick: float = 1e-12,
        antipodal_kick: float = 0.2,
        eps_h: float = 1e-6,
        sophia_rho: float = 5.0,
        scale_partners: dict | None = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            symmetrize_2d=symmetrize_2d,
            max_angular_step=max_angular_step,
            max_step=None,
            force_factored=force_factored,
            nesterov=nesterov,
            is_rotor3=is_rotor3,
            hessian_mode=hessian_mode,
            gamma_gn=gamma_gn,
            curv_weight=curv_weight,
            tau_saddle=tau_saddle,
            eps_kick=eps_kick,
            antipodal_kick=antipodal_kick,
            eps_h=eps_h,
            sophia_rho=sophia_rho,
        )
        super().__init__(params, defaults)
        if scale_partners is not None:
            for group in self.param_groups:
                group.setdefault("scale_partners", {}).update(scale_partners)

    def _foreach_diagonal_step(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        step,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        max_ang,
        is_angle,
        nesterov=False,
        is_rotor3=False,
    ):
        """Batched diagonal AdamW step using `torch._foreach_*` multi-tensor ops."""
        bc1 = 1 - beta1**step
        bc2 = 1 - beta2**step

        if wd != 0:
            torch._foreach_mul_(params, 1 - lr * wd)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        denom = torch._foreach_sqrt(torch._foreach_div(exp_avg_sqs, bc2))
        torch._foreach_add_(denom, eps)

        if is_rotor3:
            bc1_next = 1 - beta1 ** (step + 1)
            limit = (max_ang / lr) if (max_ang is not None and lr > 0) else None
            for p, g, m, d in zip(params, grads, exp_avgs, denom):
                if nesterov:
                    m_hat = m * (beta1 / bc1_next) + g * ((1 - beta1) / bc1)
                else:
                    m_hat = m / bc1
                update = m_hat / d
                if limit is not None:
                    _clamp_rotor3_update_(update, limit)
                p.add_(update, alpha=-lr)
                _wrap_rotor3_(p.data)
            return

        if nesterov:
            bc1_next = 1 - beta1 ** (step + 1)
            m_hats = torch._foreach_mul(exp_avgs, beta1 / bc1_next)
            torch._foreach_add_(m_hats, grads, alpha=(1 - beta1) / bc1)

            if max_ang is not None and lr > 0:
                limit = max_ang / lr
                updates = m_hats
                torch._foreach_div_(updates, denom)
                torch._foreach_clamp_min_(updates, -limit)
                torch._foreach_clamp_max_(updates, limit)
                torch._foreach_add_(params, updates, alpha=-lr)
            else:
                torch._foreach_addcdiv_(params, m_hats, denom, value=-lr)
        else:
            step_size = lr / bc1
            if max_ang is not None and lr > 0:
                limit = max_ang / lr
                updates = torch._foreach_div(exp_avgs, bc1)
                torch._foreach_div_(updates, denom)
                torch._foreach_clamp_min_(updates, -limit)
                torch._foreach_clamp_max_(updates, limit)
                torch._foreach_add_(params, updates, alpha=-lr)
            else:
                torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)

        if is_angle:
            for p in params:
                p.data.copy_((p.data + math.pi) % (2 * math.pi) - math.pi)

    def _foreach_hessian_angle_step(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        hess_emas,
        curv_emas,
        c_thetas,
        step,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        max_ang,
        nesterov=False,
        gamma_gn=1.0,
        curv_weight=0.25,
        tau_saddle=1e-4,
        eps_kick=1e-12,
        antipodal_kick=0.2,
        eps_h=1e-6,
        sophia_rho=5.0,
    ):
        """Batched Second-Order Riemannian (`Sophia-Torus`) step for angle parameters on `(S^1)^P`."""
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        bc1_next = 1.0 - beta1 ** (step + 1)

        if wd != 0:
            torch._foreach_mul_(params, 1.0 - lr * wd)

        # 1. Update tangential torque moment m_t and Fisher moment v_t
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        # 2. Update signed intrinsic curvature EMA c_t and full Riemannian Hessian EMA h_t
        torch._foreach_mul_(curv_emas, beta2)
        torch._foreach_add_(curv_emas, c_thetas, alpha=1.0 - beta2)

        torch._foreach_mul_(hess_emas, beta2)
        torch._foreach_add_(hess_emas, c_thetas, alpha=1.0 - beta2)
        if gamma_gn != 0.0:
            torch._foreach_addcmul_(hess_emas, grads, grads, value=(1.0 - beta2) * gamma_gn)

        limit = (max_ang / lr) if (max_ang is not None and lr > 0.0) else None
        if sophia_rho is not None and sophia_rho > 0.0:
            limit = min(limit, sophia_rho) if limit is not None else sophia_rho

        for p, g, m, v, h in zip(params, grads, exp_avgs, exp_avg_sqs, hess_emas):
            if nesterov:
                m_hat = m * (beta1 / bc1_next) + g * ((1.0 - beta1) / bc1)
            else:
                m_hat = m / bc1
            v_hat = v / bc2
            h_hat = h / bc2

            # 3. Antipodal Saddle-Point Escape (where h_hat < -tau_saddle and |m_hat| <= |h_hat|)
            # and Positive-Definite Riemannian Curvature Preconditioning (branchless on device)
            at_saddle = (h_hat < -tau_saddle) & (m_hat.abs() <= h_hat.abs())
            pos_curv = (v_hat.sqrt() + curv_weight * h_hat.clamp(min=0.0) + eps).clamp(min=eps_h)
            kick_sign = torch.where(
                m_hat.abs() > eps_kick,
                torch.sign(m_hat),
                torch.where(m_hat + eps_kick >= 0.0, torch.ones_like(m_hat), -torch.ones_like(m_hat)),
            )
            m_eff = torch.where(
                at_saddle,
                m_hat + kick_sign * (antipodal_kick * h_hat.abs()),
                m_hat,
            )
            update = m_eff / pos_curv
            if limit is not None:
                update.clamp_(-limit, limit)

            p.add_(update, alpha=-lr)
            p.data.copy_((p.data + math.pi) % (2.0 * math.pi) - math.pi)

    @torch.no_grad()
    def step(self, closure=None):
        """Single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            symmetrize = group.get("symmetrize_2d", False)
            force_factored = group.get("force_factored", False)
            force_diagonal = group.get("force_diagonal", False)
            is_angle = group.get("is_angle", False) or (group.get("group_type") == "angle")
            nesterov = group.get("nesterov", False)
            is_rotor3 = group.get("is_rotor3", False)
            hessian_mode = group.get("hessian_mode", False)
            gamma_gn = group.get("gamma_gn", 1.0)
            curv_weight = group.get("curv_weight", 0.25)
            tau_saddle = group.get("tau_saddle", 1e-4)
            eps_kick = group.get("eps_kick", 1e-12)
            antipodal_kick = group.get("antipodal_kick", 0.2)
            eps_h = group.get("eps_h", 1e-6)
            sophia_rho = group.get("sophia_rho", 5.0)
            scale_partners = group.get("scale_partners", None)

            max_ang_cfg = group.get("max_angular_step", None)
            max_step_cfg = group.get("max_step", None)
            if max_ang_cfg is not None and max_step_cfg is not None:
                raise ValueError(
                    "PairAdam param group has both max_angular_step and max_step set; "
                    "pick one (they apply the same per-element clip)."
                )
            max_ang = max_ang_cfg if max_ang_cfg is not None else max_step_cfg

            factored_list = []
            diag_params = []
            diag_grads = []
            diag_exp_avgs = []
            diag_exp_avg_sqs = []
            hess_params = []
            hess_grads = []
            hess_exp_avgs = []
            hess_exp_avg_sqs = []
            hess_emas = []
            hess_curv_emas = []
            hess_c_thetas = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PairAdam does not support sparse gradients")

                if is_rotor3 and (p.dim() == 0 or p.shape[-1] != 3):
                    raise ValueError(
                        f"PairAdam is_rotor3 requires parameter last dimension to be 3, got shape {list(p.shape)}"
                    )

                state = self.state[p]

                has_explicit_hess = (
                    getattr(p, "hessian_diag", None) is not None or getattr(p, "hess_intr", None) is not None
                )
                partner_scale, is_log_mag = (
                    _resolve_scale_partner(p, scale_partners) if hessian_mode else (None, False)
                )
                use_hessian_angle = (
                    hessian_mode and not is_rotor3 and (is_angle or has_explicit_hess or partner_scale is not None)
                )

                if use_hessian_angle:
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["hessian_ema"] = torch.zeros_like(p)
                        state["exp_avg_curv"] = torch.zeros_like(p)
                    elif "hessian_ema" not in state:
                        state["hessian_ema"] = torch.zeros_like(p)
                        state["exp_avg_curv"] = torch.zeros_like(p)

                    state["step"] += 1
                    c_theta = compute_intrinsic_angle_hessian(
                        p,
                        paired_raw_scale=partner_scale,
                        is_log_magnitude=is_log_mag,
                        group_scale_partners=scale_partners,
                    )
                    hess_params.append(p)
                    hess_grads.append(grad)
                    hess_exp_avgs.append(state["exp_avg"])
                    hess_exp_avg_sqs.append(state["exp_avg_sq"])
                    hess_emas.append(state["hessian_ema"])
                    hess_curv_emas.append(state["exp_avg_curv"])
                    hess_c_thetas.append(c_theta)
                    continue

                if force_diagonal or is_rotor3:
                    use_factored = False
                else:
                    use_factored = (p.dim() == 2) or (force_factored and p.dim() >= 2)

                if use_factored:
                    if p.dim() > 2:
                        rows_2d = 1
                        for d in p.shape[:-1]:
                            rows_2d *= d
                        cols_2d = p.shape[-1]
                    else:
                        rows_2d = p.shape[0]
                        cols_2d = p.shape[1]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["row_factor"] = torch.zeros(rows_2d, 1, device=p.device, dtype=p.dtype)
                        state["col_factor"] = torch.zeros(1, cols_2d, device=p.device, dtype=p.dtype)

                    state["step"] += 1
                    factored_list.append((p, grad, state, rows_2d, cols_2d))
                else:
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    diag_params.append(p)
                    diag_grads.append(grad)
                    diag_exp_avgs.append(state["exp_avg"])
                    diag_exp_avg_sqs.append(state["exp_avg_sq"])

            for p, grad, state, rows_2d, cols_2d in factored_list:
                step = state["step"]
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step

                if wd != 0:
                    p.mul_(1 - lr * wd)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                row_factor = state["row_factor"]
                col_factor = state["col_factor"]

                grad_2d = grad.view(rows_2d, cols_2d)
                grad_sq = grad_2d * grad_2d

                row_factor.mul_(beta2).add_(grad_sq.mean(dim=1, keepdim=True), alpha=1 - beta2)
                col_factor.mul_(beta2).add_(grad_sq.mean(dim=0, keepdim=True), alpha=1 - beta2)

                row_bc = row_factor / bc2
                col_bc = col_factor / bc2
                row_mean = row_bc.mean().clamp(min=eps)
                v_hat_2d = (row_bc * col_bc) / row_mean

                if nesterov:
                    bc1_next = 1 - beta1 ** (step + 1)
                    m_hat = exp_avg * (beta1 / bc1_next) + grad * ((1 - beta1) / bc1)
                else:
                    m_hat = exp_avg / bc1
                m_hat_2d = m_hat.view(rows_2d, cols_2d)
                update_2d = m_hat_2d / v_hat_2d.sqrt().add_(eps)

                if max_ang is not None and lr > 0:
                    limit = max_ang / lr
                    update_2d.clamp_(-limit, limit)

                update = update_2d.view_as(p)
                p.add_(update, alpha=-lr)

                if is_angle:
                    p.data.copy_((p.data + math.pi) % (2 * math.pi) - math.pi)

                if symmetrize and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.copy_((p + p.t()) * 0.5)

            if diag_params:
                step_count = max(self.state[p]["step"] for p in diag_params)
                self._foreach_diagonal_step(
                    diag_params,
                    diag_grads,
                    diag_exp_avgs,
                    diag_exp_avg_sqs,
                    step_count,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    wd,
                    max_ang,
                    is_angle,
                    nesterov=nesterov,
                    is_rotor3=is_rotor3,
                )

            if hess_params:
                step_count = max(self.state[p]["step"] for p in hess_params)
                self._foreach_hessian_angle_step(
                    hess_params,
                    hess_grads,
                    hess_exp_avgs,
                    hess_exp_avg_sqs,
                    hess_emas,
                    hess_curv_emas,
                    hess_c_thetas,
                    step_count,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    wd,
                    max_ang,
                    nesterov=nesterov,
                    gamma_gn=gamma_gn,
                    curv_weight=curv_weight,
                    tau_saddle=tau_saddle,
                    eps_kick=eps_kick,
                    antipodal_kick=antipodal_kick,
                    eps_h=eps_h,
                    sophia_rho=sophia_rho,
                )

        return loss


class HessianPairAdam(PairAdam):
    """Second-Order Riemannian PairAdam (`Sophia-Torus`) for Geometric Rotation Transformers.

    Exploits GRT's harmonic identity `d^2 R(theta)/d(theta^2) = -R(theta)` and Theorem 2.1
    (Cauchy-Riemann Polar Scale-Angle Curvature Duality) to precondition angle updates with
    the exact intrinsic toroidal Hessian `c_theta = -<grad_y L, y>` at zero extra backward
    passes, escaping antipodal saddle points (`Delta_theta = pi`) in O(1) steps.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        symmetrize_2d: bool = False,
        max_angular_step: float | None = 0.5,
        force_factored: bool = False,
        nesterov: bool = True,
        is_rotor3: bool = False,
        hessian_mode: bool = True,
        gamma_gn: float = 1.0,
        curv_weight: float = 0.05,
        tau_saddle: float = 1e-4,
        eps_kick: float = 1e-12,
        antipodal_kick: float = 0.2,
        eps_h: float = 1e-6,
        sophia_rho: float = 5.0,
        scale_partners: dict | None = None,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            symmetrize_2d=symmetrize_2d,
            max_angular_step=max_angular_step,
            force_factored=force_factored,
            nesterov=nesterov,
            is_rotor3=is_rotor3,
            hessian_mode=hessian_mode,
            gamma_gn=gamma_gn,
            curv_weight=curv_weight,
            tau_saddle=tau_saddle,
            eps_kick=eps_kick,
            antipodal_kick=antipodal_kick,
            eps_h=eps_h,
            sophia_rho=sophia_rho,
            scale_partners=scale_partners,
        )
        has_explicit_angle_group = any(
            g.get("is_angle", False) or g.get("group_type") == "angle" for g in self.param_groups
        )
        if not has_explicit_angle_group and len(self.param_groups) == 1:
            g0 = self.param_groups[0]
            if "is_angle" not in g0 and "group_type" not in g0 and not g0.get("is_rotor3", False):
                g0["is_angle"] = True
                g0["group_type"] = "angle"
                g0["weight_decay"] = 0.0


@torch.no_grad()
def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Batched quintic Newton-Schulz orthogonalization across butterfly cascade stages.

    Supports 2D `[n_stages, n_pairs]` and 3D `[n_stages, n_heads, n_pairs]` angle update tensors
    (transposing 3D tensors to `[n_heads, n_stages, n_pairs]` so each attention head's `n_stages x n_pairs`
    cascade is orthogonalized independently).
    """
    orig_dtype = g.dtype
    was_3d = g.dim() == 3
    if was_3d:
        # [K, H, P] -> [H, K, P]
        x = g.permute(1, 0, 2).to(dtype=torch.float32)
    elif g.dim() == 2:
        x = g.unsqueeze(0).to(dtype=torch.float32)
    else:
        return g

    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)

    norm = x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    x = x / norm

    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        xxt = torch.matmul(x, x.transpose(-2, -1))
        b_mat = b * xxt + c * torch.matmul(xxt, xxt)
        x = a * x + torch.matmul(b_mat, x)

    if transposed:
        x = x.transpose(-2, -1)

    if was_3d:
        return x.permute(1, 0, 2).to(dtype=orig_dtype)
    return x.squeeze(0).to(dtype=orig_dtype)


@torch.no_grad()
def _compute_polar_pair_radius_sq(
    angle_param: torch.Tensor,
    paired_raw_scale: torch.Tensor | None = None,
    is_log_magnitude: bool = False,
    group_scale_partners: dict | None = None,
) -> torch.Tensor:
    """Compute effective planar pair radius squared `r_p^2 = 0.5 * (softplus(s_{2p})^2 + softplus(s_{2p+1})^2)`."""
    if paired_raw_scale is None:
        paired_raw_scale, is_log_magnitude = _resolve_scale_partner(angle_param, group_scale_partners)
    if paired_raw_scale is None:
        return torch.ones_like(angle_param)

    work_dtype = torch.float64 if paired_raw_scale.dtype == torch.float64 else torch.float32
    if is_log_magnitude or (
        paired_raw_scale.dim() == 1
        and angle_param.dim() == 2
        and paired_raw_scale.shape[0] == angle_param.shape[0]
    ):
        r_sq = torch.exp(2.0 * paired_raw_scale.detach().to(dtype=work_dtype).clamp(-5.0, 5.0))
        return r_sq.unsqueeze(-1).expand_as(angle_param).to(dtype=angle_param.dtype)

    if paired_raw_scale.numel() == 2 * angle_param.numel():
        s_act = F.softplus(paired_raw_scale.detach().to(dtype=work_dtype))
        s_pairs = s_act.reshape(*angle_param.shape, 2)
        r_sq = 0.5 * (s_pairs[..., 0].square() + s_pairs[..., 1].square())
        return r_sq.to(dtype=angle_param.dtype)

    if paired_raw_scale.shape == angle_param.shape:
        s_act = F.softplus(paired_raw_scale.detach().to(dtype=work_dtype))
        return s_act.square().to(dtype=angle_param.dtype)

    return torch.ones_like(angle_param)


class ToroidalMuon(HessianPairAdam):
    """Stage-Orthogonalized Toroidal Optimizer (`ToroidalMuon` / `StageMuonPairAdam`) for GRT.

    Applies quintic Newton-Schulz orthogonalization across the butterfly cascade stages `[n_stages, n_pairs]`
    (or per-head `[n_stages, n_heads, n_pairs]`) of GRT rotation angle tensors so that adjacent butterfly
    stages learn mutually orthogonal phase updates on the flat torus `T^P = (S^1)^P`, blended (`muon_mix`)
    with the RMS-matched `HessianPairAdam` (`Sophia-Torus`) curvature-preconditioned step before wrapping
    into `[-pi, pi]`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_angular_step: float | None = 0.05,
        nesterov: bool = True,
        hessian_mode: bool = True,
        gamma_gn: float = 1.0,
        curv_weight: float = 0.05,
        ns_steps: int = 5,
        muon_mix: float = 0.35,
        max_cascade_stages: int = 256,
        scale_partners: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_angular_step=max_angular_step,
            nesterov=nesterov,
            hessian_mode=hessian_mode,
            gamma_gn=gamma_gn,
            curv_weight=curv_weight,
            scale_partners=scale_partners,
            **kwargs,
        )
        for g in self.param_groups:
            g.setdefault("ns_steps", ns_steps)
            g.setdefault("muon_mix", muon_mix)
            g.setdefault("max_cascade_stages", max_cascade_stages)

    def _foreach_hessian_angle_step(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        hess_emas,
        curv_emas,
        c_thetas,
        step,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        max_ang,
        nesterov=False,
        gamma_gn=1.0,
        curv_weight=0.25,
        tau_saddle=1e-4,
        eps_kick=1e-12,
        antipodal_kick=0.2,
        eps_h=1e-6,
        sophia_rho=5.0,
    ):
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        bc1_next = 1.0 - beta1 ** (step + 1)

        if wd != 0:
            torch._foreach_mul_(params, 1.0 - lr * wd)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        torch._foreach_mul_(curv_emas, beta2)
        torch._foreach_add_(curv_emas, c_thetas, alpha=1.0 - beta2)

        torch._foreach_mul_(hess_emas, beta2)
        torch._foreach_add_(hess_emas, c_thetas, alpha=1.0 - beta2)
        if gamma_gn != 0.0:
            torch._foreach_addcmul_(hess_emas, grads, grads, value=(1.0 - beta2) * gamma_gn)

        limit = (max_ang / lr) if (max_ang is not None and lr > 0.0) else None
        if sophia_rho is not None and sophia_rho > 0.0:
            limit = min(limit, sophia_rho) if limit is not None else sophia_rho

        ns_steps = self.defaults.get("ns_steps", 5)
        muon_mix = self.defaults.get("muon_mix", 0.35)
        max_cascade_stages = self.defaults.get("max_cascade_stages", 256)

        for p, g, m, v, h in zip(params, grads, exp_avgs, exp_avg_sqs, hess_emas):
            if nesterov:
                m_hat = m * (beta1 / bc1_next) + g * ((1.0 - beta1) / bc1)
            else:
                m_hat = m / bc1
            v_hat = v / bc2
            h_hat = h / bc2

            at_saddle = (h_hat < -tau_saddle) & (m_hat.abs() <= h_hat.abs())
            pos_curv = (v_hat.sqrt() + curv_weight * h_hat.clamp(min=0.0) + eps).clamp(min=eps_h)
            kick_sign = torch.where(
                m_hat.abs() > eps_kick,
                torch.sign(m_hat),
                torch.where(m_hat + eps_kick >= 0.0, torch.ones_like(m_hat), -torch.ones_like(m_hat)),
            )
            m_eff = torch.where(
                at_saddle,
                m_hat + kick_sign * (antipodal_kick * h_hat.abs()),
                m_hat,
            )
            update = m_eff / pos_curv

            # Apply Newton-Schulz stage orthogonalization on multi-stage butterfly cascades
            if (
                muon_mix > 0.0
                and p.dim() in (2, 3)
                and 1 < p.shape[0] <= max_cascade_stages
                and p.shape[-1] > 1
            ):
                ortho = _zeropower_via_newtonschulz5(m_eff, steps=ns_steps)
                # Match the RMS magnitude of the curvature-preconditioned update so lr semantics stay identical
                target_rms = update.square().mean().sqrt().clamp(min=1e-12)
                ortho_rms = ortho.square().mean().sqrt().clamp(min=1e-12)
                ortho_scaled = ortho * (target_rms / ortho_rms)
                update = (1.0 - muon_mix) * update + muon_mix * ortho_scaled

            if limit is not None:
                update.clamp_(-limit, limit)

            p.add_(update, alpha=-lr)
            p.data.copy_((p.data + math.pi) % (2.0 * math.pi) - math.pi)


class PolarNaturalAdam(HessianPairAdam):
    """Riemannian Polar Metric + Cauchy-Riemann Hessian Optimizer (`PolarNaturalAdam`) for GRT.

    Preconditions angular updates on `T^P = (S^1)^P` using both:
      1. The exact polar Riemannian metric `ds^2 = dr^2 + r^2 d(theta)^2` from the paired radial scale
         tensor `r_p^2 = 0.5 * (softplus(s_{2p})^2 + softplus(s_{2p+1})^2)` (scaling angular updates by
         `r_p^(-2 * polar_exponent)` normalized to unit mean across the stage so overall learning rate is preserved), and
      2. The analytical Cauchy-Riemann toroidal curvature `c_theta` from Theorem 2.1.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_angular_step: float | None = 0.05,
        nesterov: bool = True,
        hessian_mode: bool = True,
        gamma_gn: float = 1.0,
        curv_weight: float = 0.05,
        polar_exponent: float = 0.5,
        polar_min_radius_sq: float = 0.25,
        polar_max_radius_sq: float = 4.0,
        scale_partners: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_angular_step=max_angular_step,
            nesterov=nesterov,
            hessian_mode=hessian_mode,
            gamma_gn=gamma_gn,
            curv_weight=curv_weight,
            scale_partners=scale_partners,
            **kwargs,
        )
        self.defaults["polar_exponent"] = polar_exponent
        self.defaults["polar_min_radius_sq"] = polar_min_radius_sq
        self.defaults["polar_max_radius_sq"] = polar_max_radius_sq

    def _foreach_hessian_angle_step(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        hess_emas,
        curv_emas,
        c_thetas,
        step,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        max_ang,
        nesterov=False,
        gamma_gn=1.0,
        curv_weight=0.25,
        tau_saddle=1e-4,
        eps_kick=1e-12,
        antipodal_kick=0.2,
        eps_h=1e-6,
        sophia_rho=5.0,
    ):
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        bc1_next = 1.0 - beta1 ** (step + 1)

        if wd != 0:
            torch._foreach_mul_(params, 1.0 - lr * wd)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        torch._foreach_mul_(curv_emas, beta2)
        torch._foreach_add_(curv_emas, c_thetas, alpha=1.0 - beta2)

        torch._foreach_mul_(hess_emas, beta2)
        torch._foreach_add_(hess_emas, c_thetas, alpha=1.0 - beta2)
        if gamma_gn != 0.0:
            torch._foreach_addcmul_(hess_emas, grads, grads, value=(1.0 - beta2) * gamma_gn)

        limit = (max_ang / lr) if (max_ang is not None and lr > 0.0) else None
        if sophia_rho is not None and sophia_rho > 0.0:
            limit = min(limit, sophia_rho) if limit is not None else sophia_rho

        polar_exp = self.defaults.get("polar_exponent", 0.5)
        r_min_sq = self.defaults.get("polar_min_radius_sq", 0.25)
        r_max_sq = self.defaults.get("polar_max_radius_sq", 4.0)

        for p, g, m, v, h in zip(params, grads, exp_avgs, exp_avg_sqs, hess_emas):
            if nesterov:
                m_hat = m * (beta1 / bc1_next) + g * ((1.0 - beta1) / bc1)
            else:
                m_hat = m / bc1
            v_hat = v / bc2
            h_hat = h / bc2

            r_sq = _compute_polar_pair_radius_sq(p).clamp(min=r_min_sq, max=r_max_sq)
            inv_metric = r_sq.pow(-polar_exp)
            inv_metric = inv_metric / inv_metric.mean().clamp(min=1e-6)

            at_saddle = (h_hat < -tau_saddle) & (m_hat.abs() <= h_hat.abs())
            pos_curv = (v_hat.sqrt() + curv_weight * h_hat.clamp(min=0.0) + eps).clamp(min=eps_h)
            kick_sign = torch.where(
                m_hat.abs() > eps_kick,
                torch.sign(m_hat),
                torch.where(m_hat + eps_kick >= 0.0, torch.ones_like(m_hat), -torch.ones_like(m_hat)),
            )
            m_eff = torch.where(
                at_saddle,
                m_hat + kick_sign * (antipodal_kick * h_hat.abs()),
                m_hat,
            )
            update = (m_eff * inv_metric) / pos_curv
            if limit is not None:
                update.clamp_(-limit, limit)

            p.add_(update, alpha=-lr)
            p.data.copy_((p.data + math.pi) % (2.0 * math.pi) - math.pi)


class RotorAdam(HessianPairAdam):
    """Lie-Algebra Cayley-Retraction Optimizer (`RotorAdam` / `CayleyRotorAdam`) for GRT.

    Replaces Euclidean additive clipping on `T^P = (S^1)^P` with the smooth rational Cayley
    retraction map on `SO(2)^P`:

        `Delta theta = -2 * cayley_radius * arctan((lr * u) / (2 * cayley_radius))`

    which equals `-lr * u + O((lr * u)^3)` for small updates while smoothly saturating at
    `+-pi * cayley_radius` without hard clipping discontinuities, and applies exact `so(3)`
    geodesic wrapping on 3D rotor parameters (`is_rotor3=True`).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_angular_step: float | None = 0.05,
        nesterov: bool = True,
        hessian_mode: bool = True,
        gamma_gn: float = 1.0,
        curv_weight: float = 0.05,
        cayley_radius: float = 0.05,
        scale_partners: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_angular_step=max_angular_step,
            nesterov=nesterov,
            hessian_mode=hessian_mode,
            gamma_gn=gamma_gn,
            curv_weight=curv_weight,
            scale_partners=scale_partners,
            **kwargs,
        )
        self.defaults["cayley_radius"] = cayley_radius

    def _foreach_hessian_angle_step(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        hess_emas,
        curv_emas,
        c_thetas,
        step,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        max_ang,
        nesterov=False,
        gamma_gn=1.0,
        curv_weight=0.25,
        tau_saddle=1e-4,
        eps_kick=1e-12,
        antipodal_kick=0.2,
        eps_h=1e-6,
        sophia_rho=5.0,
    ):
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        bc1_next = 1.0 - beta1 ** (step + 1)

        if wd != 0:
            torch._foreach_mul_(params, 1.0 - lr * wd)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        torch._foreach_mul_(curv_emas, beta2)
        torch._foreach_add_(curv_emas, c_thetas, alpha=1.0 - beta2)

        torch._foreach_mul_(hess_emas, beta2)
        torch._foreach_add_(hess_emas, c_thetas, alpha=1.0 - beta2)
        if gamma_gn != 0.0:
            torch._foreach_addcmul_(hess_emas, grads, grads, value=(1.0 - beta2) * gamma_gn)

        # Smooth Cayley asymptotic radius (matches max_angular_step when set)
        asym_rad = max_ang if (max_ang is not None and max_ang > 0.0) else self.defaults.get("cayley_radius", 0.05)
        cayley_scale = 2.0 * asym_rad / math.pi

        for p, g, m, v, h in zip(params, grads, exp_avgs, exp_avg_sqs, hess_emas):
            if nesterov:
                m_hat = m * (beta1 / bc1_next) + g * ((1.0 - beta1) / bc1)
            else:
                m_hat = m / bc1
            v_hat = v / bc2
            h_hat = h / bc2

            at_saddle = (h_hat < -tau_saddle) & (m_hat.abs() <= h_hat.abs())
            pos_curv = (v_hat.sqrt() + curv_weight * h_hat.clamp(min=0.0) + eps).clamp(min=eps_h)
            kick_sign = torch.where(
                m_hat.abs() > eps_kick,
                torch.sign(m_hat),
                torch.where(m_hat + eps_kick >= 0.0, torch.ones_like(m_hat), -torch.ones_like(m_hat)),
            )
            m_eff = torch.where(
                at_saddle,
                m_hat + kick_sign * (antipodal_kick * h_hat.abs()),
                m_hat,
            )
            raw_step = lr * (m_eff / pos_curv)
            # Smooth rational Cayley retraction: maps (-inf, inf) -> (-asym_rad, +asym_rad) with unit slope at 0
            delta_theta = -2.0 * cayley_scale * torch.atan(raw_step / (2.0 * cayley_scale))
            p.add_(delta_theta)
            p.data.copy_((p.data + math.pi) % (2.0 * math.pi) - math.pi)


@torch.no_grad()
def _compute_cyclide_conic_metric_sq(
    angle_param: torch.Tensor,
    paired_raw_scale: torch.Tensor | None = None,
    is_log_magnitude: bool = False,
    group_scale_partners: dict | None = None,
    sin_theta: torch.Tensor | None = None,
    cos_theta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Dupin Cyclide conic tangent metric `g_{theta,p} = a_p^2 sin^2(theta_p) + b_p^2 cos^2(theta_p)`.

    For each 2D Givens pair `p` with semi-axes `a_p = softplus(s_{2p})` and `b_p = softplus(s_{2p+1})`,
    the Brianchon tangent envelope around the ellipse `(x / a_p)^2 + (y / b_p)^2 = 1` has exact
    phase-dependent squared tangent arc-length `a_p^2 sin^2(theta_p) + b_p^2 cos^2(theta_p)`.
    When `a_p == b_p == r_p` (zero eccentricity), this reduces to the circular torus metric `r_p^2`.
    """
    if paired_raw_scale is None:
        paired_raw_scale, is_log_magnitude = _resolve_scale_partner(angle_param, group_scale_partners)
    if paired_raw_scale is None:
        return torch.ones_like(angle_param)

    work_dtype = torch.float64 if paired_raw_scale.dtype == torch.float64 else torch.float32
    if paired_raw_scale.numel() == 2 * angle_param.numel():
        s_raw = paired_raw_scale.detach()
        if s_raw.dtype != work_dtype:
            s_raw = s_raw.to(dtype=work_dtype)
        s_act = F.softplus(s_raw)
        s_pairs = s_act.reshape(*angle_param.shape, 2)
        a_sq = s_pairs[..., 0].square()
        b_sq = s_pairs[..., 1].square()
        if sin_theta is not None and cos_theta is not None:
            s_t = sin_theta if sin_theta.dtype == work_dtype else sin_theta.to(dtype=work_dtype)
            c_t = cos_theta if cos_theta.dtype == work_dtype else cos_theta.to(dtype=work_dtype)
            sin_sq = s_t.square()
            cos_sq = c_t.square()
        else:
            theta_f = angle_param.detach()
            if theta_f.dtype != work_dtype:
                theta_f = theta_f.to(dtype=work_dtype)
            sin_sq = torch.sin(theta_f).square()
            cos_sq = 1.0 - sin_sq
        g_conic_sq = a_sq * sin_sq + b_sq * cos_sq
        return g_conic_sq if g_conic_sq.dtype == angle_param.dtype else g_conic_sq.to(dtype=angle_param.dtype)

    if (
        paired_raw_scale.dim() == angle_param.dim() + 1
        and paired_raw_scale.shape[1:-1] == angle_param.shape[:-1]
        and paired_raw_scale.shape[-1] == 2 * angle_param.shape[-1]
    ):
        s_raw = paired_raw_scale.detach()
        if s_raw.dtype != work_dtype:
            s_raw = s_raw.to(dtype=work_dtype)
        s_act = F.softplus(s_raw)
        s_pairs = s_act.reshape(paired_raw_scale.shape[0], *angle_param.shape, 2)
        a_sq = s_pairs[..., 0].square().mean(dim=0)
        b_sq = s_pairs[..., 1].square().mean(dim=0)
        theta_f = angle_param.detach()
        if theta_f.dtype != work_dtype:
            theta_f = theta_f.to(dtype=work_dtype)
        sin_sq = torch.sin(theta_f).square()
        cos_sq = 1.0 - sin_sq
        g_conic_sq = a_sq * sin_sq + b_sq * cos_sq
        return g_conic_sq if g_conic_sq.dtype == angle_param.dtype else g_conic_sq.to(dtype=angle_param.dtype)

    return _compute_polar_pair_radius_sq(
        angle_param,
        paired_raw_scale=paired_raw_scale,
        is_log_magnitude=is_log_magnitude,
        group_scale_partners=group_scale_partners,
    )


@torch.no_grad()
def _project_cyclide_deviatoric(
    angle_param: torch.Tensor,
    update: torch.Tensor,
    sin_theta: torch.Tensor | None = None,
    cos_theta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project out common-mode tangent drift across cascade stages (`dim=0`) in `O(K * P)` time.

    Decomposes multi-stage cascade updates `[n_stages, ...]` in the local tangent frame
    `e_{k,p} = (cos(theta_{k,p}), sin(theta_{k,p}))` and subtracts the coherent inter-stage
    translation `1/K * sum_j cos(theta_{k,p} - theta_{j,p}) * update_{j,p}`, leaving pure
    trace-free deviatoric twist across butterfly stages.
    """
    if angle_param.dim() not in (2, 3) or angle_param.shape[0] <= 1:
        return update
    c = cos_theta if cos_theta is not None else torch.cos(angle_param.data)
    s = sin_theta if sin_theta is not None else torch.sin(angle_param.data)
    mean_c = (c * update).mean(dim=0, keepdim=True)
    mean_s = (s * update).mean(dim=0, keepdim=True)
    common_mode = c * mean_c + s * mean_s
    return update - common_mode


class CyclideAdam(HessianPairAdam):
    """Dupin Cyclide Conic-Tangent + Deviatoric Cascade Optimizer (`CyclideAdam`) for GRT.

    Generalizes flat-torus ("Donut") optimization to a **Dupin Cyclide** manifold where each 2D
    rotation pair `p` forms an eccentric conic `(x / a_p)^2 + (y / b_p)^2 = 1` with semi-axes
    `a_p = softplus(s_{2p})` and `b_p = softplus(s_{2p+1})`:
      1. **Brianchon Conic Tangent Metric**: Preconditions angular updates by the inverse phase-dependent
         conic tangent arc-length `(a_p^2 sin^2(theta_p) + b_p^2 cos^2(theta_p))^(-cyclide_exponent)`
         normalized to unit mean across the stage,
      2. **Deviatoric (Trace-Free) Cascade Twist**: Projects out coherent common-mode phase drift across
         butterfly stages in `O(K * P)` time (`deviatoric_mix`), and
      3. **Brianchon Rational Tangent Retraction**: Smoothly maps tangent steps onto the conic envelope
         via `-2 * r_c * arctan(step / (2 * r_c))` without hard clipping discontinuities.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_angular_step: float | None = 0.05,
        nesterov: bool = True,
        hessian_mode: bool = True,
        gamma_gn: float = 1.0,
        curv_weight: float = 0.05,
        cyclide_exponent: float = 0.5,
        cyclide_min_metric_sq: float = 0.25,
        cyclide_max_metric_sq: float = 4.0,
        deviatoric_mix: float = 0.25,
        brianchon_radius: float = 0.05,
        max_cascade_stages: int = 256,
        scale_partners: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_angular_step=max_angular_step,
            nesterov=nesterov,
            hessian_mode=hessian_mode,
            gamma_gn=gamma_gn,
            curv_weight=curv_weight,
            scale_partners=scale_partners,
            **kwargs,
        )
        self.defaults["cyclide_exponent"] = cyclide_exponent
        self.defaults["cyclide_min_metric_sq"] = cyclide_min_metric_sq
        self.defaults["cyclide_max_metric_sq"] = cyclide_max_metric_sq
        self.defaults["deviatoric_mix"] = deviatoric_mix
        self.defaults["brianchon_radius"] = brianchon_radius
        self.defaults["max_cascade_stages"] = max_cascade_stages

    def _foreach_hessian_angle_step(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        hess_emas,
        curv_emas,
        c_thetas,
        step,
        lr,
        beta1,
        beta2,
        eps,
        wd,
        max_ang,
        nesterov=False,
        gamma_gn=1.0,
        curv_weight=0.25,
        tau_saddle=1e-4,
        eps_kick=1e-12,
        antipodal_kick=0.2,
        eps_h=1e-6,
        sophia_rho=5.0,
    ):
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        bc1_next = 1.0 - beta1 ** (step + 1)

        if wd != 0:
            torch._foreach_mul_(params, 1.0 - lr * wd)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        torch._foreach_mul_(curv_emas, beta2)
        torch._foreach_add_(curv_emas, c_thetas, alpha=1.0 - beta2)

        torch._foreach_mul_(hess_emas, beta2)
        torch._foreach_add_(hess_emas, c_thetas, alpha=1.0 - beta2)
        if gamma_gn != 0.0:
            torch._foreach_addcmul_(hess_emas, grads, grads, value=(1.0 - beta2) * gamma_gn)

        cyclide_exp = self.defaults.get("cyclide_exponent", 0.5)
        g_min_sq = self.defaults.get("cyclide_min_metric_sq", 0.25)
        g_max_sq = self.defaults.get("cyclide_max_metric_sq", 4.0)
        dev_mix = self.defaults.get("deviatoric_mix", 0.05)
        max_cascade_stages = self.defaults.get("max_cascade_stages", 256)
        limit = (max_ang / lr) if (max_ang is not None and lr > 0.0) else None
        if sophia_rho is not None and sophia_rho > 0.0:
            limit = min(limit, sophia_rho) if limit is not None else sophia_rho

        for p, g, m, v, h in zip(params, grads, exp_avgs, exp_avg_sqs, hess_emas):
            if nesterov:
                m_hat = m * (beta1 / bc1_next) + g * ((1.0 - beta1) / bc1)
            else:
                m_hat = m / bc1
            v_hat = v / bc2
            h_hat = h / bc2

            sin_theta = torch.sin(p.data)
            cos_theta = torch.cos(p.data)
            g_conic_sq = _compute_cyclide_conic_metric_sq(
                p, sin_theta=sin_theta, cos_theta=cos_theta
            ).clamp(min=g_min_sq, max=g_max_sq)
            inv_metric = torch.rsqrt(g_conic_sq) if cyclide_exp == 0.5 else g_conic_sq.pow(-cyclide_exp)
            inv_metric = inv_metric / inv_metric.mean().clamp(min=1e-6)

            at_saddle = (h_hat < -tau_saddle) & (m_hat.abs() <= h_hat.abs())
            pos_curv = (v_hat.sqrt() + curv_weight * h_hat.clamp(min=0.0) + eps).clamp(min=eps_h)
            kick_sign = torch.where(
                m_hat.abs() > eps_kick,
                torch.sign(m_hat),
                torch.where(m_hat + eps_kick >= 0.0, torch.ones_like(m_hat), -torch.ones_like(m_hat)),
            )
            m_eff = torch.where(
                at_saddle,
                m_hat + kick_sign * (antipodal_kick * h_hat.abs()),
                m_hat,
            )
            update = (m_eff * inv_metric) / pos_curv

            if (
                dev_mix > 0.0
                and p.dim() in (2, 3)
                and 1 < p.shape[0] <= max_cascade_stages
            ):
                dev = _project_cyclide_deviatoric(
                    p, update, sin_theta=sin_theta, cos_theta=cos_theta
                )
                target_rms = update.square().mean().sqrt().clamp(min=1e-12)
                dev_rms = dev.square().mean().sqrt().clamp(min=1e-12)
                dev_scaled = dev * (target_rms / dev_rms)
                update = (1.0 - dev_mix) * update + dev_mix * dev_scaled

            if limit is not None:
                update.clamp_(-limit, limit)

            p.add_(update, alpha=-lr)
            p.data.copy_((p.data + math.pi) % (2.0 * math.pi) - math.pi)


class ToroidalLion(Optimizer):
    """Sign-Momentum Toroidal Optimizer (`ToroidalLion` / `SignMomentumTorus`) for GRT.

    Adapts Chen et al.'s `Lion` (EvoLved Sign Momentum) to the flat torus `T^P = (S^1)^P` with:
      - Interpolated momentum `c_t = beta1 * m_{t-1} + (1 - beta1) * g_t`,
      - Optional smooth sign `tanh(c_t / sign_temperature)` (or exact `sign(c_t)` when `sign_temperature=0`)
        so every rotation plane advances at a uniform radian rate without second-moment state overhead,
      - Optional Theorem 2.1 antipodal saddle-point escape kicks on angle parameters, and
      - Exact `[-pi, pi]` toroidal wrapping and zero weight decay on angles.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-5,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
        max_angular_step: float | None = 0.05,
        sign_temperature: float = 0.0,
        hessian_mode: bool = True,
        tau_saddle: float = 1e-4,
        antipodal_kick: float = 0.2,
        scale_partners: dict | None = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            max_angular_step=max_angular_step,
            sign_temperature=sign_temperature,
            hessian_mode=hessian_mode,
            tau_saddle=tau_saddle,
            antipodal_kick=antipodal_kick,
        )
        super().__init__(params, defaults)
        if scale_partners is not None:
            for group in self.param_groups:
                group.setdefault("scale_partners", {}).update(scale_partners)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            is_angle = group.get("is_angle", False) or (group.get("group_type") == "angle")
            is_rotor3 = group.get("is_rotor3", False)
            max_ang = group.get("max_angular_step", group.get("max_step", None))
            sign_temp = group.get("sign_temperature", 0.0)
            hessian_mode = group.get("hessian_mode", False)
            tau_saddle = group.get("tau_saddle", 1e-4)
            antipodal_kick = group.get("antipodal_kick", 0.2)
            scale_partners = group.get("scale_partners", None)

            eff_lr = min(lr, max_ang) if (max_ang is not None and lr > max_ang) else lr

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]

                if wd != 0.0 and not is_angle and not is_rotor3:
                    p.mul_(1.0 - eff_lr * wd)

                c_t = exp_avg * beta1 + grad * (1.0 - beta1)

                if is_angle and hessian_mode:
                    c_theta = compute_intrinsic_angle_hessian(p, group_scale_partners=scale_partners)
                    at_saddle = (c_theta < -tau_saddle) & (c_t.abs() <= c_theta.abs())
                    kick_sign = torch.where(c_t >= 0.0, torch.ones_like(c_t), -torch.ones_like(c_t))
                    c_t = torch.where(at_saddle, c_t + kick_sign * (antipodal_kick * c_theta.abs()), c_t)

                if sign_temp > 0.0:
                    rms = c_t.square().mean().sqrt().clamp(min=1e-12)
                    update = torch.tanh(c_t / (sign_temp * rms))
                else:
                    update = torch.sign(c_t)

                if is_rotor3:
                    _clamp_rotor3_update_(update, 1.0)
                    p.add_(update, alpha=-eff_lr)
                    _wrap_rotor3_(p.data)
                else:
                    p.add_(update, alpha=-eff_lr)
                    if is_angle:
                        p.data.copy_((p.data + math.pi) % (2.0 * math.pi) - math.pi)

                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def build_grt_param_groups(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    angle_lr_mult: float = 1.0,
    scale_lr_mult: float = 1.0,
    coupling_lr_mult: float = 3.0,
    coupling_weight_decay: float | None = None,
    embedding_lr_mult: float = 1.0,
    log_magnitude_lr_mult: float = 1.0,
    output_scale_lr_mult: float = 1.0,
    max_angular_step: float | None = 0.5,
    max_scale_step: float | None = None,
    coupling_force_factored: bool = False,
    embedding_force_diagonal: bool = True,
    nesterov: bool = True,
    hessian_mode: bool = True,
) -> list[dict[str, Any]]:
    """Build parameter groups for `PairAdam` / `HessianPairAdam` on any `GRTPreTrainedModel`.

    Splits parameters into geometric groups (embedding phase angles, embedding log magnitudes,
    rotation angles on `(S^1)^P`, `so(3)` rotor vectors, radial softplus scales, stage coupling
    tensors, output temperatures, residual gates, and learnable RoPE frequencies) and pairs
    every angle tensor with its corresponding scale partner via Theorem 2.1.
    """
    wd_coupling = weight_decay if coupling_weight_decay is None else coupling_weight_decay
    emb_angle_ps, emb_scale_ps, output_scale_ps = [], [], []
    angle_ps, rotor3_ps, scale_ps, coupling_ps, matrix_ps, other_1d = [], [], [], [], [], []
    gate_ps, rope_ps = [], []

    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        if "res_gate" in name:
            gate_ps.append(p)
        elif "log_freqs" in name:
            rope_ps.append(p)
        elif "coupling" in name:
            coupling_ps.append(p)
        elif "embedding.phase_angles" in name or name.endswith("lm_head.phase_angles"):
            emb_angle_ps.append(p)
        elif "embedding.log_magnitude" in name:
            emb_scale_ps.append(p)
        elif name.endswith("output_scale"):
            output_scale_ps.append(p)
        elif p.dim() >= 1 and p.shape[-1] == 3 and ("rotvec" in name or "omega" in name):
            rotor3_ps.append(p)
        elif "angle" in name or "phase" in name:
            angle_ps.append(p)
        elif any(k in name for k in ("scale", "log_mag", "norm")):
            scale_ps.append(p)
        elif p.dim() == 2:
            matrix_ps.append(p)
        else:
            other_1d.append(p)

    scale_partners = pair_model_angles_and_scales(model)
    emb_lr = lr * embedding_lr_mult
    log_mag_lr = lr * embedding_lr_mult * log_magnitude_lr_mult
    out_scale_lr = lr * output_scale_lr_mult

    raw_groups: list[dict[str, Any]] = [
        {
            "params": emb_angle_ps,
            "lr": emb_lr,
            "weight_decay": 0.0,
            "is_angle": True,
            "group_type": "angle",
            "force_diagonal": embedding_force_diagonal,
            "hessian_mode": hessian_mode,
            "scale_partners": {id(p): scale_partners[id(p)] for p in emb_angle_ps if id(p) in scale_partners},
        },
        {"params": emb_scale_ps, "lr": log_mag_lr, "weight_decay": weight_decay, "group_type": "scale"},
        {"params": output_scale_ps, "lr": out_scale_lr, "weight_decay": weight_decay, "group_type": "scale"},
        {
            "params": angle_ps,
            "lr": lr * angle_lr_mult,
            "weight_decay": 0.0,
            "is_angle": True,
            "group_type": "angle",
            "force_diagonal": True,
            "hessian_mode": hessian_mode,
            "scale_partners": {id(p): scale_partners[id(p)] for p in angle_ps if id(p) in scale_partners},
        },
        {
            "params": rotor3_ps,
            "lr": lr * angle_lr_mult,
            "weight_decay": 0.0,
            "is_rotor3": True,
            "group_type": "rotor3",
        },
        {"params": scale_ps, "lr": lr * scale_lr_mult, "weight_decay": weight_decay, "group_type": "scale"},
        {
            "params": coupling_ps,
            "lr": lr * coupling_lr_mult,
            "weight_decay": wd_coupling,
            "force_factored": coupling_force_factored,
        },
        {"params": matrix_ps, "lr": lr, "weight_decay": weight_decay, "symmetrize_2d": True},
        {"params": other_1d, "lr": lr, "weight_decay": weight_decay},
        {"params": gate_ps, "lr": lr, "weight_decay": 0.0},
        {"params": rope_ps, "lr": lr, "weight_decay": 0.0},
    ]

    param_groups: list[dict[str, Any]] = []
    for g in raw_groups:
        if not g["params"]:
            continue
        if g.get("group_type") in ("angle", "rotor3"):
            if max_angular_step is not None:
                g["max_angular_step"] = max_angular_step
            if nesterov:
                g["nesterov"] = True
        elif g.get("group_type") == "scale" and max_scale_step is not None:
            g["max_step"] = max_scale_step
        param_groups.append(g)

    return param_groups


def build_grt_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    hessian_mode: bool = True,
    optimizer_type: str | None = None,
    angle_lr_mult: float = 1.0,
    scale_lr_mult: float = 1.0,
    coupling_lr_mult: float = 3.0,
    coupling_weight_decay: float | None = None,
    embedding_lr_mult: float = 1.0,
    log_magnitude_lr_mult: float = 1.0,
    output_scale_lr_mult: float = 1.0,
    max_angular_step: float | None = 0.5,
    max_scale_step: float | None = None,
    coupling_force_factored: bool = False,
    embedding_force_diagonal: bool = True,
    nesterov: bool = True,
    gamma_gn: float = 1.0,
    curv_weight: float = 0.05,
    tau_saddle: float = 1e-4,
    eps_kick: float = 1e-12,
    antipodal_kick: float = 0.2,
    eps_h: float = 1e-6,
    sophia_rho: float = 5.0,
    **kwargs,
) -> Optimizer:
    """Construct a GRT geometric optimizer (`HessianPairAdam`, `PairAdam`, `ToroidalMuon`, `PolarNaturalAdam`, `RotorAdam`, `CyclideAdam`, or `ToroidalLion`)."""
    param_groups = build_grt_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        angle_lr_mult=angle_lr_mult,
        scale_lr_mult=scale_lr_mult,
        coupling_lr_mult=coupling_lr_mult,
        coupling_weight_decay=coupling_weight_decay,
        embedding_lr_mult=embedding_lr_mult,
        log_magnitude_lr_mult=log_magnitude_lr_mult,
        output_scale_lr_mult=output_scale_lr_mult,
        max_angular_step=max_angular_step,
        max_scale_step=max_scale_step,
        coupling_force_factored=coupling_force_factored,
        embedding_force_diagonal=embedding_force_diagonal,
        nesterov=nesterov,
        hessian_mode=hessian_mode,
    )
    opt_key = (optimizer_type or ("hessian_pair_adam" if hessian_mode else "pair_adam")).lower()
    if opt_key == "toroidal_lion":
        return ToroidalLion(
            param_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            max_angular_step=max_angular_step,
            hessian_mode=hessian_mode,
            tau_saddle=tau_saddle,
            antipodal_kick=antipodal_kick,
            **kwargs,
        )
    opt_map = {
        "pair_adam": PairAdam,
        "hessian_pair_adam": HessianPairAdam,
        "toroidal_muon": ToroidalMuon,
        "polar_natural_adam": PolarNaturalAdam,
        "rotor_adam": RotorAdam,
        "cyclide_adam": CyclideAdam,
    }
    if opt_key not in opt_map:
        raise ValueError(f"Unknown GRT optimizer_type={optimizer_type!r}. Expected one of { sorted([*opt_map, 'toroidal_lion']) }.")
    opt_cls = opt_map[opt_key]
    return opt_cls(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        max_angular_step=max_angular_step,
        nesterov=nesterov,
        hessian_mode=hessian_mode if opt_key != "pair_adam" else False,
        gamma_gn=gamma_gn,
        curv_weight=curv_weight,
        tau_saddle=tau_saddle,
        eps_kick=eps_kick,
        antipodal_kick=antipodal_kick,
        eps_h=eps_h,
        sophia_rho=sophia_rho,
        **kwargs,
    )


__all__ = [
    "CyclideAdam",
    "HessianPairAdam",
    "PairAdam",
    "PolarNaturalAdam",
    "RotorAdam",
    "ToroidalLion",
    "ToroidalMuon",
    "build_grt_optimizer",
    "build_grt_param_groups",
    "compute_intrinsic_angle_hessian",
    "pair_model_angles_and_scales",
]
