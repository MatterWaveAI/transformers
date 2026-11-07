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
"""Tests for GRT `PairAdam` and `HessianPairAdam` (`Sophia-Torus`) optimizers."""

import math
import unittest

import torch
import torch.nn.functional as F

from transformers import (
    GRTConfig,
    GRTForCausalLM,
    GRTForSequenceClassification,
    GRTVisionConfig,
    GRTVisionModel,
    GRTVLConfig,
    GRTVLForConditionalGeneration,
)
from transformers.models.grt import (
    BeltramiAdam,
    CyclideAdam,
    GyricAdam,
    HannayAdam,
    HessianPairAdam,
    HuygensAdam,
    HuygensSymplecticAdam,
    PairAdam,
    PolarNaturalAdam,
    PonceletAdam,
    RindlerAdam,
    RotorAdam,
    SymplecticAdam,
    ToroidalLion,
    ToroidalMuon,
    build_grt_optimizer,
    build_grt_param_groups,
    compute_intrinsic_angle_hessian,
    pair_model_angles_and_scales,
)


class GRTOptimizationTest(unittest.TestCase):
    def _tiny_config(self, **overrides) -> GRTConfig:
        defaults = dict(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            n_q_rotations=2,
            n_k_rotations=2,
            n_v_rotations=2,
            n_o_rotations=2,
            n_mixer_rotations=1,
            ffn_n_rotations=2,
            max_position_embeddings=64,
        )
        defaults.update(overrides)
        return GRTConfig(**defaults)

    def test_cauchy_riemann_scale_angle_hessian_duality(self):
        """Theorem 2.1: c_{theta_p} from raw scale gradient matches exact autograd d^2 L / d theta_p^2."""
        torch.manual_seed(42)
        n_stages, n_pairs = 3, 8
        dim = 2 * n_pairs
        x = torch.randn(4, dim, dtype=torch.float64)
        target = torch.randn(4, dim, dtype=torch.float64)
        w = torch.randn(dim, dim, dtype=torch.float64)

        theta = torch.randn(n_stages, n_pairs, dtype=torch.float64, requires_grad=True)
        s_raw = torch.randn(n_stages, dim, dtype=torch.float64, requires_grad=True)

        def stage0_loss(th0, sr0):
            c = torch.cos(th0)
            s = torch.sin(th0)
            xp = x.view(4, n_pairs, 2)
            r0 = xp[..., 0] * c - xp[..., 1] * s
            r1 = xp[..., 0] * s + xp[..., 1] * c
            r = torch.stack([r0, r1], dim=-1).flatten(-2)
            y = F.softplus(sr0) * r
            # Linear-in-y downstream loss so Gauss-Newton term d^2 L / dy^2 = 0 and d^2 L / dtheta^2 == c_theta
            return ((y @ w) * target).sum()

        loss = stage0_loss(theta[0], s_raw[0])
        loss.backward()

        theta0 = theta[0].detach().clone().requires_grad_(True)
        s_raw0 = s_raw[0].detach().clone()
        s_raw0.grad = s_raw.grad[0].detach().clone()

        c_duality = compute_intrinsic_angle_hessian(theta0, paired_raw_scale=s_raw0)
        hess_exact = torch.autograd.functional.hessian(lambda t: stage0_loss(t, s_raw0), theta0)
        diag_exact = torch.diagonal(hess_exact)

        self.assertTrue(
            torch.allclose(c_duality, diag_exact, atol=1e-11, rtol=1e-9),
            f"Max diff: {(c_duality - diag_exact).abs().max().item()}",
        )

    def test_pair_model_angles_and_scales_coverage(self):
        """All angle parameters in GRTForCausalLM, MoR, SequenceClassification, and GRT-VL pair with scales."""
        for ffn_n_experts, mor_mode in [(0, "direct"), (4, "direct"), (4, "residual")]:
            cfg = self._tiny_config(ffn_n_experts=ffn_n_experts, mor_mode=mor_mode)
            model = GRTForCausalLM(cfg)
            partners = pair_model_angles_and_scales(model)
            for name, p in model.named_parameters():
                if "angle" in name:
                    self.assertIn(id(p), partners, f"Unpaired angle param {name} (mor_mode={mor_mode})")
                    self.assertIsNotNone(partners[id(p)]())

        cls_model = GRTForSequenceClassification(self._tiny_config(num_labels=3))
        cls_partners = pair_model_angles_and_scales(cls_model)
        self.assertIn(id(cls_model.score.phase_angles), cls_partners)
        self.assertIs(cls_partners[id(cls_model.score.phase_angles)](), cls_model.score.pair_scales)

        vis_cfg = GRTVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            patch_size=4,
            spatial_merge_size=2,
            patch_n_rotations=2,
            merger_n_rotations=2,
            n_q_rotations=2,
            n_k_rotations=2,
            n_v_rotations=2,
            n_o_rotations=2,
            n_mixer_rotations=1,
            ffn_n_rotations=2,
        )
        vl_cfg = GRTVLConfig(vision_config=vis_cfg, text_config=self._tiny_config())
        vl_model = GRTVLForConditionalGeneration(vl_cfg)
        vl_partners = pair_model_angles_and_scales(vl_model)
        self.assertIn(id(vl_model.visual.patch_embed.angles), vl_partners)
        self.assertIn(id(vl_model.visual.merger.expand_angles), vl_partners)

    def test_hessian_pair_adam_step_and_configure_optimizers(self):
        """GRTPreTrainedModel.configure_optimizers(hessian_mode=True) decreases loss on CausalLM."""
        torch.manual_seed(0)
        cfg = self._tiny_config(ffn_n_experts=4, mor_mode="residual")
        model = GRTForCausalLM(cfg)
        optimizer = model.configure_optimizers(lr=5e-3, weight_decay=0.01, hessian_mode=True)
        self.assertIsInstance(optimizer, HessianPairAdam)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
        out0 = model(input_ids=input_ids, labels=input_ids)
        loss0 = out0.loss.item()

        for _ in range(4):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()

        out_final = model(input_ids=input_ids, labels=input_ids)
        self.assertLess(out_final.loss.item(), loss0)

        # Verify Hessian EMA state was populated on angle parameters
        q_angles = model.model.blocks[0].attention.q_angles
        self.assertIn("hessian_ema", optimizer.state[q_angles])
        self.assertGreater(optimizer.state[q_angles]["hessian_ema"].abs().sum().item(), 0.0)

    def test_antipodal_saddle_escape(self):
        """HessianPairAdam escapes an antipodal saddle (theta = pi - 1e-5) faster than first-order PairAdam."""
        def run_opt(hessian_mode: bool) -> float:
            theta = torch.nn.Parameter(torch.tensor([math.pi - 1e-5], dtype=torch.float64))
            opt_cls = HessianPairAdam if hessian_mode else PairAdam
            opt = opt_cls([theta], lr=0.15, nesterov=False, max_angular_step=0.5, hessian_mode=hessian_mode)
            # Pre-populate realistic running Fisher second moment v_t ~ 1.0 (from prior batch variance)
            # so first-order Adam cannot divide 0/0 to take a unit step at the stationary saddle.
            opt.state[theta] = {
                "step": 10,
                "exp_avg": torch.zeros_like(theta),
                "exp_avg_sq": torch.ones_like(theta),
                "hessian_ema": torch.full_like(theta, -1.0),
                "exp_avg_curv": torch.full_like(theta, -1.0),
            }
            for _ in range(15):
                opt.zero_grad()
                loss = 1.0 - torch.cos(theta).sum()
                loss.backward()
                theta.hessian_diag = torch.cos(theta.detach())
                opt.step()
            return float((1.0 - torch.cos(theta)).item())

        loss_hess = run_opt(hessian_mode=True)
        loss_first = run_opt(hessian_mode=False)
        self.assertLess(loss_hess, loss_first * 0.25)

    def test_trainer_optimizer_registry(self):
        """OptimizerNames('hessian_pair_adam') and 'pair_adam' resolve through trainer_optimizer."""
        from types import SimpleNamespace

        from transformers.trainer_optimizer import _OPTIMIZER_HANDLERS, OptimizerContext, _parse_optim_args
        from transformers.training_args import OptimizerNames

        self.assertEqual(OptimizerNames("hessian_pair_adam"), OptimizerNames.HESSIAN_PAIR_ADAM)
        self.assertEqual(OptimizerNames("pair_adam"), OptimizerNames.PAIR_ADAM)

        model = GRTForCausalLM(self._tiny_config())
        fake_args = SimpleNamespace(
            optim=OptimizerNames.HESSIAN_PAIR_ADAM,
            learning_rate=2e-3,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            optim_args="max_angular_step=0.4,curv_weight=0.05",
        )
        ctx = OptimizerContext(
            args=fake_args,
            model=model,
            optimizer_kwargs={"lr": fake_args.learning_rate},
            adam_kwargs={"betas": (0.9, 0.999), "eps": 1e-8},
            optim_args=_parse_optim_args(fake_args.optim_args),
        )
        opt_cls, opt_kwargs = _OPTIMIZER_HANDLERS[OptimizerNames.HESSIAN_PAIR_ADAM](ctx)
        self.assertIs(opt_cls, HessianPairAdam)
        self.assertIn("params", opt_kwargs)
        params = opt_kwargs.pop("params")
        opt = opt_cls(params, **opt_kwargs)
        self.assertIsInstance(opt, HessianPairAdam)

        fake_args_pair = SimpleNamespace(
            optim=OptimizerNames.PAIR_ADAM,
            learning_rate=1e-3,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            optim_args=None,
        )
        ctx_pair = OptimizerContext(
            args=fake_args_pair,
            model=model,
            optimizer_kwargs={"lr": fake_args_pair.learning_rate},
            adam_kwargs={"betas": (0.9, 0.999), "eps": 1e-8},
            optim_args={},
        )
        opt_cls_pair, opt_kwargs_pair = _OPTIMIZER_HANDLERS[OptimizerNames.PAIR_ADAM](ctx_pair)
        self.assertIs(opt_cls_pair, PairAdam)
        params_pair = opt_kwargs_pair.pop("params")
        opt_pair = opt_cls_pair(params_pair, **opt_kwargs_pair)
        self.assertIsInstance(opt_pair, PairAdam)

    def test_toroidal_muon_polar_natural_rotor_and_lion_optimizers(self):
        """All GRT geometric optimizers decrease loss and wrap angles in [-pi, pi]."""
        for opt_type, opt_cls, lr in [
            ("toroidal_muon", ToroidalMuon, 5e-3),
            ("polar_natural_adam", PolarNaturalAdam, 5e-3),
            ("rotor_adam", RotorAdam, 5e-3),
            ("cyclide_adam", CyclideAdam, 5e-3),
            ("symplectic_adam", SymplecticAdam, 5e-3),
            ("gyric_adam", GyricAdam, 5e-3),
            ("hannay_adam", HannayAdam, 5e-3),
            ("poncelet_adam", PonceletAdam, 5e-3),
            ("huygens_adam", HuygensAdam, 5e-3),
            ("huygens_symplectic_adam", HuygensSymplecticAdam, 5e-3),
            ("beltrami_adam", BeltramiAdam, 5e-3),
            ("rindler_adam", RindlerAdam, 5e-3),
            ("toroidal_lion", ToroidalLion, 1e-3),
        ]:
            torch.manual_seed(7)
            cfg = self._tiny_config()
            model = GRTForCausalLM(cfg)
            optimizer = build_grt_optimizer(model, lr=lr, weight_decay=0.01, optimizer_type=opt_type)
            self.assertIsInstance(optimizer, opt_cls)

            input_ids = torch.randint(0, cfg.vocab_size, (2, 10))
            loss0 = model(input_ids=input_ids, labels=input_ids).loss.item()

            for _ in range(4):
                optimizer.zero_grad()
                out = model(input_ids=input_ids, labels=input_ids)
                out.loss.backward()
                optimizer.step()

            loss_final = model(input_ids=input_ids, labels=input_ids).loss.item()
            self.assertLess(loss_final, loss0, f"{opt_type} failed to decrease loss: {loss0} -> {loss_final}")

            for name, p in model.named_parameters():
                if "angle" in name:
                    self.assertLessEqual(p.data.abs().max().item(), math.pi + 1e-5, f"{opt_type} unwrapped {name}")

    def test_symplectic_adam_non_squeezing_and_cavity_step(self):
        """SymplecticAdam applies the symplectic non-squeezing floor, Casimir cavity smoothing, and Cayley retraction while decreasing loss."""
        from transformers.models.grt.optimization_grt import _apply_casimir_cavity_cascade

        torch.manual_seed(11)
        u = torch.tensor([[1.0, -1.0], [3.0, 0.0], [-1.0, 4.0], [1.0, -3.0]], dtype=torch.float32)
        u_cav = _apply_casimir_cavity_cascade(u, deviatoric_mix=0.15, casimir_mix=0.10)
        self.assertEqual(u_cav.shape, u.shape)
        self.assertTrue(torch.isfinite(u_cav).all())
        # Deviatoric projection shrinks the stage-mean magnitude by (1 - deviatoric_mix)
        self.assertTrue(
            torch.allclose(u_cav.mean(dim=0), (1.0 - 0.15) * u.mean(dim=0), atol=1e-6),
            f"Expected stage mean {(1.0 - 0.15) * u.mean(dim=0)}, got {u_cav.mean(dim=0)}",
        )

        cfg = self._tiny_config(n_q_rotations=4, ffn_n_rotations=4)
        model = GRTForCausalLM(cfg)
        optimizer = build_grt_optimizer(
            model,
            lr=5e-3,
            weight_decay=0.01,
            optimizer_type="symplectic_adam",
            zpe_floor=0.02,
            deviatoric_mix=0.15,
            casimir_mix=0.10,
        )
        self.assertIsInstance(optimizer, SymplecticAdam)

        input_ids = torch.randint(0, cfg.vocab_size, (2, 10))
        loss0 = model(input_ids=input_ids, labels=input_ids).loss.item()
        for _ in range(4):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()

        loss_final = model(input_ids=input_ids, labels=input_ids).loss.item()
        self.assertLess(loss_final, loss0)
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())

    def test_huygens_symplectic_adam_curriculum_and_cayley_bound(self):
        """HuygensSymplecticAdam applies zero-mean splay curriculum annealing, symplectic floor, and Cayley step bound."""
        from transformers.models.grt.optimization_grt import _apply_huygens_splay_torque

        torch.manual_seed(17)
        # Bunched cascade stages (R_p > splay_threshold)
        theta_bunched = torch.tensor([[0.10, -0.20], [0.15, -0.15], [0.05, -0.25], [0.12, -0.18]], dtype=torch.float32)
        raw_step = torch.tensor([[1.0, -0.5], [0.8, -0.4], [1.2, -0.6], [1.0, -0.5]], dtype=torch.float32)
        u_splay = _apply_huygens_splay_torque(theta_bunched, raw_step, splay_weight=0.1, splay_threshold=0.35)
        self.assertEqual(u_splay.shape, raw_step.shape)
        self.assertTrue(torch.isfinite(u_splay).all())
        # Zero-mean across dim=0 preserves stage mean update
        self.assertTrue(torch.allclose(u_splay.mean(dim=0), raw_step.mean(dim=0), atol=1e-6))
        self.assertGreater((u_splay - raw_step).abs().max().item(), 0.0)

        cfg = self._tiny_config(n_q_rotations=4, ffn_n_rotations=4)
        model = GRTForCausalLM(cfg)
        optimizer = build_grt_optimizer(
            model,
            lr=5e-3,
            weight_decay=0.01,
            optimizer_type="huygens_symplectic_adam",
            max_angle_step=0.08,
            zpe_floor=0.015,
            casimir_coupling=0.08,
            splay_weight=0.05,
            splay_anneal_steps=250,
        )
        self.assertIsInstance(optimizer, HuygensSymplecticAdam)

        q_angles = model.model.blocks[0].attention.q_angles
        q_before = q_angles.detach().clone()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 10))
        loss0 = model(input_ids=input_ids, labels=input_ids).loss.item()
        for _ in range(4):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()

        loss_final = model(input_ids=input_ids, labels=input_ids).loss.item()
        self.assertLess(loss_final, loss0)
        self.assertGreater((q_angles.detach() - q_before).abs().max().item(), 0.0)
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())


if __name__ == "__main__":
    unittest.main()
