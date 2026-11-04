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
    GRTAudioConfig,
    GRTConfig,
    GRTForCausalLM,
    GRTForSequenceClassification,
    GRTOmniConfig,
    GRTOmniForConditionalGeneration,
    GRTTTSConfig,
    GRTTTSForConditionalGeneration,
    GRTVisionConfig,
    GRTVisionModel,
    GRTVLConfig,
    GRTVLForConditionalGeneration,
)
from transformers.models.grt import (
    HessianPairAdam,
    PairAdam,
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
            intermediate_size=64,
            n_qkv_rotations=2,
            n_mixer_rotations=1,
            ffn_n_rotations=2,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
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
            n_qkv_rotations=2,
            n_mixer_rotations=1,
            ffn_n_rotations=2,
        )
        vl_cfg = GRTVLConfig(vision_config=vis_cfg, text_config=self._tiny_config())
        vl_model = GRTVLForConditionalGeneration(vl_cfg)
        vl_partners = pair_model_angles_and_scales(vl_model)
        self.assertIn(id(vl_model.visual.patch_embed.angles), vl_partners)
        self.assertIn(id(vl_model.visual.merger.expand_angles), vl_partners)

        aud_cfg = GRTAudioConfig(
            n_fft=30,
            hop_length=16,
            win_length=30,
            max_audio_frames=8,
            temporal_merge_size=2,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            stem_n_rotations=2,
            merger_n_rotations=2,
            vocoder_n_rotations=2,
            frames_per_token=2,
            ffn_n_rotations=2,
            n_qkv_rotations=2,
            n_mixer_rotations=1,
        )
        tts_model = GRTTTSForConditionalGeneration(GRTTTSConfig(audio_config=aud_cfg, text_config=self._tiny_config()))
        tts_partners = pair_model_angles_and_scales(tts_model)
        for name, p in tts_model.named_parameters():
            if "angle" in name:
                self.assertIn(id(p), tts_partners, f"Unpaired TTS angle param {name}")

        omni_model = GRTOmniForConditionalGeneration(
            GRTOmniConfig(vision_config=vis_cfg, audio_config=aud_cfg, text_config=self._tiny_config())
        )
        omni_partners = pair_model_angles_and_scales(omni_model)
        for name, p in omni_model.named_parameters():
            if "angle" in name:
                self.assertIn(id(p), omni_partners, f"Unpaired Omni angle param {name}")

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


if __name__ == "__main__":
    unittest.main()
