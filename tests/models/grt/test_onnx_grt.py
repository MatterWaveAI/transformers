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
"""Tests for GRT ONNX configuration, `GRTExportWrapper`, `export_grt_program`, and `export_grt_to_onnx`."""

import tempfile
import unittest
from pathlib import Path

from transformers import GRTConfig, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import GRTForCausalLM, GRTModel
    from transformers.models.grt import (
        GRTExportWrapper,
        GRTOnnxConfig,
        export_grt_program,
        export_grt_to_onnx,
    )
    from transformers.models.grt.modeling_grt import (
        GRTAttention,
        GRTCosineHead,
        GRTEmbedding,
        GRTFeedForward,
        GRTMixer,
        GRTPearlNorm,
        GRTRadialGELU,
        GRTRotaryEmbedding,
    )


def _tiny_grt_config() -> GRTConfig:
    return GRTConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        ffn_n_rotations=3,
        n_qkv_rotations=2,
        n_mixer_rotations=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@require_torch
class GRTOnnxExportTest(unittest.TestCase):
    def test_onnx_config_dummy_inputs_and_axes(self):
        config = _tiny_grt_config()
        onnx_cfg = GRTOnnxConfig(config=config, task="causal-lm")

        self.assertEqual(onnx_cfg.default_onnx_opset, 17)
        self.assertEqual(onnx_cfg.values_override, {"use_cache": False})
        self.assertIn("input_ids", onnx_cfg.inputs)
        self.assertIn("attention_mask", onnx_cfg.inputs)
        self.assertIn("logits", onnx_cfg.outputs)
        self.assertIn("input_ids", onnx_cfg.dynamic_axes)
        self.assertIn("logits", onnx_cfg.dynamic_axes)

        dummy = onnx_cfg.generate_dummy_inputs(batch_size=3, seq_length=10)
        self.assertEqual(tuple(dummy["input_ids"].shape), (3, 10))
        self.assertEqual(tuple(dummy["attention_mask"].shape), (3, 10))
        self.assertTrue((dummy["input_ids"] < config.vocab_size).all())

        base_onnx_cfg = GRTOnnxConfig(config=config, task="default")
        self.assertIn("last_hidden_state", base_onnx_cfg.outputs)

    def test_submodules_trace_cleanly(self):
        config = _tiny_grt_config()
        torch.manual_seed(0)

        x = torch.randn(2, 6, config.hidden_size)
        ids = torch.randint(0, config.vocab_size, (2, 6))
        pos = torch.arange(6).unsqueeze(0).expand(2, 6)

        for mod, args in [
            (GRTPearlNorm(config.hidden_size).eval(), (x,)),
            (GRTRadialGELU().eval(), (x,)),
            (GRTEmbedding(config).eval(), (ids,)),
            (GRTRotaryEmbedding(config).eval(), (x, pos)),
            (GRTMixer(config).eval(), (x,)),
            (GRTFeedForward(config).eval(), (x,)),
            (GRTCosineHead(config).eval(), (x,)),
        ]:
            with torch.no_grad():
                expected = mod(*args)
                traced = torch.jit.trace(mod, args)
                actual = traced(*args)
                max_diff = (expected - actual).abs().max().item()
                self.assertLess(max_diff, 1e-5)

        attn = GRTAttention(config, layer_idx=0).eval()
        with torch.no_grad():
            expected_attn, _ = attn(x, attention_mask=None)
            prog = torch.export.export(attn, (x,), strict=False)
            actual_attn, _ = prog.module()(x)
            self.assertLess((expected_attn - actual_attn).abs().max().item(), 1e-5)

    def test_export_wrapper_and_program_causal_lm(self):
        config = _tiny_grt_config()
        torch.manual_seed(42)
        model = GRTForCausalLM(config).eval()

        onnx_cfg = GRTOnnxConfig(config=config)
        dummy = onnx_cfg.generate_dummy_inputs(batch_size=2, seq_length=12)
        input_ids = dummy["input_ids"]
        attention_mask = dummy["attention_mask"]

        wrapper = GRTExportWrapper(model).eval()
        with torch.no_grad():
            eager_logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            wrapper_logits = wrapper(input_ids, attention_mask)
            self.assertIsInstance(wrapper_logits, torch.Tensor)
            self.assertLess((eager_logits - wrapper_logits).abs().max().item(), 1e-5)

            exported_program = export_grt_program(model, seq_length=12, batch_size=2)
            exported_logits = exported_program.module()(input_ids, attention_mask)
            max_diff = (eager_logits - exported_logits).abs().max().item()
            self.assertLess(max_diff, 1e-5)

    def test_export_wrapper_and_program_base_model(self):
        config = _tiny_grt_config()
        torch.manual_seed(42)
        model = GRTModel(config).eval()

        onnx_cfg = GRTOnnxConfig(config=config, task="default")
        dummy = onnx_cfg.generate_dummy_inputs(batch_size=2, seq_length=12)
        input_ids = dummy["input_ids"]
        attention_mask = dummy["attention_mask"]

        wrapper = GRTExportWrapper(model).eval()
        pooled_wrapper = GRTExportWrapper(model, return_pooled=True).eval()

        with torch.no_grad():
            eager_hidden = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            wrapper_hidden = wrapper(input_ids, attention_mask)
            self.assertIsInstance(wrapper_hidden, torch.Tensor)
            self.assertLess((eager_hidden - wrapper_hidden).abs().max().item(), 1e-5)

            hidden_out, pooled_out = pooled_wrapper(input_ids, attention_mask)
            self.assertEqual(tuple(hidden_out.shape), (2, 12, config.hidden_size))
            self.assertEqual(tuple(pooled_out.shape), (2, config.hidden_size))

            exported_program = export_grt_program(model, seq_length=12, batch_size=2)
            exported_hidden = exported_program.module()(input_ids, attention_mask)
            max_diff = (eager_hidden - exported_hidden).abs().max().item()
            self.assertLess(max_diff, 1e-5)

    def test_export_grt_to_onnx(self):
        config = _tiny_grt_config()
        torch.manual_seed(7)
        model = GRTForCausalLM(config).eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "grt_causal_lm.onnx"
            result_path = export_grt_to_onnx(model, out_path, seq_length=8, batch_size=2)
            self.assertEqual(result_path, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
