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
"""Unit tests for 100% Rotational GRT Audio (`GRTAudioModel`), TTS (`GRTTTSForConditionalGeneration`),
and Omni (`GRTOmniForConditionalGeneration`).
"""

import unittest

import torch

from transformers import (
    GRTASRConfig,
    GRTASRForConditionalGeneration,
    GRTASRModel,
    GRTAudioConfig,
    GRTAudioModel,
    GRTConfig,
    GRTOmniConfig,
    GRTOmniForConditionalGeneration,
    GRTOmniModel,
    GRTOmniProcessor,
    GRTTTSConfig,
    GRTTTSForConditionalGeneration,
    GRTVisionConfig,
)
from transformers.models.grt import HessianPairAdam


class GRTOmniTTSTest(unittest.TestCase):
    def _tiny_audio_config(self) -> GRTAudioConfig:
        return GRTAudioConfig(
            sample_rate=16000,
            n_fft=30,
            hop_length=16,
            win_length=30,
            max_audio_frames=8,
            temporal_merge_size=2,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            stem_n_rotations=3,
            merger_n_rotations=3,
            vocoder_n_rotations=3,
            frames_per_token=2,
            ffn_n_rotations=3,
            n_qkv_rotations=2,
            n_mixer_rotations=1,
        )

    def _tiny_vision_config(self) -> GRTVisionConfig:
        return GRTVisionConfig(
            image_size=28,
            patch_size=7,
            spatial_merge_size=2,
            hidden_size=32,
            intermediate_size=160,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_n_rotations=3,
            merger_n_rotations=3,
            ffn_n_rotations=3,
            n_qkv_rotations=2,
            n_mixer_rotations=1,
        )

    def _tiny_text_config(self) -> GRTConfig:
        return GRTConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            ffn_n_rotations=3,
            n_qkv_rotations=2,
            n_mixer_rotations=1,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

    def test_zero_linear_and_zero_conv1d_across_audio_tts_and_omni(self):
        """All audio, TTS, and Omni models must have 0 nn.Linear and 0 nn.Conv1d layers."""
        audio_cfg = self._tiny_audio_config()
        vision_cfg = self._tiny_vision_config()
        text_cfg = self._tiny_text_config()

        audio_model = GRTAudioModel(audio_cfg)
        tts_model = GRTTTSForConditionalGeneration(GRTTTSConfig(audio_config=audio_cfg, text_config=text_cfg))
        omni_model = GRTOmniForConditionalGeneration(
            GRTOmniConfig(vision_config=vision_cfg, audio_config=audio_cfg, text_config=text_cfg)
        )

        for name, model in (("audio", audio_model), ("tts", tts_model), ("omni", omni_model)):
            bad_layers = [
                mod_name
                for mod_name, mod in model.named_modules()
                if isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d))
            ]
            self.assertEqual(bad_layers, [], f"Found forbidden Linear/Conv1d in {name}: {bad_layers}")

    def test_audio_model_waveform_and_polar_frames(self):
        audio_cfg = self._tiny_audio_config()
        model = GRTAudioModel(audio_cfg).eval()
        self.assertFalse(model.blocks[0].attention.is_causal)

        waveform = torch.randn(2, 160)
        with torch.no_grad():
            out_wave = model(waveform)
        self.assertEqual(tuple(out_wave.last_hidden_state.shape), (2, 8, 32))
        self.assertEqual(tuple(out_wave.pooler_output.shape), (2, 4, 32))

        polar_frames = torch.randn(2, 8, audio_cfg.spec_dim)
        with torch.no_grad():
            out_frames = model(polar_frames)
        self.assertEqual(tuple(out_frames.last_hidden_state.shape), (2, 8, 32))
        self.assertEqual(tuple(out_frames.pooler_output.shape), (2, 4, 32))

    def test_tts_forward_backward_synthesize_and_hessian_optimizer(self):
        torch.manual_seed(0)
        audio_cfg = self._tiny_audio_config()
        text_cfg = self._tiny_text_config()
        tts_cfg = GRTTTSConfig(audio_config=audio_cfg, text_config=text_cfg)
        model = GRTTTSForConditionalGeneration(tts_cfg)
        self.assertIs(model.lm_head.phase_angles, model.model.embedding.phase_angles)

        optimizer = model.configure_optimizers(lr=5e-3, weight_decay=0.01, hessian_mode=True)
        self.assertIsInstance(optimizer, HessianPairAdam)

        input_ids = torch.tensor([[1, 10, 20, 30, 2]])
        target_spec = torch.randn(1, 10, audio_cfg.spec_dim)

        out0 = model(input_ids=input_ids, labels=input_ids, target_spec=target_spec, return_waveform=True)
        loss0 = out0.loss.item()
        self.assertEqual(tuple(out0.logits.shape), (1, 5, 256))
        self.assertEqual(tuple(out0.speech_spec.shape), (1, 10, audio_cfg.spec_dim))
        self.assertIsNotNone(out0.speech_waveform)

        for _ in range(3):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=input_ids, target_spec=target_spec, return_waveform=False)
            out.loss.backward()
            optimizer.step()

        out_final = model(input_ids=input_ids, labels=input_ids, target_spec=target_spec, return_waveform=False)
        self.assertLess(out_final.loss.item(), loss0)

        waveform, spec = model.synthesize_speech(input_ids=input_ids)
        self.assertEqual(tuple(spec.shape), (1, 10, audio_cfg.spec_dim))
        self.assertEqual(waveform.shape[0], 1)

    def test_omni_forward_generate_speech_and_processor(self):
        torch.manual_seed(0)
        vision_cfg = self._tiny_vision_config()
        audio_cfg = self._tiny_audio_config()
        text_cfg = self._tiny_text_config()
        omni_cfg = GRTOmniConfig(
            vision_config=vision_cfg,
            audio_config=audio_cfg,
            text_config=text_cfg,
            image_token_id=250,
            vision_start_token_id=248,
            vision_end_token_id=249,
            audio_token_id=251,
        )
        self.assertEqual(omni_cfg.num_image_tokens, 4)
        self.assertEqual(omni_cfg.num_audio_tokens, 4)

        model = GRTOmniForConditionalGeneration(omni_cfg)
        self.assertIs(model.lm_head.phase_angles, model.model.language_model.embedding.phase_angles)

        # 4 image tokens (250) + 4 audio tokens (251)
        input_ids = torch.tensor([[1, 248, 250, 250, 250, 250, 249, 251, 251, 251, 251, 10, 11]])
        pixel_values = torch.randn(1, 3, 28, 28)
        audio_values = torch.randn(1, 160)
        labels = input_ids.clone()
        labels[:, :11] = -100
        target_spec = torch.randn(1, 26, audio_cfg.spec_dim)

        optimizer = model.configure_optimizers(lr=5e-3, weight_decay=0.01, hessian_mode=True)
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            labels=labels,
            target_spec=target_spec,
            return_speech=True,
            return_waveform=True,
        )
        self.assertIsNotNone(out.loss)
        self.assertEqual(tuple(out.logits.shape), (1, 13, 256))
        self.assertEqual(tuple(out.image_hidden_states.shape), (1, 4, 32))
        self.assertEqual(tuple(out.audio_hidden_states.shape), (1, 4, 32))
        self.assertEqual(tuple(out.speech_spec.shape), (1, 26, audio_cfg.spec_dim))
        self.assertIsNotNone(out.speech_waveform)

        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            waveform, spec = model.synthesize_speech(
                input_ids=input_ids, pixel_values=pixel_values, audio_values=audio_values
            )
            self.assertEqual(tuple(spec.shape), (1, 26, audio_cfg.spec_dim))
            self.assertEqual(waveform.shape[0], 1)

            gen = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                audio_values=audio_values,
                max_new_tokens=3,
                do_sample=False,
            )
            self.assertEqual(tuple(gen.shape), (1, 16))

        proc = GRTOmniProcessor(image_seq_length=4, audio_seq_length=4)
        self.assertEqual(proc.replace_audio_token("Audio: <|video_pad|>"), "Audio: " + "<|video_pad|>" * 4)
        self.assertEqual(
            proc.replace_image_token("Img: <|image_pad|>"),
            "Img: <|vision_start|>" + "<|image_pad|>" * 4 + "<|vision_end|>",
        )

    def test_grt_asr_for_conditional_generation(self):
        torch.manual_seed(7)
        audio_cfg = self._tiny_audio_config()
        text_cfg = self._tiny_text_config()
        asr_cfg = GRTASRConfig(
            audio_config=audio_cfg,
            text_config=text_cfg,
            audio_token_id=251,
        )
        self.assertEqual(asr_cfg.model_type, "grt_asr")
        self.assertEqual(asr_cfg.num_audio_tokens, 4)

        model = GRTASRForConditionalGeneration(asr_cfg)
        self.assertIsInstance(model.model, GRTASRModel)
        self.assertIs(model.lm_head.phase_angles, model.model.language_model.embedding.phase_angles)
        for mod in model.modules():
            self.assertNotIsInstance(mod, (torch.nn.Linear, torch.nn.Conv1d))

        audio_values = torch.randn(1, 160)
        # Direct audio-to-logits forward (no input_ids)
        out_direct = model(audio_values=audio_values)
        self.assertEqual(tuple(out_direct.logits.shape), (1, 4, 256))
        self.assertEqual(tuple(out_direct.audio_hidden_states.shape), (1, 4, 32))

        # Multimodal prompt with audio placeholder tokens
        input_ids = torch.tensor([[1, 251, 251, 251, 251, 10, 11]])
        labels = input_ids.clone()
        labels[:, :5] = -100
        out = model(input_ids=input_ids, audio_values=audio_values, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(tuple(out.logits.shape), (1, 7, 256))

        pred_ids = model.transcribe(audio_values=audio_values)
        self.assertEqual(tuple(pred_ids.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
