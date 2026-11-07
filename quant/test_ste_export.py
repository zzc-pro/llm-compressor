"""Regression tests for CPU-only STE serialization and failure recovery."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel

from ste_export import save_ste_checkpoint


class ProcessorStub:
    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer_config.json").write_text("{}")


class STEExportTest(unittest.TestCase):
    def setUp(self):
        self.model = GPT2LMHeadModel(GPT2Config(
            n_layer=1, n_head=2, n_embd=8, n_positions=16, vocab_size=16,
            bos_token_id=1, eos_token_id=2,
        )).to(dtype=torch.bfloat16)
        self.model.config.quantization_config = {"runtime_only": True}
        self.state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }
        self.metadata = {
            "format_version": 1, "quantization_method": "gptq", "bits": 4,
            "groupsize": 128, "symmetric": False,
            "layers": {"probe": {
                "scale": torch.tensor([[0.125]]),
                "zero": torch.tensor([[8.0]]), "maxq": torch.tensor(15),
            }},
        }

    def test_sharded_round_trip_without_accessing_model_parameters(self):
        for method in ("awq", "gptq"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                self.metadata["quantization_method"] = method
                # Reproduce an offloaded model that cannot be traversed safely.
                with patch.object(self.model, "named_parameters", side_effect=AssertionError("onload")), \
                     patch.object(self.model, "state_dict", side_effect=AssertionError("onload")), \
                     patch.object(self.model, "num_parameters", side_effect=AssertionError("onload")):
                    save_ste_checkpoint(
                        self.model, ProcessorStub(), temp, self.state,
                        self.metadata, max_shard_size=256,
                    )
                directory = Path(temp)
                index = json.loads((directory / "model.safetensors.index.json").read_text())
                self.assertEqual(set(index["weight_map"]), set(self.state))
                actual = {}
                for shard in set(index["weight_map"].values()):
                    actual.update(load_file(directory / shard))
                for name, expected in self.state.items():
                    self.assertTrue(torch.equal(expected, actual[name]), name)
                grid = torch.load(directory / f"{method}_quantization.pt", weights_only=True)
                for key in ("scale", "zero", "maxq"):
                    self.assertTrue(torch.equal(grid["layers"]["probe"][key], self.metadata["layers"]["probe"][key]))
                self.assertTrue((directory / "tokenizer_config.json").exists())
                config = json.loads((directory / "config.json").read_text())
                self.assertNotIn("quantization_config", config)
                self.assertTrue(hasattr(self.model.config, "quantization_config"))
                loaded = GPT2LMHeadModel.from_pretrained(directory, local_files_only=True, dtype="auto")
                for name, tensor in loaded.state_dict().items():
                    self.assertTrue(torch.equal(tensor, self.state[name]), name)

    def test_qwen_vlm_native_weight_names_round_trip(self):
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        config = Qwen2_5_VLConfig(
            text_config={
                "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "bos_token_id": 1, "eos_token_id": 2,
                "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            },
            vision_config={
                "depth": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 2, "out_hidden_size": 16, "patch_size": 2,
                "temporal_patch_size": 1, "fullatt_block_indexes": [0],
            },
            image_token_id=3, video_token_id=4,
            vision_start_token_id=5, vision_end_token_id=6,
        )
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temp:
            save_ste_checkpoint(model, ProcessorStub(), temp, state, self.metadata, max_shard_size=1024)
            loaded = Qwen2_5_VLForConditionalGeneration.from_pretrained(temp, local_files_only=True).eval()
            self.assertEqual(set(loaded.state_dict()), set(state))
            for name, tensor in loaded.state_dict().items():
                self.assertTrue(torch.equal(tensor, state[name]), name)
            with torch.no_grad():
                inputs = torch.tensor([[10, 11]])
                torch.testing.assert_close(model(inputs).logits, loaded(inputs).logits, rtol=0, atol=0)

    def test_weight_write_failure_keeps_original_grids_and_processor(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("ste_export.save_torch_state_dict", side_effect=OSError("disk write failed")):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    save_ste_checkpoint(self.model, ProcessorStub(), temp, self.state, self.metadata)
            directory = Path(temp)
            grid = torch.load(directory / "gptq_quantization.pt", weights_only=True)
            self.assertTrue(torch.equal(grid["layers"]["probe"]["scale"], self.metadata["layers"]["probe"]["scale"]))
            self.assertTrue((directory / "tokenizer_config.json").exists())
            self.assertTrue((directory / "config.json").exists())
            self.assertFalse((directory / "gptq_quantization.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
