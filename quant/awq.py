"""Calibrate and quantize a local causal language model with AWQ."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from compressed_tensors.offload import align_module_device, dispatch_model
from compressed_tensors.quantization.lifecycle.forward import quantize
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier


@torch.no_grad()
def save_ste_model(model, tokenizer, save_dir):
    """Export plain HF weights and unsigned affine int4 grids for existing STE code."""
    state_dict, layers = {}, {}
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        scheme = getattr(module, "quantization_scheme", None)
        with align_module_device(module, execution_device=torch.device("cpu")):
            if scheme is not None and scheme.weights is not None:
                args = scheme.weights
                if (
                    not isinstance(module, torch.nn.Linear)
                    or args.num_bits != 4
                    or args.type != "int"
                    or args.strategy != "group"
                    or args.group_size != 128
                    or args.symmetric
                    or scheme.input_activations is not None
                    or scheme.output_activations is not None
                ):
                    raise ValueError(f"Unsupported STE quantization scheme for {name}")
                scale = module.weight_scale.detach().float().clone()
                signed_zero = module.weight_zero_point.detach()
                zero = signed_zero.float() + 8
                expected_shape = (module.out_features, module.in_features // 128)
                if scale.shape != expected_shape or zero.shape != expected_shape:
                    raise ValueError(f"Unexpected quantization grid shape for {name}")
                # compressed-tensors uses [-8, 7]; STE uses [0, 15].
                codes = quantize(
                    module.weight, module.weight_scale, signed_zero, args,
                    dtype=torch.int8,
                ).float() + 8
                weight = (
                    (codes.reshape(module.out_features, -1, 128) - zero[..., None])
                    * scale[..., None]
                ).reshape_as(module.weight).to(module.weight.dtype)
                state_dict[prefix + "weight"] = weight
                if module.bias is not None:
                    state_dict[prefix + "bias"] = module.bias.detach().cpu().clone()
                layers[name] = {
                    "scale": scale,
                    "zero": zero,
                    "maxq": torch.tensor(15),
                }
            else:
                tensors = list(module.named_parameters(recurse=False)) + [
                    (key, tensor)
                    for key, tensor in module.named_buffers(recurse=False)
                    if key not in module._non_persistent_buffers_set
                ]
                for key, tensor in tensors:
                    state_dict[prefix + key] = tensor.detach().cpu().clone()
    if not layers:
        raise ValueError("No quantized Linear layers found for STE export")
    metadata = {
        "format_version": 1,
        "quantization_method": "awq",
        "bits": 4,
        "groupsize": 128,
        "symmetric": False,
        "layers": layers,
    }
    # Bypass the compressor's save wrapper and omit its runtime configuration.
    original_config = model.config
    model.config = deepcopy(original_config)
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config
    try:
        PreTrainedModel.save_pretrained(
            model, save_dir, state_dict=state_dict, max_shard_size="5GB",
        )
    finally:
        model.config = original_config
    tokenizer.save_pretrained(save_dir)
    torch.save(metadata, save_dir / "awq_quantization.pt")
    print(f"Saved STE model and {len(layers)} quantization grids to {save_dir}")


def build_calibration_dataset(tokenizer, samples, num_samples, max_seq_length):
    """Pack documents separated by EOS into complete, fixed-length blocks."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token for document boundaries.")
    blocks, buffer = [], []
    for sample in samples:
        if "text" in sample:
            text = sample["text"]
        elif "conversations" in sample:
            # Match the project's existing AWQ ToolACE calibration text format.
            parts = [sample.get("system", "")]
            parts.extend(
                turn.get("value") or turn.get("content") or ""
                for turn in sample["conversations"]
            )
            text = "\n".join(part for part in parts if part)
        else:
            raise ValueError("Calibration data requires text or ToolACE conversations.")
        buffer.extend(
            tokenizer(text, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]
        )
        buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= max_seq_length and len(blocks) < num_samples:
            blocks.append(buffer[:max_seq_length])
            del buffer[:max_seq_length]
        if len(blocks) == num_samples:
            break
    if len(blocks) < num_samples:
        raise ValueError("Calibration dataset has insufficient tokens for full blocks.")
    return Dataset.from_dict(
        {
            "input_ids": blocks,
            "attention_mask": [[1] * max_seq_length for _ in blocks],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/ssd/zhaozichen/huggingface/Qwen/Qwen3-8B"
    )
    parser.add_argument(
        "--dataset",
        default="/ssd/zhaozichen/huggingface/datasets/en/c4-train.00000-of-01024.json.gz",
        help="Local JSON/JSONL calibration data with text or ToolACE conversations.",
    )
    parser.add_argument(
        "--num-calibration-samples", type=int, default=256,
        help="Number of complete token blocks to collect.",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--save-dir", "--output", dest="save_dir", type=Path)
    parser.add_argument(
        "--save-format", choices=["compressed", "ste"], default="compressed",
        help="STE saves plain HF weights plus asymmetric int4 awq_quantization.pt.",
    )
    args = parser.parse_args()
    if args.num_calibration_samples < 1 or args.max_seq_length < 1:
        parser.error("Calibration sample count and sequence length must be positive.")
    if args.save_format == "ste" and args.bits != 4:
        parser.error("STE export requires --bits 4.")

    # Load local weights on CPU; oneshot dispatches layers for calibration.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    # Stream documents until enough full token blocks have been collected.
    samples = load_dataset(
        "json", data_files=args.dataset, split="train", streaming=True
    )
    dataset = build_calibration_dataset(
        tokenizer, samples, args.num_calibration_samples, args.max_seq_length
    )
    print(f"Calibration: {len(dataset)} blocks x {args.max_seq_length} tokens")

    # STE requires asymmetric int4 grids; compressed export keeps the default scheme.
    quantization = {"scheme": f"W{args.bits}A16"}
    if args.save_format == "ste":
        quantization = {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4, "type": "int", "symmetric": False,
                        "strategy": "group", "group_size": 128,
                    },
                },
            },
        }
    ignore = ["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"]
    if model.config.model_type.startswith("qwen3_5"):
        # Keep the linear-attention decay and gating projections in full precision.
        ignore.extend([
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ])
    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(
            targets="Linear",
            **quantization,
            ignore=ignore,
        ),
    ]
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )

    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    inputs = tokenizer("Hello my name is", return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("==========================================")

    # Export packed weights or the plain HF model and fixed grids used by STE.
    save_dir = args.save_dir or (
        Path(__file__).resolve().parent
        / "outputs"
        / f"{Path(args.model).name}-AWQ-W{args.bits}A16"
    )
    if args.save_format == "ste":
        save_ste_model(model, tokenizer, save_dir)
    else:
        model.save_pretrained(save_dir, save_compressed=True)
        tokenizer.save_pretrained(save_dir)
    print(f"Saved AWQ model to {save_dir}")


if __name__ == "__main__":
    main()
