"""Quantize the language backbone of a local dense Qwen3.5 model with GPTQ W4A16."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from compressed_tensors.offload import align_module_device, dispatch_model
from compressed_tensors.quantization.lifecycle.forward import quantize
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    Qwen3_5ForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

from calibration_data import (
    build_calibration_dataset,
    load_eb_nav_dataset,
    load_local_dataset,
)


@torch.no_grad()
def save_ste_model(model, processor, save_dir):
    """Export plain HF weights and unsigned affine int4 grids for STE."""
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
                    or args.actorder not in (None, "static")
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
        "quantization_method": "gptq",
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
    processor.save_pretrained(save_dir)
    torch.save(metadata, save_dir / "gptq_quantization.pt")
    print(f"Saved STE model and {len(layers)} quantization grids to {save_dir}")


def data_collator(batch):
    """Preserve processor batch dimensions and variable image patch counts."""
    if len(batch) != 1:
        raise ValueError("VLM calibration requires batch size 1")
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Local dense Qwen3.5 model directory.")
    parser.add_argument(
        "--dataset", required=True,
        help="Local Flickr30k data or original EB-Nav multi-step JSON/directory.",
    )
    parser.add_argument("--dataset-type", choices=("flickr30k", "eb-nav"),
                        default="flickr30k")
    parser.add_argument(
        "--exclude-task-split", type=Path,
        help="EB-Nav SP split.json whose validation tasks must be excluded.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument(
        "--image-root", type=Path,
        help="Root for image paths or EB-Nav images.zip; required for EB-Nav.",
    )
    parser.add_argument("--prompt", default="What does the image show?")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128,
                        help="Number of weight columns quantized per GPTQ block.")
    parser.add_argument("--dampening-frac", type=float, default=0.01,
                        help="Hessian diagonal dampening fraction.")
    parser.add_argument("--save-dir", "--output", dest="save_dir", type=Path)
    parser.add_argument("--save-format", choices=["compressed", "ste"],
                        default="compressed")
    args = parser.parse_args()
    if args.num_calibration_samples < 1 or args.max_seq_length < 1:
        parser.error("Calibration sample count and sequence length must be positive.")
    if args.block_size < 1:
        parser.error("--block-size must be positive.")
    if not 0 < args.dampening_frac < 1:
        parser.error("--dampening-frac must be between 0 and 1 (exclusive).")
    if args.dataset_type == "eb-nav" and args.image_root is None:
        parser.error("--dataset-type eb-nav requires --image-root.")
    if args.dataset_type != "eb-nav" and args.exclude_task_split is not None:
        parser.error("--exclude-task-split is only used with --dataset-type eb-nav.")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    dataset_path = Path(args.dataset).expanduser().resolve()
    image_root = args.image_root or (
        dataset_path if dataset_path.is_dir() else dataset_path.parent
    )
    if args.dataset_type == "eb-nav":
        samples = load_eb_nav_dataset(dataset_path, args.exclude_task_split)
    else:
        samples = load_local_dataset(dataset_path, args.split).shuffle(seed=42)
    image_column = "image" if args.dataset_type == "eb-nav" else args.image_column
    dataset = build_calibration_dataset(
        processor, samples, args.num_calibration_samples, args.max_seq_length,
        image_column, args.prompt, image_root,
    )
    print(f"Calibration: {len(dataset)} complete image-text samples")

    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, dtype="auto", local_files_only=True,
        )

    quantization = {"scheme": "W4A16"}
    if args.save_format == "ste":
        quantization = {"config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4, "type": "int", "symmetric": False,
                "strategy": "group", "group_size": 128,
            },
        }}}
    recipe = [
        GPTQModifier(
            targets="Linear", **quantization,
            block_size=args.block_size, dampening_frac=args.dampening_frac,
            # Static ordering preserves contiguous groups for STE export.
            actorder="static",
            ignore=[
                "re:.*lm_head", "re:.*visual.*",
                r"re:.*linear_attn\.in_proj_a$",
                r"re:.*linear_attn\.in_proj_b$",
            ],
        ),
    ]
    oneshot(
        model=model, processor=processor, dataset=dataset, recipe=recipe,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length, batch_size=1,
        data_collator=data_collator,
        sequential_targets=["Qwen3_5DecoderLayer"],
        sequential_offload_device="cpu",
        sequential_prefetch=False,
    )

    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    generation_dataset = build_calibration_dataset(
        processor, samples, 1, args.max_seq_length,
        image_column, args.prompt, image_root, generation_prompt=True,
    )
    inputs = {key: value.to(model.device)
              for key, value in data_collator([generation_dataset[0]]).items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    print(processor.decode(
        output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    ))

    save_dir = args.save_dir or (
        Path(__file__).resolve().parent / "outputs"
        / f"{Path(args.model).name}-GPTQ-W4A16"
    )
    if args.save_format == "ste":
        save_ste_model(model, processor, save_dir)
    else:
        model.save_pretrained(save_dir, save_compressed=True)
        processor.save_pretrained(save_dir)
    print(f"Saved GPTQ VLM to {save_dir}")


if __name__ == "__main__":
    main()
