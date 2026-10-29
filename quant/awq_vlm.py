"""Quantize the language backbone of a local dense Qwen3.5 model with AWQ W4A16."""

import argparse
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

from awq import save_ste_model
from calibration_data import (
    build_calibration_dataset, load_eb_nav_dataset, load_local_dataset,
)


class MemoryEfficientAWQModifier(AWQModifier):
    """Offload search outputs without retaining whole-dataset attention matrices."""

    @torch.no_grad()
    def _run_samples(self, module: torch.nn.Module) -> list[torch.Tensor]:
        outputs = []
        # Deliberately avoid prefetch: only one sample needs to be on device.
        for batch in self._parent_args_cache[module]:
            result = module(**batch)
            hidden_states = result[0] if isinstance(result, tuple) else result
            # Float32 also lets the inherited MSE loss run on CPU for bf16 models.
            outputs.append(hidden_states.detach().to(device="cpu", dtype=torch.float32))
            # The base implementation collects every complete return tuple first;
            # drop its unused attention weights before running the next sample.
            del result, hidden_states, batch
        return outputs


def data_collator(batch):
    """Preserve processor batch dimensions and variable image patch counts."""
    if len(batch) != 1:
        raise ValueError("VLM calibration requires batch size 1")
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local dense Qwen3.5 model directory.")
    parser.add_argument("--dataset", required=True,
                        help="Local Flickr30k data or original EB-Nav multi-step JSON/directory.")
    parser.add_argument("--dataset-type", choices=("flickr30k", "eb-nav"), default="flickr30k")
    parser.add_argument("--exclude-task-split", type=Path,
                        help="EB-Nav SP split.json whose validation tasks must be excluded.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--image-root", type=Path,
                        help="Root for image paths or EB-Nav images.zip; required for EB-Nav.")
    parser.add_argument("--prompt", default="What does the image show?")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-dir", "--output", dest="save_dir", type=Path)
    parser.add_argument("--save-format", choices=["compressed", "ste"],
                        default="compressed")
    args = parser.parse_args()
    if args.num_calibration_samples < 1 or args.max_seq_length < 1:
        parser.error("Calibration sample count and sequence length must be positive.")

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
    dataset = build_calibration_dataset(
        processor, samples, args.num_calibration_samples, args.max_seq_length,
        "image" if args.dataset_type == "eb-nav" else args.image_column, args.prompt, image_root,
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
        MemoryEfficientAWQModifier(
            duo_scaling=False, offload_device=torch.device("cpu"),
        ),
        QuantizationModifier(
            targets="Linear", **quantization,
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
        "image" if args.dataset_type == "eb-nav" else args.image_column, args.prompt, image_root,
        generation_prompt=True,
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
        / f"{Path(args.model).name}-AWQ-W4A16"
    )
    if args.save_format == "ste":
        save_ste_model(model, processor, save_dir)
    else:
        model.save_pretrained(save_dir, save_compressed=True)
        processor.save_pretrained(save_dir)
    print(f"Saved AWQ VLM to {save_dir}")


if __name__ == "__main__":
    main()
