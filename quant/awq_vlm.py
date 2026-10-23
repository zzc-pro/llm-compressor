"""Quantize the language backbone of a local Qwen3-VL model with AWQ W4A16."""

import argparse
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

from awq import save_ste_model


def load_local_dataset(path, split):
    """Read a saved HF dataset or a local JSON/JSONL/Parquet file."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Calibration dataset does not exist: {path}")
    if path.is_dir():
        dataset = load_from_disk(str(path))
        return dataset[split] if isinstance(dataset, DatasetDict) else dataset
    loader = "parquet" if path.suffix == ".parquet" else "json"
    return load_dataset(loader, data_files=str(path), split="train")


def build_calibration_dataset(processor, samples, num_samples, max_seq_length,
                              image_column, prompt, image_root):
    """Keep complete image-text examples without truncating image tokens."""
    rows = []
    for sample in samples:
        image = sample[image_column]
        if isinstance(image, str):
            path = Path(image).expanduser()
            if not path.is_absolute():
                path = image_root / path
            path = path.resolve()
            if not path.is_file():
                raise ValueError(f"Calibration image does not exist: {path}")
            from PIL import Image

            with Image.open(path) as source:
                image = source.convert("RGB")
        if image is None:
            raise ValueError(f"Missing image in column {image_column!r}")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        encoded = processor(
            text=[text], images=[image], padding=False, truncation=False,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[-1] > max_seq_length:
            continue
        rows.append({key: value.tolist() for key, value in encoded.items()})
        if len(rows) == num_samples:
            return Dataset.from_list(rows)
    raise ValueError(
        f"Only {len(rows)} usable image-text samples; requested {num_samples}. "
        "Increase --max-seq-length or provide more calibration images."
    )


def data_collator(batch):
    """Preserve processor batch dimensions and variable image patch counts."""
    if len(batch) != 1:
        raise ValueError("VLM calibration requires batch size 1")
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3-VL model directory.")
    parser.add_argument("--dataset", required=True,
                        help="Local saved HF dataset or JSON/JSONL/Parquet file.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--image-root", type=Path,
                        help="Root for image paths; defaults to the dataset directory.")
    parser.add_argument("--prompt", default="What does the image show?")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-dir", "--output", dest="save_dir", type=Path)
    parser.add_argument("--save-format", choices=["compressed", "ste"],
                        default="compressed")
    args = parser.parse_args()
    if args.num_calibration_samples < 1 or args.max_seq_length < 1:
        parser.error("Calibration sample count and sequence length must be positive.")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype="auto", local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    dataset_path = Path(args.dataset).expanduser().resolve()
    image_root = args.image_root or (
        dataset_path if dataset_path.is_dir() else dataset_path.parent
    )
    samples = load_local_dataset(dataset_path, args.split).shuffle(seed=42)
    dataset = build_calibration_dataset(
        processor, samples, args.num_calibration_samples, args.max_seq_length,
        args.image_column, args.prompt, image_root,
    )
    print(f"Calibration: {len(dataset)} complete image-text samples")

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
        AWQModifier(duo_scaling=False),
        QuantizationModifier(
            targets="Linear", **quantization,
            ignore=["re:.*lm_head", "re:.*visual.*"],
        ),
    ]
    oneshot(
        model=model, processor=processor, dataset=dataset, recipe=recipe,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length, batch_size=1,
        data_collator=data_collator,
        sequential_targets=["Qwen3VLTextDecoderLayer"],
    )

    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    inputs = {key: value.to(model.device)
              for key, value in data_collator([dataset[0]]).items()}
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
