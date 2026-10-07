"""Calibrate and quantize a local causal language model with AWQ."""

import argparse
from pathlib import Path

from compressed_tensors.offload import dispatch_model
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier


def build_calibration_dataset(tokenizer, samples, num_samples, max_seq_length):
    """Pack documents separated by EOS into complete, fixed-length blocks."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token for document boundaries.")
    blocks, buffer = [], []
    for sample in samples:
        if "text" not in sample:
            raise ValueError("Calibration dataset must contain a text column.")
        buffer.extend(
            tokenizer(sample["text"], add_special_tokens=False, truncation=False)[
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
        help="Local JSON/JSONL calibration data with a text column.",
    )
    parser.add_argument(
        "--num-calibration-samples", type=int, default=256,
        help="Number of complete token blocks to collect.",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--save-dir", "--output", dest="save_dir", type=Path)
    args = parser.parse_args()
    if args.num_calibration_samples < 1 or args.max_seq_length < 1:
        parser.error("Calibration sample count and sequence length must be positive.")

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

    # Use symmetric integer weights with group size 128; keep MoE routers unquantized.
    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(
            targets="Linear",
            scheme=f"W{args.bits}A16",
            ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
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

    # Save packed weights and tokenizer in compressed-tensors format.
    save_dir = args.save_dir or (
        Path(__file__).resolve().parent
        / "outputs"
        / f"{Path(args.model).name}-AWQ-W{args.bits}A16"
    )
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved AWQ model to {save_dir}")


if __name__ == "__main__":
    main()
