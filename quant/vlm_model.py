"""Shared architecture selection for local VLM quantization entry points."""

from transformers import (
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
)


_VLM_ARCHITECTURES = {
    "qwen2_5_vl": (Qwen2_5_VLForConditionalGeneration, "Qwen2_5_VLDecoderLayer"),
    "qwen3_5": (Qwen3_5ForConditionalGeneration, "Qwen3_5DecoderLayer"),
}


def get_vlm_architecture(model_path):
    """Resolve the model class and calibration layer from the local config."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    try:
        return _VLM_ARCHITECTURES[config.model_type]
    except KeyError:
        raise ValueError(
            f"Unsupported VLM model_type {config.model_type!r}; "
            f"expected one of {', '.join(_VLM_ARCHITECTURES)}."
        ) from None
