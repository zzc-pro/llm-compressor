"""Save STE checkpoints without accessing offloaded model parameters."""

from copy import deepcopy
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict


def save_ste_checkpoint(
    model, processor, save_dir, state_dict, metadata, *, max_shard_size="5GB",
):
    """Persist fixed grids first, then shard the already materialized CPU weights."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if any(tensor.device.type != "cpu" for tensor in state_dict.values()):
        raise ValueError("STE export requires an entirely CPU state_dict")

    # Preserve the original grids even if a later checkpoint write fails.
    grid_path = save_dir / f"{metadata['quantization_method']}_quantization.pt"
    temporary_grid = grid_path.with_suffix(".pt.tmp")
    torch.save(metadata, temporary_grid)
    temporary_grid.replace(grid_path)
    processor.save_pretrained(save_dir)

    config = deepcopy(model.config)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config.architectures = [type(model).__name__]
    config.save_pretrained(save_dir)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(save_dir)

    # Avoid save_pretrained(), whose parameter counting can onload the model.
    save_torch_state_dict(
        state_dict, save_dir, max_shard_size=max_shard_size,
        safe_serialization=True,
    )
