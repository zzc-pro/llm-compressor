"""Local Flickr30k and EB-Nav image-text calibration data."""

from contextlib import ExitStack
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image


OUTPUT_FORMAT = (
    '\n\nOutput only a JSON object with these fields: '
    '"visual_state_description": string, "reasoning_and_reflection": string, '
    '"language_plan": string, "executable_plan": '
    '[{"action_id": integer from 0 to 7, "action_name": string}]. '
    'Do not include action outcomes, image paths, or environment feedback in the output.'
)


def load_local_dataset(path, split):
    """Read a saved HF dataset, a Parquet repository, or a local data file."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Calibration dataset does not exist: {path}")
    if path.is_dir():
        if (path / "dataset_dict.json").is_file() or (path / "dataset_info.json").is_file():
            dataset = load_from_disk(str(path))
            return dataset[split] if isinstance(dataset, DatasetDict) else dataset
        files = sorted(path.glob(f"data/{split}-*.parquet"))
        if not files:
            files = sorted(path.glob(f"{split}-*.parquet"))
        if not files:
            raise ValueError(f"No Parquet shards for split {split!r} in {path}")
        return load_dataset(
            "parquet", data_files={split: [str(file) for file in files]}, split=split,
        )
    loader = "parquet" if path.suffix == ".parquet" else "json"
    return load_dataset(loader, data_files=str(path), split="train")


def _action_values(action):
    action_id, action_name = action["action"]
    if type(action_id) is not int or not 0 <= action_id <= 7:
        raise ValueError(f"Invalid EB-Nav action ID: {action_id!r}")
    if not isinstance(action_name, str) or not action_name.strip():
        raise ValueError("EB-Nav action names must be non-empty strings.")
    return {"action_id": action_id, "action_name": action_name}


def load_eb_nav_dataset(path, exclude_task_split=None):
    """Expand original base episodes, including successes and failures, before shuffling."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "eb-nav_dataset_multi_step.json"
    excluded = set()
    if exclude_task_split is not None:
        split = json.loads(Path(exclude_task_split).expanduser().read_text())
        excluded = {str(task).removeprefix("base/") for task in split["validation_task_ids"]}
    samples = []
    for episode in json.loads(path.read_text()):
        if episode["eval_set"] != "base" or str(episode["episode_id"]) in excluded:
            continue
        history = []
        for index, decision in enumerate(episode["trajectory"]):
            prompt = episode["input"].strip()
            if not prompt or episode["instruction"] not in prompt:
                raise ValueError("EB-Nav input must include the navigation instruction.")
            if history:
                prompt = prompt.replace(
                    "Aim for about 1-2 actions in this step.",
                    "Aim for about 5-6 actions in this step to be closer to the target object.",
                )
                prompt += "\n\nThe action history (already executed):\n" + "\n".join(history)
            member = PurePosixPath(decision["input_image_path"])
            if member.is_absolute() or ".." in member.parts or "base" not in member.parts:
                raise ValueError(f"Invalid base observation path: {member}")
            actions = [_action_values(action) for action in decision["executable_plan"]]
            answer = {
                "visual_state_description": decision.get("visual_description") or "",
                "reasoning_and_reflection": decision.get("reasoning_and_reflection") or "",
                "language_plan": decision.get("language_plan") or "",
                "executable_plan": actions,
            }
            if isinstance(answer["language_plan"], list) and all(
                isinstance(step, str) for step in answer["language_plan"]
            ):
                answer["language_plan"] = "\n".join(answer["language_plan"])
            if any(not isinstance(answer[key], str) for key in (
                "visual_state_description", "reasoning_and_reflection", "language_plan",
            )):
                raise ValueError("EB-Nav description, reasoning and language plan must be strings.")
            samples.append({
                "task_id": f"base/{episode['episode_id']}",
                "trajectory_id": f"{episode['model_name']}/base/{episode['episode_id']}",
                "decision_index": index,
                "image": member.as_posix(),
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": prompt + OUTPUT_FORMAT},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": json.dumps(answer, ensure_ascii=False)},
                    ]},
                ],
            })
            # Current outcomes become visible only in later decisions.
            for action, values in zip(decision["executable_plan"], actions):
                history.append(
                    f"Step {len(history) + 1}, action id {values['action_id']}, "
                    f"{values['action_name']}, env feedback: {action.get('env_feedback', '')}"
                )
    if not samples:
        raise ValueError("EB-Nav data contains no base decisions after task exclusion.")
    random.Random(42).shuffle(samples)
    return samples


def build_calibration_dataset(processor, samples, num_samples, max_seq_length,
                              image_column, prompt, image_root, *, generation_prompt=False):
    """Keep complete image-text examples without truncating image tokens."""
    rows = []
    image_root = Path(image_root).expanduser().resolve()
    with ExitStack() as stack:
        archive = None
        for sample in samples:
            image = sample[image_column]
            if isinstance(image, str):
                path = Path(image).expanduser()
                path = path if path.is_absolute() else image_root / path
                if path.is_file():
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                elif "messages" in sample and (image_root / "images.zip").is_file():
                    if archive is None:
                        archive = stack.enter_context(zipfile.ZipFile(image_root / "images.zip"))
                    with archive.open(sample[image_column]) as stream, Image.open(stream) as source:
                        image = source.convert("RGB")
                else:
                    raise ValueError(f"Calibration image does not exist: {path}")
            if image is None:
                raise ValueError(f"Missing image in column {image_column!r}")
            messages = sample.get("messages")
            if messages is None:
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt},
                ]}]
            if generation_prompt:
                messages = [message for message in messages if message["role"] != "assistant"]
            template_kwargs = {"enable_thinking": False} if "messages" in sample else {}
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=messages[-1]["role"] != "assistant",
                **template_kwargs,
            )
            encoded = processor(
                text=[text], images=[image], padding=False, truncation=False,
                return_tensors="pt",
            )
            if encoded["input_ids"].shape[-1] > max_seq_length:
                continue
            encoded.pop("token_type_ids", None)
            rows.append({key: value.tolist() for key, value in encoded.items()})
            if len(rows) == num_samples:
                return Dataset.from_list(rows)
    raise ValueError(
        f"Only {len(rows)} usable image-text samples; requested {num_samples}. "
        "Increase --max-seq-length or provide more calibration images."
    )
