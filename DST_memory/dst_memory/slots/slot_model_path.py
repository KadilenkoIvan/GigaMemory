"""Resolve slot model directory or HuggingFace id (shared by serving and tooling)."""

from pathlib import Path


def _is_hf_repo_id(s: str) -> bool:
    """
    True for HuggingFace hub ids like 'Qwen/Qwen2.5-0.5B' (namespace/model).
    False for local paths (absolute, or multi-segment relative paths).
    """
    if "/" not in s:
        return False
    if Path(s).is_absolute():
        return False
    parts = s.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False
    if Path(s).exists():
        return False
    return True


def resolve_slot_model_path(model_path: str) -> str:
    """
    Return a path/id suitable for AutoTokenizer/AutoModel.from_pretrained.

    - Existing directory (relative or absolute) -> resolved absolute path.
    - HuggingFace repo id -> unchanged string.
    - Otherwise -> FileNotFoundError with a clear message.
    """
    raw = str(model_path).strip()
    if not raw:
        raise ValueError("slot model path is empty")

    local = Path(raw).expanduser()
    if local.is_dir():
        return str(local.resolve())

    if _is_hf_repo_id(raw):
        return raw

    raise FileNotFoundError(
        f"Slot model directory not found: {model_path!r}. "
        "Use an existing folder with tokenizer + weights, a HuggingFace id (e.g. org/model), "
        "or --slot-use-stub."
    )
