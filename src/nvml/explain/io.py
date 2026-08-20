import pickle
from pathlib import Path

from nvml.explain.interfaces import RawExplanation
from nvml.utils import make_dir


def explanation_path(save_loc: Path, tag: str) -> Path:
    return save_loc / "explanations" / f"explanations_{tag}.pkl"


def save_explanations(raw_explanations: list[RawExplanation], path: Path) -> Path:
    make_dir(path)
    with open(path, "wb") as file:
        pickle.dump(raw_explanations, file)
    return path


def load_explanations(path: Path) -> list[RawExplanation]:
    with open(path, "rb") as file:
        return pickle.load(file)
