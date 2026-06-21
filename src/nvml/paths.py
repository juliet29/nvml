import os
from pathlib import Path


class StoragePaths:
    base = Path(os.environ["DIR_STORAGE"])
    nvflow_latest = base / "nvflow/260508_1101"
