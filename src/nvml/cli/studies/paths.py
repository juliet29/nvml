import os
from pathlib import Path


class StoragePaths:
    base = Path(os.environ["DIR_STORAGE"])
    nvflow_latest = base / "nvflow/260508_1101"


BASE = Path(os.environ["DIR_STORAGE"]) / "nvml"


class TempFiguresPaths:
    figs_base = BASE / "figures_temp"
    qdim_corr = figs_base / "qdim_corr"


class ProjectPaths:  # for now assuming all are temp
    graphs = BASE / "jun24"
