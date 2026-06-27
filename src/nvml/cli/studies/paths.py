from pathlib import Path

PATH_TO_STORAGE = "/oak/stanford/groups/risheej/jnwagwu/storage"


class StoragePaths:
    base = Path(PATH_TO_STORAGE)
    nvflow_latest = base / "nvflow/260508_1101"  # has idf, sql models..


BASE = Path(PATH_TO_STORAGE) / "nvml"


class TempFiguresPaths:
    figs_base = BASE / "figures_temp"
    qdim_corr = figs_base / "qdim_corr"


class ProjectPaths:  # for now assuming all are temp
    graphs = BASE / "jun24"
    figs = TempFiguresPaths
