from pathlib import Path

from nvml.cli.paths import PATH_TO_STORAGE, PATH_TO_TEMP


class External:
    base = Path(PATH_TO_STORAGE)
    nvflow_latest = base / "nvflow/260508_1101"  # has idf, sql models..


class Figures:
    base = PATH_TO_TEMP / "figures_temp"
    qdim_corr = base / "qdim_corr"
    tsne = base / "tsne/tests"


class GraphModels:
    base = PATH_TO_TEMP / "gnn"
    test = base / "tests"
    case50 = base / "case50"


class Data:
    base = PATH_TO_TEMP
    graphs_jun24 = base / "jun24"
    graphs_jun30 = base / "jun30"
    qdim_test = base / "qdim/tests"
    tsne = base / "tsne/tests"
    gnn = GraphModels
    # gnn = base / "gnn/tests"


class StudyPaths:  # for now assuming all are temp
    figs = Figures
    data = Data
    external = External
