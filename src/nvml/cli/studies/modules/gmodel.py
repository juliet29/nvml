import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

import matplotlib.pyplot as plt
from cyclopts import App
from icecream import ic

from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.cli.studies.helpers import cfg
from nvml.cli.studies.study_paths import StudyPaths
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model_interfaces import GraphModelParams
from nvml.gmodel.model_utils import (
    cross_validate,
    evaluate,
    init_model,
    load_data,
    split_data,
    train_model,
)
from nvml.gmodel.viz import CrossValidationReport, TrainingWatcher, VisualizeTraining
from nvml.utils import save_mpl_fig

gmod = App("gmod")


@gmod.command()
def fc():
    cfg = CONFIGS_DICT["msd_aug20"]
    r = cfg.get_one_case_data().sql
    assert r.exists()
    return cfg


@gmod.command()
def fd():
    # test dataset
    return FlowGraphDataset(cfg, StudyPaths.data.gnn.test)


@gmod.command()
def fe():
    # larger test dataset
    return FlowGraphDataset(cfg, StudyPaths.data.gnn.aug20)


@gmod.command()
def ff(
    cross_validation: bool = False,
    n_folds: int = 5,
    epochs: int = 100,
    show_live: bool = False,
):
    cfg = CONFIGS_DICT["msd_aug20"]
    p = StudyPaths.data.gnn.aug20
    figures = p / "figures"

    bsize = 8
    hidden_channels = 16

    ds = load_data(cfg, p, ClusterModelParams("N", 2))
    ic(ds.num_classes)

    graph_params = GraphModelParams.make(hidden_channels, ds)

    if cross_validation:
        fold_accuracies = cross_validate(ds, graph_params, bsize, p, n_folds, epochs)
        report = CrossValidationReport(fold_accuracies)
        report.report()
        fig = report.make_plot()
        save_mpl_fig(fig, figures / "gnn_cross_validation.png")
    else:
        sdl = split_data(ds, bsize)
        model_and_details = init_model(graph_params)
        watcher = TrainingWatcher(
            save_path=figures / "gnn_convergence.png", show_live=show_live
        )
        train_model(sdl.train, model_and_details, p, epochs, val_loader=sdl.val, watcher=watcher)
        evaluate(sdl.test, model_and_details.model)
        fig = VisualizeTraining(watcher).make_plot()
        save_mpl_fig(fig, figures / "gnn_convergence.png")

    plt.show()
    return fig
