import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

import matplotlib.pyplot as plt
import xarray as xr
from cyclopts import App
from icecream import ic
from loguru import logger

from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.cli.studies.helpers import cfg
from nvml.cli.studies.study_paths import StudyPaths
from nvml.cluster.spectral import silhouette_sweep
from nvml.cluster.tsne import setup_for_clustering
from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model_interfaces import GraphModelParams
from nvml.gmodel.model_utils import (
    compute_class_weights,
    cross_validate,
    evaluate,
    evaluate_best_model,
    init_model,
    load_data,
    split_data,
    train_model,
)
from nvml.gmodel.processing import GModelNames
from nvml.gmodel.viz import (
    CrossValidationReport,
    TrainingWatcher,
    VisualizeClusters,
    VisualizeConfusion,
    VisualizeCrossValidationConvergence,
    VisualizeQDimByWind,
    VisualizeSilhouette,
    VisualizeTraining,
)
from nvml.qdim.wind import WindDirectionBins
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
def ksweep(wind_sector: str = "N", max_k: int = 6, seed: int = 1204):
    p = StudyPaths.data.gnn.aug20
    silhouette_dir = p / "figures" / "silhouette"
    zarr_path = GModelNames.make_cluster_path(p / "processed") / FileNames.zarr

    da = setup_for_clustering(zarr_path)
    scores = silhouette_sweep(da, wind_sector, list(range(2, max_k + 1)), seed)
    for k, score in scores.items():
        logger.info(f"k={k} silhouette={score:.3f}")
    logger.info(f"Best silhouette at k={max(scores, key=scores.get)}")

    fig = VisualizeSilhouette(scores).make_plot()
    save_mpl_fig(fig, silhouette_dir / f"{wind_sector}.png")
    plt.show()
    return fig


@gmod.command()
def qcompare(
    case_a: str,
    case_b: str,
    wind_sectors: tuple[str, ...] = ("N", "E", "S", "W"),
):
    p = StudyPaths.data.gnn.aug20
    qdim_dir = p / "figures" / "profiles"
    zarr_path = GModelNames.make_cluster_path(p / "processed") / FileNames.zarr

    ds = xr.open_zarr(zarr_path)
    canonical_order = list(WindDirectionBins.labels)
    ordered_sectors = sorted(wind_sectors, key=canonical_order.index)
    qdim_by_wind = ds[dn.q_dim_median].sel(
        {dn.case_name: [case_a, case_b], dn.wind_sector: ordered_sectors}
    )

    viz = VisualizeQDimByWind(qdim_by_wind, [case_a, case_b], ordered_sectors)
    fig = viz.make_plot()
    save_mpl_fig(fig, qdim_dir / f"{case_a}_vs_{case_b}.png")
    plt.show()
    return fig


@gmod.command()
def cviz(wind_sector: str = "N", n_clusters: int = 3):
    p = StudyPaths.data.gnn.aug20
    clusters_dir = p / "figures" / "clusters" / wind_sector / str(n_clusters)

    cluster_path = GModelNames.make_cluster_path(p / "processed")
    da = setup_for_clustering(cluster_path / FileNames.zarr)
    labels_da = xr.open_dataarray(
        GModelNames.make_cluster_name(cluster_path, wind_sector, n_clusters)
    )

    ranked_qdim = da.sel({dn.wind_sector: wind_sector})
    case_order = ranked_qdim[dn.case_name]
    labels = labels_da.sel({dn.case_name: case_order}).values
    qdim_median = ranked_qdim.median(dim=dn.space_ix).values
    qdim_slope = (
        ranked_qdim.isel({dn.space_ix: 0}) - ranked_qdim.isel({dn.space_ix: -1})
    ).values

    viz = VisualizeClusters(labels, qdim_median, qdim_slope, ranked_qdim.values)
    save_mpl_fig(viz.make_qdim_distribution_plot(), clusters_dir / "qdim_by_class.png")
    save_mpl_fig(
        viz.make_qdim_slope_distribution_plot(),
        clusters_dir / "qdim_slope_by_class.png",
    )
    save_mpl_fig(viz.make_level_evenness_plot(), clusters_dir / "level_evenness.png")
    save_mpl_fig(viz.make_rank_profile_plot(), clusters_dir / "rank_profile.png")
    plt.show()


@gmod.command()
def ff(
    wind_sector: str = "N",
    n_clusters: int = 2,
    batch_size: int = 8,
    hidden_channels: int = 16,
    lr: float = 1e-3,
    cross_validation: bool = False,
    n_folds: int = 5,
    epochs: int = 100,
    show_live: bool = True,
    seed: int = 1204,
):
    cfg = CONFIGS_DICT["msd_aug20"]
    p = StudyPaths.data.gnn.aug20
    gnn_dir = p / "figures" / "gnn" / wind_sector / str(n_clusters)

    model_params = ClusterModelParams(wind_sector, n_clusters)
    tag = model_params.tag
    ds = load_data(cfg, p, model_params, seed)
    ic(ds.num_classes)

    graph_params = GraphModelParams.make(hidden_channels, ds)

    if cross_validation:
        fold_scores, fold_watchers, cv_confusion = cross_validate(
            ds,
            graph_params,
            batch_size,
            p,
            tag,
            n_folds,
            epochs,
            lr=lr,
            show_live=show_live,
            seed=seed,
        )
        report = CrossValidationReport(fold_scores)
        report.report()
        fig = report.make_plot()
        save_mpl_fig(fig, gnn_dir / "cross_validation.png")

        convergence_fig = VisualizeCrossValidationConvergence(fold_watchers).make_plot()
        save_mpl_fig(convergence_fig, gnn_dir / "cross_validation_convergence.png")

        confusion_fig = VisualizeConfusion(cv_confusion).make_plot()
        save_mpl_fig(confusion_fig, gnn_dir / "confusion.png")
    else:
        sdl = split_data(ds, batch_size, seed)
        train_labels = [
            int(sdl.train.dataset[i].y) for i in range(len(sdl.train.dataset))
        ]
        class_weights = compute_class_weights(train_labels, ds.num_classes)

        model_and_details = init_model(graph_params, class_weights, lr)
        watcher = TrainingWatcher(
            save_path=gnn_dir / "convergence.png",
            show_live=show_live,
        )
        train_model(
            sdl.train,
            model_and_details,
            p,
            tag,
            epochs,
            val_loader=sdl.val,
            watcher=watcher,
        )
        evaluate(sdl.test, model_and_details.model, label="Final model TEST")
        _, best_test_metrics = evaluate_best_model(
            p, graph_params, tag, sdl.val, sdl.test
        )

        fig = VisualizeTraining(watcher).make_plot()
        save_mpl_fig(fig, gnn_dir / "convergence.png")

        confusion_fig = VisualizeConfusion(best_test_metrics.confusion).make_plot()
        save_mpl_fig(confusion_fig, gnn_dir / "confusion.png")

    plt.show()
    return fig
