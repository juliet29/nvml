import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

import matplotlib.pyplot as plt
from cyclopts import App
from icecream import ic
from loguru import logger
from plan2eplus.ezcase.ez import EZ
from plan2eplus.visuals.simple_plots import make_base_plot

from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.cli.studies.study_paths import StudyPaths
from nvml.explain.aggregate import DEFAULT_MIN_SUPPORT, aggregate_explanations
from nvml.explain.explainer import explain_dataset
from nvml.explain.io import explanation_path, load_explanations, save_explanations
from nvml.explain.overlay import FloorPlanMotifOverlay, save_motif_overlays
from nvml.explain.profile import (
    build_cluster_signatures,
    extract_connectivity,
    save_cluster_contrast,
    save_signature_overlays,
)
from nvml.explain.report import report_top_motifs
from nvml.explain.subgraph import (
    DEFAULT_AREA_CUTS,
    DEFAULT_EDGE_QUANTILE,
    DEFAULT_MAX_EDGES,
    DEFAULT_MOTIF_DISTANCE,
    DEFAULT_WL_ITERATIONS,
    extract_instance,
)
from nvml.explain.viz import save_class_reports, save_motif_galleries
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model_interfaces import GraphModelParams
from nvml.gmodel.model_utils import load_best_model_state, load_data
from nvml.gmodel.processing import load_flow_graph

geexplain = App("gex")


@geexplain.command()
def explain(
    wind_sector: str = "N",
    n_clusters: int = 3,
    hidden_channels: int = 16,
    explainer_epochs: int = 200,
    edge_quantile: float = DEFAULT_EDGE_QUANTILE,
    max_edges: int = DEFAULT_MAX_EDGES,
    area_cuts: tuple[float, ...] = DEFAULT_AREA_CUTS,
    area_labels: tuple[str, ...] = (),
    wl_iterations: int = DEFAULT_WL_ITERATIONS,
    motif_distance: float = DEFAULT_MOTIF_DISTANCE,
    min_support: float = DEFAULT_MIN_SUPPORT,
    workers: int | None = None,
    seed: int = 1204,
):
    cfg = CONFIGS_DICT["msd_aug20"]
    p = StudyPaths.data.gnn.aug20
    figures = p / "figures"

    model_params = ClusterModelParams(wind_sector, n_clusters)
    tag = model_params.tag
    ds = load_data(cfg, p, model_params, seed)
    ic(ds.num_classes)

    graph_params = GraphModelParams.make(hidden_channels, ds)
    model = load_best_model_state(p, graph_params, tag)

    raw_explanations = explain_dataset(
        model, ds, list(range(len(ds))), explainer_epochs, workers, seed
    )
    saved_explanations = save_explanations(raw_explanations, explanation_path(p, tag))
    logger.info(f"Saved explanations to {saved_explanations}")

    cuts = tuple(cut for cut in area_cuts if cut > 0) or None
    labels = tuple(area_labels) or None
    instances = [
        extract_instance(raw, edge_quantile, cuts, wl_iterations, labels, max_edges)
        for raw in raw_explanations
    ]
    class_explanations = aggregate_explanations(
        instances, motif_distance, wl_iterations, min_support
    )
    report_top_motifs(class_explanations)
    model_label = f"wind {wind_sector}, k={n_clusters}"
    saved = save_class_reports(class_explanations, figures, model_label, tag)
    logger.info(f"Saved {len(saved)} explanation figures to {figures / 'explanation' / tag}")

    plt.show()


@geexplain.command()
def overlay(
    wind_sector: str = "N",
    n_clusters: int = 3,
    edge_quantile: float = DEFAULT_EDGE_QUANTILE,
    max_edges: int = DEFAULT_MAX_EDGES,
    area_cuts: tuple[float, ...] = DEFAULT_AREA_CUTS,
    area_labels: tuple[str, ...] = (),
    wl_iterations: int = DEFAULT_WL_ITERATIONS,
    motif_distance: float = DEFAULT_MOTIF_DISTANCE,
    min_support: float = DEFAULT_MIN_SUPPORT,
    top_k: int = 4,
):
    cfg = CONFIGS_DICT["msd_aug20"]
    p = StudyPaths.data.gnn.aug20
    figures = p / "figures"

    model_params = ClusterModelParams(wind_sector, n_clusters)
    tag = model_params.tag
    raw_explanations = load_explanations(explanation_path(p, tag))

    cuts = tuple(cut for cut in area_cuts if cut > 0) or None
    labels = tuple(area_labels) or None
    instances = [
        extract_instance(raw, edge_quantile, cuts, wl_iterations, labels, max_edges)
        for raw in raw_explanations
    ]
    class_explanations = aggregate_explanations(
        instances, motif_distance, wl_iterations, min_support
    )

    plan_overlay = FloorPlanMotifOverlay(
        load_graph=lambda name: load_flow_graph(cfg, name, True),
        make_plan_plot=lambda name: make_base_plot(
            EZ(cfg.make_case_data(name).idf), cfg.cardinal_expansion_factor
        ),
    )
    saved = save_motif_overlays(class_explanations, plan_overlay, figures, tag, top_k)
    logger.info(
        f"Saved {len(saved)} motif overlays to {figures / 'explanation' / tag / 'overlays'}"
    )

    plt.show()


@geexplain.command()
def contrast(
    wind_sector: str = "N",
    n_clusters: int = 3,
    area_cuts: tuple[float, ...] = DEFAULT_AREA_CUTS,
    area_labels: tuple[str, ...] = (),
):
    cfg = CONFIGS_DICT["msd_aug20"]
    p = StudyPaths.data.gnn.aug20
    figures = p / "figures"

    model_params = ClusterModelParams(wind_sector, n_clusters)
    tag = model_params.tag
    raw_explanations = load_explanations(explanation_path(p, tag))

    cuts = tuple(cut for cut in area_cuts if cut > 0) or None
    labels = tuple(area_labels) or None
    profiles = [extract_connectivity(raw, cuts, labels) for raw in raw_explanations]
    signatures = build_cluster_signatures(profiles, cuts, labels)

    model_label = f"wind {wind_sector}, k={n_clusters}"
    contrast_path = save_cluster_contrast(signatures, figures, model_label, tag)
    logger.info(f"Saved cluster contrast to {contrast_path}")

    plan_overlay = FloorPlanMotifOverlay(
        load_graph=lambda name: load_flow_graph(cfg, name, True),
        make_plan_plot=lambda name: make_base_plot(
            EZ(cfg.make_case_data(name).idf), cfg.cardinal_expansion_factor
        ),
    )
    saved = save_signature_overlays(signatures, plan_overlay, figures, tag)
    logger.info(
        f"Saved {len(saved)} cluster overlays to {figures / 'explanation' / tag / 'cluster_contrast'}"
    )

    plt.show()


@geexplain.command()
def plot(
    wind_sector: str = "N",
    n_clusters: int = 3,
    edge_quantile: float = DEFAULT_EDGE_QUANTILE,
    max_edges: int = DEFAULT_MAX_EDGES,
    area_cuts: tuple[float, ...] = DEFAULT_AREA_CUTS,
    area_labels: tuple[str, ...] = (),
    wl_iterations: int = DEFAULT_WL_ITERATIONS,
    motif_distance: float = DEFAULT_MOTIF_DISTANCE,
    min_support: float = DEFAULT_MIN_SUPPORT,
):
    p = StudyPaths.data.gnn.aug20
    figures = p / "figures"

    model_params = ClusterModelParams(wind_sector, n_clusters)
    tag = model_params.tag
    raw_explanations = load_explanations(explanation_path(p, tag))

    cuts = tuple(cut for cut in area_cuts if cut > 0) or None
    labels = tuple(area_labels) or None
    instances = [
        extract_instance(raw, edge_quantile, cuts, wl_iterations, labels, max_edges)
        for raw in raw_explanations
    ]
    class_explanations = aggregate_explanations(
        instances, motif_distance, wl_iterations, min_support
    )
    report_top_motifs(class_explanations)
    model_label = f"wind {wind_sector}, k={n_clusters}"
    saved = save_class_reports(class_explanations, figures, model_label, tag)
    logger.info(f"Saved {len(saved)} explanation figures to {figures / 'explanation' / tag}")

    plt.show()


@geexplain.command()
def sweep(
    wind_sector: str = "N",
    n_clusters: int = 3,
    edge_quantiles: tuple[float, ...] = (0.5, 0.6, 0.7),
    max_edges_values: tuple[int, ...] = (4, 6, 8),
    motif_distances: tuple[float, ...] = (0.3, 0.4, 0.5),
    wl_iterations: int = DEFAULT_WL_ITERATIONS,
):
    p = StudyPaths.data.gnn.aug20
    figures = p / "figures"

    model_params = ClusterModelParams(wind_sector, n_clusters)
    tag = model_params.tag
    raw_explanations = load_explanations(explanation_path(p, tag))
    model_label = f"wind {wind_sector}, k={n_clusters}"
    tests_dir = figures / "explanation" / tag / "tests"

    combos = [
        (edge_quantile, max_edges, motif_distance)
        for edge_quantile in edge_quantiles
        for max_edges in max_edges_values
        for motif_distance in motif_distances
    ]
    logger.info(f"Sweeping {len(combos)} combos (area off) into {tests_dir}")

    saved_total = 0

    def run_combo(edge_quantile: float, max_edges: int, motif_distance: float):
        nonlocal saved_total
        instances = [
            extract_instance(raw, edge_quantile, None, wl_iterations, None, max_edges)
            for raw in raw_explanations
        ]
        class_explanations = aggregate_explanations(instances, motif_distance, wl_iterations)
        name = f"eq{edge_quantile}_me{max_edges}_md{motif_distance}"
        saved = save_motif_galleries(
            class_explanations, tests_dir / name, f"{model_label} — {name}"
        )
        saved_total += len(saved)
        logger.info(f"  {name}: {len(saved)} galleries")

    for edge_quantile, max_edges, motif_distance in combos:
        run_combo(edge_quantile, max_edges, motif_distance)
    logger.info(f"Saved {saved_total} galleries across {len(combos)} tests to {tests_dir}")
