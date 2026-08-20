import matplotlib

matplotlib.use("module://matplotlib-backend-kitty")

import matplotlib.pyplot as plt
from cyclopts import App
from icecream import ic
from loguru import logger

from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.cli.studies.study_paths import StudyPaths
from nvml.explain.aggregate import aggregate_explanations
from nvml.explain.explainer import explain_dataset
from nvml.explain.io import explanation_path, load_explanations, save_explanations
from nvml.explain.subgraph import DEFAULT_EDGE_QUANTILE, extract_instance
from nvml.explain.viz import save_class_reports
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model_interfaces import GraphModelParams
from nvml.gmodel.model_utils import load_best_model_state, load_data

geexplain = App("geexplain")


@geexplain.command()
def explain(
    wind_sector: str = "N",
    n_clusters: int = 2,
    hidden_channels: int = 16,
    explainer_epochs: int = 200,
    edge_quantile: float = DEFAULT_EDGE_QUANTILE,
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

    instances = [extract_instance(raw, edge_quantile) for raw in raw_explanations]
    class_explanations = aggregate_explanations(instances)
    model_label = f"wind {wind_sector}, k={n_clusters}"
    saved = save_class_reports(class_explanations, figures, model_label, tag)
    for path in saved:
        logger.info(f"Saved explanation figure to {path}")

    plt.show()
    return class_explanations


@geexplain.command()
def plot(
    wind_sector: str = "N",
    n_clusters: int = 2,
    edge_quantile: float = DEFAULT_EDGE_QUANTILE,
):
    p = StudyPaths.data.gnn.aug20
    figures = p / "figures"

    model_params = ClusterModelParams(wind_sector, n_clusters)
    tag = model_params.tag
    raw_explanations = load_explanations(explanation_path(p, tag))

    instances = [extract_instance(raw, edge_quantile) for raw in raw_explanations]
    class_explanations = aggregate_explanations(instances)
    model_label = f"wind {wind_sector}, k={n_clusters}"
    saved = save_class_reports(class_explanations, figures, model_label, tag)
    for path in saved:
        logger.info(f"Saved explanation figure to {path}")

    plt.show()
    return class_explanations
