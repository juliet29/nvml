import hashlib
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

import torch
from loguru import logger
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
from tqdm import tqdm

from nvml.constants import DataNames
from nvml.explain.interfaces import RawExplanation
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.hetero import CARDINAL_DIRECTIONS, EXTERNAL, NODE_TYPES, ZONE
from nvml.gmodel.model import HeterogGCN
from nvml.utils import default_worker_count

DEFAULT_EPOCHS = 200


def build_explainer(model: HeterogGCN, epochs: int) -> Explainer:
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )


def explain_instance(
    explainer: Explainer, model: HeterogGCN, data: HeteroData, seed: int = 1204
) -> RawExplanation:
    case_name = getattr(data, DataNames.case_name)
    case_digest = int(hashlib.sha1(case_name.encode()).hexdigest(), 16)
    torch.manual_seed(seed + case_digest % 1_000_000)

    batch_dict = {
        node_type: torch.zeros(data[node_type].num_nodes, dtype=torch.long)
        for node_type in NODE_TYPES
    }

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict, batch_dict)
    predicted_class = int(logits.argmax(dim=1))

    explanation = explainer(
        data.x_dict, data.edge_index_dict, batch_dict=batch_dict, index=0
    )

    direction_index = data[EXTERNAL].x.argmax(dim=1).tolist()
    external_directions = [CARDINAL_DIRECTIONS[index] for index in direction_index]

    relation_edges: dict = {}
    relation_edge_importance: dict = {}

    def record_relation(relation, mask):
        relation_edges[relation] = [
            (int(u), int(v)) for u, v in data[relation].edge_index.t().tolist()
        ]
        relation_edge_importance[relation] = mask.tolist()

    for relation, mask in explanation.edge_mask_dict.items():
        record_relation(relation, mask)

    return RawExplanation(
        case_name=case_name,
        predicted_class=predicted_class,
        zone_areas=data[ZONE].x.squeeze(-1).tolist(),
        zone_importance=explanation[ZONE].node_mask.squeeze(-1).tolist(),
        external_directions=external_directions,
        external_importance=explanation[EXTERNAL].node_mask.squeeze(-1).tolist(),
        relation_edges=relation_edges,
        relation_edge_importance=relation_edge_importance,
    )


_worker_state: dict = {}


def init_explain_worker(model: HeterogGCN, epochs: int, seed: int):
    torch.set_num_threads(1)
    model.eval()
    _worker_state["model"] = model
    _worker_state["explainer"] = build_explainer(model, epochs)
    _worker_state["seed"] = seed


def explain_worker(data: HeteroData) -> RawExplanation:
    return explain_instance(
        _worker_state["explainer"], _worker_state["model"], data, _worker_state["seed"]
    )


def explain_dataset(
    model: HeterogGCN,
    dataset: FlowGraphDataset,
    indices: list[int],
    epochs: int = DEFAULT_EPOCHS,
    workers: int | None = None,
    seed: int = 1204,
) -> list[RawExplanation]:
    model.eval()
    data_list = [dataset[index] for index in indices]
    max_workers = workers if workers is not None else default_worker_count()

    if max_workers == 1:
        explainer = build_explainer(model, epochs)
        return [
            explain_instance(explainer, model, data, seed)
            for data in tqdm(data_list, desc="explaining")
        ]

    explanations: list[RawExplanation] = []
    failures: list[str] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_explain_worker,
        initargs=(model, epochs, seed),
    ) as executor:
        data_by_future = {
            executor.submit(explain_worker, data): getattr(data, DataNames.case_name)
            for data in data_list
        }

        def record(completed: Future[RawExplanation]):
            case_name = data_by_future[completed]
            try:
                explanations.append(completed.result())
            except Exception as e:
                failures.append(f"{case_name}: {e}")

        for completed in tqdm(
            as_completed(data_by_future), total=len(data_by_future), desc="explaining"
        ):
            record(completed)

    if failures:
        logger.error(f"Failed to explain {len(failures)} cases: {failures}")
    return explanations
