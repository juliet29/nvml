from cyclopts import App
from icecream import ic

from nvml.cli.config.cstore import CONFIGS_DICT
from nvml.cli.studies.helpers import cfg
from nvml.cli.studies.study_paths import StudyPaths
from nvml.gmodel.dataset import FlowGraphDataset
from nvml.gmodel.dataset_interfaces import ClusterModelParams
from nvml.gmodel.model_interfaces import GraphModelParams
from nvml.gmodel.model_utils import init_model, load_data, split_data, train_model

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
def ff():
    cfg = CONFIGS_DICT["msd_aug20"]
    p = StudyPaths.data.gnn.aug20

    bsize = 8
    n_train = bsize * 4
    hidden_channels = 16

    ds = load_data(cfg, p, ClusterModelParams("N", 2))
    ic(ds.num_node_features)

    graph_params = GraphModelParams.make(hidden_channels, ds)

    sdl = split_data(ds, bsize, n_train)
    model_and_details = init_model(graph_params)
    train_model(sdl.train, model_and_details, p)
