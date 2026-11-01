from pathlib import Path
from typing import NamedTuple

import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
from plan2eplus.ezcase.ez import EZ
from plan2eplus.ops.zones.ezobject import WallNormal
from plyze.metrics.registries import FlowMetricRegistry as fmr
from plyze.metrics.registries import PlanMetricRegistry as pmr
from tqdm import tqdm
from utils4plans.io import make_dir
from utils4plans.lists import filter_none

from nvml._study.config.config import MakeConfig
from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.features.area import calc_flow_data_metrics, calc_plan_area_metrics
from nvml.features.coinc import COINCIDENCE_POLARS_ENUM, get_coinc
from nvml.features.qoi import calc_qdim_quantity
from nvml.io import get_ambient_data_as_ds, read_graph
from nvml.qdim.wind import WindDirectionBinNames, add_wind_sector_coord

direction_map: dict[WindDirectionBinNames, WallNormal] = {
    "N": WallNormal.NORTH,
    "E": WallNormal.EAST,
    "S": WallNormal.SOUTH,
    "W": WallNormal.WEST,
}


def get_ambient_ds(ix: int, cfg: MakeConfig):
    res = get_ambient_data_as_ds(cfg.get_one_case_data(ix).sql)
    res = add_wind_sector_coord(res)
    return res


class TableEntry(NamedTuple):
    case_name: str
    space_mean_time_mean: float
    space_sum_time_mean: float
    coincidence: str
    area: float
    num_rooms: int
    num_paths: int
    avg_path_length: float
    mode_path_length: int
    len_longest_path: int
    len_shortest_path: int
    degree_dom_node: int


class dfc:
    case_name = "case_name"
    space_mean_time_mean = "space_mean_time_mean"
    space_sum_time_mean = "space_sum_time_mean"
    coincidence = "coincidence"
    area = "area"

    # --- added on later
    ww = "ww"
    lw = "lw"
    sd = "sd"


class DataMaker(NamedTuple):
    ambient_ds: xr.Dataset
    direction: WindDirectionBinNames
    cfg: MakeConfig
    case_ix: int

    @property
    def wall_normal(self) -> WallNormal:
        wall_normal = direction_map[self.direction]
        assert wall_normal is not None, (
            f"{self.direction} needs to be a cardinal direction"
        )
        return wall_normal

    @property
    def graph(self):
        graph_path = self.cfg.get_one_graph_path(self.case_ix)
        return read_graph(graph_path)

    @property
    def case_name(self):
        return self.cfg.get_one_case(self.case_ix)

    def get_target(self):
        da = calc_qdim_quantity(self.graph, self.ambient_ds, self.direction)
        space_mean_time_mean = (
            da.mean(dim=dn.space_name).mean(dim=dn.wind_sector).values
        )
        space_sum_time_mean = da.sum(dim=dn.space_name).mean(dim=dn.wind_sector).values
        return float(space_mean_time_mean), float(space_sum_time_mean)

    def get_features(self):
        case_data = self.cfg.get_one_case_data(self.case_ix)
        case = EZ(case_data.idf)
        coincidence = get_coinc(self.graph, case, self.wall_normal)
        plan_metrics = calc_plan_area_metrics(self.graph)
        flow_metrics = calc_flow_data_metrics(self.graph, self.wall_normal)
        return coincidence, plan_metrics, flow_metrics

    def make_table_entry(self):
        coincidence, plan, flow = self.get_features()

        return TableEntry(
            self.case_name,
            *self.get_target(),
            coincidence,
            area=plan[pmr.area.name],
            num_rooms=plan[pmr.num_rooms.name],  # pyright: ignore[reportArgumentType]
            avg_path_length=flow[fmr.avg_path_length.name],
            num_paths=flow[fmr.num_paths.name],  # pyright: ignore[reportArgumentType]
            mode_path_length=flow[fmr.mode_path_length.name],  # pyright: ignore[reportArgumentType]
            len_longest_path=flow[fmr.len_longest_path.name],  # pyright: ignore[reportArgumentType]
            len_shortest_path=flow[fmr.len_shortest_path.name],  # pyright: ignore[reportArgumentType]
            degree_dom_node=flow[fmr.degree_dom_node.name],  # pyright: ignore[reportArgumentType]
        )


def save_data(
    cfg: MakeConfig,
    direction: WindDirectionBinNames,
    path: Path,
    true_run: bool = False,
    save: bool = False,
    njobs: int = 3,
):
    path = path / FileNames.regression_data
    make_dir(path)
    if save:
        logger.info(f"Will save to {path}")
    else:
        logger.info(f"Not planning to save this, but the passed path is {path}")

    ambient_ds = get_ambient_ds(0, cfg)

    if true_run:
        rng = range(len(cfg))
    else:
        rng = range(len(cfg))[21:26]

    def process(ix: int):
        try:
            dm = DataMaker(ambient_ds, direction, cfg, ix)
            entry = dm.make_table_entry()
        except Exception as e:
            # TODO: tdqm print!
            print(f"Issue for {dm.case_name}: {e}")
            return None

        return entry

    entries = Parallel(n_jobs=njobs)(
        delayed(process)(ix) for ix in tqdm(rng, "Processing data")
    )
    valid_entries = filter_none(entries)

    df = pl.DataFrame(valid_entries).with_columns(
        pl.col(dfc.coincidence).cast(COINCIDENCE_POLARS_ENUM)
    )

    if save:
        df.write_csv(path)

    return df
