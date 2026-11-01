from pathlib import Path
from typing import NamedTuple

import polars as pl
import xarray as xr
from icecream import ic
from joblib import Parallel, delayed
from loguru import logger
from plan2eplus.ezcase.ez import EZ
from plan2eplus.ops.zones.ezobject import WallNormal
from tqdm import tqdm
from utils4plans.io import make_dir

from nvml._study.config.config import MakeConfig
from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.features.coinc import (
    calc_plan_coinc,
    tabulate_coinc_features,
)
from nvml.features.qoi import calc_qdim_quantity
from nvml.io import get_ambient_data_as_ds, read_graph
from nvml.qdim.wind import WindDirectionBinNames, add_wind_sector_coord

COINCIDENCE_POLARS_ENUM = pl.Enum(
    categories=["ww", "lw", "sd", "ww_lw", "ww_sd", "lw_sd", "ww_lw_sd"]
)


def get_ambient_ds(ix: int, cfg: MakeConfig):
    res = get_ambient_data_as_ds(cfg.get_one_case_data(ix).sql)
    res = add_wind_sector_coord(res)
    return res


direction_map: dict[WindDirectionBinNames, WallNormal] = {
    "N": WallNormal.NORTH,
    "E": WallNormal.EAST,
    "S": WallNormal.SOUTH,
    "W": WallNormal.WEST,
}


class TableEntry(NamedTuple):
    case_name: str
    space_mean_time_mean: float
    space_sum_time_mean: float
    feature: str


class DataMaker(NamedTuple):
    ambient_ds: xr.Dataset
    direction: WindDirectionBinNames
    cfg: MakeConfig
    case_ix: int

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
        wall_normal = direction_map[self.direction]

        assert wall_normal is not None, (
            f"{self.direction} needs to be a cardinal direction"
        )

        coinc_da = calc_plan_coinc(self.graph, case, wall_normal)
        coinc_feature = tabulate_coinc_features(coinc_da)
        return coinc_feature.name

    def make_table_entry(self):
        return TableEntry(
            self.case_name,
            *self.get_target(),
            self.get_features(),
        )


def save_data(cfg: MakeConfig, direction: WindDirectionBinNames, path: Path):
    path = path / FileNames.regression_data
    make_dir(path)
    logger.info(f"Will save to {path}")

    ambient_ds = get_ambient_ds(0, cfg)

    rng = range(len(cfg))[0:3]

    def process(ix: int):
        try:
            # with logger.contextualize(silence=True):
            dm = DataMaker(ambient_ds, direction, cfg, ix)
            entry = dm.make_table_entry()
        except Exception as e:
            print(f"Issue for {dm.case_name}: {e}")
            # logger.critical(f"Issue for {dm.case_name}: {e}")

        return entry

    entries = Parallel(n_jobs=3)(
        delayed(process)(ix) for ix in tqdm(rng, "Processing data")
    )

    df = pl.DataFrame(entries).with_columns(
        pl.col("feature").cast(COINCIDENCE_POLARS_ENUM)
    )
    ic(df)
    return df
    #
    # df.write_csv(path)
