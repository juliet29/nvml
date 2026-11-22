from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from plan2eplus.ezcase.ez import EZ
from utils4plans.computation.loops2 import DatasetMaker
from utils4plans.io import make_dir

from nvml._study.config.config import MakeConfig
from nvml.io import get_ambient_ds_from_cfg, read_graph
from nvml.fexp.make_fexp.gather import gather_constant_data, gather_wind_dependent_data


class RegDatasetMaker(DatasetMaker):
    def __init__(
        self,
        cfg: MakeConfig,
        wdep_path: Path,
        const_path: Path,
        true_run: bool = False,
        save: bool = False,
        njobs: int = 3,
        test_rng: tuple[int, int] = (0, 5),
    ) -> None:
        self.cfg = cfg
        file_name = ""
        super().__init__(cfg, wdep_path, file_name, true_run, save, njobs, test_rng)
        self.ambient_ds = get_ambient_ds_from_cfg(self.cfg)

        self.wdep_path = wdep_path

        make_dir(const_path)
        self.const_path = const_path

    def process_fx(self, obj: None | Any = None, ix: int = 0):
        try:
            case_name = self.cfg.get_one_case(ix)
            G = read_graph(self.cfg.get_one_graph_path(ix))
            if not G:
                logger.critical(f"No graph for {case_name}, skipping..")
                return None

            ez_case = EZ(self.cfg.get_one_case_data(ix).idf)

            wdf = gather_wind_dependent_data(ez_case, G, case_name)
            cd = gather_constant_data(G, case_name)

        except Exception as e:
            logger.critical(f"Issue for {case_name}: {e}")
            return None

        return (wdf, cd)

    def agglom_fx(self, entries: list[tuple[pl.DataFrame, dict]]):
        logger.info("Agglomerating data!")
        wdfs = [i[0] for i in entries]
        cds = [i[1] for i in entries]

        wind_dep_df = pl.concat(wdfs)
        constant_df = pl.DataFrame(cds)
        return (wind_dep_df, constant_df)

    def save_fx(self, df_pair: tuple[pl.DataFrame, pl.DataFrame]):
        logger.info("Starting to save")
        wdf, cd = df_pair
        wdf.write_parquet(self.wdep_path)
        cd.write_parquet(self.const_path)

        logger.success("Saved regression data!")
