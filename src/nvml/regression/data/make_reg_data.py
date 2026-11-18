from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from nvml.regression.data.interfaces import TableEntry, dfc
from nvml.regression.data.single import FeaturesMaker
from utils4plans.computation.loops2 import DatasetMaker

from nvml._study.config.config import MakeConfig
from nvml.io import get_ambient_ds_from_cfg


class RegDatasetMaker(DatasetMaker):
    def __init__(
        self,
        cfg: MakeConfig,
        dir_path: Path,
        file_name: str = "",
        true_run: bool = False,
        save: bool = False,
        njobs: int = 3,
        test_rng: tuple[int, int] = (0, 5),
    ) -> None:
        self.cfg = cfg
        super().__init__(cfg, dir_path, file_name, true_run, save, njobs, test_rng)
        self.ambient_ds = get_ambient_ds_from_cfg(self.cfg)

    def process_fx(self, obj: None | Any = None, ix: int = 0):
        try:
            dm = FeaturesMaker(self.ambient_ds, direction, self.cfg, ix)
            entry = dm.make_table_entry()
        except Exception as e:
            # TODO: tdqm print!
            print(f"Issue for {dm.case_name}: {e}")
            return None

        return entry

    def agglom_fx(self, entries: list[TableEntry]):
        logger.info("Agglomerating data!")

        self.df = pl.DataFrame(entries).with_columns(
            pl.col(dfc.coincidence).cast(COINCIDENCE_POLARS_ENUM)
        )
        return self.df

    def save_fx(self, df: pl.DataFrame):
        logger.success("Saved graphs!")
        df.write_csv(self.save_path)
