from pathlib import Path
from typing import Any

import xarray as xr
from loguru import logger
from utils4plans.computation.loops2 import DatasetMaker

from nvml import dn
from nvml._study.config.config import MakeConfig
from nvml.flow_qoi.defns import calc_flow_qois
from nvml.io import get_ambient_ds_from_cfg, read_graph
from nvml.qdim.wind import wind_sector_as_str


class FlowQOIDatasetMaker(DatasetMaker):
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
        path = self.cfg.get_one_graph_path(ix)
        case_name = self.cfg.get_one_case(ix)
        try:
            flow_graph = read_graph(path)
            if not flow_graph:
                return None
            flow_qois = calc_flow_qois(flow_graph, self.ambient_ds)
            flow_qois[dn.label] = case_name
            return flow_qois
        except Exception as e:
            logger.critical(f"Issue proccessing {ix}:{case_name} -- {e}")

    def agglom_fx(self, datasets: list[xr.Dataset]):
        logger.info("Agglomerating data!")
        names = [str(i[dn.label]) for i in datasets]
        self.ds = (
            xr.concat(datasets, dim=dn.case_name)
            .assign_coords({dn.case_name: names})
            .pipe(wind_sector_as_str)
        )
        return self.ds

    def save_fx(self, ds: xr.Dataset):
        self.ds.to_netcdf(self.save_path)

        logger.success("Saved graphs!")
