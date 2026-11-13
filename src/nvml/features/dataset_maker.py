from typing import NamedTuple

import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from nvml._study.config.config import MakeConfig
from nvml.constants import DataNames as dn
from nvml.constants import FileNames
from nvml.features.data_maker import handle_path
from nvml.features.qoi import (
    add_ambient_data_to_da,
    add_temp_ambient_data_to_da,
    gather_qdim,
    gather_temp,
)
from nvml.io import get_ambient_ds_from_cfg, read_graph


# Should go up to make config as eaier access thing..
class DataMaker(NamedTuple):
    cfg: MakeConfig
    case_ix: int

    @property
    def graph(self):
        graph_path = self.cfg.get_one_graph_path(self.case_ix)
        return read_graph(graph_path)

    @property
    def case_name(self):
        return self.cfg.get_one_case(self.case_ix)


def process(cfg: MakeConfig, ix: int):
    try:
        dm = DataMaker(cfg, ix)
        da = gather_qdim(dm.graph)
        da.name = dm.case_name

    except Exception as e:
        # TODO: tdqm print!
        print(f"Issue for {dm.case_name}: {e}")
        return None

    return da


def make_dataset(cfg, dir_path, true_run, save, njobs):
    dir_path = handle_path(dir_path, FileNames.qoi_dataArray, save=save)
    pass

    if true_run:
        rng = range(len(cfg))
    else:
        rng = range(len(cfg))[0:20]

    # arrs = [process(cfg, i) for i in rng]

    arrs = Parallel(n_jobs=njobs)(
        delayed(process)(cfg, ix) for ix in tqdm(rng, "Processing data")
    )

    valid_arrs = [i for i in arrs if i is not None]

    # dataset to array to dataset..
    case_names = [i.name for i in valid_arrs]
    concat_arr = xr.concat(valid_arrs, dim=dn.case_name, join="outer").assign_coords(
        {dn.case_name: case_names}
    )
    concat_arr.name = dn.zone_dimless_flow
    ambient_ds = get_ambient_ds_from_cfg(cfg, 0)
    out = add_ambient_data_to_da(concat_arr, ambient_ds)

    if save:
        out[dn.wind_sector] = out[dn.wind_sector].astype(str)
        out.to_netcdf(dir_path)

    return out


def process_tdim(cfg: MakeConfig, ix: int):
    try:
        dm = DataMaker(cfg, ix)
        da = gather_temp(dm.graph)
        da.name = dm.case_name

    except Exception as e:
        # TODO: tdqm print!
        print(f"Issue for {dm.case_name}: {e}")
        return None

    return da


def make_tdim_dataset(cfg, dir_path, true_run, save, njobs):
    dir_path = handle_path(dir_path, FileNames.tdim_zone_dataArray, save=save)
    pass

    if true_run:
        rng = range(len(cfg))
    else:
        rng = range(len(cfg))[0:5]

    # arrs = [process(cfg, i) for i in rng]

    arrs = Parallel(n_jobs=njobs)(
        delayed(process_tdim)(cfg, ix) for ix in tqdm(rng, "Processing temp data")
    )

    valid_arrs = [i for i in arrs if i is not None]

    # dataset to array to dataset..
    case_names = [i.name for i in valid_arrs]
    concat_arr = xr.concat(valid_arrs, dim=dn.case_name, join="outer").assign_coords(
        {dn.case_name: case_names}
    )
    ambient_ds = get_ambient_ds_from_cfg(cfg, 0)
    da = add_temp_ambient_data_to_da(concat_arr, ambient_ds)

    # now make it dimensionless
    # da_space_average = da.mean(dim=dn.space_ix)
    da_dim = (da - da[dn.t_out]) / da[dn.t_out]
    # print(da_dim)
    # breakpoint()

    da_dim.name = dn.zone_dimless_temp

    if save:
        da_dim.to_netcdf(dir_path)

    return da_dim
