from pathlib import Path

import numpy as np
import xarray as xr

from nvml.constants import DataNames as dn
from nvml.io import graph_to_ds


# TODO: think about if there are ways to parralelize graph reading..
def make_space_name_by_wind_sector_da(
    case_name: str, graph_path: Path, ambient_ds: xr.Dataset
):

    qoi_ds = graph_to_ds(graph_path)

    # TODO: move this check higher, but since q_dim diveides by velocity, we get infinity whenever U = 0
    q_dim = qoi_ds[dn.zone_dimless_flow]

    space_ixes = np.arange(qoi_ds[dn.space_name].size)
    qds = (
        q_dim.where(np.isfinite(q_dim), other=0)
        .assign_coords({dn.space_ix: (dn.space_name, space_ixes)})
        .swap_dims(
            {dn.space_name: dn.space_ix}
        )  # make space_ix the dimension and drop space names
        .drop_vars(dn.space_name)
    )

    qdw = (
        qds.assign_coords(
            {dn.wind_sector: (dn.datetime, ambient_ds[dn.wind_sector].data)}
        )
        .swap_dims({dn.datetime: dn.wind_sector})
        .drop_vars(dn.datetime)
    )

    # Aggregate over space names and wind sectors => really group by wind sectors while preserving the space_ix axis..
    qa_flat = qdw.groupby([dn.wind_sector, dn.space_ix])  # .sum(dn.wind_sector)
    qa_final = qa_flat.median(skipna=True).assign_attrs({dn.case_name: case_name})

    qa_final.name = dn.q_dim_median

    return qa_final
