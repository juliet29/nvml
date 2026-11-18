import xarray as xr
from plyze.flow_graph.interfaces import Edge, FlowGraph

from nvml import dn


# class FlowQOI(NamedTuple):
#     intake: xr.DataArray
#     dist_ratio: xr.DataArray
#     coeff_var: xr.DataArray
def calc_qdim_edge(edge: Edge, U: xr.DataArray):
    ed = edge.data
    f_in = ed.flow_in.sel({dn.datetime: U[dn.datetime]}) / (ed.surface_area * U)
    return f_in


def plan_qdim(graph: FlowGraph, U: xr.DataArray):
    U_1 = U.where(
        cond=U >= 1, drop=True
    )  # NOTE: dropping U < 1 m/s -> can do some quantile analysis later

    flows = [calc_qdim_edge(edge, U_1) for edge in graph.edges_with_data]
    is_ext = [1 if i.is_external else 0 for i in graph.edges_with_data]
    edge_nums = list(range(len(flows)))
    da = xr.concat(flows, dim=dn.edge_num).assign_coords(
        {dn.edge_num: edge_nums, dn.is_external: ("edge_num", is_ext)}
    )

    return da


def calc_intake(da: xr.DataArray):
    da1 = da.sel({dn.is_external: 1}).sum(dim=dn.edge_num)
    return da1


def calc_distribution_ratio(da: xr.DataArray):
    intake = calc_intake(da)
    internal = da.sel({dn.is_external: 0}).sum(dim=dn.edge_num)
    return internal / intake


def calc_coeff_var(da: xr.DataArray):
    mu = da.mean(dim=dn.edge_num)
    std = da.std(dim=dn.edge_num)
    return mu / std


def calc_flow_qois(
    graph: FlowGraph,
    ambient_ds: xr.Dataset,
):
    U = ambient_ds[dn.wind_speed]
    da = plan_qdim(graph, U)

    vals = {
        dn.intake: calc_intake(da),
        dn.dist: calc_distribution_ratio(da),
        dn.cv: calc_coeff_var(da),
    }
    return xr.Dataset(vals)
