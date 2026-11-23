import xarray as xr
from plyze.flow_graph.interfaces import Edge, FlowGraph, ZoneNode

from nvml import dn
from nvml.df_names import dfc


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
        {dn.edge_num: edge_nums, dn.is_external: (dn.edge_num, is_ext)}
    )

    return da


def calc_intake(da: xr.DataArray):
    da1 = da.sel({dn.is_external: 1}).sum(dim=dn.edge_num)
    return da1


def calc_interzonal_ratio(da: xr.DataArray):
    intake = calc_intake(da)
    internal = da.sel({dn.is_external: 0}).sum(dim=dn.edge_num)
    return internal / (intake + internal)


def calc_zone_inflow_across_plan(G: FlowGraph, U: xr.DataArray):
    def calc(zone: ZoneNode):
        # get all edges
        edge_names = G.get_edges_of_zone(zone)
        edges = [
            i
            for i in G.edges_with_data
            if i.as_tuple in edge_names or i.as_tuple_reverse in edge_names
        ]
        edge_qdims = [calc_qdim_edge(i, U_1) for i in edges]
        edge_nums = list(range(len(edge_qdims)))

        qdim_da = xr.concat(edge_qdims, dim=dn.edge_num).assign_coords(
            {dn.edge_num: edge_nums}
        )

        net_inflow_per_area = qdim_da.sum(dim=dn.edge_num) / zone.data.area
        return net_inflow_per_area

    U_1 = U.where(
        cond=U >= 1, drop=True
    )  # NOTE: dropping U < 1 m/s -> can do some quantile analysis later, need to put in config
    inflows = [calc(i) for i in G.zone_nodes]
    zone_ix = list(range(len(inflows)))
    plan_da = xr.concat(inflows, dim=dn.space_ix).assign_coords({dn.space_ix: zone_ix})
    return plan_da


def calc_uniformity_index(G: FlowGraph, U: xr.DataArray):
    da = calc_zone_inflow_across_plan(G, U)

    p75 = da.quantile(0.75, dim=dn.space_ix)
    p25 = da.quantile(0.25, dim=dn.space_ix)
    variation = (p75 - p25) / (p75 + p25)
    uniformity = 1 - variation
    return uniformity


def calc_flow_qois(
    graph: FlowGraph,
    ambient_ds: xr.Dataset,
):
    U = ambient_ds[dn.wind_speed]
    da = plan_qdim(graph, U)

    vals = {
        dfc.intake: calc_intake(da),
        dfc.iz_ratio: calc_interzonal_ratio(da),
        dfc.ui: calc_uniformity_index(graph, U),
    }
    return xr.Dataset(
        vals
    )  # NEED TO CHECK ALL PLACES THIS IS CALLED FOR DN REF.. OR JUST BREAK
