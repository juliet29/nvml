from pathlib import Path
from snakemake.utils import validate

validate(config, schema="../schemas/qdim.schema.yaml")

def get_graphs_data():
  loc = Path(config["pathvars"]["path_to_graphs"]) # assuming they are nested here in flat..
  case_names = [i.name for i in loc.iterdir()]
  return [{"case_name": case_name, "case_name_idx": idx} for idx, case_name in enumerate(case_names)] 

def make_graph_path():
  pass
gd = get_graphs_data()
print(gd)

rule qdim_create:
  input:
    zarr_path = config[savedir] / zarr_name
    graph_path = make_graph_path(wildcards.sample)
    ambient_ds_path = config[savedir] / ambient_name
  output:
    [] # TODO zarr output
  params:
    case_name_idx = 0 # TODO get from wild cards
    case_name = 0
  shell:
    """
    uv run nvmlsmk qdim create \
        --case-name-idx {params.case_name_idx}
        --case_name {params.case_name}
        --graph_path {input.graph_path}
        --ambient-ds-path {input.ambient-ds-path}

    """


