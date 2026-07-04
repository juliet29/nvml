from pathlib import Path
from snakemake.utils import validate

validate(config, schema="../schemas/qdim.schema.yaml")

def get_case_names():
  loc = Path(config["pathvars"]["path_to_graphs"])
  return sorted(p.name for p in loc.iterdir() if p.is_dir())

def make_graph_path(wildcards):
  return Path(config["pathvars"]["path_to_graphs"]) / wildcards.case_name / config["graph_name"]

CASE_NAMES = get_case_names()

SAVEDIR = Path(config["pathvars"]["savedir"])
ZARR_PATH = SAVEDIR / config["zarr_name"]
AMBIENT_PATH = SAVEDIR / config["ambient_name"]
SENTINEL = str(SAVEDIR / ".done" / "{case_name}.done")

rule qdim_create_target:
  input:
    expand(SENTINEL, case_name=CASE_NAMES),

rule qdim_init:
  input:
    sql = config["pathvars"]["path_to_sql"],
  output:
    zarr = directory(ZARR_PATH),
    ambient = AMBIENT_PATH,
  params:
    savedir = SAVEDIR,
    case_names = " ".join(CASE_NAMES),
  shell:
    """
    uv run nvmlsmk qdim init {params.savedir} {input.sql} {params.case_names}
    """

rule qdim_create:
  input:
    zarr_path = ZARR_PATH,
    ambient_ds_path = AMBIENT_PATH,
    graph_path = make_graph_path,
  output:
    sentinel = touch(SENTINEL),
  shell:
    """
    uv run nvmlsmk qdim create \
        --case-name {wildcards.case_name} \
        --zarr-path {input.zarr_path} \
        --graph-path {input.graph_path} \
        --ambient-ds-path {input.ambient_ds_path}
    """
