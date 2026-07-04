from pathlib import Path

configfile: "smkconfig/qdim/01.yaml"
validate(config, schema="workflow/schemas/qdim.schema.yaml")

def get_graphs_data():
  loc = Path(config["pathvars"]["path_to_graphs"]) # assuming they are nested here in flat..
  case_names = [i.name for i in loc.iterdir()]
  return [{"case_name": i, "case_name_idx": idx} for idx, case_name in enumerate(case_names)] 

def make_graph_path():
  pass

rule qdim_create:
  input:
    ""
  output:
    ""
  shell:
    """
    echo hello

    """


