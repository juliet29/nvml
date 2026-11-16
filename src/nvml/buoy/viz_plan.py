from plan2eplus.ezcase.ez import EZ
from plan2eplus.visuals.simple_plots import make_base_plot

from nvml._study.config.config import MakeConfig


def plot_one(cfg: MakeConfig, case_name: str):
    cd = cfg.make_case_data(case_name)
    case = EZ(cd.idf)
    bp = make_base_plot(case)
    bp.finalize()
    bp.axes.set_title(case_name)
    return bp.fig
