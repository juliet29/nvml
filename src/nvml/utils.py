from pathlib import Path

import hvplot
from holoviews.core import Dimensioned
from matplotlib.figure import Figure


def make_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_mpl_fig(fig: Figure, path: Path, dpi: int = 300, **kwargs) -> Path:
    make_dir(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    return path


def save_hvplot(plot: Dimensioned, path: Path, **kwargs) -> Path:
    make_dir(path)
    hvplot.save(plot, path, **kwargs)
    return path
