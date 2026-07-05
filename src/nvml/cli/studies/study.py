import matplotlib.pyplot as plt
from cyclopts import App
from icecream import ic, install
from loguru import logger

# from plyze.plots.altair_helpers import AltairRenderers
# from plyze.plots.theme import default_theme
from rich.pretty import pretty_repr
from utils4plans.logconfig import logset

from nvml.cli.studies.cluster import cluster
from nvml.cli.studies.create import create
from nvml.cli.studies.flow import flow
from nvml.cli.studies.qdim import qdim

app = App()
app.command(qdim)
app.command(create)
app.command(flow)
app.command(cluster)

install()  # allow icecream to be used everywhere


def keep():
    # default_theme()
    logger.debug("")
    plt.plot()

    pretty_repr("")


def fc():
    pass


def main():
    # AltairRenderers.set_renderer()
    # alt.theme.enable("default_theme")
    ic.configureOutput(outputFunction=logger.debug)
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
