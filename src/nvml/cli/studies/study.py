from cyclopts import App
from icecream import ic, install  # pyright: ignore[reportPrivateImportUsage]
from loguru import logger
from utils4plans.logs import logset

# from nvml.cli.studies.create import create
# from nvml.cli.studies.modules.cluster import cluster
# from nvml.cli.studies.modules.flow import flow
from nvml.cli.studies.modules.gmodel import gmod

# from nvml.cli.studies.modules.qdim import qdim

app = App()
# app.command(qdim)
# app.command(create)
# app.command(flow)
# app.command(cluster)
app.command(gmod)

install()  # allow icecream to be used everywhere


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
