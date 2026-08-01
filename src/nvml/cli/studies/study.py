import warnings

from cyclopts import App
from icecream import ic
from loguru import logger
from utils4plans.logs import logset
from zarr.core.dtype.common import UnstableSpecificationWarning

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


def suppress_zarr_warnings():
    warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
    warnings.filterwarnings("ignore", message=".*Consolidated metadata.*")


def main():
    # TODO: get new utils4plans
    suppress_zarr_warnings()
    ic.configureOutput(outputFunction=logger.debug)
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
