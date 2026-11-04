from cyclopts import App
from icecream import ic
from loguru import logger
from utils4plans.logs import logset

from nvml._study.coinc import coinc_app
from nvml._study.qoi import qoi_app

# from nvml._study.reg import reg_app
from nvml._study.reg_data import reg_data_app
from nvml._study.reg_qoi import reg_qoi_app
from nvml._study.rex import qoi_explore_app

# from nvml._study._root.modules.qdim import qdim

app = App()
app.command(coinc_app)
# app.command(reg_app)
app.command(qoi_app)
app.command(reg_data_app)
app.command(reg_qoi_app)
app.command(qoi_explore_app)


def main():
    ic.configureOutput(outputFunction=logger.debug)
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
