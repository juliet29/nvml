import matplotlib
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa
from cyclopts import App
from icecream import ic
from loguru import logger
from utils4plans.logs import logset

from nvml._study.buoy import by
from nvml._study.f2 import f2_app
from nvml._study.q2s import q2_app
from nvml._study.wl import wl_app

matplotlib.use("module://matplotlib-backend-kitty")
app = App()

app.command(by)
app.command(wl_app)
app.command(q2_app)
app.command(f2_app)


def main():
    ic.configureOutput(outputFunction=logger.debug)
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
