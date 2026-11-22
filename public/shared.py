from cyclopts import App
from icecream import ic
from loguru import logger
from utils4plans.logs import logset
from utils4plots import set_dynamic_params

from nvml.settings import settings
from public.figures.qoi.task_valid import fig_qoi_valid

app = App()
# app.command(qois)
# app.command(features)
# app.command(glm)

figs = App("figs")
app.command(figs)

figs.command(fig_qoi_valid)


def main():
    ic.configureOutput(outputFunction=logger.debug)
    logset(to_stderr=True)

    set_dynamic_params(settings.figure_profile)
    app()


if __name__ == "__main__":
    main()
