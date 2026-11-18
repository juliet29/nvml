from cyclopts import App
from icecream import ic
from loguru import logger
from utils4plans.logs import logset

from public.data.task_features import features
from public.data.task_flow_qoi import qois

app = App()
app.command(qois)
app.command(features)


def main():
    ic.configureOutput(outputFunction=logger.debug)
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
