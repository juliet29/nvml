from cyclopts import App
from icecream import ic
from loguru import logger
from utils4plans.logs import logset

from nvml._study.buoy import by

app = App()

app.command(by)


def main():
    ic.configureOutput(outputFunction=logger.debug)
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
