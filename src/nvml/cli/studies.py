import altair as alt
import matplotlib.pyplot as plt
from cyclopts import App
from loguru import logger
from plyze.plots.altair_helpers import AltairRenderers
from plyze.plots.theme import default_theme
from rich.pretty import pretty_repr
from utils4plans.logconfig import logset

app = App()


def keep():
    default_theme()
    logger.debug("")
    plt.plot()

    pretty_repr("")


### ------- START KEEP COMMANDS ---------
@app.command
def fc():
    pass


### ------- START TEMP COMMANDS ---------


### ------- END COMMANDS ---------


def main():
    AltairRenderers.set_renderer()
    alt.theme.enable("default_theme")
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
