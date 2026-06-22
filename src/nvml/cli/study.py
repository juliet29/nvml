import matplotlib.pyplot as plt
from cyclopts import App
from loguru import logger

# from plyze.plots.altair_helpers import AltairRenderers
# from plyze.plots.theme import default_theme
from rich.pretty import pretty_repr
from utils4plans.logconfig import logset

from nvml.constants import FileNames
from nvml.model.data import arrange_data, create_train_test_dataset
from nvml.model.make import evaluate, init_model, train_model
from nvml.paths import StoragePaths

app = App()


def keep():
    # default_theme()
    logger.debug("")
    plt.plot()

    pretty_repr("")


@app.command
def fc():
    dir_path = StoragePaths.nvflow_latest
    metrics_path = dir_path / FileNames.metrics_path
    qois_path = dir_path / FileNames.qois_path
    SEED = 12345
    df = arrange_data(metrics_path, qois_path)
    train, test = create_train_test_dataset(df, SEED)
    mad = init_model(train)
    train_model(train, mad, 500)
    evaluate(test, mad)


def main():
    # AltairRenderers.set_renderer()
    # alt.theme.enable("default_theme")
    logset(to_stderr=True)
    app()


if __name__ == "__main__":
    main()
