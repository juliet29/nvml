from cyclopts import App
from nvml.paths import StoragePaths
from sympy import evaluate

from nvml.constants import FileNames
from nvml.model.data import arrange_data, create_train_test_dataset
from nvml.model.make import init_model, train_model

model = App("model")


@model.command
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
