from pathlib import Path

_BasePath = type(Path())


class GraphRepDir(_BasePath):
    def gk_graphs_joblib(self, test: bool = True) -> Path:
        if test:
            return self / "test_gk_graphs.joblib"
        return self / "gk_graphs.joblib"

    @property
    def kernel_matrices_h5(self) -> Path:
        return self / "kernel_matrices.h5"

    @property
    def sm_kernel_npy(self) -> Path:
        return self / "sm_kernel.npy"

    @property
    def sm_clusters_npy(self) -> Path:
        return self / "sm_clusters.npy"


class RegressionDir(_BasePath):
    @property
    def regression_data(self):
        return self / "data.csv"

    @property
    def qoi_dataArray(self):
        return self / "qoi.nc"

    @property
    def tdim_dataArray(self):
        return self / "tdim.nc"

    # TOD): opportunity to have this be a method and pass variable!
    @property
    def tdim_zone_dataArray(self):
        return self / "tdim_zone.nc"
