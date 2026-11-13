import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score


class SilhouettePlot:
    """
    Modified from [sklearn example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py)
    """

    def __init__(
        self,
        kernel: np.ndarray,
        n_clusters: int,
        color_map: Colormap = cm.nipy_spectral,
    ) -> None:
        pass

        self.kernel = kernel  # this is actually the kernel with distances
        self.n_clusters = n_clusters
        self.color_map: Colormap = color_map
        self.random_state: int = 10
        self.y_low = 10

    def cluster(self):
        # this can happen on init if make this a real class..
        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            random_state=self.random_state,
            assign_labels="cluster_qr",  # Todo look this up
        )
        self.labels = self.model.fit_predict(self.kernel)

    def score(self):
        return silhouette_score(self.kernel, self.labels)

    def sil_plot(self, ax: Axes):
        y_low = self.y_low

        def handle_cluster(i: int, y_low: int):
            values: np.ndarray = sil_values[self.labels == i]
            values.sort()
            sz: int = values.shape[0]

            y_upper = y_low + sz
            color = self.color_map(
                float(i) / self.n_clusters
            )  # TODO: replace with own colormap

            ax.fill_betweenx(
                np.arange(y_low, y_upper),
                0,
                values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax.text(x=-0.05, y=y_low + 0.5 * sz, s=str(i))

            y_low = y_upper + y_low
            return y_low

        ax.set_xlim(-0.1, 1)
        y_space_for_clusters = len(self.kernel) + (self.n_clusters + 1) * self.y_low
        ax.set_ylim(0, y_space_for_clusters)
        sil_values = silhouette_samples(self.kernel, self.labels)

        for i in range(self.n_clusters):
            y_low = handle_cluster(i, y_low)

        # plot average
        ax.axvline(x=self.score(), color="red", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # def cluster_plot(self, ax: Axes):
    #     # plot the data
    #     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    #     # can imagine it being just a single axis plot..
    #     xs = self.X[:, 0]
    #
    #     # plot the centers
    #     centers = self.model.cluster_centers_
    #     xs = centers[:, 0]
    #     ys = centers[:, 1]
    #     ax.scatter(xs, ys, marker="o", c="white", s=200, edgecolors="k")
    #
    #     for ix, center in enumerate(centers):
    #         ax.scatter(c[0], c[1], marker=f"{ix}", edgecolors="k", s=50)
    #
    def plot(self):
        fig, ax1 = plt.subplots(1)
        fig.set_size_inches(18, 7)
        self.sil_plot(ax1)
