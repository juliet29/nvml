import matplotlib.cm as cm
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score

RANDOM_STATE = 10


class spc:
    k = "k"
    score = "score"


def cluster(X: np.ndarray, n_clusters: int, random_state: int = RANDOM_STATE):
    # this can happen on init if make this a real class..
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="cluster_qr",  # Todo look this up
    )
    labels = model.fit_predict(X)
    return model, labels


def study_sil_score(X: np.ndarray, k_to_try: list[int]):
    def mk(k):
        model, labels = cluster(X, k)
        score = silhouette_score(X, labels)
        return score

    scores = [mk(i) for i in k_to_try]
    df = pl.DataFrame(
        {spc.k: k_to_try, spc.score: scores},
        schema_overrides={spc.k: pl.Categorical},
        strict=False,
    )
    return df


def score_plot(df: pl.DataFrame, ax: Axes):
    sns.pointplot(data=df, x=spc.k, y=spc.score, ax=ax)


def sil_dist_plot(
    X: np.ndarray,
    k: int,
    ax: Axes,
    color_map: Colormap = cm.nipy_spectral,
    random_state: int = RANDOM_STATE,
):
    """
    Modified from [sklearn example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py)
    """

    def handle_cluster(i: int, y_low: int):
        cluster_coeffs: np.ndarray = coeffs[labels == i]  # pyright: ignore[reportAssignmentType]
        cluster_coeffs.sort()
        sz: int = cluster_coeffs.shape[0]

        y_upper = y_low + sz
        color = color_map(float(i) / k)

        ax.fill_betweenx(
            np.arange(y_low, y_upper),
            0,
            cluster_coeffs,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(x=-0.05, y=y_low + 0.5 * sz, s=str(i))

        y_low = y_upper + y_low
        return y_low

    model, labels = cluster(X, k, random_state)
    score = silhouette_score(X, labels)
    coeffs = silhouette_samples(X, labels)
    y_low = 10

    ax.set_xlim(-0.1, 1)
    y_space_for_clusters = len(X) + (k + 1) * y_low
    ax.set_ylim(0, y_space_for_clusters)

    for i in range(k):
        y_low = handle_cluster(i, y_low)

        # plot average
    ax.axvline(x=0, color="k")
    ax.axvline(x=score, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_title(f"k={k}")
