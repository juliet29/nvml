import matplotlib.pyplot as plt  # noqa
import polars as pl

import numpy as np


from sklearn.metrics import r2_score
from sklearn.decomposition import KernelPCA
import statsmodels.api as sm
import statsmodels.formula.api as smf
import iplotx as ipx  # noqa

import matplotlib

from sklearn.model_selection import KFold
from nvml._study.wl2 import fe3

matplotlib.use("module://matplotlib-backend-kitty")
labels, qoi, features, kern = fe3()

K = kern
df0 = features.with_columns(gclass=labels)
kpca = KernelPCA(n_components=10, kernel="precomputed", random_state=42)
graph_pcs = kpca.fit_transform(K)
print(df0.columns)

df = df0.with_columns(
    **{f"graph_pc{i + 1}": graph_pcs[:, i] for i in range(10)},
)
print(df["coincidence"].value_counts().sort("count", descending=True))

df_clean = df.with_columns(
    has_ww=pl.col("coincidence").str.contains("ww").cast(pl.Int8),
    has_lw=pl.col("coincidence").str.contains("lw").cast(pl.Int8),
    has_sd=pl.col("coincidence").str.contains("sd").cast(pl.Int8),
)

pdf = df_clean.to_pandas()
formula_upgraded = """
space_sum_time_mean ~ np.log1p(num_paths) * bs(degree_dom_node, df=3)
                    + np.log1p(avg_path_length)
                    + np.log(num_rooms)
                    + has_ww + has_lw + has_sd
                    + C(gclass) * graph_pc2
                    + graph_pc1 + graph_pc3 + graph_pc6
"""
glm_up = smf.glm(
    formula=formula_upgraded,
    data=pdf,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

print(glm_up.summary())
print(
    "In-sample Original-scale R2:"
    f" {r2_score(pdf['space_sum_time_mean'], glm_up.predict()):.4f}"
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []
for train_idx, test_idx in cv.split(pdf):
    m = smf.glm(
        formula_upgraded,
        data=pdf.iloc[train_idx],
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    preds = m.predict(pdf.iloc[test_idx])
    cv_r2.append(r2_score(pdf.iloc[test_idx]["space_sum_time_mean"], preds))

print(f"5-Fold CV R2 (Upgraded Gamma GLM): {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
# Ensure graph_pc1..8 are in your Polars DataFrame and convert to Pandas
# df = df0.with_columns(
#     gclass=labels,
#     **{f"graph_pc{i + 1}": graph_pcs[:, i] for i in range(8)},
# )
# pdf = df.to_pandas()
# pdf["log_y"] = np.log(pdf["space_sum_time_mean"])
# pdf["log_num_paths"] = np.log1p(pdf["num_paths"])
#
# # 1. 2x3 Grid of Your Top 6 Most Important Features (by Permutation Importance)
# top_6 = [
#     "num_paths",
#     "graph_pc1",
#     "graph_pc6",
#     "degree_dom_node",
#     "graph_pc8",
#     "graph_pc2",
# ]
# fig, axes = plt.subplots(2, 3, figsize=(16, 9))
#
# for ax, feat in zip(axes.flat, top_6):
#     for cls, sub in pdf.groupby("gclass"):
#         sns.regplot(
#             data=sub,
#             x=feat,
#             y="log_y",
#             lowess=True,
#             scatter_kws={"alpha": 0.3, "s": 18},
#             line_kws={"linewidth": 2.5},
#             label=f"gclass={cls}",
#             ax=ax,
#         )
#     ax.set_title(f"log(y) vs. {feat}")
#     ax.set_ylabel("log(space_sum_time_mean)")
#     ax.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 2. Side-by-Side Check: Raw num_paths vs. log1p(num_paths)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
#
# sns.regplot(
#     data=pdf,
#     x="num_paths",
#     y="log_y",
#     lowess=True,
#     scatter_kws={"alpha": 0.3, "s": 18},
#     line_kws={"color": "crimson", "linewidth": 2.5},
#     ax=ax1,
# )
# ax1.set_title("Raw num_paths vs. log(y) (Curved)")
#
# sns.regplot(
#     data=pdf,
#     x="log_num_paths",
#     y="log_y",
#     lowess=True,
#     scatter_kws={"alpha": 0.3, "s": 18},
#     line_kws={"color": "seagreen", "linewidth": 2.5},
#     ax=ax2,
# )
# ax2.set_title("log1p(num_paths) vs. log(y) (Linearized)")
#
# plt.tight_layout()
# plt.show()
