import matplotlib.pyplot as plt  # noqa

import numpy as np
import seaborn as sns

from sklearn.inspection import permutation_importance

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import KernelPCA
import statsmodels.api as sm
import statsmodels.formula.api as smf
import iplotx as ipx  # noqa

import matplotlib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from nvml._study.wl2 import fe3

matplotlib.use("module://matplotlib-backend-kitty")
labels, qoi, features, kern = fe3()

K = kern
df0 = features


df = features.with_columns(gclass=labels)
# Extract top 10 KPCA components instead of just 3
kpca10 = KernelPCA(n_components=10, kernel="precomputed").fit_transform(K)
X_tab = df.select(["gclass", "num_paths", "degree_dom_node"]).to_numpy()
X_all = np.hstack([X_tab, kpca10])
y = df["space_sum_time_mean"].to_numpy()


cv = KFold(n_splits=5, shuffle=True, random_state=42)
hgb = HistGradientBoostingRegressor(loss="gamma", random_state=42)
scores = cross_val_score(hgb, X_all, y, cv=cv, scoring="r2")
print(f"5-Fold CV R2 (Gamma GBDT): {scores.mean():.4f} ± {scores.std():.4f}")

hgb.fit(X_all, y)
perm = permutation_importance(hgb, X_all, y, n_repeats=10, random_state=42)

feature_names = ["gclass", "num_paths", "degree_dom_node"] + [
    f"graph_pc{i + 1}" for i in range(10)
]
for name, imp in sorted(zip(feature_names, perm.importances_mean), key=lambda x: -x[1]):
    print(f"{name:18s}: {imp:.4f}")


pdf = df.to_pandas()
pdf["log_y"] = np.log(pdf["space_sum_time_mean"])

kpca = KernelPCA(n_components=5, kernel="precomputed")
graph_pcs = kpca.fit_transform(K)
df = df0.with_columns(
    gclass=labels,
    **{f"graph_pc{i + 1}": graph_pcs[:, i] for i in range(graph_pcs.shape[1])},
)
features = [
    "num_paths",
    "degree_dom_node",
    "graph_pc1",
    "graph_pc2",
    "graph_pc3",
    "graph_pc4",
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, feat in zip(axes.flat, features):
    for cls, sub in pdf.groupby("gclass"):
        sns.regplot(
            data=sub,
            x=feat,
            y="log_y",
            lowess=True,  # Smooth curve reveals exact non-linear bends
            scatter_kws={"alpha": 0.3, "s": 18},
            line_kws={"linewidth": 2.5},
            label=f"gclass={cls}",
            ax=ax,
        )
    ax.set_title(f"log(y) vs. {feat}")
    ax.set_ylabel("log(space_sum_time_mean)")
    ax.legend()

plt.tight_layout()
plt.show()

breakpoint()

kpca = KernelPCA(n_components=5, kernel="precomputed")
graph_pcs = kpca.fit_transform(K)
df = df0.with_columns(
    gclass=labels,
    **{f"graph_pc{i + 1}": graph_pcs[:, i] for i in range(graph_pcs.shape[1])},
)

print(df.describe())

formula = """
space_sum_time_mean ~ C(gclass) + num_paths + degree_dom_node 
                    + graph_pc1 + graph_pc2 + graph_pc3 + graph_pc4 + graph_pc5
"""

glm_gamma = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

print(glm_gamma.summary())


# 1. Compute original-scale R2 and MAE (directly comparable to OLS)
y_true = df["space_sum_time_mean"].to_numpy()
y_pred = glm_gamma.predict()
print(f"Original-scale R2:  {r2_score(y_true, y_pred):.4f}")
print(f"Original-scale MAE: {mean_absolute_error(y_true, y_pred):.4f}")


#
# # 1. Check deviance residuals to spot any extreme graphs
# dev_resid = glm_gamma.resid_deviance
# print("Max deviance residual:", np.max(np.abs(dev_resid)))
# print("Top 5 outlier row indices:", np.argsort(np.abs(dev_resid))[-5:])
#
# # 2. Optional: check if adding a nonlinear term (e.g., degree_dom_node**2) boosts raw R2
