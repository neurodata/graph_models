#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from graspy.datasets import load_drosophila_left
from graspy.models import RDPGEstimator, SBEstimator, DCSBEstimator
from graspy.utils import binarize
from graspy.plot import heatmap


adj, labels = load_drosophila_left(return_labels=True)
adj = binarize(adj)
heatmap(adj, inner_hier_labels=labels, cbar=False)


def run_mses(adj, estimator, rangelist, kws):
    mse = np.zeros(len(rangelist))
    n_params = np.zeros(len(rangelist))
    for i, n in enumerate(rangelist):
        if estimator == "SBM":
            e = SBEstimator(min_comm=n, max_comm=n, **kws)
        elif estimator == "RDPG":
            e = RDPGEstimator(n_components=n, **kws)
        elif estimator == "DCSBM":
            e = DCSBEstimator(min_comm=n, max_comm=n, **kws)
        else:
            raise TypeError()
        e.fit(adj)
        mse[i] = e.mse(adj) + e._n_parameters() * np.log(adj.size)
        n_params[i] = e._n_parameters()
    return np.stack((mse, n_params), axis=1)


exps = []
names = []

# SBM NORMAL ######
name = "SBM"
max_comm = 60
comm_range = list(range(1, max_comm))
sb_kws = dict(directed=True, loops=False)

sb_out = run_mses(adj, "SBM", comm_range, sb_kws)
exps.append(sb_out)
names.append(name)

# # dDCSBM NORMAL ######
# name = "dDCSBM"
# max_comm = 60
# comm_range = list(range(1, max_comm))
# dcsb_kws = dict(directed=True, loops=False, degree_directed=True)

# sb_out = run_mses(adj, "DCSBM", comm_range, dcsb_kws)
# exps.append(sb_out)
# names.append(name)

# DCSBM NORMAL ######
name = "DCSBM"
max_comm = 60
comm_range = list(range(1, max_comm))
dcsb_kws = dict(directed=True, loops=False, degree_directed=False)

sb_out = run_mses(adj, "DCSBM", comm_range, dcsb_kws)
exps.append(sb_out)
names.append(name)

# SBM NORMAL ######
# name = "SBM Undirected"
# max_comm = 60
# comm_range = list(range(1, max_comm))
# sb_kws = dict(directed=False, loops=False)

# sb_out = run_mses(adj, "SBM", comm_range, sb_kws)
# exps.append(sb_out)
# names.append(name)

# RDPG RAW ######
name = "RDPG Raw"
max_components = 10
rdpg_kws = dict(loops=False, diag_aug_weight=0, plus_c_weight=0)
comp_range = list(range(1, max_components))

rdpg_raw_out = run_mses(adj, "RDPG", comp_range, rdpg_kws)
exps.append(rdpg_raw_out)
names.append(name)

# # RDPG Diag Aug ######
# name = "RDPG Diag Aug"
# max_components = 10
# rdpg_kws = dict(loops=False, diag_aug_weight=1, plus_c_weight=0)
# comp_range = list(range(1, max_components))

# rdpg_raw_out = run_mses(adj, "RDPG", comp_range, rdpg_kws)
# exps.append(rdpg_raw_out)
# names.append(name)

# # RDPG Diag Aug and C #####
# name = "RDPG Diag Aug + C"
# max_components = 10
# rdpg_kws = dict(loops=False, diag_aug_weight=1, plus_c_weight=1)
# comp_range = list(range(1, max_components))

# rdpg_raw_out = run_mses(adj, "RDPG", comp_range, rdpg_kws)
# exps.append(rdpg_raw_out)
# names.append(name)

# Plot ######

data = np.concatenate(exps, axis=0)

label_vecs = []
for e, n in zip(exps, names):
    label_vecs.append(np.array(e.shape[0] * [n]))
labels = np.concatenate(label_vecs)

plot_df = pd.DataFrame(data=data, columns=("MSE", "# Parameters"))
plot_df["Model"] = labels
plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1)
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=plot_df, x="# Parameters", y="MSE", hue="Model", palette="Set1", linewidth=0
)
plt.savefig(
    "graph_models/drosophila_experiments/figs/2model_comparison.pdf",
    format="pdf",
    facecolor="w",
)

#%% Sims
from graspy.simulations import p_from_latent, sample_edges
from graspy.plot import pairplot

p_kwargs = {}
sample_kwargs = {}
n_verts = 1000
show_graphs = False
show_latent = False
names = []
graph_sims = []


def get_graph(latent):
    if type(latent) is tuple:
        left_latent = latent[0]
        right_latent = latent[1]
    else:
        left_latent = latent
        right_latent = None
    true_P = p_from_latent(left_latent, right_latent, **p_kwargs)
    graph = sample_edges(true_P, **sample_kwargs)
    if show_graphs:
        heatmap(graph)
    if show_latent:
        if right_latent is not None:
            labels = np.array(
                len(left_latent) * ["left"] + len(right_latent) * ["right"]
            )
            # print(left_latent.shape)
            # print(right_latent.shape)
            latent = np.concatenate((left_latent, right_latent), axis=0)
            pairplot(latent, labels=labels)
        else:
            pairplot(left_latent)
    return graph


# Single point in latent space
# this should be an ER model
latent = np.array(n_verts * [0.5])
latent = latent[:, np.newaxis]  # to make it n x d
graph = get_graph(latent)
names.append("Latent point")
graph_sims.append(graph)

# Line in 1d
# should be a degree corrected ER
latent = np.random.uniform(0.25, 0.75, n_verts)
latent = latent[:, np.newaxis]
graph = get_graph(latent)
names.append("Latent line - uniform")
graph_sims.append(graph)

# Line in 1d, but gaussian
latent = np.random.normal(0.5, 0.1, n_verts)
latent = latent[:, np.newaxis]
graph = get_graph(latent)
names.append("Latent line - gaussian")
graph_sims.append(graph)

# directed latent lines
left_latent = np.random.uniform(0.25, 0.75, n_verts)
left_latent = left_latent[:, np.newaxis]
right_latent = np.random.uniform(0.25, 0.75, n_verts)
right_latent = right_latent[:, np.newaxis]
latent = (left_latent, right_latent)
graph = get_graph(latent)
names.append("Directed latent lines - same uniform")
graph_sims.append(graph)

# directed latent lines, different uniform
left_latent = np.random.uniform(0.4, 0.8, n_verts)
left_latent = left_latent[:, np.newaxis]
right_latent = np.random.uniform(0.2, 0.5, n_verts)
right_latent = right_latent[:, np.newaxis]
latent = (left_latent, right_latent)
graph = get_graph(latent)
names.append("Directed latent lines - same uniform")
graph_sims.append(graph)

# directed latent lines, different gaussian
left_latent = np.random.normal(0.4, 0.1, n_verts)
left_latent = left_latent[:, np.newaxis]
right_latent = np.random.normal(0.8, 0.05, n_verts)
right_latent = right_latent[:, np.newaxis]
latent = (left_latent, right_latent)
graph = get_graph(latent)
names.append("Directed latent lines - same uniform")
graph_sims.append(graph)

# sbm simple, 2 block
point1 = [0.1, 0.6]
point2 = [0.6, 0.1]
points = np.array([point1, point2])
inds = np.array(int(n_verts / 2) * [0] + int(n_verts / 2) * [1])
latent = np.array(points[inds])
graph = get_graph(latent)
names.append("SBM - 2 block")
graph_sims.append(graph)


# dcsbm, 2 line, uniform
thetas = np.array([0 * np.pi, 0.5 * np.pi])
distances = np.random.uniform(0.2, 0.9, n_verts)
vec1 = np.array([np.cos(thetas[0]), np.sin(thetas[0])])
vec2 = np.array([np.cos(thetas[1]), np.sin(thetas[1])])
latent1 = np.multiply(distances[: int(n_verts / 2)][:, np.newaxis], vec1[np.newaxis, :])
latent2 = np.multiply(distances[int(n_verts / 2) :][:, np.newaxis], vec2[np.newaxis, :])
latent = np.concatenate((latent1, latent2), axis=0)
graph = get_graph(latent)
names.append("DCSBM - 2 line uniform")
graph_sims.append(graph)


# dcsbm, 2 line, beta
thetas = np.array([0.1 * np.pi, 0.4 * np.pi])
distances = np.random.beta(0.5, 0.5, n_verts)
vec1 = np.array([np.cos(thetas[0]), np.sin(thetas[0])])
vec2 = np.array([np.cos(thetas[1]), np.sin(thetas[1])])
latent1 = np.multiply(distances[: int(n_verts / 2)][:, np.newaxis], vec1[np.newaxis, :])
latent2 = np.multiply(distances[int(n_verts / 2) :][:, np.newaxis], vec2[np.newaxis, :])
latent = np.concatenate((latent1, latent2), axis=0)
graph = get_graph(latent)
names.append("DCSBM - 2 line beta")
graph_sims.append(graph)

inds = np.array(int(n_verts / 2) * [0] + int(n_verts / 2) * [1])
# for graph, name in zip(graph_sims, names):
#     evaluate_models(graph, inds, title=name)


def evaluate_models(adj, title):
    exps = []
    names = []

    # SBM NORMAL ######
    name = "SBM"
    max_comm = 60
    comm_range = list(range(1, max_comm))
    sb_kws = dict(directed=True, loops=False)

    sb_out = run_mses(adj, "SBM", comm_range, sb_kws)
    exps.append(sb_out)
    names.append(name)

    # # dDCSBM NORMAL ######
    # name = "dDCSBM"
    # max_comm = 60
    # comm_range = list(range(1, max_comm))
    # dcsb_kws = dict(directed=True, loops=False, degree_directed=True)

    # sb_out = run_mses(adj, "DCSBM", comm_range, dcsb_kws)
    # exps.append(sb_out)
    # names.append(name)

    # DCSBM NORMAL ######
    name = "DCSBM"
    max_comm = 60
    comm_range = list(range(1, max_comm))
    dcsb_kws = dict(directed=True, loops=False, degree_directed=False)

    sb_out = run_mses(adj, "DCSBM", comm_range, dcsb_kws)
    exps.append(sb_out)
    names.append(name)

    # RDPG RAW ######
    name = "RDPG Raw"
    max_components = 10
    rdpg_kws = dict(loops=False, diag_aug_weight=0, plus_c_weight=0)
    comp_range = list(range(1, max_components))

    rdpg_raw_out = run_mses(adj, "RDPG", comp_range, rdpg_kws)
    exps.append(rdpg_raw_out)
    names.append(name)

    # Plot ######

    data = np.concatenate(exps, axis=0)

    label_vecs = []
    for e, n in zip(exps, names):
        label_vecs.append(np.array(e.shape[0] * [n]))
    labels = np.concatenate(label_vecs)

    plot_df = pd.DataFrame(data=data, columns=("MSE", "# Parameters"))
    plot_df["Model"] = labels
    plt.style.use("seaborn-white")
    sns.set_context("talk", font_scale=1)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=plot_df,
        x="# Parameters",
        y="MSE",
        hue="Model",
        palette="Set1",
        linewidth=0,
    )
    plt.title(title)
    plt.show()
    # plt.savefig(
    #     "graph_models/drosophila_experiments/figs/2model_comparison.pdf",
    #     format="pdf",
    #     facecolor="w",
    # )


for graph, name in zip(graph_sims, names):
    evaluate_models(graph, name)

#%%

#%%
s, v = np.linalg.eig(adj)
inds = np.argsort(np.abs(s))
inds = inds[::-1]
plt.plot(s[inds[:10]], ".")

