#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import svd

from graspy.embed import AdjacencySpectralEmbed
from graspy.plot import heatmap, pairplot
from graspy.simulations import p_from_latent, sample_edges
from graspy.utils import is_fully_connected

sns.set_context("talk", font_scale=1)
plt.style.use("seaborn-white")
sns.set_palette("Set1")
np.random.seed(8888)
n_verts = 200
p_kwargs = {}
sample_kwargs = {"directed": True}
show_graphs = False
show_latent = False
names = []
graph_sims = []


def get_graph(latent, labels=None, title=None):
    if type(latent) is tuple:
        left_latent = latent[0]
        right_latent = latent[1]
    else:
        left_latent = latent
        right_latent = None
    true_P = p_from_latent(left_latent, right_latent, **p_kwargs)
    graph = sample_edges(true_P, **sample_kwargs)
    if not is_fully_connected(graph):
        raise ValueError("unconnected graph")
    if show_graphs:
        heatmap(graph, inner_hier_labels=labels, title=title)
    if show_latent:
        if right_latent is not None:
            labels = np.array(
                len(left_latent) * ["left"] + len(right_latent) * ["right"]
            )
            latent = np.concatenate((left_latent, right_latent), axis=0)
            pairplot(latent, labels=labels, title=title)
        else:
            pairplot(left_latent, labels=labels, title=title)
    return graph


# Line in 1d
# should be a degree corrected ER
latent = np.random.uniform(0.25, 0.75, n_verts)
latent = latent[:, np.newaxis]
graph = get_graph(latent, title="Latent line")

n_try = list(range(1, graph.shape[0]))

for j in range(1):
    mse_ase = np.zeros(len(n_try))
    mse_svd = np.zeros(len(n_try))
    mse_my_ase = np.zeros(len(n_try))
    for i, n_components in enumerate(n_try):
        ase = AdjacencySpectralEmbed(n_components=n_components, algorithm="full")
        latent = ase.fit_transform(graph)
        mse_ase[i] = np.linalg.norm(graph - latent[0] @ latent[1].T) ** 2
        u, s, vh = svd(graph, full_matrices=False)
        mse_svd[i] = (
            np.linalg.norm(
                graph
                - u[:, :n_components] @ np.diag(s[:n_components]) @ vh[:n_components, :]
            )
            ** 2
        )
        X = u[:, :n_components] @ np.diag(np.sqrt(s[:n_components]))
        Y = vh.T[:, :n_components] @ np.diag(np.sqrt(s[:n_components]))
        mse_my_ase[i] = np.linalg.norm(graph - X @ Y.T) ** 2
    sns.scatterplot(x=n_try, y=mse_ase, linewidth=0, alpha=0.7, label="ASE")
    sns.scatterplot(x=n_try, y=mse_svd, linewidth=0, alpha=0.7, label="SVD")
    sns.scatterplot(x=n_try, y=mse_my_ase, linewidth=0, alpha=0.7, label="P-ASE")
    plt.legend()
    plt.show()
