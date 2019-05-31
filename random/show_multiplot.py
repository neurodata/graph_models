#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

from graspy.plot import heatmap, gridplot
from graspy.simulations import sbm

n = [50, 50]
B = np.array([[0.9, 0.1], [0.1, 0.9]])
graphs = [sbm(n, B) for i in range(4)]
labels = np.array(50 * [0] + 50 * [1])

sns.set_context("talk")
plt.style.use("seaborn-white")
fig, axs = plt.subplots(2, 2, figsize=(12, 14))

# colormapping
cmap = cm.get_cmap("RdBu_r")
center = 0
vmin = 0
vmax = 1
norm = mpl.colors.Normalize(0, 1)
cc = np.linspace(0.5, 1, 256)
cmap = mpl.colors.ListedColormap(cmap(cc))

# heatmapping
heatmap_kws = dict(
    inner_hier_labels=labels,
    vmin=0,
    vmax=1,
    cbar=False,
    cmap=cmap,
    center=None,
    hier_label_fontsize=20,
    title_pad=40.1,
    font_scale=1.5,
)

heatmap(graphs[0], ax=axs[0][0], title="Donkey", **heatmap_kws)
heatmap(graphs[1], ax=axs[0][1], title="Emu", **heatmap_kws)
heatmap(graphs[2], ax=axs[1][0], title="Pony", **heatmap_kws)
heatmap(graphs[3], ax=axs[1][1], title="Goat", **heatmap_kws)


# add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(graphs[0])
fig.colorbar(sm, ax=axs, orientation="horizontal", pad=0.05, shrink=0.8, fraction=0.1)
plt.savefig("show_multiplot.png", format="png", dpi=300, facecolor="w")
plt.show()

# another way
# cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
# cb1 = mpl.colorbar.ColorbarBase(cbar_ax, )
#%%
gridplot(graphs, title="All", title_pad=0)
