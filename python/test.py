import numpy as np
import pandas as pd
from gfem import GFEMModel
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sid = "sm128"
source = "assets/data/synthetic-samples-mixing-middle.csv"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# helper functions !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_assemblage_centers(assemblage_array, assemblage_list, top_n=13):
    # Convert assemblage_array to integers
    assemblage_array = assemblage_array.astype(int)

    counts = np.bincount(assemblage_array.ravel())
    unique_values = np.arange(len(counts))

    # Sort unique values by counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    unique_values_sorted = unique_values[sorted_indices]

    # Select top n unique values
    top_values = unique_values_sorted[:top_n]

    centers = {}

    for value in top_values:
        count = counts[value]
        positions = np.column_stack(np.where(assemblage_array == value))
        center_y, center_x = positions.mean(axis=0)
        assemblage = assemblage_list[value - 1]
        centers[value] = (center_x, center_y, assemblage)

    # Sort centers by value before returning
    centers = {k: v for k, v in sorted(centers.items(), key=lambda item: item[0])}

    return centers

def convert_assemblage_centers_to_PT(center, extent, res):
    """
    """
    center_x, center_y = center
    T_min, T_max, P_min, P_max = extent

    T_val = T_min + (T_max - T_min) * (center_x / res)
    P_val = P_min + (P_max - P_min) * (center_y / res)

    return T_val, P_val

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 22
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.facecolor"] = "0.9"
plt.rcParams["legend.frameon"] = "False"
plt.rcParams["legend.facecolor"] = "0.9"
plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["legend.fontsize"] = "small"
plt.rcParams["figure.autolayout"] = "True"

model = GFEMModel("hp02", sid, source)
model._get_target_array()

res = model.res
targets = model.targets
results = model.results
target_array = model.target_array
P, T = results["P"], results["T"]
model_out_dir = model.model_out_dir
assemblages = model._read_perplex_assemblages()
extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

for i, target in enumerate(targets):
    if target == "assemblage":
        img = target_array[:, i].reshape(res + 1, res + 1)

centers = find_assemblage_centers(img, assemblages)

fig, ax = plt.subplots(figsize=(6.3, 4.725 * 2))
im = ax.imshow(img, extent=extent, aspect="auto", cmap="bone", origin="lower")

legend_handles = []

for value, (center_x, center_y, assemblage) in centers.items():
    T_val, P_val = convert_assemblage_centers_to_PT((center_x, center_y), extent, res)
    ax.text(T_val, P_val, value, color="white", ha="center", va="center", fontweight="bold")
    legend_handles.append(mlines.Line2D([0], [0], color="black", linewidth=0,
                                        label=f"{value}: {assemblage}"))

ax.set_xlabel("T (K)")
ax.set_ylabel("P (GPa)")
plt.colorbar(im, ax=ax, label="")
plt.legend(title="", handles=legend_handles, loc="upper center", handleheight=1.2,
           bbox_to_anchor=(0.5, -0.2), columnspacing=0.2, handletextpad=-0.1,
           fontsize=22 * 0.633)

plt.savefig("test.png")

model = GFEMModel("hp633", sid, source)
model._get_target_array()

res = model.res
targets = model.targets
results = model.results
target_array = model.target_array
P, T = results["P"], results["T"]
model_out_dir = model.model_out_dir
assemblages = model._read_perplex_assemblages()
extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

for i, target in enumerate(targets):
    if target == "assemblage":
        img = target_array[:, i].reshape(res + 1, res + 1)

centers = find_assemblage_centers(img, assemblages)

fig, ax = plt.subplots(figsize=(6.3, 4.725 * 2))
im = ax.imshow(img, extent=extent, aspect="auto", cmap="bone", origin="lower")

legend_handles = []

for value, (center_x, center_y, assemblage) in centers.items():
    T_val, P_val = convert_assemblage_centers_to_PT((center_x, center_y), extent, res)
    ax.text(T_val, P_val, value, color="white", ha="center", va="center", fontweight="bold")
    legend_handles.append(mlines.Line2D([0], [0], color="black", linewidth=0,
                                        label=f"{value}: {assemblage}"))

ax.set_xlabel("T (K)")
ax.set_ylabel("P (GPa)")
plt.colorbar(im, ax=ax, label="")
plt.legend(title="", handles=legend_handles, loc="upper center", handleheight=1.2,
           bbox_to_anchor=(0.5, -0.2), columnspacing=0.2, handletextpad=-0.1,
           fontsize=22 * 0.633)

plt.savefig("test1.png")