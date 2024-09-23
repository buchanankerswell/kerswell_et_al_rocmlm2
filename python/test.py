import os
import numpy as np
import pandas as pd
from gfem import GFEMModel, get_sampleids, build_gfem_models
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main():
    """
    """
    ####################################################################################
    # Building training dataset
    ####################################################################################
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Testing gfem models
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    res, src, samp, gts = 128, "assets/synth-mids.csv", "sm005-loi005", "mor"

    # hp model
    P_min, P_max, T_min, T_max = 0.1, 8.1, 273, 1973
    model_shallow = GFEMModel("hp02", samp, src, res, P_min, P_max, T_min, T_max, gts)
    model_shallow.build_model()
    model_shallow.visualize_model()

    # stx model
    P_min, P_max, T_min, T_max = 8.1, 136.1, 773, 4273
    model_deep = GFEMModel("stx21", samp, src, res, P_min, P_max, T_min, T_max, gts)
    model_deep.build_model()
    model_deep.visualize_model()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GFEM models
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    gfems = []
#    sids = get_sampleids(src)
#
#    # stx models
#    P_min, P_max, T_min, T_max = 8.1, 136.1, 773, 4273
#    gfems.extend(build_gfem_models(src, sids, "stx21", res, P_min, P_max, T_min, T_max, gts))
#
#    # hp models
#    P_min, P_max, T_min, T_max = 0.1, 8.1, 273, 1973
#    gfems.extend(build_gfem_models(src, sids, "hp02", res, P_min, P_max, T_min, T_max, gts))
#
#    # visualize models
#    for m in gfems:
#        m.visualize_model()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Visualize training dataset design
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_training_dataset_design(model_shallow, model_deep):
        """
        """
        res = model_shallow.res
        P_min_sh, P_max_sh = model_shallow.P_min, model_shallow.P_max
        P_min_dp, P_max_dp = model_deep.P_min, model_deep.P_max
        T_min_sh, T_max_sh = model_shallow.T_min, model_shallow.T_max
        T_min_dp, T_max_dp = model_deep.T_min, model_deep.T_max

        T_sh_grid = np.linspace(T_min_sh, T_max_sh, res)
        P_sh_grid = np.linspace(P_min_sh, P_max_sh, res)
        T_sh, P_sh = np.meshgrid(T_sh_grid, P_sh_grid)

        T_dp_grid = np.linspace(T_min_dp, T_max_dp, res)
        P_dp_grid = np.linspace(P_min_dp, P_max_dp, res)
        T_dp, P_dp = np.meshgrid(T_dp_grid, P_dp_grid)

        sub1 = model_shallow._get_subduction_geotherm("Kamchatka")
        sub2 = model_shallow._get_subduction_geotherm("Central_Cascadia", position="top")
        mantle1 = model_deep._get_mantle_geotherm(693)
        mantle2 = model_shallow._get_mantle_geotherm(1773)
        mantle3 = model_deep._get_mantle_geotherm(1773)

        y = np.linspace(P_min_sh, P_max_sh, res + 1)
        x = (y * 35 * 2) + 273

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["font.size"] = 14
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["axes.facecolor"] = "0.9"
        plt.rcParams["legend.frameon"] = "False"
        plt.rcParams["legend.facecolor"] = "0.9"
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["figure.autolayout"] = "True"

        fig, ax = plt.subplots(figsize=(6.3, 5))

        width_sh = T_max_sh - T_min_sh
        height_sh = P_max_sh - P_min_sh
        shallow_rect = Rectangle((T_min_sh, P_min_sh), width_sh, height_sh, linewidth=1,
                                 edgecolor="darkblue", facecolor="none", label="hp02 wet")

        width_dp = T_max_dp - T_min_dp
        height_dp = P_max_dp - P_min_dp
        deep_rect = Rectangle((T_min_dp, P_min_dp), width_dp, height_dp, linewidth=1,
                              edgecolor="darkorange", facecolor="none", label="stx21 dry")

        ax.add_patch(deep_rect)
        ax.add_patch(shallow_rect)
        ax.scatter(T_dp, P_dp, color="darkorange", s=0.1, marker=".")
        ax.scatter(T_sh, P_sh, color="darkblue", s=0.01, marker=".")
        ax.plot(sub1["T"], sub1["P"], color="black", label="subduction")
        ax.plot(sub2["T"], sub2["P"], color="black")
        ax.plot(mantle1["T"], mantle1["P"], color="black", linestyle=":")
        ax.plot(x, y, color="black", linestyle=":", label="forbidden")
        ax.plot(mantle2["T"], mantle2["P"], color="black", linestyle="--",
                label="sub-cont. plume")
        ax.plot(mantle3["T"], mantle3["P"], color="black", linestyle="--")

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (GPa)")

        ax.legend(loc="upper left", bbox_to_anchor=(0.15, -0.03, 1, 1))

        plt.rcParams["font.size"] = 12

        ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower right",
                              bbox_to_anchor=(0, 0.06, 1, 1), bbox_transform=ax.transAxes)
        ax_inset.add_patch(Rectangle((T_min_sh, P_min_sh), width_sh, height_sh, linewidth=1,
                                     edgecolor="darkblue", facecolor="none"))
        ax_inset.scatter(T_sh, P_sh, color="darkblue", s=0.01, marker=".")
        ax_inset.plot(sub1["T"], sub1["P"], color="black")
        ax_inset.plot(sub2["T"], sub2["P"], color="black")
        ax_inset.plot(x, y, color="black", linestyle=":")
        ax_inset.plot(mantle2["T"], mantle2["P"], color="black", linestyle="--")
        ax_inset.set_xlim(T_min_sh, T_max_sh)
        ax_inset.set_ylim(P_min_sh, P_max_sh)

        plt.savefig("test.png")

#    ####################################################################################
#    Test rocmlm inference speed
#    ####################################################################################
#    xi = [1]
#    h2o_in = [1]
#    P = [15]
#    T = [1773]
#
#    print("STX speed")
#    for _ in range(10):
#        pred = mod_st.predict_old(xi, P, T)
#    print("HP speed")
#    for _ in range(10):
#        pred = mod_hp.predict(xi, h2o_in, P, T)
#
#    ####################################################################################
#    End test rocmlm inference speed
#    ####################################################################################

    return None

if __name__ == "__main__":
    main()