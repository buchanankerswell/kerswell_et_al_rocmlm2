import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hymatz import HyMaTZ
from rocmlm import RocMLM, train_rocmlms, visualize_rocmlm_performance, compose_rocmlm_plots
from gfem import (GFEMModel, get_sampleids, build_gfem_models, compose_gfem_plots,
                  visualize_prem_comps)

def main():
    """
    """
#    model = HyMaTZ(1573, "Pyrolite", 50)
#
#    ####################################################################################
#    Building and training rocmlms
#    ####################################################################################
    model = GFEMModel("hp633", f"sm005-loi005", "assets/synth-mids.csv", res=32, P_min=0, P_max=10, T_min=298, T_max=1273)
#    model = GFEMModel("hp633", f"sm014-loi001", "assets/synth-mids.csv", res=32)
#    model.build_model()
#
#    models = []
#    for i in range(0, 8):
#        models.append(GFEMModel("hp633", f"sm000-loi00{i}", "assets/synth-mids.csv"))
#        models.append(GFEMModel("koma06", f"sm000-loi00{i}", "assets/synth-mids.csv"))
#        models.append(GFEMModel("stx21", f"sm000-loi00{i}", "assets/synth-mids.csv"))
#    compose_gfem_plots(models)
#
#    gfems = {}
#    sources = {"m": "assets/synth-mids.csv"}
#    for name, source in sources.items():
#        sids = get_sampleids(source)[:24]
#        gfems[name] = build_gfem_models(source, sids, perplex_db="hp633", res=64)
#    visualize_prem_comps(gfems["m"])
#
#    gfems = {}
#    sources = {"m": "assets/synth-mids.csv", "r": "assets/synth-rnds.csv"}
#    for name, source in sources.items():
#        sids = get_sampleids(source, "all")
#        gfems[name] = build_gfem_models(source, sids)
#    training_data = gfems["m"] + gfems["r"]
#    rocmlms = train_rocmlms(training_data, PT_steps=[1], X_steps=[1])
#    visualize_rocmlm_performance()
#    for model in rocmlms: compose_rocmlm_plots(model)
#
#    ####################################################################################
#    End building and training rocmlms
#    ####################################################################################
#
#    ####################################################################################
#    Visualize iwamori, hymatz, and rocmlm profiles
#    ####################################################################################
#    model = GFEMModel("hp633", f"sm000-loi000", "assets/synth-mids.csv", res=64)
#    ref_models = model._get_1d_reference_models()
#    prem_P = ref_models["prem"]["P"].to_numpy()
#    prem_rho = ref_models["prem"]["rho"].to_numpy()
#    prem_Vp = ref_models["prem"]["Vp"].to_numpy()
#    prem_Vs = ref_models["prem"]["Vs"].to_numpy()
#    stw105_P = ref_models["stw105"]["P"].to_numpy()
#    stw105_rho = ref_models["stw105"]["rho"].to_numpy()
#    stw105_Vp = ref_models["stw105"]["Vp"].to_numpy()
#    stw105_Vs = ref_models["stw105"]["Vs"].to_numpy()
#    iwamori_P, iwamori_T, iwamori_h2o = model._get_1d_profile(iwamori=True)
#    hymatz_P, hymatz_T, hymatz_h2o = model._get_1d_profile(hymatz_input=["Pyrolite", 100], target="h2o")
#    _, _, hymatz_rho = model._get_1d_profile(hymatz_input=["Pyrolite", 100], target="rho")
#    _, _, hymatz_Vp = model._get_1d_profile(hymatz_input=["Pyrolite", 100], target="Vp")
#    _, _, hymatz_Vs = model._get_1d_profile(hymatz_input=["Pyrolite", 100], target="Vs")
#    hymatz50_P, hymatz50_T, hymatz50_h2o = model._get_1d_profile(hymatz_input=["Pyrolite", 50], target="h2o")
#
#    profile = pd.read_csv("assets/hymatz-1573K-Pyrolite-0H2O", sep="\t")
#
#    P = profile["P"].to_numpy()
#    T = profile["T"].to_numpy()
#
#    mod_hp = RocMLM.load_pretrained_model("rocmlms/synthetic-NN3-S248-W65.pkl")
#    mod_st = RocMLM.load_pretrained_model("rocmlms/perp-synthetic-NN3-S129-W129.pkl")
#
#    xi = np.repeat(0.98, len(profile["T"]))
#    h2o_in = np.repeat(0, len(profile["T"]))
#    pred_hp = mod_hp.predict(xi, h2o_in, P, T)
#    pred_st = mod_st.predict_old(xi, P, T)
#    rho_hp, Vp_hp, Vs_hp, h2o_hp, melt_hp = pred_hp.T
#    rho_st, Vp_st, Vs_st = pred_st.T
#
#    xi = np.repeat(0.85, len(profile["T"]))
#    h2o_in = np.repeat(0, len(profile["T"]))
#    pred_hp2 = mod_hp.predict(xi, h2o_in, P, T)
#    pred_st2 = mod_st.predict_old(xi, P, T)
#    rho_hp2, Vp_hp2, Vs_hp2, h2o_hp2, melt_hp2 = pred_hp2.T
#    rho_st2, Vp_st2, Vs_st2 = pred_st2.T
#
#    xi = np.repeat(0.85, len(profile["T"]))
#    h2o_in = np.repeat(1.5, len(profile["T"]))
#    pred_hp3 = mod_hp.predict(xi, h2o_in, P, T)
#    pred_st3 = mod_st.predict_old(xi, P, T)
#    rho_hp3, Vp_hp3, Vs_hp3, h2o_hp3, melt_hp3 = pred_hp3.T
#    rho_st3, Vp_st3, Vs_st3 = pred_st3.T
#
#    fig, axs = plt.subplots(2, 3, figsize=(10, 6.5))
#
#    axs[0, 0].plot(prem_rho, prem_P, label="PREM", color="black")
#    axs[0, 0].plot(stw105_rho, stw105_P, label="stw105", color="black", linestyle="dashed")
#    axs[0, 0].plot(hymatz_rho, hymatz_P, label="hymatz", color="black", linestyle="dotted")
#    axs[0, 0].plot(rho_hp, P, label="xi=0.98, dry")
#    axs[0, 0].plot(rho_hp2, P, label="xi=0.85, dry")
#    axs[0, 0].plot(rho_hp3, P, label="xi=0.85, h2o=1.5")
#    axs[0, 0].set_ylabel("P (GPa)")
#    axs[0, 0].set_title("rho")
#    axs[0, 0].set_xlim(np.min(rho_hp) - 0.05 * np.min(rho_hp), np.max(rho_hp) + 0.05 * np.max(rho_hp))
#    axs[0, 0].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[0, 1].plot(prem_Vp, prem_P, label="PREM", color="black")
#    axs[0, 1].plot(stw105_Vp, stw105_P, label="stw105", color="black", linestyle="dashed")
#    axs[0, 1].plot(hymatz_Vp, hymatz_P, label="hymatz", color="black", linestyle="dotted")
#    axs[0, 1].plot(Vp_hp, P)
#    axs[0, 1].plot(Vp_hp2, P)
#    axs[0, 1].plot(Vp_hp3, P)
#    axs[0, 1].set_ylabel("P (GPa)")
#    axs[0, 1].set_title("Vp")
#    axs[0, 1].set_xlim(np.min(Vp_hp) - 0.05 * np.min(Vp_hp), np.max(Vp_hp) + 0.05 * np.max(Vp_hp))
#    axs[0, 1].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[0, 2].plot(prem_Vs, prem_P, label="PREM", color="black")
#    axs[0, 2].plot(stw105_Vs, stw105_P, label="stw105", color="black", linestyle="dashed")
#    axs[0, 2].plot(hymatz_Vs, hymatz_P, label="hymatz", color="black", linestyle="dotted")
#    axs[0, 2].plot(Vs_hp, P)
#    axs[0, 2].plot(Vs_hp2, P)
#    axs[0, 2].plot(Vs_hp3, P)
#    axs[0, 2].set_ylabel("P (GPa)")
#    axs[0, 2].set_title("Vs")
#    axs[0, 2].set_xlim(np.min(Vs_hp) - 0.05 * np.min(Vs_hp), np.max(Vs_hp) + 0.05 * np.max(Vs_hp))
#    axs[0, 2].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[1, 0].plot(iwamori_h2o, iwamori_P, label="Iwamori", color="black", linestyle="dashdot")
#    axs[1, 0].plot(hymatz_h2o, hymatz_P, label="hymatz100", color="black", linestyle="dotted")
#    axs[1, 0].plot(hymatz50_h2o, hymatz50_P, label="hymatz50", color="black", linestyle=(0, (1,1)))
#    axs[1, 0].plot(h2o_hp, P)
#    axs[1, 0].plot(h2o_hp2, P)
#    axs[1, 0].plot(h2o_hp3, P)
#    axs[1, 0].set_ylabel("P (GPa)")
#    axs[1, 0].set_title("h2o")
#    axs[1, 0].set_xlim(np.min(h2o_hp) - 0.1 * np.min(h2o_hp), np.max(hymatz_h2o) + 0.05 * np.max(hymatz_h2o))
#    axs[1, 0].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[1, 1].plot(melt_hp, P)
#    axs[1, 1].plot(melt_hp2, P)
#    axs[1, 1].plot(melt_hp3, P)
#    axs[1, 1].set_ylabel("P (GPa)")
#    axs[1, 1].set_title("melt")
#    axs[1, 1].set_xlim(np.min(melt_hp) - 0.1 * np.min(melt_hp), np.max(melt_hp) + 0.05 * np.max(melt_hp))
#    axs[1, 1].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[1, 2].axis("off")
#
#    handles_00, labels_00 = axs[0, 0].get_legend_handles_labels()
#    handles_10, labels_10 = axs[1, 0].get_legend_handles_labels()
#    handles = handles_00 + handles_10
#    labels = labels_00 + labels_10
#    axs[1, 2].legend(handles, labels, loc="center")
#    plt.tight_layout()
#    plt.savefig("test1.png")
#    plt.close()
#
#    thresh = 20
#    fig, axs = plt.subplots(2, 3, figsize=(10, 6.5))
#
#    axs[0, 0].plot(prem_rho, prem_P, label="PREM", color="black")
#    axs[0, 0].plot(stw105_rho, stw105_P, label="stw105", color="black", linestyle="dashed")
#    axs[0, 0].plot(hymatz_rho, hymatz_P, label="hymatz", color="black", linestyle="dotted")
#    axs[0, 0].plot(rho_hp3, P)
#    axs[0, 0].plot(rho_st2, P)
#    axs[0, 0].set_ylabel("P (GPa)")
#    axs[0, 0].set_title("rho")
#    axs[0, 0].set_xlim(np.min(rho_hp) - 0.05 * np.min(rho_hp), np.max(rho_hp) + 0.05 * np.max(rho_hp))
#    axs[0, 0].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[0, 1].plot(prem_Vp, prem_P, label="PREM", color="black")
#    axs[0, 1].plot(stw105_Vp, stw105_P, label="stw105", color="black", linestyle="dashed")
#    axs[0, 1].plot(hymatz_Vp, hymatz_P, label="hymatz", color="black", linestyle="dotted")
#    axs[0, 1].plot(Vp_hp3, P)
#    axs[0, 1].plot(Vp_st2, P)
#    axs[0, 1].set_ylabel("P (GPa)")
#    axs[0, 1].set_title("Vp")
#    axs[0, 1].set_xlim(np.min(Vp_hp) - 0.05 * np.min(Vp_hp), np.max(Vp_hp) + 0.05 * np.max(Vp_hp))
#    axs[0, 1].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[0, 2].plot(prem_Vs, prem_P, label="PREM", color="black")
#    axs[0, 2].plot(stw105_Vs, stw105_P, label="stw105", color="black", linestyle="dashed")
#    axs[0, 2].plot(hymatz_Vs, hymatz_P, label="hymatz", color="black", linestyle="dotted")
#    axs[0, 2].plot(Vs_hp3, P)
#    axs[0, 2].plot(Vs_st2, P)
#    axs[0, 2].set_ylabel("P (GPa)")
#    axs[0, 2].set_title("Vs")
#    axs[0, 2].set_xlim(np.min(Vs_hp) - 0.05 * np.min(Vs_hp), np.max(Vs_hp) + 0.05 * np.max(Vs_hp))
#    axs[0, 2].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[1, 0].plot(rho_hp3, P, alpha=0.2, color="darkblue")
#    axs[1, 0].plot(rho_hp3[P < thresh], P[P < thresh])
#    axs[1, 0].plot(rho_st2, P, alpha=0.2, color="darkorange")
#    axs[1, 0].plot(rho_st2[P > thresh], P[P > thresh])
#    axs[1, 0].set_ylabel("P (GPa)")
#    axs[1, 0].set_title("rho")
#    axs[1, 0].set_xlim(np.min(rho_hp) - 0.05 * np.min(rho_hp), np.max(rho_hp) + 0.05 * np.max(rho_hp))
#    axs[1, 0].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[1, 1].plot(Vp_hp3, P, alpha=0.2, color="darkblue")
#    axs[1, 1].plot(Vp_hp3[P < thresh], P[P < thresh])
#    axs[1, 1].plot(Vp_st2, P, alpha=0.2, color="darkorange")
#    axs[1, 1].plot(Vp_st2[P > thresh], P[P > thresh])
#    axs[1, 1].set_ylabel("P (GPa)")
#    axs[1, 1].set_title("Vp")
#    axs[1, 1].set_xlim(np.min(Vp_hp) - 0.05 * np.min(Vp_hp), np.max(Vp_hp) + 0.05 * np.max(Vp_hp))
#    axs[1, 1].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    axs[1, 2].plot(Vs_hp3, P, alpha=0.2, color="darkblue")
#    axs[1, 2].plot(Vs_hp3[P < thresh], P[P < thresh], label="hp633, xi=0.85, h2o=1.5")
#    axs[1, 2].plot(Vs_st2, P, alpha=0.2, color="darkorange")
#    axs[1, 2].plot(Vs_st2[P > thresh], P[P > thresh], label="stx21, xi=0.85, dry")
#    axs[1, 2].set_ylabel("P (GPa)")
#    axs[1, 2].set_title("Vs")
#    axs[1, 2].set_xlim(np.min(Vs_hp) - 0.05 * np.min(Vs_hp), np.max(Vs_hp) + 0.05 * np.max(Vs_hp))
#    axs[1, 2].set_ylim(np.min(P) - 0.05 * np.min(P), np.max(P) + 0.05 * np.max(P))
#
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig("test2.png")
#    plt.close()
#
#    # Load CSV file
#    df = pd.read_csv("~/Downloads/IwamoriH2O.csv")
#    df["h2o"] = df["h2o"] / 0.3
#
#    # Extract unique values for P and T to define grid dimensions
#    p_unique = np.sort(df["P"].unique())
#    t_unique = np.sort(df["T"].unique())
#
#    # Initialize an empty grid
#    grid = np.empty((len(t_unique), len(p_unique)))
#
#    # Fill the grid with h2o values
#    for _, row in df.iterrows():
#        p_index = np.where(p_unique == row["P"])[0][0]
#        t_index = np.where(t_unique == row["T"])[0][0]
#        grid[t_index, p_index] = row["h2o"]
#
#    plt.imshow(grid.T, cmap="Blues", origin="lower", aspect="auto",
#               extent=[t_unique.min(), t_unique.max(), p_unique.min(), p_unique.max()])
#    plt.colorbar(label="h2o")
#    plt.xlabel("T")
#    plt.ylabel("P")
#    plt.savefig("iwamori.png")
#    plt.close()
#
#    ####################################################################################
#    End visualize iwamori, hymatz, and rocmlm profiles
#    ####################################################################################
#
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