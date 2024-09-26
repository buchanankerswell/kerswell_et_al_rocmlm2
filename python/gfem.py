#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import io
import os
import time
import glob
import shutil
import textwrap
import traceback
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures as cf
from contextlib import redirect_stdout
from scipy.interpolate import LinearNDInterpolator

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#######################################################
## .1.              GFEMModel class              !!! ##
#######################################################
class GFEMModel:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, perplex_db, sid, source, res=128, P_min=1, P_max=28, T_min=773,
                 T_max=2273, verbose=1):
        """
        """
        # Input
        self.res = res
        self.sid = sid
        self.P_min = P_min
        self.P_max = P_max
        self.T_min = T_min
        self.T_max = T_max
        self.source = source
        self.verbose = verbose

        # Global options
        self.seed = 42
        self.digits = 3

        # Mantle potential T and subduction segments for depth profiles
        self.pot_Ts = [1173, 1573, 1773]
        self.segs = ["Central_Cascadia", "Kamchatka"]

        # Check perplex db
        if perplex_db not in ["hp02", "hp11", "hp622", "hp633", "stx21"]:
            print("Unrecognized thermodynamic dataset ! Defaulting to hp02 ...")
            self.perplex_db = "hp02"
        else:
            self.perplex_db = perplex_db

        # Configure perplex
        dataset_configs = {
            "hp02": {
                "T_melt": 1100,
                "melt_mod": "melt(HGPH)",
                "em_exclude": ["anL", "enL", "foL", "fo8L", "foHL", "diL", "woGL", "liz",
                               "ak", "pswo", "wo"],
                "sl_include": ["O(HGP)", "Cpx(HGP)", "Omph(GHP)", "Opx(HGP)", "Sp(HP)",
                               "Gt(HGP)", "Maj", "feldspar", "cAmph(G)", "Chl(W)",
                               "Atg(PN)", "A-phase", "B", "T", "melt(HGPH)"]
            },
            "hp11": {
                "T_melt": 1100,
                "melt_mod": "melt(HGPH)",
                "em_exclude": ["foWL", "fojL", "foL", "fa8L", "faTL", "foTL", "perL", "neL",
                               "fo8L", "diL", "dijL", "abL", "jdjL", "enL", "naph", "prl",
                               "liz", "ne", "anl", "tap", "cg", "hen", "cen", "glt", "cgh",
                               "dsp", "fctd"],
                "sl_include": ["O(HGP)", "Ring", "Wus", "Cpx(HGP)", "Omph(GHP)", "Opx(HGP)",
                               "Sp(HGP)", "Gt(HGP)", "Maj", "feldspar", "cAmph(G)",
                               "Chl(W)", "Atg(PN)", "A-phase", "B", "T", "Anth",
                               "melt(HGPH)"]
            },
            "stx21": {
                "T_melt": None,
                "melt_mod": None,
                "em_exclude": ["ca-pv"],
                "sl_include": ["C2/c", "Wus", "Pv", "Pl", "Sp", "O", "Wad", "Ring", "Opx",
                               "Cpx", "Aki", "Gt", "Ppv", "CF", "NaAl"]}
        }
        dataset_configs["hp622"] = dataset_configs["hp11"]
        dataset_configs["hp633"] = dataset_configs["hp11"]

        # Load Perple_X config
        config = dataset_configs[self.perplex_db]
        self.T_melt = config["T_melt"]
        self.melt_mod = config["melt_mod"]
        self.em_exclude = sorted(config["em_exclude"])
        self.sl_include = sorted(config["sl_include"])

        # Global Perple_X options
        self.ox_gfem = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "NA2O", "LOI"]
        self.melt_is_fluid = "T" # Treat melts as fluids ?
        self.fluid_properties = "N" # (N: rock properties without melt fraction)
        self.fluid_assemblages = "Y" # (Y: assemblages with melt and H2O fraction)

        # Werami output map
        self.werami_output_map = {
            "T(K)":                "T",
            "P(bar)":              "P",
            "N,g":                 "mass",
            "n,mol":               "moles",
            "rho,kg/m3":           "density",
           f"{self.melt_mod},vo%": "melt_fraction",
            "alpha,1/K":           "expansivity",
            "beta,1/bar":          "compressibility",
            "H,J/mol":             "molar_enthalpy",
            "S,J/K/mol":           "molar_entropy",
            "V,J/bar/mol":         "molar_volume",
            "cp,J/K/mol":          "molar_heat_capacity",
            "cp/cv":               "heat_capacity_ratio",
            "vp,km/s":             "Vp",
            "vp_P":                "Vp_dP",
           r"vp_{T}":              "Vp_dT",
            "vs,km/s":             "Vs",
           r"vs_{P}":              "Vs_dP",
           r"vs_{T}":              "Vs_dT",
            "v0,km/s":             "sound_velocity",
           r"v0_{P}":              "sound_velocity_dP",
           r"v0_{T}":              "sound_velocity_dT",
            "vp/vs":               "Vp/Vs",
            "Ks,bar":              "bulk_modulus",
           r"Ks_{P}":              "bulk_modulus_dP",
           r"Ks_{T},bar/K":        "bulk_modulus_dT",
            "Gs,bar":              "shear_modulus",
            "Gs_P":                "shear_modulus_dP",
           r"Gs_{T},bar/K":        "shear_modulus_dT",
            "G,J/mol":             "molar_gibbs_free_energy",
            "Gruneisen_T":         "gruneisen_thermal_ratio",
            "assemblage_i":        "assemblage_index"
        }

        # Add oxides to werami_output_map
        for ox in self.ox_gfem:
            if ox == "LOI": ox = "H2O"
            self.werami_output_map[f"{ox},wt%"] = ox

        self.target_units_map = {
            "T":                         "K",
            "P":                         "bar",
            "mass":                      "g",
            "moles":                     "mol",
            "density":                   "kg/m3",
            "melt_fraction":             "vol%",
            "expansivity":               "1/K",
            "compressibility":           "1/bar",
            "molar_enthalpy":            "J/mol",
            "molar_entropy":             "J/K/mol",
            "molar_volume":              "J/bar/mol",
            "molar_heat_capacity":       "J/K/mol",
            "heat_capacity_ratio":       "",
            "Vp":                        "km/s",
            "Vp_dP":                     "km/s/bar",
            "Vp_dT":                     "km/s/K",
            "Vs":                        "km/s",
            "Vs_dP":                     "km/s/bar",
            "Vs_dT":                     "km/s/K",
            "sound_velocity":            "km/s",
            "sound_velocity_dP":         "km/s/bar",
            "sound_velocity_dT":         "km/s/K",
            "Vp/Vs":                     "",
            "bulk_modulus":              "bar",
            "bulk_modulus_dP":           "",
            "bulk_modulus_dT":           "bar/K",
            "shear_modulus":             "bar",
            "shear_modulus_dP":          "",
            "shear_modulus_dT":          "bar/K",
            "molar_gibbs_free_energy":   "kg/m3",
            "gruneisen_thermal_ratio":   "",
            "assemblage_index":          "",
            "phase_assemblage_variance": ""
        }

        # Add oxides to target_units_map
        for ox in self.ox_gfem:
            if ox == "LOI": ox = "H2O"
            self.target_units_map[ox] = "wt.%"

        self.target_digits_map = {
            "T":                         "%.1f",
            "P":                         "%.1f",
            "mass":                      "%.1f",
            "moles":                     "%.1f",
            "density":                   "%.1f",
            "melt_fraction":             "%.0f",
            "expansivity":               "%.1f",
            "compressibility":           "%.1f",
            "molar_enthalpy":            "%.1f",
            "molar_entropy":             "%.0f",
            "molar_volume":              "%.1f",
            "molar_heat_capacity":       "%.0f",
            "heat_capacity_ratio":       "%.1f",
            "Vp":                        "%.1f",
            "Vp_dP":                     "%.1f",
            "Vp_dT":                     "%.1f",
            "Vs":                        "%.1f",
            "Vs_dP":                     "%.1f",
            "Vs_dT":                     "%.1f",
            "sound_velocity":            "%.1f",
            "sound_velocity_dP":         "%.1f",
            "sound_velocity_dT":         "%.1f",
            "Vp/Vs":                     "%.1f",
            "bulk_modulus":              "%.1f",
            "bulk_modulus_dP":           "%.1f",
            "bulk_modulus_dT":           "%.1f",
            "shear_modulus":             "%.1f",
            "shear_modulus_dP":          "%.1f",
            "shear_modulus_dT":          "%.1f",
            "molar_gibbs_free_energy":   "%.1f",
            "gruneisen_thermal_ratio":   "%.1f",
            "assemblage_index":          "%.0f",
            "phase_assemblage_variance": "%.0f"
        }

        # Add oxides to target_digits_map
        for ox in self.ox_gfem:
            if ox == "LOI": ox = "H2O"
            self.target_digits_map[ox] = "%.1f"

        self.target_labels_map = {
            "T":                         "T",
            "P":                         "P",
            "mass":                      "Mass",
            "moles":                     "Moles",
            "density":                   "Density",
            "melt_fraction":             "Melt",
            "expansivity":               "Expansivity",
            "compressibility":           "Compressibility",
            "molar_enthalpy":            "Enthalpy",
            "molar_entropy":             "Entropy",
            "molar_volume":              "Volume",
            "molar_heat_capacity":       "Cp",
            "heat_capacity_ratio":       "Cp/Cv",
            "Vp":                        "Vp",
            "Vp_dP":                     "dVp/dP",
            "Vp_dT":                     "dVp/dT",
            "Vs":                        "Vs",
            "Vs_dP":                     "dVs/dP",
            "Vs_dT":                     "dVs/dT",
            "sound_velocity":            "V0",
            "sound_velocity_dP":         "dV0/dP",
            "sound_velocity_dT":         "dV0/dT",
            "Vp/Vs":                     "Vp/Vs",
            "bulk_modulus":              "Ks",
            "bulk_modulus_dP":           "dKs/dP",
            "bulk_modulus_dT":           "dKs/dT",
            "shear_modulus":             "Gs",
            "shear_modulus_dP":          "dGs/dP",
            "shear_modulus_dT":          "dGs/dT",
            "molar_gibbs_free_energy":   "G",
            "gruneisen_thermal_ratio":   "Gruneisen",
            "assemblage_index":          "Assemblage Index",
            "phase_assemblage_variance": "Assemblage Variance"
        }

        # Add oxides to target_labels_map
        for ox in self.ox_gfem:
            if ox == "LOI": ox = "H2O"
            self.target_labels_map[ox] = ox

        # GFEM target variables
        self.targets = [target for target in self.target_labels_map.keys()
                        if target not in ["P", "T"]]
        self.targets_to_visualize = ["density", "Vp", "Vs", "melt_fraction", "H2O",
                                     "molar_entropy", "molar_volume", "molar_heat_capacity",
                                     "assemblage_index", "phase_assemblage_variance"]

        # Perplex dirs and filepaths
        self.model_out_dir = f"gfems/{self.sid}_{self.perplex_db}_{self.res}"

        # Output file paths
        self.data_dir = "assets"
        self.fig_dir = f"figs/{self.model_out_dir}"
        self.log_file = f"{self.model_out_dir}/log-{self.sid}"

        # Results
        self.xi = None
        self.loi = None
        self.results = {}
        self.features = []
        self.sample_features = []
        self.norm_sample_comp = []
        self.pt_array = np.array([])
        self.target_array = np.array([])

        # Errors
        self.model_error = None
        self.model_built = False
        self.timeout = (res**2) * 3
        self.model_build_error = False

        self._get_normalized_sample_comp()
        self._get_sample_features()
        self._check_existing_model()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get normalized sample composition !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_normalized_sample_comp(self):
        """
        """
        # Get self attributes
        sid = self.sid
        source = self.source
        digits = self.digits
        ox_gfem = self.ox_gfem

        try:
            # Get oxides from PCA
            df = pd.read_csv(source)
            start_idx = df.columns.get_loc("SAMPLEID") + 1
            end_idx = df.columns.get_loc("PC1")
            ox_pca = df.columns[start_idx:end_idx]

            # Check input
            if len(ox_gfem) > len(ox_pca):
                raise Exception("Not enough oxides in PCA to satisfy ox_gfem !")

            # Define the oxide order for perplex
            ox_order = ["K2O", "NA2O", "CAO", "FEO", "MGO", "AL2O3", "SIO2", "TIO2", "CR2O3"]

            # Create a mapping from element to index based on ox_order
            ox_mapping = {oxide: idx for idx, oxide in enumerate(ox_order)}

            # Reorder ox_gfem to match perplex input order
            ox_pca = sorted(ox_pca, key=lambda x: ox_mapping.get(x, float("inf")))
            ox_gfem = sorted(ox_gfem, key=lambda x: ox_mapping.get(x, float("inf")))
            self.ox_gfem = ox_gfem

            # Get sample data from PCA
            subset_df = df[df["SAMPLEID"] == sid]

            if subset_df.empty:
                raise Exception("Sample name not found in the dataset !")

            # Get Fertility Index and LOI
            self.loi = float(subset_df["LOI"].values[0])
            self.xi = float(subset_df["XI_FRAC"].values[0])

            # Get sample composition
            sample_comp = []

            for oxide in ox_pca:
                if oxide in subset_df.columns and pd.notnull(subset_df[oxide].iloc[0]):
                    sample_comp.append(float(subset_df[oxide].iloc[0]))
                else:
                    sample_comp.append(0)

            # No normalizing if all components in PCA are used for perplex input
            if len(sample_comp) == len(ox_gfem):
                self.norm_sample_comp = sample_comp
                return sample_comp

            # Create a mapping from element to index based on ox_order
            pca_mapping = {oxide: idx for idx, oxide in enumerate(ox_pca)}

            # Filter components and set negative concentrations to zero
            sub_comp = [sample_comp[pca_mapping[o]] for o in ox_gfem if o in pca_mapping]
            sub_comp = [c if c >= 0 else 0 for c in sub_comp]

            # Get total oxides excluding LOI
            tot_comp = sum([c for c, o in zip(sub_comp, ox_gfem) if c != 0 and o != "LOI"])

            # Normalize excluding LOI
            norm_comp = [
                round(((c / tot_comp) * 100 if c != 0 and o != "LOI" else c), digits)
                for c, o in zip(sub_comp, ox_gfem)]

            # Check input
            if len(norm_comp) != len(ox_gfem):
                raise Exception("Normalized sample has incorrect number of oxides !")

            self.norm_sample_comp = norm_comp

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_normalized_sample_comp() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get sample features !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_sample_features(self):
        """
        """
        # Get self attributes
        sid = self.sid
        source = self.source

        try:
            # Read the data file
            df = pd.read_csv(source)

            # Get sample features names
            features = [col for col in df.columns if col != "SAMPLEID"]
            self.features = features

            # Subset the df based on the sample name
            subset_df = df[df["SAMPLEID"] == sid]

            if subset_df.empty:
                raise Exception(f"Sample {sid} not found in the dataset {source}!")

            # Get features for selected sample
            sample_features = subset_df[features].values.flatten().tolist()

            self.sample_features = sample_features

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_sample_features() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check existing model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_existing_model(self):
        # Get self attributes
        sid = self.sid
        verbose = self.verbose
        perplex_db = self.perplex_db
        model_out_dir = self.model_out_dir

        if os.path.exists(model_out_dir):
            if (os.path.exists(f"{model_out_dir}/results.csv") and
                os.path.exists(f"{model_out_dir}/assemblages.csv")):
                self.model_built = True
                if verbose >= 1:
                    print(f"  Found {perplex_db} GFEM model for sample {sid} !")
                try:
                    self._get_results()
                    self._get_target_array()
                    self._get_pt_array()
                except Exception as e:
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"!!! ERROR in _check_existing_model() !!!")
                    print(f"{e}")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    traceback.print_exc()
                    return None
            else:
                shutil.rmtree(model_out_dir)
                os.makedirs(model_out_dir, exist_ok=True)
        else:
            os.makedirs(model_out_dir, exist_ok=True)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get subduction geotherm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_subduction_geotherm(self, segment="Central_Cascadia", slab_position="slabmoho"):
        """
        """
        # Get self attributes
        P_min = self.P_min
        P_max = self.P_max
        T_min = self.T_min
        T_max = self.T_max
        data_dir = f"{self.data_dir}/D80"

        try:
            if not os.path.exists(data_dir):
                raise Exception(f"Data not found at {data_dir} !")

            # Reference geotherm paths
            path = f"{data_dir}/{segment}.txt"

            # Define column headers
            ref_cols = ["slab_depth", "unk", "depth", "T"]
            columns_to_keep = ["P", "T"]

            # Load reference geotherm
            if not os.path.exists(path):
                raise Exception(f"Subduction geotherm {segment} not found at {path} !")

            # Read reference geotherm
            geotherm = pd.read_csv(path, header=None, names=ref_cols, sep=r"\s+")

            if slab_position == "slabmoho":
                geotherm = geotherm[geotherm["slab_depth"] == 7]
            elif slab_position == "slabtop":
                geotherm = geotherm[geotherm["slab_depth"] == 0]
            else:
                raise Exception(f"Unrecognized position argument '{position}' !")

            # Truncate top of geotherm
            geotherm = geotherm[geotherm["depth"] < 240]

            # Calculate P from depth
            litho_P_gradient = 35 # (km/GPa)
            geotherm["P"] = geotherm["depth"] / litho_P_gradient

            # Transform units
            geotherm["T"] = geotherm["T"] + 273
            geotherm.sort_values(by=["P"], inplace=True)

            # Cropping profile to same length as GFEM
            geotherm = geotherm[(geotherm["P"] >= P_min) & (geotherm["P"] <= P_max)]
            geotherm = geotherm[(geotherm["T"] >= T_min) & (geotherm["T"] <= T_max)]

            # Clean up df
            geotherm = geotherm[columns_to_keep]
            geotherm = geotherm.round(3)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_subduction_geotherm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return geotherm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get mantle geotherm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_mantle_geotherm(self, mantle_potential=1573, Qs=55e-3, Ts=273, A1=1.0e-6,
                             A2=2.2e-8, k1=2.3, k2=3.0, mantle_adiabat=0.5e-3,
                             crust_thickness=35e3, litho_thickness=150e3):
        """
        """
        # Get self attributes
        res = self.res
        P_min = self.P_min
        P_max = self.P_max

        try:
            # Define array size
            array_size = (res + 1) * 10

            # 1D layered lithospheric cooling model parameters
            # Qs:                                  Surface heat flux (W/m2)
            # Ts:                                  Surface T (K)
            # A1:                                  Layer 1 heat production (W/m3)
            # A2:                                  Layer 2 heat production (W/m3)
            # k1:                                  Layer 1 thermal conductivity (W/mK)
            # k2:                                  Layer 2 thermal conductivity (W/mK)
            # mantle_adiabat:                      Mantle adiabatic gradient (K/m)
            # crust_thickness:                     Crustal thickness (m)
            # litho_thickness:                     Lithoshperic thickness (m)
            litho_P_gradient = 35e3              # Lithostatic P gradient (m/GPa)
            Z_min = P_min * litho_P_gradient     # Min depth (m)
            Z_max = P_max * litho_P_gradient     # Max depth (m)

            # Initialize depth and temperature arrays
            z = np.linspace(Z_min, Z_max, array_size)
            T_geotherm = np.zeros(array_size)

            # Layer1 (crust)
            # A1 Radiogenic heat production (W/m^3)
            # k1 Thermal conductivity (W/mK)
            D1 = crust_thickness

            # Layer2 (lithospheric mantle)
            # A2 Radiogenic heat production (W/m^3)
            # k2 Thermal conductivity (W/mK)
            D2 = litho_thickness

            # Calculate heat flow at the top of each layer
            Qt2 = Qs - (A1 * D1)
            Qt1 = Qs

            # Calculate T at the top of each layer
            Tt1 = Ts
            Tt2 = Tt1 + (Qt1 * D1 / k1) - (A1 / 2 / k1 * D1**2)
            Tt3 = Tt2 + (Qt2 * D2 / k2) - (A2 / 2 / k2 * D2**2)

            # Calculate T within each layer
            for j in range(array_size):
                potential_temp = mantle_potential + mantle_adiabat * z[j]
                if z[j] <= D1:
                    T_geotherm[j] = Tt1 + (Qt1 / k1 * z[j]) - (A1 / (2 * k1) * z[j]**2)
                    if T_geotherm[j] >= potential_temp:
                        T_geotherm[j] = potential_temp
                elif D1 < z[j] <= D2 + D1:
                    T_geotherm[j] = Tt2 + (Qt2 / k2 * (z[j] - D1)) - (A2 / (2 * k2) *
                                                                      (z[j] - D1)**2)
                    if T_geotherm[j] >= potential_temp:
                        T_geotherm[j] = potential_temp
                elif z[j] > D2 + D1:
                    T_geotherm[j] = Tt3 + mantle_adiabat * (z[j] - D1 - D2)
                    if T_geotherm[j] >= potential_temp:
                        T_geotherm[j] = potential_temp

            P_geotherm = z / litho_P_gradient

            geotherm = pd.DataFrame(
                {"P": P_geotherm, "T": T_geotherm}).sort_values(by=["P", "T"])

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_mantle_geotherm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return geotherm

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.1.          Perple_X Functions             !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # write perplex config !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_config(self):
        """
        """
        # Get self attributes
        sid = self.sid
        res = self.res
        segs = self.segs
        P_min = self.P_min
        T_melt = self.T_melt
        pot_Ts = self.pot_Ts
        ox_gfem = self.ox_gfem
        melt_mod = self.melt_mod
        data_dir = self.data_dir
        perplex_db = self.perplex_db
        melt_is_fluid = self.melt_is_fluid
        model_out_dir = self.model_out_dir
        T_min, T_max = self.T_min, self.T_max
        sl_include = "\n".join(self.sl_include)
        em_exclude = "\n".join(self.em_exclude)
        fluid_properties = self.fluid_properties
        norm_sample_comp = self.norm_sample_comp
        fluid_assemblages = self.fluid_assemblages
        P_min, P_max = self.P_min * 1e4, self.P_max * 1e4
        norm_sample_comp = " ".join(map(str, norm_sample_comp))

        # Build options
        # https://www.perplex.ethz.ch/perplex_options.html
        o = (f"composition_system     wt\n"
             f"composition_phase      wt\n"
             f"intermediate_savdyn    T\n"
             f"intermediate_savrpc    T\n"
             f"warn_no_limit          F\n"
             f"grid_levels            1 1\n"
             f"x_nodes                {int(res / 4)} {res + 1}\n"
             f"y_nodes                {int(res / 4)} {res + 1}\n"
             f"bounds                 VRH\n"
             f"vrh/hs_weighting       0.5\n"
             f"Anderson-Gruneisen     F\n"
             f"explicit_bulk_modulus  T\n"
             f"melt_is_fluid          {melt_is_fluid}\n"
             f"T_melt                 {T_melt}\n"
             f"poisson_test           F\n"
             f"poisson_ratio          on 0.31\n"
             f"seismic_output         some\n"
             f"auto_refine_file       F\n"
             f"seismic_data_file      F\n")

        # Write build options
        with open(f"{model_out_dir}/build-options", "w") as file:
            file.write(o)

        # Write plot options
        # https://www.perplex.ethz.ch/perplex_PLOT_options.html
        with open(f"{model_out_dir}/perplex_plot_option.dat", "w") as file:
            file.write("numeric_field_label T")

        # Write vertex minimize
        with open(f"{model_out_dir}/vertex-minimize", "w") as file:
            file.write(f"{sid}")

        # Write pssect-draw
        with open(f"{model_out_dir}/pssect-draw", "w") as file:
            file.write(f"{sid}\nN")

        # Oxides string
        oxides = []
        if "K2O" in ox_gfem:
            oxides.append("K2O")
        if "NA2O" in ox_gfem:
            oxides.append("Na2O")
        if "CAO" in ox_gfem:
            oxides.append("CaO")
        if "FEO" in ox_gfem:
            oxides.append("FeO")
        if "MGO" in ox_gfem:
            oxides.append("MgO")
        if "AL2O3" in ox_gfem:
            oxides.append("Al2O3")
        if "SIO2" in ox_gfem:
            oxides.append("SiO2")
        if "TIO2" in ox_gfem:
            oxides.append("TiO2")
        if "CR2O3" in ox_gfem:
            oxides.append("Cr2O3")
        if "LOI" in ox_gfem:
            oxides.append("H2O")
        oxides_string = "\n".join(oxides)

        if perplex_db not in ["hp622", "hp633"]:
            oxides_string = oxides_string.upper()

        if perplex_db in ["hp11", "hp622", "hp633"]:
            # Build
            b = (f"{sid}\n"               # Proj name
                 f"td-data\n"             # Thermodynamic data file
                 f"build-options\n"       # Build options file
                 f"N\n"                   # Transform components ?
                 f"2\n"                   # Computational mode (2: Constrained 2D grid)
                 f"N\n"                   # Calculations with saturated fluid ?
                 f"N\n"                   # Use chemical potentials as ind variables ?
                 f"N\n"                   # Calculations with saturated components ?
                 )
            b += (oxides_string + "\n")   # Select components (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"5\n"                  # Select fluid EOS (5: H2O-CO2 CORK H&P 91, 98)
                  f"N\n"                  # Calculate along a geotherm ?
                  f"2\n"                  # X-axis variable (2: T(K))
                  f"{T_min} {T_max}\n"    # Enter min and max T(K)
                  f"{P_min} {P_max}\n"    # Enter min and max P(bar)
                  f"Y\n"                  # Specify component amounts by mass ?
                  f"{norm_sample_comp}\n" # Enter mass amounts of components
                  f"N\n"                  # Print output file ?
                  f"Y\n"                  # Exclude pure and/or endmember phases ?
                  f"N\n"                  # Prompt for phases ?
                  )
            b += (em_exclude + "\n")      # Enter names (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"Y\n"                  # Include solution models ?
                  f"solution-models\n"    # Solution model file
                  )
            b += (sl_include + "\n")      # Enter names (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"{sid}\n"              # Calculation title
                  )
            # Copy thermodynamic data
            if perplex_db == "hp11":
                shutil.copy(f"{data_dir}/hp11-td", f"{model_out_dir}/td-data")
            elif perplex_db == "hp622":
                shutil.copy(f"{data_dir}/hp622-td", f"{model_out_dir}/td-data")
            elif perplex_db == "hp633":
                shutil.copy(f"{data_dir}/hp633-td", f"{model_out_dir}/td-data")
            shutil.copy(f"{data_dir}/hp-sl", f"{model_out_dir}/solution-models")
        elif perplex_db == "hp02":
            # Build
            b = (f"{sid}\n"               # Proj name
                 f"td-data\n"             # Thermodynamic data file
                 f"build-options\n"       # Build options file
                 f"N\n"                   # Transform components ?
                 f"2\n"                   # Computational mode (2: Constrained 2D grid)
                 f"N\n"                   # Calculations with saturated fluid ?
                 f"N\n"                   # Calculations with saturated components ?
                 f"N\n"                   # Use chemical potentials as ind variables ?
                 )
            b += (oxides_string + "\n")   # Select components (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"5\n"                  # Select fluid EOS (5: H2O-CO2 CORK H&P 91, 98)
                  f"N\n"                  # Calculate along a geotherm ?
                  f"2\n"                  # X-axis variable (2: T(K))
                  f"{T_min} {T_max}\n"    # Enter min and max T(K)
                  f"{P_min} {P_max}\n"    # Enter min and max P(bar)
                  f"Y\n"                  # Specify component amounts by mass ?
                  f"{norm_sample_comp}\n" # Enter mass amounts of components
                  f"N\n"                  # Print output file ?
                  f"Y\n"                  # Exclude pure and/or endmember phases ?
                  f"N\n"                  # Prompt for phases ?
                  )
            b += (em_exclude + "\n")      # Enter names (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"Y\n"                  # Include solution models ?
                  f"solution-models\n"    # Solution model file
                  )
            b += (sl_include + "\n")      # Enter names (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"{sid}\n"              # Calculation title
                  )
            # Copy thermodynamic data
            shutil.copy(f"{data_dir}/hp02-td", f"{model_out_dir}/td-data")
            shutil.copy(f"{data_dir}/hp-sl", f"{model_out_dir}/solution-models")
        elif perplex_db == "stx21":
            # Build
            b = (f"{sid}\n"               # Proj name
                 f"td-data\n"             # Thermodynamic data file
                 f"build-options\n"       # Build options file
                 f"N\n"                   # Transform components ?
                 f"2\n"                   # Computational mode (2: Constrained 2D grid)
                 f"N\n"                   # Calculations with saturated fluid ?
                 f"N\n"                   # Calculations with saturated components ?
                 )
            b += (oxides_string + "\n")   # Select components (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"N\n"                  # Calculate along a geotherm ?
                  f"2\n"                  # X-axis variable (2: T(K))
                  f"{T_min} {T_max}\n"    # Enter min and max T(K)
                  f"{P_min} {P_max}\n"    # Enter min and max P(bar)
                  f"Y\n"                  # Specify component amounts by mass ?
                  f"{norm_sample_comp}\n" # Enter mass amounts of components
                  f"N\n"                  # Print output file ?
                  f"Y\n"                  # Exclude pure and/or endmember phases ?
                  f"N\n"                  # Prompt for phases ?
                  )
            b += (em_exclude + "\n")      # Enter names (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"Y\n"                  # Include solution models ?
                  f"solution-models\n"    # Solution model file
                  )
            b += (sl_include + "\n")      # Enter names (1 per line)
            b += (f"\n"                   # Enter to finish
                  f"{sid}\n"              # Calculation title
                  )
            # Copy thermodynamic data
            shutil.copy(f"{data_dir}/stx21-td", f"{model_out_dir}/td-data")
            shutil.copy(f"{data_dir}/stx21-sl", f"{model_out_dir}/solution-models")
        else:
            raise Exception("Unrecognized thermodynamic dataset !")

        # Write build config
        with open(f"{model_out_dir}/build-config", "w") as file:
            file.write(b)

        if "hp" in perplex_db:
            # Werami targets
            w = (f"{sid}\n"              # Proj name
                 f"2\n"                  # Operational mode (2: properties on 2D grid)
                 f"36\n"                 # Select a property
                 f"1\n"                  # Properties of system
                 f"{fluid_properties}\n" # Include fluid in comp ?
                 f"N\n"                  # Change default variable range ?
                 f"\n"                   # Select the grid resolution (enter to continue)
                 f"2\n"                  # Operational mode (2: properties on 2D grid)
                 f"7\n"                  # Select a property (7: mode of phase)
                 f"{melt_mod}\n"         # Select a property
                 f"24\n"                 # Select a property (24: assemblage index)
                 f"0\n"                  # Zero to finish
                 f"N\n"                  # Change default variable range ?
                 f"\n"                   # Select the grid resolution (enter to continue)
                 f"5\n"                  # Average immiscible melt (if applicable)
                 f"0\n"                  # Zero to exit
                 )
        elif perplex_db == "stx21":
            # Werami targets
            w = (f"{sid}\n"              # Proj name
                 f"2\n"                  # Operational mode (2: properties on 2D grid)
                 f"36\n"                 # Select a property
                 f"1\n"                  # Properties of system
                 f"N\n"                  # Change default variable range ?
                 f"\n"                   # Select the grid resolution (enter to continue)
                 f"2\n"                  # Operational mode (2: properties on 2D grid)
                 f"24\n"                 # Select a property (24: assemblage index)
                 f"0\n"                  # Zero to finish
                 f"N\n"                  # Change default variable range ?
                 f"\n"                   # Select the grid resolution (enter to continue)
                 f"0\n"                  # Zero to exit
                 )

        # Write werami targets
        with open(f"{model_out_dir}/werami-targets", "w") as file:
            file.write(w)

        # Write subduction geotherms to tsv files
        for seg in segs:
            geotherm_top = self._get_subduction_geotherm(seg, slab_position="slabtop")
            geotherm_moho = self._get_subduction_geotherm(seg, slab_position="slabmoho")
            geotherm_top["P"] = geotherm_top["P"] * 1e4
            geotherm_moho["P"] = geotherm_moho["P"] * 1e4
            geotherm_top = geotherm_top[["T", "P"]]
            geotherm_moho = geotherm_moho[["T", "P"]]
            geotherm_top.to_csv(f"{model_out_dir}/gt-slabtop-{seg}", sep="\t",
                                index=False, header=False, float_format="%.6E")
            geotherm_moho.to_csv(f"{model_out_dir}/gt-slabmoho-{seg}", sep="\t",
                                 index=False, header=False, float_format="%.6E")
        for pot in pot_Ts:
            geotherm = self._get_mantle_geotherm(pot)
            geotherm["P"] = geotherm["P"] * 1e4
            geotherm = geotherm[["T", "P"]]
            geotherm.to_csv(f"{model_out_dir}/gt-craton-{pot}", sep="\t",
                            index=False, header=False, float_format="%.6E")
        for pot in pot_Ts:
            geotherm = self._get_mantle_geotherm(
                pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                crust_thickness=7e3, litho_thickness=1e3)
            geotherm["P"] = geotherm["P"] * 1e4
            geotherm = geotherm[["T", "P"]]
            geotherm.to_csv(f"{model_out_dir}/gt-mor-{pot}", sep="\t",
                            index=False, header=False, float_format="%.6E")

        # Initialize werami geotherms
        werami_geotherms_top = []
        werami_geotherms_mor = []
        werami_geotherms_moho = []
        werami_geotherms_craton = []

        if "hp" in perplex_db:
            # Werami phase
            f = (f"{sid}\n"               # Proj name
                 f"2\n"                   # Operational mode (2: properties on 2D grid)
                 f"25\n"                  # Select a property (25: Modes of all phases)
                 f"N\n"                   # Output cumulative modes ?
                 f"{fluid_assemblages}\n" # Include fluid in computation of properties ?
                 f"N\n"                   # Change default variable range ?
                 f"\n"                    # Select grid resolution (enter to continue)
                 f"0\n"                   # Zero to exit
                 )
            # Werami geotherm
            for seg in segs:
                if os.path.exists(f"{model_out_dir}/gt-slabtop-{seg}"):
                    g = (f"{sid}\n"                # Proj name
                         f"4\n"                    # Op mode (4: prop along a 1d path)
                         f"2\n"                    # Path (2: a file with T-P points)
                         f"gt-slabtop-{seg}\n"     # Enter filename
                         f"1\n"                    # How many nth points to plot ?
                         f"25\n"                   # Select a property (25: Modes of all)
                         f"N\n"                    # Output cumulative modes ?
                         f"{fluid_assemblages}\n"  # Include fluid in computation ?
                         f"0\n"                    # Zero to exit
                         )
                    werami_geotherms_top.append(g)
                if os.path.exists(f"{model_out_dir}/gt-slabmoho-{seg}"):
                    g = (f"{sid}\n"                # Proj name
                         f"4\n"                    # Op mode (4: prop along a 1d path)
                         f"2\n"                    # Path (2: a file with T-P points)
                         f"gt-slabmoho-{seg}\n"    # Enter filename
                         f"1\n"                    # How many nth points to plot ?
                         f"25\n"                   # Select a property (25: Modes of all)
                         f"N\n"                    # Output cumulative modes ?
                         f"{fluid_assemblages}\n"  # Include fluid in computation ?
                         f"0\n"                    # Zero to exit
                         )
                    werami_geotherms_moho.append(g)
            for pot in pot_Ts:
                if os.path.exists(f"{model_out_dir}/gt-craton-{pot}"):
                    g = (f"{sid}\n"               # Proj name
                         f"4\n"                   # Op mode (4: properties along a 1d path)
                         f"2\n"                   # Path (2: a file with T-P points)
                         f"gt-craton-{pot}\n"     # Enter filename
                         f"1\n"                   # How many nth points to plot ?
                         f"25\n"                  # Select a property (25: Modes of all)
                         f"N\n"                   # Output cumulative modes ?
                         f"{fluid_assemblages}\n" # Include fluid in computation ?
                         f"0\n"                   # Zero to exit
                         )
                    werami_geotherms_craton.append(g)
            for pot in pot_Ts:
                if os.path.exists(f"{model_out_dir}/gt-mor-{pot}"):
                    g = (f"{sid}\n"               # Proj name
                         f"4\n"                   # Op mode (4: properties along a 1d path)
                         f"2\n"                   # Path (2: a file with T-P points)
                         f"gt-mor-{pot}\n"        # Enter filename
                         f"1\n"                   # How many nth points to plot ?
                         f"25\n"                  # Select a property (25: Modes of all)
                         f"N\n"                   # Output cumulative modes ?
                         f"{fluid_assemblages}\n" # Include fluid in computation ?
                         f"0\n"                   # Zero to exit
                         )
                    werami_geotherms_mor.append(g)
        elif perplex_db == "stx21":
            # Werami phase
            f = (f"{sid}\n" # Proj name
                 f"2\n"     # Operational mode (2: properties on 2D grid)
                 f"25\n"    # Select a property (25: Modes of all phases)
                 f"N\n"     # Output cumulative modes ?
                 f"N\n"     # Change default variable range ?
                 f"\n"      # Select grid resolution (enter to continue)
                 f"0\n"     # Zero to exit
                 )
            # Werami geotherm
            for seg in segs:
                if os.path.exists(f"{model_out_dir}/gt-slabtop-{seg}"):
                    g = (f"{sid}\n"            # Proj name
                         f"4\n"                # Op mode (4: properties along a 1d path)
                         f"2\n"                # Path (2: a file with T-P points)
                         f"gt-slabtop-{seg}\n" # Enter filename
                         f"1\n"                # How many nth points to plot ?
                         f"25\n"               # Select a property (25: Modes of all)
                         f"N\n"                # Output cumulative modes ?
                         f"0\n"                # Zero to exit
                         )
                    werami_geotherms_top.append(g)
                if os.path.exists(f"{model_out_dir}/gt-slabmoho-{seg}"):
                    g = (f"{sid}\n"             # Proj name
                         f"4\n"                 # Op mode (4: properties along a 1d path)
                         f"2\n"                 # Path (2: a file with T-P points)
                         f"gt-slabmoho-{seg}\n" # Enter filename
                         f"1\n"                 # How many nth points to plot ?
                         f"25\n"                # Select a property (25: Modes of all)
                         f"N\n"                 # Output cumulative modes ?
                         f"0\n"                 # Zero to exit
                         )
                    werami_geotherms_moho.append(g)
            for pot in pot_Ts:
                if os.path.exists(f"{model_out_dir}/gt-craton-{pot}"):
                    g = (f"{sid}\n"           # Proj name
                         f"4\n"               # Op mode (4: properties along a 1d path)
                         f"2\n"               # Path (2: a file with T-P points)
                         f"gt-craton-{pot}\n" # Enter filename
                         f"1\n"               # How many nth points to plot ?
                         f"25\n"              # Select a property (25: Modes of all)
                         f"N\n"               # Output cumulative modes ?
                         f"0\n"               # Zero to exit
                         )
                    werami_geotherms_craton.append(g)
            for pot in pot_Ts:
                if os.path.exists(f"{model_out_dir}/gt-mor-{pot}"):
                    g = (f"{sid}\n"        # Proj name
                         f"4\n"            # Op mode (4: properties along a 1d path)
                         f"2\n"            # Path (2: a file with T-P points)
                         f"gt-mor-{pot}\n" # Enter filename
                         f"1\n"            # How many nth points to plot ?
                         f"25\n"           # Select a property (25: Modes of all)
                         f"N\n"            # Output cumulative modes ?
                         f"0\n"            # Zero to exit
                         )
                    werami_geotherms_mor.append(g)
        else:
            raise Exception("Unrecognized thermodynamic dataset !")

        # Write werami phase
        with open(f"{model_out_dir}/werami-phases", "w") as file:
            file.write(f)

        # Write werami geotherms
        if werami_geotherms_top:
            for i, g in enumerate(werami_geotherms_top):
                with open(f"{model_out_dir}/werami-gt-slabtop-{segs[i]}", "w") as file:
                    file.write(g)
        if werami_geotherms_moho:
            for i, g in enumerate(werami_geotherms_moho):
                with open(f"{model_out_dir}/werami-gt-slabmoho-{segs[i]}", "w") as file:
                    file.write(g)
        if werami_geotherms_craton:
            for i, g in enumerate(werami_geotherms_craton):
                with open(f"{model_out_dir}/werami-gt-craton-{pot_Ts[i]}", "w") as file:
                    file.write(g)
        if werami_geotherms_mor:
            for i, g in enumerate(werami_geotherms_mor):
                with open(f"{model_out_dir}/werami-gt-mor-{pot_Ts[i]}", "w") as file:
                    file.write(g)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # configure perplex model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_perplex_model(self):
        """
        """
        # Get self attributes
        sid = self.sid
        segs = self.segs
        pot_Ts = self.pot_Ts
        ox_gfem = self.ox_gfem
        timeout = self.timeout
        verbose = self.verbose
        log_file = self.log_file
        em_exclude = self.em_exclude
        sl_include = self.sl_include
        perplex_db = self.perplex_db
        model_built = self.model_built
        model_out_dir = self.model_out_dir
        norm_sample_comp = self.norm_sample_comp

        try:
            # Check if model is built
            if model_built:
                return None

            # Write perplex configuration files
            self._write_perplex_config()

            if verbose >= 1:
                print(f"  Configuring Perple_X with {perplex_db} database and comp (wt.%):")
                max_oxide_width = max(len(oxide) for oxide in ox_gfem)
                max_comp_width = max(len(str(comp)) for comp in norm_sample_comp)
                max_width = max(max_oxide_width, max_comp_width)
                print(" ".join([f"  {oxide:<{max_width}}" for oxide in ox_gfem]))
                print(" ".join([f"  {comp:<{max_width}}" for comp in norm_sample_comp]))
                print("  --------------------")
                emwrp = textwrap.fill(", ".join(em_exclude), width=80,
                                      subsequent_indent="    ")
                slwrp = textwrap.fill(", ".join(sl_include), width=80,
                                      subsequent_indent="    ")
                print(f"  Excluded endmembers:")
                print(f"    {emwrp}")
                print("  --------------------")
                print(f"  Included solution models:")
                print(f"    {slwrp}")
                print("  --------------------")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _configure_perplex_model() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # print perplex info !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_perplex_info(self):
        """
        """
        # Get self attributes
        res = self.res
        sid = self.sid
        P_min = self.P_min
        P_max = self.P_max
        T_min = self.T_min
        T_max = self.T_max
        source = self.source
        ox_gfem = self.ox_gfem
        features = self.features
        perplex_db = self.perplex_db
        model_out_dir = self.model_out_dir
        targets = [target for target in self.target_labels_map.values()
                   if target not in ["P", "T"]]

        oxwrp = textwrap.fill(", ".join(ox_gfem), width=80)
        tgwrp = textwrap.fill(", ".join(targets), width=80,
                              subsequent_indent="                  ")
        ftwrp = textwrap.fill(", ".join(features), width=80,
                              subsequent_indent="                  ")

        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Perple_X model: {sid} {perplex_db}")
        print("---------------------------------------------")
        print(f"  PT resolution:  {res}")
        print(f"  P range:        {P_min:.1f} - {P_max:.1f} GPa")
        print(f"  T range:        {T_min:.0f} - {T_max:.0f} K")
        print(f"  Sampleid:       {sid}")
        print(f"  Source:         {source}")
        print(f"  GFEM sys.:      {oxwrp}")
        print(f"  Targets:        {tgwrp}")
        print(f"  Features:       {ftwrp}")
        print(f"  Thermo. data:   {perplex_db}")
        print(f"  Model out dir:  {model_out_dir}")
        print("  --------------------")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # replace in file !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _replace_in_file(self, filepath, replacements):
        """
        """
        try:
            with open(filepath, "r") as file:
                file_data = file.read()

                for key, value in replacements.items():
                    file_data = file_data.replace(key, value)

            with open(filepath, "w") as file:
                file.write(file_data)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _replace_in_file() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run command line program !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_command_line_program(self, program_path, config_file):
        # Get self attributes
        timeout = self.timeout
        verbose = self.verbose
        log_file = self.log_file
        model_out_dir = self.model_out_dir

        # Get relative program path
        relative_program_path = f"../../{program_path}"

        try:
            # Set permissions
            os.chmod(program_path, 0o755)

            if verbose >= 1:
                print(f"  Running {program_path} with {config_file} ...")

            # Open the subprocess and redirect input from the input file
            with open(config_file, "rb") as input_stream:
                process = subprocess.Popen(
                    [relative_program_path], stdin=input_stream,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    shell=True, cwd=model_out_dir)

            # Wait for the process to complete and capture its output
            stdout, stderr = process.communicate(timeout=timeout)

            if verbose >= 2:
                print(f"{stdout.decode()}")

            # Write to logfile
            with open(log_file, "a") as log:
                log.write(stdout.decode())
                log.write(stderr.decode())

            if process.returncode != 0:
                raise RuntimeError(f"Error with perplex program '{program_path}'!")

            elif verbose >= 2:
                print(f"{program_path} output:")
                print(f"{stdout.decode()}")

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex build !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex_build(self):
        """
        """
        # Get self attributes
        sid = self.sid
        verbose = self.verbose
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        try:
            # Check if model is built
            if model_built:
                return None

            if not os.path.exists(f"{model_out_dir}/build-config"):
                raise Exception("No config! Call _configure_perplex_model() first ...")

            # Check for existing
            build_file = f"{model_out_dir}/{sid}.dat"
            if os.path.exists(build_file):
                if verbose >= 1:
                    print(f"  Perple_X build found !")
                return None

            # Get config file
            config_file = f"{model_out_dir}/build-config"

            # Get program path
            program_path = f"Perple_X/build"

            # Set build option
            self._replace_in_file(f"{model_out_dir}/build-options",
                                  {"Anderson-Gruneisen     T":
                                   "Anderson-Gruneisen     F"})

            # Run program
            self._run_command_line_program(program_path, config_file)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _run_perplex_build() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex vertex !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex_vertex(self):
        """
        """
        # Get self attributes
        sid = self.sid
        verbose = self.verbose
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        try:
            # Check if model is built
            if model_built:
                return None

            if not os.path.exists(f"{model_out_dir}/build-config"):
                raise Exception("No config! Call _configure_perplex_model() first ...")

            if not os.path.exists(f"{model_out_dir}/{sid}.dat"):
                raise Exception("No build output! Call _run_perplex_build() first ...")

            # Check for existing
            vertex_file = f"{model_out_dir}/{sid}.tof"
            if os.path.exists(vertex_file):
                if verbose >= 1:
                    print(f"  Perple_X vertex found !")
                return None

            # Get config files
            config_file = f"{model_out_dir}/vertex-minimize"

            # Get program path
            program_path = f"Perple_X/vertex"

            # Set build option
            self._replace_in_file(f"{model_out_dir}/build-options",
                                  {"Anderson-Gruneisen     T":
                                   "Anderson-Gruneisen     F"})

            # Run program
            self._run_command_line_program(program_path, config_file)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _run_perplex_vertex() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex werami !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex_werami(self):
        """
        """
        # Get self attributes
        sid = self.sid
        segs = self.segs
        P_min = self.P_min
        pot_Ts = self.pot_Ts
        perplex_db = self.perplex_db
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        try:
            # Check if model is built
            if model_built:
                return None

            if not os.path.exists(f"{model_out_dir}/build-config"):
                raise Exception("No config! Call _configure_perplex_model() first ...")

            if not os.path.exists(f"{model_out_dir}/{sid}.dat"):
                raise Exception("No build output! Call _run_perplex_build() first ...")

            if not os.path.exists(f"{model_out_dir}/{sid}.tof"):
                raise Exception("No build output! Call _run_perplex_vertex() first ...")

            # Get config files
            config_files = {}
            config_files["targets"] = f"{model_out_dir}/werami-targets"
            config_files["phases"] = f"{model_out_dir}/werami-phases"
            for seg in segs:
                if P_min < 6:
                    config_files[f"slabtop-{seg}"] = (
                        f"{model_out_dir}/werami-gt-slabtop-{seg}")
                    config_files[f"slabmoho-{seg}"] = (
                        f"{model_out_dir}/werami-gt-slabmoho-{seg}")
            for pot in pot_Ts:
                config_files[f"craton-{pot}"] = f"{model_out_dir}/werami-gt-craton-{pot}"
            for pot in pot_Ts:
                config_files[f"mor-{pot}"] = f"{model_out_dir}/werami-gt-mor-{pot}"

            # Set build option
            self._replace_in_file(f"{model_out_dir}/build-options",
                                  {"Anderson-Gruneisen     F":
                                   "Anderson-Gruneisen     T"})

            # Get program path
            program_path = f"Perple_X/werami"

            for name, config in config_files.items():
                # Run program
                self._run_command_line_program(program_path, config)

                if name == "targets":
                    # Rename werami output
                    shutil.copy(f"{model_out_dir}/{sid}_1.tab",
                                f"{model_out_dir}/{name}.tab")
                    shutil.copy(f"{model_out_dir}/{sid}_2.tab",
                                f"{model_out_dir}/supplemental.tab")

                    # Remove old output
                    os.remove(f"{model_out_dir}/{sid}_1.tab")
                    os.remove(f"{model_out_dir}/{sid}_2.tab")
                else:
                    # Rename werami output
                    shutil.copy(f"{model_out_dir}/{sid}_1.tab",
                                f"{model_out_dir}/{name}.tab")

                    # Remove old output
                    os.remove(f"{model_out_dir}/{sid}_1.tab")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _run_perplex_werami() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex pssect !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex_pssect(self):
        """
        """
        # Get self attributes
        sid = self.sid
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        try:
            # Check if model is built
            if model_built:
                return None

            if not os.path.exists(f"{model_out_dir}/build-config"):
                raise Exception("No config! Call _configure_perplex_model() first ...")

            # Get config file
            config_file = f"{model_out_dir}/pssect-draw"

            # Get program path
            program_path = f"Perple_X/pssect"

            # Run program
            self._run_command_line_program(program_path, config_file)

            # Rename pssect assemblages output
            shutil.copy(f"{model_out_dir}/"
                        f"{sid}_assemblages.txt",
                        f"{model_out_dir}/assemblages.txt")

            # Remove old output
            os.remove(f"{model_out_dir}/{sid}_assemblages.txt")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _run_perplex_build() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex(self):
        """
        """
        # Get self attributes
        model_out_dir = self.model_out_dir

        try:
            # Run programs
            self._configure_perplex_model()
            self._run_perplex_build()
            self._run_perplex_vertex()
            self._run_perplex_werami()
            self._run_perplex_pssect()

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _run_perplex() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.        Post-process GFEM Models         !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read perplex targets !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_targets(self):
        """
        """
        # Get self attributes
        model_out_dir = self.model_out_dir
        werami_output_map = self.werami_output_map
        werami_targets = f"{model_out_dir}/targets.tab"
        werami_supplemental = f"{model_out_dir}/supplemental.tab"

        try:
            # Initialize results
            results = {v: [] for v in werami_output_map.values()}

            # Open werami targets file
            for w, werami_output in enumerate([werami_targets, werami_supplemental]):
                with open(werami_output, "r") as file:
                    # Read headers
                    headers = None
                    for line in file:
                        if "T(K)" in line and "P(bar)" in line:
                            headers = line.split()
                            break

                    # Ensure headers are found
                    if headers is None:
                        raise Exception("No column headers found !")

                    # Create a mapping of column index to variable names
                    index_map = {}
                    for i, header in enumerate(headers):
                        if header in werami_output_map:
                            index_map[i] = werami_output_map[header]

                    # Read the data lines
                    for line in file:
                        vals = line.split()

                        # Read values and map them to the correct variables
                        for i, val in enumerate(vals):
                            if i in index_map:
                                variable_name = index_map[i]
                                if w > 0 and variable_name in ["T", "P"]:
                                    continue
                                try:
                                    # Convert to float or handle as nan if necessary
                                    value = (float(val) if not np.isnan(float(val)) else
                                             np.nan)

                                    # Special conversions (pressure, density, etc.)
                                    if variable_name == "P":
                                        value /= 1e4  # Convert bar to GPa
                                    if variable_name == "density":
                                        value /= 1e3  # Convert kg/m3 to g/cm3
                                    if variable_name == "assemblage_index":
                                        value = int(value)

                                    # Append to the corresponding list in results
                                    results[variable_name].append(value)

                                except ValueError:
                                    results[variable_name].append(np.nan)

            # Get max length of results
            max_len = max(len(vals) for vals in results.values())

            # Fill empty columns with np.nan
            for k, v in results.items():
                if len(v) == 0: results[k] = [np.nan] * max_len

            if all(len(vals) == 0 for vals in results.values()):
                raise Exception(f"No data was read from the file {perplex_targets} !")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _read_perplex_targets() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return results

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read perplex assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_assemblages(self):
        """
        """
        # Get self attributes
        model_built = self.model_built
        model_out_dir = self.model_out_dir
        perplex_assemblages = f"{self.model_out_dir}/assemblages.txt"
        perplex_assemblages_csv = f"{self.model_out_dir}/assemblages.csv"

        try:
            # Check if model is built
            if model_built:
                return None

            if os.path.exists(perplex_assemblages_csv):
                df = pd.read_csv(perplex_assemblages_csv)
                assemblage_dict = df["assemblage"].to_list()
            else:
                # Initialize dictionary to store assemblage info
                assemblage_dict = {}

                # Open assemblage file
                with open(perplex_assemblages, "r") as file:
                    for i, line in enumerate(file, start=1):
                        cleaned_line = line.strip()
                        if " - " in cleaned_line:
                            phases_string = cleaned_line.split(" - ", 1)[1]
                        else:
                            phases_string = cleaned_line

                        assemblages = phases_string.split()

                        # Add assemblage to dict
                        assemblage_dict[i] = assemblages

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _read_perplex_assemblages() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return assemblage_dict

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # encode assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _encode_assemblages(self, assemblages):
        """
        """
        # Get self attributes
        model_out_dir = self.model_out_dir

        try:
            unique_assemblages = {}
            encoded_assemblages = []

            # Encoding unique phase assemblages
            for assemblage in assemblages:
                assemblage_tuple = tuple(sorted(assemblage))

                if assemblage_tuple and not any(
                        np.isnan(item) for item in assemblage_tuple if
                        isinstance(item, (int, float))):
                    if assemblage_tuple not in unique_assemblages:
                        unique_assemblages[assemblage_tuple] = len(unique_assemblages) + 1

            # Create dataframe
            df = pd.DataFrame(list(unique_assemblages.items()),
                              columns=["assemblage", "index"])

            # Put spaces between phases
            df["assemblage"] = df["assemblage"].apply(" ".join)

            # Reorder columns
            df = df[["index", "assemblage"]]

            # Save to csv
            assemblages_csv = f"{model_out_dir}/assemblages.csv"
            df.to_csv(assemblages_csv, index=False)

            # Encoding phase assemblage numbers
            for assemblage in assemblages:
                if assemblage == "":
                    encoded_assemblages.append(np.nan)

                else:
                    encoded_assemblage = unique_assemblages[tuple(sorted(assemblage))]
                    encoded_assemblages.append(encoded_assemblage)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _encode_assemblages() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return encoded_assemblages

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process perplex results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_perplex_results(self):
        """
        """
        # Get self attributes
        verbose = self.verbose
        model_built = self.model_built
        model_out_dir = self.model_out_dir
        perplex_targets = f"{model_out_dir}/targets.tab"
        perplex_assemblages = f"{model_out_dir}/assemblages.txt"

        try:
            # Check if model is built
            if model_built:
                return None

            if not os.path.exists(perplex_targets):
                raise Exception("No models to process! Call _run_perplex() first ...")

            if not os.path.exists(perplex_assemblages):
                raise Exception("No models to process! Call _run_perplex() first ...")

            if verbose >= 1:
                print(f"  Reading Perple_X output from {model_out_dir} ...")

            # Read results
            results = self._read_perplex_targets()

            # Get assemblages from file
            assemblages = self._read_perplex_assemblages()

            # Parse assemblages by index
            phase_assemblages = []
            for index in results.get("assemblage_index"):
                if np.isnan(index):
                    phase_assemblages.append("")
                else:
                    phases = sorted(set(assemblages[index]))
                    phase_assemblages.append(" ".join(phases))

            # Add phase assemblage to results
            results["phase_assemblage"] = phase_assemblages

            # Count unique phases (assemblage variance)
            assemblage_variance = []
            for assemblage in results.get("phase_assemblage"):
                if assemblage is None:
                    assemblage_variance.append(np.nan)
                else:
                    unique_phases = set(assemblage)
                    count = len(unique_phases)

                    assemblage_variance.append(count)

            # Add phase assemblage variance to results
            results["phase_assemblage_variance"] = assemblage_variance

            # Remove assemblage index
            results.pop("assemblage_index")

            # Encode assemblage
            encoded_assemblages = self._encode_assemblages(results["phase_assemblage"])

            # Replace assemblage with encoded assemblages
            results["assemblage_index"] = encoded_assemblages

            # Save as pandas df
            df = pd.DataFrame.from_dict(results)

            if verbose >= 1:
                print(f"  Writing Perple_X results to {model_out_dir}/results.csv ...")

            # Write to csv file
            df.to_csv(f"{model_out_dir}/results.csv", index=False)

            self.model_built = True

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _process_perplex_results() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_results(self):
        """
        """
        # Get self attributes
        verbose = self.verbose
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        try:
            if not model_built:
                raise Exception("No GFEM model! Call build() first ...")

            # Get filepaths for gfem output
            filepath = f"{model_out_dir}/results.csv"

            if not os.path.exists(filepath):
                raise Exception("No results to read!")

            if verbose >= 2:
                print(f"  Reading results from {filepath} ...")

            # Read results
            df = pd.read_csv(filepath)

            # Convert to df into dict
            self.results = {column: np.array(values)
                            for column, values in df.to_dict(orient="list").items()}

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_results() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get pt array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_pt_array(self):
        """
        """
        # Get self attributes
        results = self.results
        verbose = self.verbose
        model_built = self.model_built

        try:
            if not model_built:
                raise Exception("No GFEM model! Call build() first ...")

            if not results:
                raise Exception("No GFEM model results! Call get_results() first ...")

            if verbose >= 2:
                print(f"  Getting PT array ...")

            # Get P T arrays
            P, T = results["P"].copy(), results["T"].copy()

            # Stack PT arrays
            self.pt_array = np.stack((P, T), axis=-1).copy()

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_pt_array() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get target array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_target_array(self):
        """
        """
        # Get self attributes
        res = self.res
        targets = self.targets
        results = self.results
        verbose = self.verbose
        model_built = self.model_built
        targets_exclude = ["phase_assemblage"]

        try:
            if not model_built:
                raise Exception("No GFEM model! Call build() first ...")

            if not results:
                raise Exception("No GFEM model results! Call get_results() first ...")

            if verbose >= 2:
                print(f"  Getting target array ...")

            target_array_list = [
                results[key] for key in targets if key not in targets_exclude]

            self.target_array = np.stack(target_array_list, axis=-1).copy()

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_target_array() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # extract target along geotherm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _extract_target_along_geotherm(self, target, geotherm):
        """
        """
        # Get self attributes
        results = self.results

        try:
            # Get GFEM PTs and targets
            P_vals, T_vals, target_vals = results["P"], results["T"], results[target]
            gfem_points = np.vstack((T_vals, P_vals)).T

            # Get geotherm PTs
            geo_P, geo_T = geotherm["P"], geotherm["T"]
            geo_points = np.array([geo_T, geo_P]).T

            # Create interpolator
            interpolator = LinearNDInterpolator(gfem_points, target_vals)

            # Interpolate target values at the geotherm PT points
            target_interp = interpolator(geo_points)

            # Save in dataframe
            df = pd.DataFrame({"P": geo_P, "T": geo_T, target: target_interp})

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _extract_target_along_geotherm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return df

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.3.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model array images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_array_images(self, type="sub", gradient=False):
        """
        """
        # Get model data
        sid = self.sid
        fig_dir = self.fig_dir
        targets = self.targets
        perplex_db = self.perplex_db
        targets_to_visualize = self.targets_to_visualize
        if gradient:
            targets_exclude = ["phase_assemblage", "assemblage_index",
                               "phase_assemblage_variance"]
        else:
            targets_exclude = ["phase_assemblage"]
        targets_to_visualize = [t for t in targets_to_visualize if t not in targets_exclude]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            if target not in targets_to_visualize:
                continue

            if gradient:
                path = (f"{fig_dir}/{sid}-{perplex_db}-{target.replace("_", "-")}-grad-"
                        f"{type}.png")
            else:
                path = (f"{fig_dir}/{sid}-{perplex_db}-{target.replace("_", "-")}-"
                        f"{type}.png")

            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if len(existing_figs) == len(targets_to_visualize):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model array surfs !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_array_surfs(self):
        """
        """
        # Get model data
        sid = self.sid
        fig_dir = self.fig_dir
        targets = self.targets
        perplex_db = self.perplex_db
        targets_to_visualize = self.targets_to_visualize
        targets_exclude = ["phase_assemblage"]
        targets_to_visualize = [t for t in targets_to_visualize if t not in targets_exclude]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            if target not in targets_to_visualize:
                continue

            path = f"{fig_dir}/{sid}-{perplex_db}-{target.replace("_", "-")}-surf.png"

            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if len(existing_figs) == len(targets_to_visualize):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model depth profile images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_depth_profile_images(self):
        """
        """
        # Get model data
        sid = self.sid
        P_min = self.P_min
        fig_dir = self.fig_dir
        targets = self.targets
        perplex_db = self.perplex_db
        targets_to_visualize = self.targets_to_visualize
        targets_exclude = ["phase_assemblage", "assemblage_index",
                           "phase_assemblage_variance"]
        targets_to_visualize = [t for t in targets_to_visualize if t not in targets_exclude]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            if target not in targets_to_visualize:
                continue

            if P_min < 6:
                path = (f"{fig_dir}/{sid}-{perplex_db}-{target.replace("_", "-")}-"
                        f"depth-profile-sub-slabtop.png")
                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

                path = (f"{fig_dir}/{sid}-{perplex_db}-{target.replace("_", "-")}-"
                        f"depth-profile-sub-slabmoho.png")
                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

            path = (f"{fig_dir}/{sid}-{perplex_db}-{target.replace("_", "-")}-"
                    f"depth-profile-craton.png")
            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

            path = (f"{fig_dir}/{sid}-{perplex_db}-{target.replace("_", "-")}-"
                    f"depth-profile-mor.png")
            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if P_min < 6:
            if len(existing_figs) == (len(targets_to_visualize) * 4):
                return True
        elif P_min >= 6:
            if len(existing_figs) == (len(targets_to_visualize) * 2):
                return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model gt assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_gt_assemblages_images(self):
        """
        """
        # Get model data
        sid = self.sid
        segs = self.segs
        P_min = self.P_min
        pot_Ts = self.pot_Ts
        fig_dir = self.fig_dir
        perplex_db = self.perplex_db

        # Check for existing plots
        existing_figs = []
        for i, seg in enumerate(segs):
            if P_min < 6:
                seg_lab = seg.replace("_", "-").lower()
                path = (f"{fig_dir}/{sid}-{perplex_db}-slabtop-{seg_lab}-assemblages.png")

                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

                seg_lab = seg.replace("_", "-").lower()
                path = (f"{fig_dir}/{sid}-{perplex_db}-slabmoho-{seg_lab}-assemblages.png")

                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

        for i, pot in enumerate(pot_Ts):
            path = f"{fig_dir}/{sid}-{perplex_db}-craton-{pot}-assemblages.png"

            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        for i, pot in enumerate(pot_Ts):
            path = f"{fig_dir}/{sid}-{perplex_db}-mor-{pot}-assemblages.png"

            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if P_min < 6:
            if len(existing_figs) == ((len(segs) * 2) + (len(pot_Ts) * 2)):
                return True
        elif P_min >= 6:
            if len(existing_figs) == (len(pot_Ts) * 2):
                return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize array image  !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_image(self, type="sub", palette="bone", gradient=False,
                               figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get self attributes
        sid = self.sid
        res = self.res
        segs = self.segs
        P_min = self.P_min
        pot_Ts = self.pot_Ts
        results = self.results
        fig_dir = self.fig_dir
        targets = self.targets
        perplex_db = self.perplex_db
        model_built = self.model_built
        target_array = self.target_array
        target_units_map = self.target_units_map
        target_digits_map = self.target_digits_map
        target_labels_map = self.target_labels_map
        targets_to_visualize = self.targets_to_visualize

        P, T = results["P"], results["T"]
        extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

        if not model_built:
            raise Exception("No GFEM model! Call build() first ...")

        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        if target_array is None or target_array.size == 0:
            raise Exception("No GFEM model target array! Call get_target_array() first ...")

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Filter targets for gradient images
        if gradient:
            targets_exclude = ["assemblage_index", "phase_assemblage_variance"]
        else:
            targets_exclude = ["phase_assemblage"]
        targets_to_visualize = [t for t in targets_to_visualize if t not in targets_exclude]

        # Get geotherm
        if type == "sub":
            sub_gtt = {}
            sub_gtm = {}
            for seg in segs:
                if P_min < 6:
                    sub_gtt[seg] = self._get_subduction_geotherm(
                        seg, slab_position="slabtop")
                    sub_gtm[seg] = self._get_subduction_geotherm(
                        seg, slab_position="slabmoho")
                else:
                    print("  P_min too high to plot subduction geotherms !")
                    return None
        elif type == "craton":
            ad_gt = {}
            for pot in pot_Ts:
                ad_gt[pot] = self._get_mantle_geotherm(pot)
        elif type == "mor":
            ad_gt = {}
            for pot in pot_Ts:
                ad_gt[pot] = self._get_mantle_geotherm(
                    pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                    crust_thickness=7e3, litho_thickness=1e3)

        # Set plot style and settings
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["axes.facecolor"] = "0.9"
        plt.rcParams["legend.frameon"] = "False"
        plt.rcParams["legend.facecolor"] = "0.9"
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["figure.autolayout"] = "True"

        linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]

        for i, target in enumerate(targets):
            if target not in targets_to_visualize:
                continue

            # Target label
            target_label = target_labels_map[target]

            # Set filename
            filename = f"{sid}-{perplex_db}-{target.replace("_", "-")}-{type}.png"
            if target not in ["assemblage_index", "phase_assemblage_variance"]:
                title = f"{target_label} ({target_units_map[target]})"
            else:
                title = f"{target_label}"

            # Reshape targets into square array
            square_target = target_array[:, i].reshape(res + 1, res + 1)

            # Sobel filter for gradient images
            if gradient:
                original_image = square_target.copy()

                # Apply Sobel edge detection
                edges_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
                edges_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate the magnitude of the gradient
                if np.any(~np.isnan(original_image)):
                    max_val = np.nanmax(original_image)
                    if max_val > 0:
                        square_target = np.sqrt(edges_x**2 + edges_y**2) / max_val
                    else:
                        square_target = np.zeros_like(edges_x)
                else:
                    square_target = np.full_like(edges_x, np.nan)

                filename = f"{sid}-{perplex_db}-{target.replace("_", "-")}-grad-{type}.png"
                title = f"{target_label} Gradient"

            # Use discrete colorscale
            if target in ["assemblage_index", "phase_assemblage_variance"]:
                color_discrete = True
            else:
                color_discrete = False

            # Reverse color scale
            if palette in ["grey"]:
                if target in ["phase_assemblage_variance"]:
                    color_reverse = True
                else:
                    color_reverse = False
            else:
                if target in ["phase_assemblage_variance"]:
                    color_reverse = False
                else:
                    color_reverse = True

            # Set colorbar limits for better comparisons
            if not color_discrete:
                non_nan_values = square_target[np.logical_not(np.isnan(square_target))]
                if non_nan_values.size > 0:
                    vmin = np.min(non_nan_values)
                    vmax = np.max(non_nan_values)
                else:
                    vmin = 0
                    vmax = 0
            else:
                vmin = int(np.nanmin(np.unique(square_target)))
                vmax = int(np.nanmax(np.unique(square_target)))

            if color_discrete:
                # Discrete color palette
                num_colors = vmax - vmin
                num_colors = max(num_colors, num_colors // 4)

                if palette == "viridis":
                    if color_reverse:
                        pal = plt.colormaps["viridis_r"]
                    else:
                        pal = plt.colormaps["viridis"]
                elif palette == "bone":
                    if color_reverse:
                        pal = plt.colormaps["bone_r"]
                    else:
                        pal = plt.colormaps["bone"]
                elif palette == "pink":
                    if color_reverse:
                        pal = plt.colormaps["pink_r"]
                    else:
                        pal = plt.colormaps["pink"]
                elif palette == "seismic":
                    if color_reverse:
                        pal = plt.colormaps["seismic_r"]
                    else:
                        pal = plt.colormaps["seismic"]
                elif palette == "grey":
                    if color_reverse:
                        pal = plt.colormaps["Greys_r"]
                    else:
                        pal = plt.colormaps["Greys"]
                elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
                    if color_reverse:
                        pal = plt.colormaps["Blues_r"]
                    else:
                        pal = plt.colormaps["Blues"]

                # Descritize
                color_palette = pal(np.linspace(0, 1, num_colors))
                cmap = ListedColormap(color_palette)

                # Set nan color
                cmap.set_bad(color="0.9")

                # Plot as a raster using imshow
                fig, ax = plt.subplots(figsize=(figwidth, figheight))

                im = ax.imshow(square_target, extent=extent, aspect="auto", cmap=cmap,
                               origin="lower", vmin=vmin, vmax=vmax)
                if type == "sub":
                    for seg, gt in sub_gtt.items():
                        ax.plot(gt["T"], gt["P"], linestyle="-", color="black",
                                linewidth=2, label=seg)
                    for seg, gt in sub_gtm.items():
                        ax.plot(gt["T"], gt["P"], linestyle="--", color="black",
                                linewidth=2, label=seg)
                elif type == "craton":
                    for i, (pot, gt) in enumerate(ad_gt.items()):
                        ax.plot(gt["T"], gt["P"], linestyle=linestyles[i], color="black",
                                linewidth=2, label=pot)
                elif type == "mor":
                    for i, (pot, gt) in enumerate(ad_gt.items()):
                        ax.plot(gt["T"], gt["P"], linestyle=linestyles[i], color="black",
                                linewidth=2, label=pot)

                ax.set_xlabel("T (K)")
                ax.set_ylabel("P (GPa)")
                plt.colorbar(im, ax=ax, label="")

            else:
                # Continuous color palette
                if palette == "viridis":
                    if color_reverse:
                        cmap = "viridis_r"
                    else:
                        cmap = "viridis"
                elif palette == "bone":
                    if color_reverse:
                        cmap = "bone_r"
                    else:
                        cmap = "bone"
                elif palette == "pink":
                    if color_reverse:
                        cmap = "pink_r"
                    else:
                        cmap = "pink"
                elif palette == "seismic":
                    if color_reverse:
                        cmap = "seismic_r"
                    else:
                        cmap = "seismic"
                elif palette == "grey":
                    if color_reverse:
                        cmap = "Greys_r"
                    else:
                        cmap = "Greys"
                elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
                    if color_reverse:
                        cmap="Blues_r"
                    else:
                        cmap="Blues"

                # Adjust diverging colorscale to center on zero
                if palette == "seismic":
                    vmin = -np.max(
                        np.abs(square_target[np.logical_not(np.isnan(square_target))]))
                    vmax = np.max(
                        np.abs(square_target[np.logical_not(np.isnan(square_target))]))
                else:
                    vmin, vmax = vmin, vmax

                    # Adjust vmin close to zero
                    if vmin <= 1e-4: vmin = 0

                # Set nan color
                cmap = plt.colormaps[cmap]
                cmap.set_bad(color="0.9")

                # Plot as a raster using imshow
                fig, ax = plt.subplots()

                im = ax.imshow(square_target, extent=extent, aspect="auto", cmap=cmap,
                               origin="lower", vmin=vmin, vmax=vmax)
                if type == "sub":
                    for seg, gt in sub_gtt.items():
                        ax.plot(gt["T"], gt["P"], linestyle="-", color="black",
                                linewidth=2, label=seg)
                    for seg, gt in sub_gtm.items():
                        ax.plot(gt["T"], gt["P"], linestyle="--", color="black",
                                linewidth=2, label=seg)
                elif type == "craton":
                    for i, (pot, gt) in enumerate(ad_gt.items()):
                        ax.plot(gt["T"], gt["P"], linestyle=linestyles[i], color="black",
                                linewidth=2, label=pot)
                elif type == "mor":
                    for i, (pot, gt) in enumerate(ad_gt.items()):
                        ax.plot(gt["T"], gt["P"], linestyle=linestyles[i], color="black",
                                linewidth=2, label=pot)
                ax.set_xlabel("T (K)")
                ax.set_ylabel("P (GPa)")

                # Diverging colorbar
                if palette == "seismic":
                    cbar = plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], label="")

                # Continuous colorbar
                else:
                    cbar = plt.colorbar(im, ax=ax, ticks=np.linspace(vmin, vmax, num=4),
                                        label="")

                # Set colorbar limits and number formatting
                cbar.ax.yaxis.set_major_formatter(
                    plt.FormatStrFormatter(target_digits_map[target]))

            plt.title(title)

            # Vertical text spacing
            text_margin_x = 0.04
            text_margin_y = 0.15
            text_spacing_y = 0.1

            # Save the plot to a file
            plt.savefig(f"{fig_dir}/{filename}")

            # Close device
            plt.close()
            print(f"  Figure saved to: {filename} ...")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize target surf !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_target_surf(self, palette="bone", figwidth=6.3, figheight=4.725,
                               fontsize=22):
        """
        """
        # Get self attributes
        sid = self.sid
        res = self.res
        results = self.results
        fig_dir = self.fig_dir
        targets = self.targets
        perplex_db = self.perplex_db
        model_built = self.model_built
        target_array = self.target_array
        target_units_map = self.target_units_map
        target_digits_map = self.target_digits_map
        target_labels_map = self.target_labels_map
        targets_to_visualize = self.targets_to_visualize

        P = results["P"].reshape(res + 1, res + 1)
        T = results["T"].reshape(res + 1, res + 1)

        if not model_built:
            raise Exception("No GFEM model! Call build() first ...")

        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        if target_array is None or target_array.size == 0:
            raise Exception("No GFEM model target array! Call get_target_array() first ...")

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Filter non-numeric targets
        targets_exclude = ["phase_assemblage"]
        targets_to_visualize = [t for t in targets_to_visualize if t not in targets_exclude]

        # Set plot style and settings
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["axes.facecolor"] = "0.9"
        plt.rcParams["legend.frameon"] = "False"
        plt.rcParams["legend.facecolor"] = "0.9"
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["figure.autolayout"] = "True"

        for i, target in enumerate(targets):
            if target not in targets_to_visualize:
                continue

            # Target label
            target_label = target_labels_map[target]

            # Set filename
            filename = f"{sid}-{perplex_db}-{target.replace("_", "-")}-surf.png"
            if target not in ["assemblage_index", "phase_assemblage_variance"]:
                title = f"{target_label} ({target_units_map[target]})"
            else:
                title = f"{target_label}"

            # Reshape targets into square array
            square_target = target_array[:, i].reshape(res + 1, res + 1)

            # Use discrete colorscale
            if target in ["assemblage_index", "phase_assemblage_variance"]:
                color_discrete = True
            else:
                color_discrete = False

            # Reverse color scale
            if palette in ["grey"]:
                if target in ["phase_assemblage_variance"]:
                    color_reverse = True
                else:
                    color_reverse = False
            else:
                if target in ["phase_assemblage_variance"]:
                    color_reverse = False
                else:
                    color_reverse = True

            # Set colorbar limits for better comparisons
            if not color_discrete:
                non_nan_values = square_target[np.logical_not(np.isnan(square_target))]
                if non_nan_values.size > 0:
                    vmin = np.min(non_nan_values)
                    vmax = np.max(non_nan_values)
                else:
                    vmin = 0
                    vmax = 0
            else:
                vmin = int(np.nanmin(np.unique(square_target)))
                vmax = int(np.nanmax(np.unique(square_target)))

            if color_discrete:
                # Discrete color palette
                num_colors = vmax - vmin
                num_colors = max(num_colors, num_colors // 4)

                if palette == "viridis":
                    if color_reverse:
                        pal = plt.get_cmap("viridis_r", num_colors)
                    else:
                        pal = plt.get_cmap("viridis", num_colors)
                elif palette == "bone":
                    if color_reverse:
                        pal = plt.get_cmap("bone_r", num_colors)
                    else:
                        pal = plt.get_cmap("bone", num_colors)
                elif palette == "pink":
                    if color_reverse:
                        pal = plt.get_cmap("pink_r", num_colors)
                    else:
                        pal = plt.get_cmap("pink", num_colors)
                elif palette == "seismic":
                    if color_reverse:
                        pal = plt.get_cmap("seismic_r", num_colors)
                    else:
                        pal = plt.get_cmap("seismic", num_colors)
                elif palette == "grey":
                    if color_reverse:
                        pal = plt.get_cmap("Greys_r", num_colors)
                    else:
                        pal = plt.get_cmap("Greys", num_colors)
                elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
                    if color_reverse:
                        pal = plt.get_cmap("Blues_r", num_colors)
                    else:
                        pal = plt.get_cmap("Blues", num_colors)

                # Descritize
                color_palette = pal(np.linspace(0, 1, num_colors))
                cmap = ListedColormap(color_palette)

                # Set nan color
                cmap.set_bad(color="0.9")

                # 3D surface
                fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
                ax = fig.add_subplot(111, projection="3d")

                surf = ax.plot_surface(T, P, square_target, cmap=cmap, vmin=vmin, vmax=vmax)

                ax.set_xlabel("T (K)", labelpad=18)
                ax.set_ylabel("P (GPa)", labelpad=18)
                ax.set_zlabel("")
                ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                plt.tick_params(axis="x", which="major")
                plt.tick_params(axis="y", which="major")
                plt.title(title, y=0.95)
                ax.view_init(20, -145)
                ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
                ax.set_facecolor("white")
                cbar = fig.colorbar(surf, ax=ax, label="", shrink=0.6)
                cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                cbar.ax.set_ylim(vmax, vmin)

            else:
                # Continuous color palette
                if palette == "viridis":
                    if color_reverse:
                        cmap = "viridis_r"
                    else:
                        cmap = "viridis"
                elif palette == "bone":
                    if color_reverse:
                        cmap = "bone_r"
                    else:
                        cmap = "bone"
                elif palette == "pink":
                    if color_reverse:
                        cmap = "pink_r"
                    else:
                        cmap = "pink"
                elif palette == "seismic":
                    if color_reverse:
                        cmap = "seismic_r"
                    else:
                        cmap = "seismic"
                elif palette == "grey":
                    if color_reverse:
                        cmap = "Greys_r"
                    else:
                        cmap = "Greys"
                elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
                    if color_reverse:
                        cmap="Blues_r"
                    else:
                        cmap="Blues"

                # Adjust diverging colorscale to center on zero
                if palette == "seismic":
                    vmin = -np.max(
                        np.abs(square_target[np.logical_not(np.isnan(square_target))]))
                    vmax = np.max(
                        np.abs(square_target[np.logical_not(np.isnan(square_target))]))
                else:
                    vmin, vmax = vmin, vmax

                    # Adjust vmin close to zero
                    if vmin <= 1e-4: vmin = 0

                # Set nan color
                cmap = plt.get_cmap(cmap)
                cmap.set_bad(color="0.9")

                # 3D surface
                fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
                ax = fig.add_subplot(111, projection="3d")

                surf = ax.plot_surface(T, P, square_target, cmap=cmap, vmin=vmin, vmax=vmax)

                ax.set_xlabel("T (K)", labelpad=18)
                ax.set_ylabel("P (GPa)", labelpad=18)
                ax.set_zlabel("")
                if vmin != vmax:
                    ax.set_zlim(vmin - (vmin * 0.05), vmax + (vmax * 0.05))
                plt.tick_params(axis="x", which="major")
                plt.tick_params(axis="y", which="major")
                plt.title(title, y=0.95)
                ax.view_init(20, -145)
                ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
                ax.set_facecolor("white")

                # Diverging colorbar
                if palette == "seismic":
                    cbar = fig.colorbar(surf, ax=ax, ticks=[vmin, 0, vmax], label="",
                                        shrink=0.6)

                # Continous colorbar
                else:
                    cbar = fig.colorbar(surf, ax=ax, ticks=np.linspace(vmin, vmax, num=4),
                                        label="", shrink=0.6)

                # Set z and colorbar limits and number formatting
                cbar.ax.yaxis.set_major_formatter(
                    plt.FormatStrFormatter(target_digits_map[target]))
                ax.zaxis.set_major_formatter(
                    plt.FormatStrFormatter(target_digits_map[target]))

                if vmin != vmax:
                    cbar.ax.set_ylim(vmin, vmax)

            # Save the plot to a file
            plt.savefig(f"{fig_dir}/{filename}")

            # Close fig
            plt.close()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize depth profiles !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_depth_profiles(self, type="sub", slab_position="slabtop", figwidth=6.3,
                                  figheight=4.725, fontsize=22):
        """
        """
        # Get model data
        sid = self.sid
        res = self.res
        segs = self.segs
        P_min = self.P_min
        P_max = self.P_max
        pot_Ts = self.pot_Ts
        fig_dir = self.fig_dir
        targets = self.targets
        data_dir = self.data_dir
        perplex_db = self.perplex_db
        model_built = self.model_built
        model_out_dir = self.model_out_dir
        target_units_map = self.target_units_map
        target_digits_map = self.target_digits_map
        target_labels_map = self.target_labels_map
        targets_to_visualize = self.targets_to_visualize

        if not model_built:
            raise Exception("No GFEM model! Call build() first ...")

        if not os.path.exists(data_dir):
            raise Exception(f"Data not found at {data_dir} !")

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Filter targets
        targets_exclude = ["phase_assemblage", "assebmlage_index",
                           "phase_assemblage_variance"]
        targets_to_visualize = [t for t in targets_to_visualize if t not in targets_exclude]

        # Get synthetic endmember compositions
        sids = ["sm000-loi000", f"sm{str(res).zfill(3)}-loi000"]
        df_mids = pd.read_csv("assets/synth-mids.csv")
        df_synth_bench = df_mids[df_mids["SAMPLEID"].isin(sids) & (df_mids["LOI"] == 0)]

        # Mixing array endmembers
        bend = df_synth_bench["SAMPLEID"].iloc[0]
        tend = df_synth_bench["SAMPLEID"].iloc[-1]

        # Set plot style and settings
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["axes.facecolor"] = "0.9"
        plt.rcParams["legend.frameon"] = "False"
        plt.rcParams["legend.facecolor"] = "0.9"
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["figure.autolayout"] = "True"

        # Change endmember sampleids
        if sid == tend:
            sid_lab = "DSUM"
        elif sid == bend:
            sid_lab = "PSUM"
        else:
            sid_lab = sid

        # Colormap
        colormap = plt.colormaps["tab10"]

        for i, target in enumerate(targets):
            if target not in targets_to_visualize:
                continue

                # Plotting
                fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

                Pprof, tprof, labels = [], [], []

                # Plot assemblages and rock properties along geotherms
                if type == "sub":
                    for j, seg in enumerate(segs):
                        if slab_position == "slabtop":
                            filename = (f"{sid}-{perplex_db}-{target.replace("_", "-")}-"
                                        f"depth-profile-sub-slabtop.png")
                            gt = self._get_subduction_geotherm(seg, slab_position="slabtop")
                        elif slab_position == "slabmoho":
                            filename = (f"{sid}-{perplex_db}-{target.replace("_", "-")}-"
                                        f"depth-profile-sub-slabmoho.png")
                            gt = self._get_subduction_geotherm(seg, slab_position="slabmoho")
                        df_gt = self._extract_target_along_geotherm(target, gt)
                        seg_lab = seg.replace("_", " ").lower()
                        labels.append(seg_lab)
                        Pprof.append(df_gt["P"])
                        tprof.append(df_gt[target])
                elif type == "craton":
                    filename = (f"{sid}-{perplex_db}-{target.replace("_", "-")}-"
                                f"depth-profile-craton.png")
                    for j, pot in enumerate(pot_Ts):
                        gt = self._get_mantle_geotherm(pot)
                        df_gt = self._extract_target_along_geotherm(target, gt)
                        labels.append(pot)
                        Pprof.append(df_gt["P"])
                        tprof.append(df_gt[target])
                elif type == "mor":
                    filename = (f"{sid}-{perplex_db}-{target.replace("_", "-")}-"
                                f"depth-profile-mor.png")
                    for j, pot in enumerate(pot_Ts):
                        gt = self._get_mantle_geotherm(
                            pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                            crust_thickness=7e3, litho_thickness=1e3)
                        df_gt = self._extract_target_along_geotherm(target, gt)
                        labels.append(pot)
                        Pprof.append(df_gt["P"])
                        tprof.append(df_gt[target])

                for j, (Pp, tp, lab) in enumerate(zip(Pprof, tprof, labels)):
                    # Plot GFEM model profiles
                    ax1.plot(tp, Pp, "-", linewidth=2, color=colormap(j), label=lab)

                # Target label
                target_label = target_labels_map[target]

                ax1.set_xlabel(f"{target_label} ({target_units_map[target]})")
                ax1.set_ylabel("P (GPa)")
                ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
                ax1.xaxis.set_major_formatter(
                    ticker.FormatStrFormatter(target_digits_map[target]))

                # Vertical text spacing
                text_margin_x = 0.04
                text_margin_y = 0.15
                text_spacing_y = 0.1

                # Convert the primary y-axis data (pressure) to depth
                depth_conversion = lambda P: P * 30
                depth_values = depth_conversion(np.linspace(P_min, P_max, len(Pp)))

                # Create the secondary y-axis and plot depth on it
                ax2 = ax1.secondary_yaxis(
                    "right", functions=(depth_conversion, depth_conversion))
                ax2.set_ylabel("Depth (km)")

                plt.legend(loc="upper left", columnspacing=0, handletextpad=0.2,
                           fontsize=fontsize * 0.833)

                plt.title("Depth Profile")

                # Save the plot to a file
                plt.savefig(f"{fig_dir}/{filename}")

                # Close device
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize geotherm assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_geotherm_assemblages(self, type="sub", slab_position="slabtop",
                                        modal_thresh=5, figwidth=6.3, figheight=4.725,
                                        fontsize=22):
        """
        """
        # Get self attributes
        xi = self.xi
        sid = self.sid
        res = self.res
        loi = self.loi
        segs = self.segs
        P_min = self.P_min
        pot_Ts = self.pot_Ts
        fig_dir = self.fig_dir
        perplex_db = self.perplex_db
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        if not model_built:
            raise Exception("No GFEM model! Call build() first ...")

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        if type not in ["sub", "craton", "mor"]:
            raise Exception("Unrecognized type argument !")

        if type == "sub":
            for seg in segs:
                if P_min < 6:
                    path_top = f"{model_out_dir}/slabmoho-{seg}.tab"
                    path_moho = f"{model_out_dir}/slabtop-{seg}.tab"
                    if not os.path.exists(path_top):
                        raise Exception(f"No werami data found at {path_top} !")
                    if not os.path.exists(path_moho):
                        raise Exception(f"No werami data found at {path_moho} !")
                else:
                    print("  P_min too high to plot subduction geotherms !")
                    return None

        if type == "craton":
            for pot in pot_Ts:
                path = f"{model_out_dir}/craton-{pot}.tab"
                if not os.path.exists(path):
                    raise Exception(f"No werami data found at {path} !")

        if type == "mor":
            for pot in pot_Ts:
                path = f"{model_out_dir}/mor-{pot}.tab"
                if not os.path.exists(path):
                    raise Exception(f"No werami data found at {path} !")

        # Set plot style and settings
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["font.size"] = 22
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["axes.facecolor"] = "0.9"
        plt.rcParams["legend.frameon"] = "False"
        plt.rcParams["legend.facecolor"] = "0.9"
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["figure.autolayout"] = "True"

        # Get unique phases that meet modal_thresh for all gfem models
        all_phases = []

        if "hp" in perplex_db:
            file_patterns = [
                os.path.join("gfems", "*hp*", "mor*.tab"),
                os.path.join("gfems", "*hp*", "craton*.tab"),
                os.path.join("gfems", "*hp*", "slab*.tab"),
            ]
        elif perplex_db == "stx21":
            file_patterns = [
                os.path.join("gfems", "*stx*", "mor*.tab"),
                os.path.join("gfems", "*stx*", "craton*.tab"),
                os.path.join("gfems", "*stx*", "slab*.tab"),
            ]
        else:
            raise Exception("Unrecognized thermodynamic dataset !")

        file_paths = []
        for pattern in file_patterns:
            file_paths.extend(glob.glob(pattern, recursive=True))

        for file_path in file_paths:
            # Read wearmi file
            df = pd.read_csv(file_path, sep="\\s+", skiprows=8)
            df = df.dropna(axis=1, how="all")
            df = df.fillna(0)
            df = df.drop(["T(K)", "P(bar)"], axis=1)

            # Combine duplicate columns
            normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
            duplicate_columns = normalized_columns[
                normalized_columns.duplicated()].unique()
            for base_name in duplicate_columns:
                cols_to_combine = df.loc[:, normalized_columns == base_name]
                combined_col = cols_to_combine.sum(axis=1)
                df[base_name] = combined_col
                df = df.drop(cols_to_combine.columns[1:], axis=1)
                normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)

            # Drop minor phases
            df = df.drop(columns=[col for col in df.columns if
                                  (df[col] < modal_thresh).all()])
            all_phases.extend(df.columns)

        phase_names = sorted(set(all_phases))

        # Assign unique colors to phases that meet modal_thresh
        num_colors = len(phase_names)
        colormap = plt.colormaps["tab20"]
        colors = [colormap(i) for i in range(num_colors)]
        color_map = {col_name: colors[idx] for idx, col_name in enumerate(phase_names)}

        tabfiles, filenames, gts, labels = [], [], [], []

        # Plot assemblages and rock properties along geotherms
        if type == "sub":
            for seg in segs:
                seg_lab = seg.replace("_", "-").lower()
                labels.append(seg_lab)
                if slab_position == "slabtop":
                    tabfiles.append(f"{model_out_dir}/slabtop-{seg}.tab")
                    filenames.append(f"{sid}-{perplex_db}-slabtop-{seg_lab}-assemblages.png")
                    gt = self._get_subduction_geotherm(seg, slab_position="slabtop")
                    gts.append(gt)
                elif slab_position == "slabmoho":
                    tabfiles.append(f"{model_out_dir}/slabmoho-{seg}.tab")
                    filenames.append(f"{sid}-{perplex_db}-slabmoho-{seg_lab}-"
                                     f"assemblages.png")
                    gt = self._get_subduction_geotherm(seg, slab_position="slabmoho")
                    gts.append(gt)
        elif type == "craton":
            for pot in pot_Ts:
                labels.append(pot)
                tabfiles.append(f"{model_out_dir}/craton-{pot}.tab")
                filenames.append(f"{sid}-{perplex_db}-craton-{pot}-assemblages.png")
                gt = self._get_mantle_geotherm(pot)
                gts.append(gt)
        elif type == "mor":
            for pot in pot_Ts:
                labels.append(pot)
                tabfiles.append(f"{model_out_dir}/mor-{pot}.tab")
                filenames.append(f"{sid}-{perplex_db}-mor-{pot}-assemblages.png")
                gt = self._get_mantle_geotherm(
                    pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                    crust_thickness=7e3, litho_thickness=1e3)
                gts.append(gt)

        for tabfile, filename, gt, lab in zip(tabfiles, filenames, gts, labels):
            # Read wearmi file
            df = pd.read_csv(tabfile, sep="\\s+", skiprows=8)
            df = df.dropna(axis=1, how="all")
            df = df.fillna(0)

            # Combine duplicate columns
            normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
            duplicate_columns = normalized_columns[normalized_columns.duplicated()].unique()
            for base_name in duplicate_columns:
                cols_to_combine = df.loc[:, normalized_columns == base_name]
                combined_col = cols_to_combine.sum(axis=1)
                df[base_name] = combined_col
                df = df.drop(cols_to_combine.columns[1:], axis=1)
                normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)

            # Drop minor phases
            df = df.drop(columns=[col for col in df.columns if
                                  (df[col] < modal_thresh).all()])

            # Get rock property profiles
            df_gt = self._extract_target_along_geotherm("density", gt)
            Pg, Tg, rhog = df_gt["P"], df_gt["T"], df_gt["density"]

            # Get water profile
            df_gt = self._extract_target_along_geotherm("H2O", gt)
            H2Og = df_gt["H2O"]

            # Plot assemblages and rock properties
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(figwidth * 2, figheight * 2))
            colors_plot = [color_map[col] for col in
                           df.drop(["T(K)", "P(bar)"], axis=1).columns]

            ax_stack = axes[0]
            ax_stack.stackplot(df["P(bar)"].values / 1e4,
                               df.drop(["T(K)", "P(bar)"], axis=1).values.T,
                               labels=df.drop(["T(K)", "P(bar)"], axis=1).columns,
                               colors=colors_plot)
            cumulative = np.cumsum(df.drop(["T(K)", "P(bar)"], axis=1).values, axis=1)
            for col, color in enumerate(colors_plot):
                ax_stack.plot(df["P(bar)"].values / 1e4, cumulative[:, col],
                              color="black", lw=0.8)

            ax_stack.set_ylim(0, 100)
            ax_stack.set_xlabel("")
            ax_stack.set_xticks([])
            ax_stack.set_ylabel("Cumulative %")
            ax_stack.set_title(
                f"Sample composition: ({xi:.2f} $\\xi$, {loi:.2f} wt.% H$_2$O)")

            ax_line = axes[1]
            ax_line.plot(Pg, rhog, color="black", linewidth=2, label=f"GFEM $\\rho$")
            ax_line.set_xlabel("Pressure (GPa)")
            ax_line.set_ylabel("Density (g/cm$^3$)")
            lines1, labels1 = ax_line.get_legend_handles_labels()

            ax_line_sec = ax_line.twinx()
            ax_line_sec.plot(Pg, H2Og, color="blue", linewidth=2, label="GFEM H$_2$O")
            ax_line_sec.set_ylabel("H$_2$O (wt.%)")
            ax_line_sec.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            if perplex_db == "stx21" or np.all(H2Og == 0):
                ax_line_sec.set_ylim(-0.04, 1)
                ax_line_sec.set_yticks([0])
            lines2, labels2 = ax_line_sec.get_legend_handles_labels()
            ax_line.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
            ax_line.set_title(lab)

            handles, labels = ax_stack.get_legend_handles_labels()
            sorted_handles_labels = sorted(zip(handles, labels),
                                           key=lambda x: phase_names.index(x[1]))
            handles, labels = zip(*sorted_handles_labels)
            labels = [label.split("(")[0].strip() for label in labels]

            fig.legend(handles=handles, labels=labels, loc="upper left",
                       bbox_to_anchor=(0.9, 0.95), ncol=2, title="Stable phases")

            # Vertical text spacing
            text_margin_x = 0.02
            text_margin_y = 0.52
            text_spacing_y = 0.1

            # Save the plot to a file
            plt.savefig(f"{fig_dir}/{filename}")

            # Close device
            plt.close()
            print(f"  Figure saved to: {filename} ...")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize(self):
        """
        """
        # Get self attributes
        results = self.results
        model_built = self.model_built

        try:
            if not model_built:
                raise Exception("No GFEM model! Call build() first ...")
            if not results:
                self._get_results()
                self._get_target_array()
                self._get_pt_array()
            if not self._check_model_array_images(type="mor", gradient=False):
                self._visualize_array_image(type="mor", gradient=False)
            if not self._check_model_array_images(type="sub", gradient=False):
                self._visualize_array_image(type="sub", gradient=False)
            if not self._check_model_array_images(type="craton", gradient=False):
                self._visualize_array_image(type="craton", gradient=False)
            if not self._check_model_array_images(type="mor", gradient=True):
                self._visualize_array_image(type="mor", gradient=True)
            if not self._check_model_array_images(type="sub", gradient=True):
                self._visualize_array_image(type="sub", gradient=True)
            if not self._check_model_array_images(type="craton", gradient=True):
                self._visualize_array_image(type="craton", gradient=True)
            if not self._check_model_array_surfs():
                self._visualize_target_surf()
            if not self._check_model_depth_profile_images():
                self._visualize_depth_profiles(type="mor")
                self._visualize_depth_profiles(type="craton")
                self._visualize_depth_profiles(type="sub", slab_position="slabtop")
                self._visualize_depth_profiles(type="sub", slab_position="slabmoho")
            if not self._check_model_gt_assemblages_images():
                self._visualize_geotherm_assemblages(type="mor")
                self._visualize_geotherm_assemblages(type="craton")
                self._visualize_geotherm_assemblages(type="sub", slab_position="slabtop")
                self._visualize_geotherm_assemblages(type="sub", slab_position="slabmoho")
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in visualize() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.4.           Build GFEM Models             !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # build !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def build(self):
        """
        """
        max_retries = 3
        for retry in range(max_retries):
            if self.model_built:
                break
            try:
                self._print_perplex_info()
                self._run_perplex()
                self._process_perplex_results()
                break
            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"!!! ERROR in build() !!!")
                print(f"{e}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                traceback.print_exc()
                if retry < max_retries - 1:
                    print(f"Retrying in 5 seconds ...")
                    time.sleep(5)
                else:
                    self.model_build_error = True
                    self.model_built = False
                    self.model_error = e
                    return None

        return None

#######################################################
## .2.   Build GFEM for RocMLM training data     !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get sampleids !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sampleids(filepath, batch="all", n_batches=8):
    """
    """
    try:
        if not os.path.exists(filepath):
            raise Exception("Sample data source does not exist!")
        df = pd.read_csv(filepath)
        sampleids = df["SAMPLEID"].values
    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in get_sampleids() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()
        return None

    return sampleids

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gfem itr !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gfem_itr(args):
    """
    """
    try:
        stdout_buffer = io.StringIO()
        perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax = args
        with redirect_stdout(stdout_buffer):
            iteration = GFEMModel(perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax)
            if not iteration.model_built:
                iteration.build()
    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in gfem_itr() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()
        return None

    return iteration, stdout_buffer.getvalue()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build gfem models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_gfem_models(source, perplex_db="hp02", res=64, Pmin=0.1, Pmax=28, Tmin=773,
                      Tmax=2273, sampleids=None, nprocs=os.cpu_count() - 2, verbose=1):
    """
    """
    try:
        if os.path.exists(source) and sampleids is None:
            sampleids = get_sampleids(source)
        elif os.path.exists(source) and sampleids is not None:
            sids = get_sampleids(source)
            if not set(sampleids).issubset(sids):
                raise Exception(f"Sampleids {sampleids} not in source: {source}!")
        else:
            raise Exception(f"Source {source} does not exist!")

        models = []
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Building {perplex_db} GFEM models for {len(sampleids)} samples ...")

        if nprocs is None or nprocs > os.cpu_count():
            nprocs = os.cpu_count() - 2
        if nprocs > len(sampleids):
            nprocs = len(sampleids)

        run_args = [(perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax) for
                    sampleid in sampleids]

        with cf.ProcessPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(gfem_itr, args) for args in run_args]

            for future in tqdm(cf.as_completed(futures), total=len(futures)):
                iteration, stdout_output = future.result()
                models.append(iteration)
                if verbose >= 2:
                    print(stdout_output, end="")

        # Get successful models
        gfems = [model for model in models if not model.model_build_error]
        error_count = len([model for model in models if model.model_build_error])

        if error_count > 0:
            print(f"Total GFEM models with errors: {error_count}")
        else:
            print("All GFEM models built successfully !")

        print(":::::::::::::::::::::::::::::::::::::::::::::")

    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in build_gfem_models() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()
        return None

    return gfems

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    try:
        gfems = {}
        sources = {"m": "assets/synth-mids.csv", "r": "assets/synth-rnds.csv"}
        res = 64

        # Build GFEM models
        for name, src in sources.items():
            P_min, P_max, T_min, T_max = 0.1, 8.1, 273, 1973
            gfems[name + "hp"] = build_gfem_models(
                src, "hp02", res, P_min, P_max, T_min, T_max)

            P_min, P_max, T_min, T_max = 8.1, 136.1, 773, 4273
            gfems[name + "stx"] = build_gfem_models(
                src, "stx21", res, P_min, P_max, T_min, T_max)

        # Visualize models
        for name, models in gfems.items():
            for m in models:
                m.visualize()

    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in main() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()
        return None

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("GFEM models built and visualized !")
    print("=============================================")

    return None

if __name__ == "__main__":
    main()
