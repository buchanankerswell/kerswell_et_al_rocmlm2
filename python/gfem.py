#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import time
import glob
import yaml
import shutil
import textwrap
import traceback
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures as cf
from scipy.interpolate import LinearNDInterpolator

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

#######################################################
## .1.              GFEMModel class              !!! ##
#######################################################
class GFEMModel:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, sid, config_yaml, verbose=1):
        """
        """
        self.sid = sid
        self.config_yaml = config_yaml
        self.verbose = verbose

        self.model_built = False

        self._load_global_options()
        self._load_perplex_options()
        self._load_target_maps()
        self._get_sample_features()
        self._get_normalized_sample_comp()
        self._generate_perplex_config_strings()
        self._check_for_existing_model()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_global_options(self):
        """
        """
        try:
            if not os.path.exists(self.config_yaml):
                raise Exception(f"No config_yaml found at {self.config_yaml}!")

            with open(self.config_yaml, "r") as file:
                config_data = yaml.safe_load(file)

            global_options = config_data["global_options"]

            self.segs = global_options["segs"]
            self.seed = global_options["seed"]
            self.digits = global_options["digits"]
            self.pot_Ts = global_options["pot_Ts"]
            self.targets_to_visualize = global_options["targets_to_visualize"]

            # Plot settings
            plt.rcParams.update({
                "figure.dpi": 300,
                "savefig.bbox": "tight",
                "axes.facecolor": "0.9",
                "legend.frameon": False,
                "legend.facecolor": "0.9",
                "legend.loc": "upper left",
                "legend.fontsize": "small",
                "figure.autolayout": True
            })

        except Exception as e:
            print(f"Error in _load_perplex_options(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_perplex_options(self):
        """
        """
        try:
            if not os.path.exists(self.config_yaml):
                raise Exception(f"No config_yaml found at {self.config_yaml}!")

            with open(self.config_yaml, "r") as file:
                config_data = yaml.safe_load(file)

            perplex_options = config_data["perplex_options"]
            perplex_datasets = config_data["perplex_datasets"]

            self.res = perplex_options["res"]
            self.P_min = perplex_options["P_min"]
            self.P_max = perplex_options["P_max"]
            self.T_min = perplex_options["T_min"]
            self.T_max = perplex_options["T_max"]
            self.source = perplex_options["source"]
            self.ox_gfem = perplex_options["ox_gfem"]
            self.perplex_db = perplex_options["perplex_db"]

            if self.perplex_db not in ["hp02", "hp11", "hp622", "hp633", "stx21"]:
                raise ValueError(f"Unrecognized thermodynamic dataset: {self.perplex_db}")

            self.timeout = (self.res**2) * 3
            self.model_out_dir = f"gfems/{self.sid}_{self.perplex_db}_{self.res}"
            self.log_file = f"{self.model_out_dir}/log-{self.sid}"
            self.fig_dir = f"figs/{self.model_out_dir}"

            # Get the configuration for the selected database
            config = perplex_datasets[self.perplex_db]
            if not config:
                raise ValueError(f"Unknown database {self.perplex_db}")

            # Assign values from config
            self.T_melt = config["T_melt"]
            self.melt_mod = config["melt_mod"]
            self.td_data_file = config["td_data_file"]
            self.sl_data_file = config["sl_data_file"]
            self.melt_is_fluid = config["melt_is_fluid"]
            self.em_exclude = sorted(config["em_exclude"])
            self.sl_include = sorted(config["sl_include"])
            self.fluid_in_properties = config["fluid_in_properties"]
            self.fluid_in_assemblages = config["fluid_in_assemblages"]

        except Exception as e:
            print(f"Error in _load_perplex_options(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_target_maps(self):
        """
        """
        try:
            # Initialize werami output maps
            self.werami_output_map = {
               f"T(K)":                "T",
               f"P(bar)":              "P",
               f"N,g":                 "mass",
               f"n,mol":               "moles",
               f"rho,kg/m3":           "density",
               f"{self.melt_mod},vo%": "melt_fraction",
               f"alpha,1/K":           "expansivity",
               f"beta,1/bar":          "compressibility",
               f"H,J/mol":             "molar_enthalpy",
               f"S,J/K/mol":           "molar_entropy",
               f"V,J/bar/mol":         "molar_volume",
               f"cp,J/K/mol":          "molar_heat_capacity",
               f"cp/cv":               "heat_capacity_ratio",
               f"vp,km/s":             "Vp",
               f"vp_P":                "Vp_dP",
               r"vp_{T}":              "Vp_dT",
               f"vs,km/s":             "Vs",
               r"vs_{P}":              "Vs_dP",
               r"vs_{T}":              "Vs_dT",
               f"v0,km/s":             "sound_velocity",
               r"v0_{P}":              "sound_velocity_dP",
               r"v0_{T}":              "sound_velocity_dT",
               f"vp/vs":               "Vp/Vs",
               f"Ks,bar":              "bulk_modulus",
               r"Ks_{P}":              "bulk_modulus_dP",
               r"Ks_{T},bar/K":        "bulk_modulus_dT",
               f"Gs,bar":              "shear_modulus",
               f"Gs_P":                "shear_modulus_dP",
               r"Gs_{T},bar/K":        "shear_modulus_dT",
               f"G,J/mol":             "molar_gibbs_free_energy",
               f"Gruneisen_T":         "gruneisen_thermal_ratio",
               f"assemblage_i":        "assemblage_index"
            }

            # Initialize target maps for units, formatting, and labels
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

            # Add oxide components from ox_gfem
            for ox in self.ox_gfem:
                self.werami_output_map[f"{ox},wt%"] = ox
                self.target_units_map[ox] = "wt.%"
                self.target_digits_map[ox] = "%.1f"
                self.target_labels_map[ox] = ox

            # Set targets excluding P and T
            self.targets = [t for t in self.target_labels_map.keys() if t not in ["P", "T"]]

        except Exception as e:
            print(f"Error in _load_target_maps(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_sample_features(self):
        """
        """
        try:
            df = pd.read_csv(self.source)

            if self.sid not in df["SAMPLEID"].values:
                raise ValueError(f"Sample {self.sid} not found in dataset {self.source}!")

            self.features = [col for col in df.columns if col != "SAMPLEID"]
            self.sample_features = df.loc[
                df["SAMPLEID"] == self.sid, self.features].values.flatten().tolist()

        except Exception as e:
            print(f"Error in _get_sample_features(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_normalized_sample_comp(self):
        """
        """
        try:
            df = pd.read_csv(self.source)

            # Get numeric columns
            ox_pca = list(df.select_dtypes(include=[float, int]).columns)

            if len(self.ox_gfem) > len(ox_pca):
                raise ValueError("Not enough oxides in PCA to satisfy ox_gfem!")

            # Define oxide ordering and sort
            ox_order = ["K2O", "NA2O", "CAO", "FEO", "MGO", "AL2O3", "SIO2", "TIO2", "CR2O3"]
            ox_mapping = {oxide: idx for idx, oxide in enumerate(ox_order)}
            ox_pca = sorted(ox_pca, key=lambda x: ox_mapping.get(x, float("inf")))
            ox_gfem = sorted(self.ox_gfem, key=lambda x: ox_mapping.get(x, float("inf")))

            subset_df = df[df["SAMPLEID"] == self.sid]
            if subset_df.empty:
                raise ValueError(f"Sample {self.sid} not found in dataset {self.source}!")

            self.ox_gfem = ox_gfem
            self.h2o = float(subset_df["H2O"].iloc[0]) if "H2O" in ox_pca else None
            self.xi = float(subset_df["XI_FRAC"].iloc[0]) if "XI_FRAC" in ox_pca else None

            # Get sample composition
            sample_comp = [float(subset_df[oxide].iloc[0])
                           if oxide in subset_df.columns else 0 for oxide in ox_pca]

            # Normalize if required
            if len(sample_comp) != len(ox_gfem):
                sample_comp = [sample_comp[ox_pca.index(o)]
                               if o in ox_pca else 0 for o in ox_gfem]
            tot_comp = sum(c for c, o in zip(sample_comp, ox_gfem) if o != "H2O" and c > 0)
            self.norm_sample_comp = [
                round((c / tot_comp) * 100, self.digits) if c > 0 and o != "H2O" else c
                for c, o in zip(sample_comp, ox_gfem)
            ]

        except Exception as e:
            print(f"Error in _get_normalized_sample_comp(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _generate_perplex_config_strings(self):
        """
        """
        try:
            oxides_list = {"K2O": "K2O", "NA2O": "Na2O", "CAO": "CaO", "FEO": "FeO",
                           "MGO": "MgO", "AL2O3": "Al2O3", "SIO2": "SiO2", "TIO2": "TiO2",
                           "CR2O3": "Cr2O3", "H2O": "H2O"}
            oxides = [oxides_list[key] for key in oxides_list if key in self.ox_gfem]
            oxides_string = "\n".join(oxides)
            norm_sample_comp_string = " ".join(map(str, self.norm_sample_comp))
            em_exclude_string = "\n".join(self.em_exclude)
            sl_include_string = "\n".join(self.sl_include)
            if self.perplex_db not in ["hp11", "hp622", "hp633"]:
                oxides_string = oxides_string.upper()

            # Generate configuration strings
            if "hp" in self.perplex_db:
                self.build_config = (
                    f"{self.sid}\n"
                    f"td-data\n"
                    f"build-options\n"
                    f"N\n"
                    f"2\n"
                    f"N\n"
                    f"N\n"
                    f"N\n"
                    f"{oxides_string}\n\n"
                    f"5\n"
                    f"N\n"
                    f"2\n"
                    f"{self.T_min} {self.T_max}\n"
                    f"{self.P_min * 1e4} {self.P_max * 1e4}\n"
                    f"Y\n"
                    f"{norm_sample_comp_string}\n"
                    f"N\n"
                    f"Y\n"
                    f"N\n"
                    f"{em_exclude_string}\n\n"
                    f"Y\n"
                    f"solution-models\n"
                    f"{sl_include_string}\n\n"
                    f"{self.sid}\n"
                )

                self.werami_target_config = (
                    f"{self.sid}\n"
                    f"2\n"
                    f"36\n"
                    f"1\n"
                    f"{self.fluid_in_properties}\n"
                    f"N\n\n"
                    f"2\n"
                    f"7\n"
                    f"{self.melt_mod}\n"
                    f"24\n"
                    f"0\n"
                    f"N\n\n"
                    f"5\n"
                    f"0\n"
                )

                self.werami_phase_config = (
                    f"{self.sid}\n"
                    f"2\n"
                    f"25\n"
                    f"N\n"
                    f"{self.fluid_in_assemblages}\n"
                    f"N\n\n"
                    f"0\n"
                )

            elif self.perplex_db == "stx21":
                self.build_config = (
                    f"{self.sid}\n"
                    f"td-data\n"
                    f"build-options\n"
                    f"N\n"
                    f"2\n"
                    f"N\n"
                    f"N\n"
                    f"N\n"
                    f"{oxides_string}\n\n"
                    f"N\n"
                    f"2\n"
                    f"{self.T_min} {self.T_max}\n"
                    f"{self.P_min * 1e4} {self.P_max * 1e4}\n"
                    f"Y\n"
                    f"{norm_sample_comp_string}\n"
                    f"N\n"
                    f"Y\n"
                    f"N\n"
                    f"{em_exclude_string}\n\n"
                    f"Y\n"
                    f"solution-models\n"
                    f"{sl_include_string}\n\n"
                    f"{self.sid}\n"
                )

                self.werami_target_config = (
                    f"{self.sid}\n"
                    f"2\n"
                    f"36\n"
                    f"1\n"
                    f"N\n\n"
                    f"2\n"
                    f"24\n"
                    f"0\n"
                    f"N\n\n"
                    f"0\n"
                )

                self.werami_phase_config = (
                    f"{self.sid}\n"
                    f"2\n"
                    f"25\n"
                    f"N\n"
                    f"N\n\n"
                    f"0\n"
                )

        except Exception as e:
            print(f"Error in _generate_perplex_config_strings(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_for_existing_model(self):
        """
        """
        if os.path.exists(self.model_out_dir):
            if (os.path.exists(f"{self.model_out_dir}/results.csv") and
                os.path.exists(f"{self.model_out_dir}/assemblages.csv")):
                self.model_built = True

                if self.verbose >= 2:
                    print(f"  Found {self.perplex_db} GFEM model for sample {self.sid}!")

                try:
                    self._get_results()
                    self._get_target_array()
                    self._get_pt_array()

                except Exception as e:
                    print(f"Error in _check_for_existing_model(): {e}")
                    traceback.print_exc()

            else:
                shutil.rmtree(self.model_out_dir)
                os.makedirs(self.model_out_dir, exist_ok=True)
        else:
            os.makedirs(self.model_out_dir, exist_ok=True)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.1.                Perple_X                 !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_subduction_geotherm(self, segment="Central_Cascadia", slab_position="slabmoho"):
        """
        """
        try:
            path = f"assets/D80/{segment}.txt"

            if not os.path.exists(path):
                raise Exception(f"Subduction geotherm {segment} not found at {path}!")

            ref_cols = ["slab_depth", "unk", "depth", "T"]
            litho_P_gradient = 35 # (km/GPa)
            columns_to_keep = ["P", "T"]

            # Load and filter geotherm data
            geotherm = pd.read_csv(path, header=None, names=ref_cols, sep=r"\s+")
            slab_depth_filter = (7 if slab_position == "slabmoho" else 0
                                 if slab_position == "slabtop" else None)

            if slab_depth_filter is None:
                raise Exception(f"Unrecognized position argument '{slab_position}'!")

            geotherm = geotherm[geotherm["slab_depth"] == slab_depth_filter]
            geotherm = geotherm[geotherm["depth"] < 240]
            geotherm["P"] = geotherm["depth"] / litho_P_gradient
            geotherm["T"] = geotherm["T"] + 273
            geotherm.sort_values(by=["P"], inplace=True)

            # Filter based on minimum and maximum P and T values
            geotherm = geotherm[
                (geotherm["P"] >= self.P_min) & (geotherm["P"] <= self.P_max) &
                (geotherm["T"] >= self.T_min) & (geotherm["T"] <= self.T_max)
            ]

            # Select relevant columns and round values
            geotherm = geotherm[["P", "T"]].round(3)

        except Exception as e:
            print(f"Error in _get_subduction_geotherm(): {e}")
            return None

        return geotherm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_mantle_geotherm(self, mantle_potential=1573, Qs=55e-3, Ts=273, A1=1.0e-6,
                             A2=2.2e-8, k1=2.3, k2=3.0, mantle_adiabat=0.5e-3,
                             crust_thickness=35e3, litho_thickness=150e3):
        """
        """
        try:
            array_size = (self.res) * 10
            litho_P_gradient = 35e3
            Z_min = self.P_min * litho_P_gradient
            Z_max = self.P_max * litho_P_gradient
            z = np.linspace(Z_min, Z_max, array_size)

            # Initialize temperature array
            T_geotherm = np.zeros(array_size)

            # Calculate temperatures at different layers
            Qt2 = Qs - (A1 * crust_thickness)
            Tt2 = Ts + (Qs * crust_thickness / k1) - (A1 / 2 / k1 * crust_thickness**2)
            Tt3 = Tt2 + (Qt2 * litho_thickness / k2) - (A2 / 2 / k2 * litho_thickness**2)
            for j in range(array_size):
                potential_temp = mantle_potential + mantle_adiabat * z[j]

                if z[j] <= crust_thickness:
                    T_geotherm[j] = Ts + (Qs / k1 * z[j]) - (A1 / (2 * k1) * z[j]**2)
                elif crust_thickness < z[j] <= litho_thickness + crust_thickness:
                    T_geotherm[j] = (Tt2 + (Qt2 / k2 * (z[j] - crust_thickness)) -
                                     (A2 / (2 * k2) * (z[j] - crust_thickness)**2))
                elif z[j] > litho_thickness + crust_thickness:
                    T_geotherm[j] = (Tt3 + mantle_adiabat *
                                     (z[j] - crust_thickness - litho_thickness))

                # Ensure temperature does not exceed potential temperature
                if T_geotherm[j] >= potential_temp:
                    T_geotherm[j] = potential_temp

            # Calculate pressure values
            P_geotherm = z / litho_P_gradient

            # Create a DataFrame for the geotherm data
            geotherm = pd.DataFrame(
                {"P": P_geotherm, "T": T_geotherm}).sort_values(by=["P", "T"])

        except Exception as e:
            print(f"Error in _get_mantle_geotherm(): {e}")
            return None

        return geotherm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_td_data(self):
        """
        """
        try:
            shutil.copy(self.td_data_file, f"{self.model_out_dir}/td-data")
            shutil.copy(self.sl_data_file, f"{self.model_out_dir}/solution-models")

        except FileNotFoundError as e:
            print(f"Error: One or more source files not found: {e}")

        except Exception as e:
            print(f"An error occurred while copying files: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_build_options(self):
        """
        """
        try:
            build_options = (
                f"composition_system     wt\n"
                f"composition_phase      wt\n"
                f"intermediate_savdyn    T\n"
                f"intermediate_savrpc    T\n"
                f"warn_no_limit          F\n"
                f"grid_levels            1 1\n"
                f"x_nodes                {self.res // 4} {self.res}\n"
                f"y_nodes                {self.res // 4} {self.res}\n"
                f"bounds                 VRH\n"
                f"vrh/hs_weighting       0.5\n"
                f"Anderson-Gruneisen     F\n"
                f"explicit_bulk_modulus  T\n"
                f"melt_is_fluid          {self.melt_is_fluid}\n"
                f"T_melt                 {self.T_melt}\n"
                f"poisson_test           F\n"
                f"poisson_ratio          on 0.31\n"
                f"seismic_output         some\n"
                f"auto_refine_file       F\n"
                f"seismic_data_file      F\n"
            )

            with open(f"{self.model_out_dir}/build-options", "w") as file:
                file.write(build_options)

            with open(f"{self.model_out_dir}/perplex_plot_option.dat", "w") as file:
                file.write("numeric_field_label T")

        except IOError as e:
            print(f"Error writing BUILD options files: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_build_config(self):
        """
        """
        try:
            with open(f"{self.model_out_dir}/build-config", "w") as file:
                file.write(self.build_config)

        except IOError as e:
            print(f"Error writing BUILD configuration file: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_vertex_config(self):
        """
        """
        try:
            with open(f"{self.model_out_dir}/vertex-minimize", "w") as file:
                file.write(f"{self.sid}")

        except IOError as e:
            print(f"Error writing VERTEX configuration file: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_werami_config(self):
        """
        """
        try:
            with open(f"{self.model_out_dir}/werami-targets", "w") as file:
                file.write(self.werami_target_config)

            with open(f"{self.model_out_dir}/werami-phases", "w") as file:
                file.write(self.werami_phase_config)

        except IOError as e:
            print(f"Error writing WERAMI configuration files: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_werami_geotherm_configs(self):
        """
        """
        try:
            werami_geotherms_top = []
            werami_geotherms_mor = []
            werami_geotherms_moho = []
            werami_geotherms_craton = []

            for seg in self.segs:
                # Retrieve geotherms for slab top and slab moho positions
                geotherm_top = self._get_subduction_geotherm(seg, slab_position="slabtop")
                geotherm_moho = self._get_subduction_geotherm(seg, slab_position="slabmoho")

                # Convert pressure from GPa to bar and keep only relevant columns
                geotherm_top["P"] *= 1e4
                geotherm_top = geotherm_top[["T", "P"]]
                geotherm_moho["P"] *= 1e4
                geotherm_moho = geotherm_moho[["T", "P"]]

                # Save geotherm data to CSV files
                geotherm_top.to_csv(f"{self.model_out_dir}/gt-slabtop-{seg}", sep="\t",
                                    index=False, header=False, float_format="%.6E")
                geotherm_moho.to_csv(f"{self.model_out_dir}/gt-slabmoho-{seg}", sep="\t",
                                     index=False, header=False, float_format="%.6E")

                # Create Werami configuration entries based on perplex database type
                for position in ["top", "moho"]:
                    if "hp" in self.perplex_db:
                        g = (f"{self.sid}\n4\n2\ngt-slab{position}-{seg}\n1\n25\nN\n"
                             f"{self.fluid_in_assemblages}\n0\n")
                    elif self.perplex_db == "stx21":
                        g = f"{self.sid}\n4\n2\ngt-slab{position}-{seg}\n1\n25\nN\n0\n"
                    if position == "top":
                        werami_geotherms_top.append(g)
                    else:
                        werami_geotherms_moho.append(g)

            # Process geotherms for cratonic regions
            for pot in self.pot_Ts:
                geotherm = self._get_mantle_geotherm(pot)
                geotherm["P"] *= 1e4
                geotherm = geotherm[["T", "P"]]
                geotherm.to_csv(f"{self.model_out_dir}/gt-craton-{pot}", sep="\t",
                                index=False, header=False, float_format="%.6E")

                # Additional mantle geotherm for mor
                geotherm = self._get_mantle_geotherm(
                    pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                    crust_thickness=7e3, litho_thickness=1e3)
                geotherm["P"] *= 1e4
                geotherm = geotherm[["T", "P"]]
                geotherm.to_csv(f"{self.model_out_dir}/gt-mor-{pot}", sep="\t",
                                index=False, header=False, float_format="%.6E")

                # Create Werami configuration entries for craton and mor
                for type, pot in zip(["craton", "mor"], [pot, pot]):
                    if "hp" in self.perplex_db:
                        g = (f"{self.sid}\n4\n2\ngt-{type}-{pot}\n1\n25\nN\n"
                             f"{self.fluid_in_assemblages}\n0\n")
                    elif self.perplex_db == "stx21":
                        g = f"{self.sid}\n4\n2\ngt-{type}-{pot}\n1\n25\nN\n0\n"
                    if type == "craton":
                        werami_geotherms_craton.append(g)
                    else:
                        werami_geotherms_mor.append(g)

            # Write out the Werami configuration files
            for i, g in enumerate(werami_geotherms_top):
                with open(f"{self.model_out_dir}/werami-slabtop-{self.segs[i]}", "w") as f:
                    f.write(g)
            for i, g in enumerate(werami_geotherms_moho):
                with open(f"{self.model_out_dir}/werami-slabmoho-{self.segs[i]}", "w") as f:
                    f.write(g)
            for i, g in enumerate(werami_geotherms_craton):
                with open(f"{self.model_out_dir}/werami-craton-{self.pot_Ts[i]}", "w") as f:
                    f.write(g)
            for i, g in enumerate(werami_geotherms_mor):
                with open(f"{self.model_out_dir}/werami-mor-{self.pot_Ts[i]}", "w") as f:
                    f.write(g)

        except Exception as e:
            print(f"Error writing WERAMI geotherm configuration files: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_pssect_config(self):
        """
        """
        try:
            with open(f"{self.model_out_dir}/pssect-draw", "w") as file:
                file.write(f"{self.sid}\nN")

        except IOError as e:
            print(f"Error writing PSSECT configuration file: {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _write_perplex_config(self):
        """
        """
        try:
            if self.model_built:
                return None

            self._write_perplex_td_data()
            self._write_perplex_build_options()
            self._write_perplex_build_config()
            self._write_perplex_vertex_config()
            self._write_perplex_werami_config()
            self._write_perplex_werami_geotherm_configs()
            self._write_perplex_pssect_config()

        except Exception as e:
            print(f"Error in _write_perplex_config(): {e}")
            traceback.print_exc()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_perplex_info(self):
        """
        """
        try:
            # Prepare formatted strings for output
            oxwrp = textwrap.fill(", ".join(self.ox_gfem), width=80)
            tgwrp = textwrap.fill(", ".join(self.targets), width=80,
                                  subsequent_indent="                  ")
            ftwrp = textwrap.fill(", ".join(self.features), width=80,
                                  subsequent_indent="                  ")
            emwrp = textwrap.fill(", ".join(self.em_exclude), width=80,
                                  subsequent_indent="    ")
            slwrp = textwrap.fill(", ".join(self.sl_include), width=80,
                                  subsequent_indent="    ")

            # Print model information
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Perple_X model: {self.sid} {self.perplex_db}")
            print("---------------------------------------------")
            print(f"  PT resolution:  {self.res}")
            print(f"  P range:        {self.P_min:.1f} - {self.P_max:.1f} GPa")
            print(f"  T range:        {self.T_min:.0f} - {self.T_max:.0f} K")
            print(f"  Sampleid:       {self.sid}")
            print(f"  Source:         {self.source}")
            print(f"  GFEM sys.:      {oxwrp}")
            print(f"  Thermo. data:   {self.perplex_db}")
            print(f"  Model out dir:  {self.model_out_dir}")
            print(f"  Targets:        {tgwrp}")
            print(f"  Features:       {ftwrp}")
            print("  --------------------")

            # Print configuration details
            print(f"  Configuring Perple_X with {self.perplex_db} database and comp (wt.%):")
            max_oxide_width = max(len(oxide) for oxide in self.ox_gfem)
            max_comp_width = max(len(str(comp)) for comp in self.norm_sample_comp)
            max_width = max(max_oxide_width, max_comp_width)
            print(" ".join([f"  {oxide:<{max_width}}" for oxide in self.ox_gfem]))
            print(" ".join([f"  {comp:<{max_width}}" for comp in self.norm_sample_comp]))
            print("  --------------------")

            # Print excluded endmembers and included solution models
            print(f"  Excluded endmembers:")
            print(f"    {emwrp}")
            print("  --------------------")
            print(f"  Included solution models:")
            print(f"    {slwrp}")
            print("  --------------------")

        except Exception as e:
            print(f"Error in _print_perplex_info(): {e}")

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
            print(f"Error in _replace_in_file(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_command_line_program(self, program_path, config_file):
        """
        """
        try:
            os.chmod(program_path, 0o755)
            relative_program_path = f"../../{program_path}"

            if self.verbose >= 1:
                print(f"  Running {program_path} with {config_file} ...")

            with open(config_file, "rb") as input_stream:
                process = subprocess.Popen(
                    [relative_program_path],
                    stdin=input_stream,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    cwd=self.model_out_dir
                )

            stdout, stderr = process.communicate(timeout=self.timeout)

            # Log output and errors
            with open(self.log_file, "a") as log:
                log.write(stdout.decode())
                log.write(stderr.decode())

            if process.returncode != 0:
                raise RuntimeError(f"Error with perplex program '{program_path}'!")

            if self.verbose >= 2:
                print(f"{stdout.decode()}")

        except subprocess.CalledProcessError as e:
            print(f"Error in _run_command_line_program(): {e}")
        except IOError as e:
            print(f"IO Error occurred in _run_command_line_program(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _perplex_build(self):
        """
        """
        try:
            if self.model_built:
                return None

            # Check for existence of configuration file
            config_path = f"{self.model_out_dir}/build-config"
            if not os.path.exists(config_path):
                raise Exception("No config! Call _configure_perplex_model() first ...")

            # Check for build output
            build_file = f"{self.model_out_dir}/{self.sid}.dat"
            if os.path.exists(build_file):
                if self.verbose >= 1:
                    print(f"  Perple_X build found!")
                return None

            # Run build
            self._run_command_line_program("Perple_X/build", config_path)

        except Exception as e:
            print(f"Error in _perplex_build(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _perplex_vertex(self):
        """
        """
        try:
            if self.model_built:
                return None

            # Check for existence of configuration file
            config_path = f"{self.model_out_dir}/vertex-minimize"
            if not os.path.exists(config_path):
                raise Exception("No config! Call _configure_perplex_model() first ...")

            # Check for build output
            build_file = f"{self.model_out_dir}/{self.sid}.dat"
            if not os.path.exists(build_file):
                raise Exception("No build output! Call _perplex_build() first ...")

            # Check for vertex file
            vertex_file = f"{self.model_out_dir}/{self.sid}.tof"
            if os.path.exists(vertex_file):
                if self.verbose >= 1:
                    print(f"  Perple_X vertex found!")
                return None

            # Run vertex
            self._run_command_line_program("Perple_X/vertex", config_path)

        except Exception as e:
            print(f"Error in _perplex_vertex(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _perplex_werami(self):
        """
        """
        try:
            if self.model_built:
                return None

            config_files = {}
            config_files["targets"] = f"{self.model_out_dir}/werami-targets"
            config_files["phases"] = f"{self.model_out_dir}/werami-phases"

            # Check for existence of configuration files
            for name, path in config_files.items():
                if not os.path.exists(path):
                    raise Exception("No config! Call _configure_perplex_model() first ...")

            # Check for build output
            build_file = f"{self.model_out_dir}/{self.sid}.dat"
            if not os.path.exists(build_file):
                raise Exception("No build output! Call _perplex_build() first ...")

            # Check for vertex file
            vertex_file = f"{self.model_out_dir}/{self.sid}.tof"
            if not os.path.exists(vertex_file):
                raise Exception("No build output! Call _perplex_vertex() first ...")

            # Configure slabtop and slabmoho files for segments if applicable
            for seg in self.segs:
                if self.P_min < 6:
                    config_files[f"slabtop-{seg}"] = (
                        f"{self.model_out_dir}/werami-slabtop-{seg}")
                    config_files[f"slabmoho-{seg}"] = (
                        f"{self.model_out_dir}/werami-slabmoho-{seg}")

            # Add craton and mor configurations for potential temperatures
            for pot in self.pot_Ts:
                config_files[f"craton-{pot}"] = f"{self.model_out_dir}/werami-craton-{pot}"
                config_files[f"mor-{pot}"] = f"{self.model_out_dir}/werami-mor-{pot}"

            # Modify build options
            self._replace_in_file(f"{self.model_out_dir}/build-options",
                                  {"Anderson-Gruneisen     F": "Anderson-Gruneisen     T"})

            # Run werami for each configuration file and manage output files
            for name, path in config_files.items():
                self._run_command_line_program("Perple_X/werami", path)

                if name == "targets":
                    shutil.copy(f"{self.model_out_dir}/{self.sid}_1.tab",
                                f"{self.model_out_dir}/{name}.tab")
                    shutil.copy(f"{self.model_out_dir}/{self.sid}_2.tab",
                                f"{self.model_out_dir}/supplemental.tab")
                    os.remove(f"{self.model_out_dir}/{self.sid}_1.tab")
                    os.remove(f"{self.model_out_dir}/{self.sid}_2.tab")
                else:
                    shutil.copy(f"{self.model_out_dir}/{self.sid}_1.tab",
                                f"{self.model_out_dir}/{name}.tab")
                    os.remove(f"{self.model_out_dir}/{self.sid}_1.tab")

        except Exception as e:
            print(f"Error in _perplex_werami(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _perplex_pssect(self):
        """
        """
        try:
            if self.model_built:
                return None

            # Check for existence of configuration file
            config_path = f"{self.model_out_dir}/pssect-draw"
            if not os.path.exists(config_path):
                raise Exception("No config! Call _configure_perplex_model() first ...")

            # Run pssect
            self._run_command_line_program("Perple_X/pssect", config_path)

            # Manage output file
            shutil.copy(f"{self.model_out_dir}/"
                        f"{self.sid}_assemblages.txt",
                        f"{self.model_out_dir}/assemblages.txt")
            os.remove(f"{self.model_out_dir}/{self.sid}_assemblages.txt")

        except Exception as e:
            print(f"Error in _perplex_build(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _build_perplex_model(self):
        """
        """
        try:
            if self.model_built:
                return None

            self._write_perplex_config()
            self._print_perplex_info()
            self._perplex_build()
            self._perplex_vertex()
            self._perplex_werami()
            self._perplex_pssect()

        except Exception as e:
            print(f"Error in _build_perplex_model(): {e}")

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.        Post-process GFEM Models         !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_perplex_targets(self):
        """
        """
        try:
            werami_targets = f"{self.model_out_dir}/targets.tab"
            werami_supplemental = f"{self.model_out_dir}/supplemental.tab"
            results = {v: [] for v in self.werami_output_map.values()}

            for w, werami_output in enumerate([werami_targets, werami_supplemental]):
                with open(werami_output, "r") as file:
                    headers = None

                    # Find headers in the file
                    for line in file:
                        if "T(K)" in line and "P(bar)" in line:
                            headers = line.split()
                            break

                    if headers is None:
                        raise Exception("No column headers found!")

                    index_map = {i: self.werami_output_map[header]
                                 for i, header in enumerate(headers)
                                 if header in self.werami_output_map}

                    # Read and process data lines
                    for line in file:
                        vals = line.split()
                        for i, val in enumerate(vals):
                            if i in index_map:
                                variable_name = index_map[i]
                                if w > 0 and variable_name in ["T", "P"]:
                                    continue

                                try:
                                    value = (
                                        float(val) if not np.isnan(float(val)) else np.nan)
                                    # Unit conversions
                                    if variable_name == "P":
                                        value /= 1e4  # Convert bar to GPa
                                    if variable_name == "density":
                                        value /= 1e3  # Convert kg/m3 to g/cm3
                                    if variable_name == "assemblage_index":
                                        value = int(value)

                                    results[variable_name].append(value)

                                except ValueError:
                                    results[variable_name].append(np.nan)

            # Ensure all results lists are of equal length
            max_len = max(len(vals) for vals in results.values())
            for k, v in results.items():
                if len(v) == 0:
                    results[k] = [np.nan] * max_len

            if all(len(vals) == 0 for vals in results.values()):
                raise Exception(f"No data was read from the files!")

        except Exception as e:
            print(f"Error in _read_perplex_targets(): {e}")
            return None

        return results

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_assemblages(self):
        """
        """
        try:
            if self.model_built:
                return None

            assemblage_dict = {}
            perplex_assemblages_csv = f"{self.model_out_dir}/assemblages.csv"

            # Read from CSV if it exists
            if os.path.exists(perplex_assemblages_csv):
                df = pd.read_csv(perplex_assemblages_csv)
                return df["assemblage"].to_list()

            # Read from text file if CSV does not exist
            perplex_assemblages = f"{self.model_out_dir}/assemblages.txt"
            with open(perplex_assemblages, "r") as file:
                for i, line in enumerate(file, start=1):
                    cleaned_line = line.strip()
                    if " - " in cleaned_line:
                        phases_string = cleaned_line.split(" - ", 1)[1]
                    else:
                        phases_string = cleaned_line

                    assemblages = phases_string.split()
                    assemblage_dict[i] = assemblages

        except Exception as e:
            print(f"Error in _read_perplex_assemblages(): {e}")
            return None

        return assemblage_dict

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _encode_assemblages(self, assemblages):
        """
        """
        try:
            unique_assemblages = {}
            encoded_assemblages = []

            # Generate unique assemblage indices
            for assemblage in assemblages:
                assemblage_tuple = tuple(sorted(assemblage))
                if assemblage_tuple and not any(
                        np.isnan(item) for item in assemblage_tuple if
                        isinstance(item, (int, float))):
                    if assemblage_tuple not in unique_assemblages:
                        unique_assemblages[assemblage_tuple] = len(unique_assemblages) + 1

            # Create DataFrame and save to CSV
            df = pd.DataFrame(list(unique_assemblages.items()),
                              columns=["assemblage", "index"])
            df["assemblage"] = df["assemblage"].apply(" ".join)
            df = df[["index", "assemblage"]]
            assemblages_csv = f"{self.model_out_dir}/assemblages.csv"
            df.to_csv(assemblages_csv, index=False)

            # Encode the input assemblages
            for assemblage in assemblages:
                if assemblage == "":
                    encoded_assemblages.append(np.nan)
                else:
                    encoded_assemblage = unique_assemblages[tuple(sorted(assemblage))]
                    encoded_assemblages.append(encoded_assemblage)

        except Exception as e:
            print(f"Error in _encode_assemblages(): {e}")
            return None

        return encoded_assemblages

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_perplex_results(self):
        """
        """
        try:
            if self.model_built:
                return None

            # Check for required files
            perplex_targets = f"{self.model_out_dir}/targets.tab"
            if not os.path.exists(perplex_targets):
                raise Exception("No models found! Call _build_perplex_model() first ...")

            perplex_assemblages = f"{self.model_out_dir}/assemblages.txt"
            if not os.path.exists(perplex_assemblages):
                raise Exception("No models found! Call _build_perplex_model() first ...")

            if self.verbose >= 1:
                print(f"  Reading Perple_X output from {self.model_out_dir} ...")

            # Read results and assemblages
            results = self._read_perplex_targets()
            assemblages = self._read_perplex_assemblages()

            # Construct phase assemblages
            phase_assemblages = []
            for index in results.get("assemblage_index"):
                if np.isnan(index):
                    phase_assemblages.append("")
                else:
                    phases = sorted(set(assemblages[index]))
                    phase_assemblages.append(" ".join(phases))

            # Calculate assemblage variance
            results["phase_assemblage"] = phase_assemblages
            assemblage_variance = []
            for assemblage in results.get("phase_assemblage"):
                if assemblage is None:
                    assemblage_variance.append(np.nan)
                else:
                    unique_phases = set(assemblage)
                    count = len(unique_phases)
                    assemblage_variance.append(count)

            results["phase_assemblage_variance"] = assemblage_variance
            results.pop("assemblage_index")

            encoded_assemblages = self._encode_assemblages(results["phase_assemblage"])
            results["assemblage_index"] = encoded_assemblages

            # Create results df
            results_df = pd.DataFrame.from_dict(results)

            # Create feature df
            f_cols = [c + "_FEAT" for c in self.features]
            f_vals = self.sample_features
            feature_data = {c: [v] * len(results_df) for c, v in zip(f_cols, f_vals)}
            feature_df = pd.DataFrame(feature_data)

            # Concat feature df with results
            df = pd.concat([results_df, feature_df], axis=1)

            if self.verbose >= 1:
                print(f"  Writing Perple_X results to {self.model_out_dir}/results.csv ...")

            # Write results to CSV
            df.to_csv(f"{self.model_out_dir}/results.csv", index=False)
            self.model_built = True

        except Exception as e:
            print(f"Error in _process_perplex_results(): {e}")
            traceback.print_exc()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_results(self):
        """
        """
        try:
            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")

            filepath = f"{self.model_out_dir}/results.csv"
            if not os.path.exists(filepath):
                raise Exception("No results to read!")

            if self.verbose >= 2:
                print(f"  Reading results from {filepath} ...")

            df = pd.read_csv(filepath)
            self.results = {column: np.array(values)
                            for column, values in df.to_dict(orient="list").items()}

        except Exception as e:
            print(f"Error in _get_results(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_pt_array(self):
        """
        """
        try:
            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")

            if not self.results:
                raise Exception("No GFEM model results! Call get_results() first ...")

            P, T = self.results["P"].copy(), self.results["T"].copy()

            self.pt_array = np.stack((P, T), axis=-1).copy()

        except Exception as e:
            print(f"Error in _get_pt_array(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_target_array(self):
        """
        """
        try:
            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")

            if not self.results:
                raise Exception("No GFEM model results! Call get_results() first ...")

            targets_exclude = ["phase_assemblage"]
            target_array_list = [
                self.results[key] for key in self.targets if key not in targets_exclude
            ]

            self.target_array = np.stack(target_array_list, axis=-1).copy()

        except Exception as e:
            print(f"Error in _get_target_array(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _extract_target_along_geotherm(self, target, geotherm):
        """
        """
        try:
            P_vals = self.results["P"]
            T_vals = self.results["T"]
            target_vals = self.results[target]

            gfem_points = np.vstack((T_vals, P_vals)).T

            geo_P, geo_T = geotherm["P"], geotherm["T"]
            geo_points = np.array([geo_T, geo_P]).T

            interpolator = LinearNDInterpolator(gfem_points, target_vals)
            target_interp = interpolator(geo_points)

            df = pd.DataFrame({"P": geo_P, "T": geo_T, target: target_interp})

        except Exception as e:
            print(f"Error in _extract_target_along_geotherm(): {e}")
            return None

        return df

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.3.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _check_model_array_images(self, geotherm_type="sub", gradient=False):
        """
        """
        # Define targets to exclude based on whether gradients are requested
        targets_exclude = ["phase_assemblage"] + (
            ["assemblage_index", "phase_assemblage_variance"] if gradient else []
        )

        # Filter targets to visualize based on exclusions
        self.targets_to_visualize = [
            t for t in self.targets_to_visualize if t not in targets_exclude
        ]

        existing_figs = []
        for i, target in enumerate(self.targets):
            if target not in self.targets_to_visualize:
                continue

            # Construct file path
            path = (f"{self.fig_dir}/{self.sid}-{self.perplex_db}-"
                    f"{target.replace('_', '-')}"
                    f"{'-grad' if gradient else ''}-{geotherm_type}.png")

            # Check for existence of the image file
            if os.path.exists(path):
                existing_figs.append(path)

        # Check if all required images exist
        return len(existing_figs) == len(self.targets_to_visualize)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_array_surfs(self):
        """
        """
        try:
            targets_exclude = ["phase_assemblage"]

            # Filter out excluded targets
            self.targets_to_visualize = [
                t for t in self.targets_to_visualize if t not in targets_exclude]

            existing_figs = [
                target for target in self.targets_to_visualize
                if os.path.exists(f"{self.fig_dir}/{self.sid}-{self.perplex_db}-"
                                  f"{target.replace('_', '-')}-surf.png")
            ]

        except Exception as e:
            print(f"Error in _check_model_array_surfs(): {e}")
            return None

        # Check if the number of existing figures matches the number of targets to visualize
        return len(existing_figs) == len(self.targets_to_visualize)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_depth_profile_images(self):
        """
        """
        # Define targets to exclude
        targets_exclude = [
            "phase_assemblage", "assemblage_index", "phase_assemblage_variance"]

        self.targets_to_visualize = [
            t for t in self.targets_to_visualize if t not in targets_exclude]

        existing_figs = []
        for i, target in enumerate(self.targets):
            if target not in self.targets_to_visualize:
                continue

            # Check paths based on pressure conditions
            if self.P_min < 6:
                for position in ["sub-slabtop", "sub-slabmoho"]:
                    path = (f"{self.fig_dir}/{self.sid}-{self.perplex_db}-"
                            f"{target.replace('_', '-')}-depth-profile-{position}.png")
                    if os.path.exists(path):
                        existing_figs.append(path)

            for geotherm_type in ["craton", "mor"]:
                path = (f"{self.fig_dir}/{self.sid}-{self.perplex_db}-"
                        f"{target.replace('_', '-')}-depth-profile-{geotherm_type}.png")
                if os.path.exists(path):
                    existing_figs.append(path)

        # Determine the expected number of existing images based on pressure
        expected_count = len(self.targets_to_visualize) * (4 if self.P_min < 6 else 2)

        return len(existing_figs) == expected_count

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_gt_assemblages_images(self):
        """
        """
        existing_figs = []

        # Check for images based on segments
        for seg in self.segs:
            if self.P_min < 6:
                for position in ["slabtop", "slabmoho"]:
                    seg_lab = seg.replace("_", "-").lower()
                    path = (f"{self.fig_dir}/{self.sid}-{self.perplex_db}-{position}-"
                            f"{seg_lab}-assemblages.png")
                    if os.path.exists(path):
                        existing_figs.append(path)

        # Check for images based on potential temperatures
        for pot in self.pot_Ts:
            for type in ["craton", "mor"]:
                path = (f"{self.fig_dir}/{self.sid}-{self.perplex_db}-{type}-"
                        f"{pot}-assemblages.png")
                if os.path.exists(path):
                    existing_figs.append(path)

        # Determine the expected number of existing images based on pressure
        if self.P_min < 6:
            expected_count = (len(self.segs) * 2) + (len(self.pot_Ts) * 2)
        else:
            expected_count = len(self.pot_Ts) * 2

        return len(existing_figs) == expected_count

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_geotherms_for_array_image(self, geotherm_type):
        """
        """
        try:
            geotherms = {}
            if geotherm_type == "sub":
                for seg in self.segs:
                    if self.P_min < 6:
                        geotherms[seg] = {
                            "slabtop": self._get_subduction_geotherm(
                                seg, slab_position="slabtop"),
                            "slabmoho": self._get_subduction_geotherm(
                                seg, slab_position="slabmoho"),
                        }
            elif geotherm_type == "craton":
                for pot in self.pot_Ts:
                    geotherms[pot] = self._get_mantle_geotherm(pot)
            elif geotherm_type == "mor":
                for pot in self.pot_Ts:
                    geotherms[pot] = self._get_mantle_geotherm(
                        pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                        crust_thickness=7e3, litho_thickness=1e3)

        except Exception as e:
            print(f"Error in _get_geotherms_for_array_image(): {e}")
            return None

        return geotherms

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_square_target_for_array_image(self, index, gradient):
        """
        """
        try:
            square_target = self.target_array[:, index].reshape(self.res, self.res)

            if gradient:
                edges_x = cv2.Sobel(square_target, cv2.CV_64F, 1, 0, ksize=3)
                edges_y = cv2.Sobel(square_target, cv2.CV_64F, 0, 1, ksize=3)
                max_val = np.nanmax(square_target) if np.any(~np.isnan(square_target)) else 0
                square_target = (np.sqrt(edges_x**2 + edges_y**2) / max_val
                                 if max_val > 0 else np.zeros_like(edges_x))

        except Exception as e:
            print(f"Error in _get_square_target_for_array_image(): {e}")
            return None

        return square_target

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_colormap_for_array_image(self, palette, reverse, discrete):
        """
        """
        try:
            palettes = {
                "viridis": "viridis_r" if reverse else "viridis",
                "bone": "bone_r" if reverse else "bone",
                "pink": "pink_r" if reverse else "pink",
                "seismic": "seismic_r" if reverse else "seismic",
                "grey": "Greys_r" if reverse else "Greys",
                "default": "Blues_r" if reverse else "Blues"
            }
            pal = palettes.get(palette, palettes["default"])
            cmap = (ListedColormap(plt.colormaps[pal](np.linspace(0, 1, 256)))
                    if discrete else plt.get_cmap(pal))
            cmap.set_bad(color="0.9")

        except Exception as e:
            print(f"Error in _get_colormap_for_array_image(): {e}")
            return None

        return cmap

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_vmin_vmax_for_array_image(self, square_target, discrete):
        """
        """
        try:
            if discrete:
                vmin = int(np.nanmin(np.unique(square_target)))
                vmax = int(np.nanmax(np.unique(square_target)))
            else:
                non_nan_values = square_target[~np.isnan(square_target)]
                vmin, vmax = ((np.min(non_nan_values), np.max(non_nan_values))
                              if non_nan_values.size > 0 else (0, 0))

        except Exception as e:
            print(f"Error in _get_vmin_vmax_for_array_image(): {e}")
            return None, None

        return vmin, vmax

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _plot_geotherms_on_array_image(self, ax, geotherms, geotherm_type, linestyles):
        """
        """
        try:
            if geotherm_type == "sub":
                for seg, gt in geotherms.items():
                    ax.plot(gt["slabtop"]["T"], gt["slabtop"]["P"], linestyle="-",
                            color="black", linewidth=2, label=seg)
                    ax.plot(gt["slabmoho"]["T"], gt["slabmoho"]["P"], linestyle="--",
                            color="black", linewidth=2)
            elif geotherm_type in ["craton", "mor"]:
                for i, (pot, gt) in enumerate(geotherms.items()):
                    ax.plot(gt["T"], gt["P"], linestyle=linestyles[i], color="black",
                            linewidth=2, label=pot)

        except Exception as e:
            print(f"Error in _plot_geotherms_on_array_image(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_image(self, geotherm_type="sub", palette="bone", gradient=False,
                               figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        try:
            plt.rcParams["font.size"] = fontsize
            linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]
            P, T = self.results["P"], self.results["T"]
            extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")
            if not self.results:
                raise Exception("No results! Call get_results() first ...")
            if self.target_array is None or self.target_array.size == 0:
                raise Exception("No target array! Call get_target_array() first ...")
            if geotherm_type not in ["sub", "craton", "mor"]:
                raise Exception("Unrecognized geotherm_type argument!")

            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir, exist_ok=True)

            # Filter targets
            exclude_targets = (
                ["assemblage_index", "phase_assemblage_variance"] if gradient else
                ["phase_assemblage"]
            )
            trg_vis = [t for t in self.targets_to_visualize if t not in exclude_targets]

            # Get geotherms
            geotherms = self._get_geotherms_for_array_image(geotherm_type)

            for i, target in enumerate(self.targets):
                if target not in trg_vis:
                    continue

                target_label = self.target_labels_map[target]
                title = (f"{target_label} ({self.target_units_map.get(target, '')})")
                filename = (f"{self.sid}-{self.perplex_db}-{target.replace('_', '-')}-"
                            f"{geotherm_type}.png")

                # Get square array
                square_target = self._get_square_target_for_array_image(i, gradient)

                if gradient:
                    filename = (f"{self.sid}-{self.perplex_db}-{target.replace('_', '-')}-"
                                f"grad-{geotherm_type}.png")
                    title = f"{target_label} Gradient"

                # Determine color mapping
                color_discrete = target in ["assemblage_index", "phase_assemblage_variance"]
                color_reverse = target not in ["phase_assemblage_variance"]
                cmap = self._get_colormap_for_array_image(
                    palette, color_reverse, color_discrete)
                vmin, vmax = self._get_vmin_vmax_for_array_image(
                    square_target, color_discrete)

                # Plot the array in 2d
                fig, ax = plt.subplots(figsize=(figwidth, figheight))
                im = ax.imshow(square_target, extent=extent, aspect="auto", cmap=cmap,
                               origin="lower", vmin=vmin, vmax=vmax)

                # Plot geotherms based on type
                self._plot_geotherms_on_array_image(ax, geotherms, geotherm_type, linestyles)

                # Finalize the plot
                ax.set_xlabel("T (K)")
                ax.set_ylabel("P (GPa)")
                cbar = plt.colorbar(
                    im, ax=ax, ticks=np.linspace(vmin, vmax, num=4), label="")
                cbar.ax.yaxis.set_major_formatter(
                    plt.FormatStrFormatter(self.target_digits_map[target]))
                plt.title(title)

                # Save fig
                plt.savefig(f"{self.fig_dir}/{filename}")
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print(f"Error in _visualize_array_image(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_surf(self, palette="bone", gradient=False, figwidth=6.3,
                              figheight=4.725, fontsize=22):
        """
        """
        try:
            plt.rcParams["font.size"] = fontsize
            P = self.results["P"].reshape(self.res, self.res)
            T = self.results["T"].reshape(self.res, self.res)

            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")
            if not self.results:
                raise Exception("No results! Call get_results() first ...")
            if self.target_array is None or self.target_array.size == 0:
                raise Exception("No target array! Call get_target_array() first ...")

            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir, exist_ok=True)

            # Filter targets
            targets_exclude = ["phase_assemblage"]
            trg_vis = [t for t in self.targets_to_visualize if t not in targets_exclude]

            for i, target in enumerate(self.targets):
                if target not in trg_vis:
                    continue

                target_label = self.target_labels_map[target]
                filename = (f"{self.sid}-{self.perplex_db}-{target.replace('_', '-')}-"
                            f"surf.png")
                title = (f"{target_label} ({self.target_units_map.get(target, '')})")

                # Get square array
                square_target = self._get_square_target_for_array_image(i, gradient)

                if gradient:
                    filename = (f"{self.sid}-{self.perplex_db}-{target.replace('_', '-')}-"
                                f"grad-surf.png")
                    title = f"{target_label} Gradient"

                # Determine color mapping
                color_discrete = target in ["assemblage_index", "phase_assemblage_variance"]
                color_reverse = target not in ["phase_assemblage_variance"]
                cmap = self._get_colormap_for_array_image(
                    palette, color_reverse, color_discrete)
                vmin, vmax = self._get_vmin_vmax_for_array_image(
                    square_target, color_discrete)

                # Plot the array in 3d
                fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
                ax = fig.add_subplot(111, projection="3d")
                surf = ax.plot_surface(T, P, square_target, cmap=cmap, vmin=vmin, vmax=vmax)

                # Finalize the plot
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
                cbar = fig.colorbar(
                    surf, ax=ax, ticks=np.linspace(vmin, vmax, num=4), label="", shrink=0.6)
                cbar.ax.yaxis.set_major_formatter(
                    plt.FormatStrFormatter(self.target_digits_map[target]))
                ax.zaxis.set_major_formatter(
                    plt.FormatStrFormatter(self.target_digits_map[target]))

                # Save fig
                plt.savefig(f"{self.fig_dir}/{filename}")
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print(f"Error in _visualize_array_surf(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _generate_subduction_profiles(self, target, slab_position):
        """
        """
        try:
            Pprof, tprof, labels = [], [], []
            for j, seg in enumerate(self.segs):
                filename = (f"{self.sid}-{self.perplex_db}-{target.replace('_', '-')}-"
                            f"depth-profile-sub-{slab_position}.png")

                gt = self._get_subduction_geotherm(seg, slab_position=slab_position)

                df_gt = self._extract_target_along_geotherm(target, gt)

                labels.append(seg.replace("_", " ").lower())
                Pprof.append(df_gt["P"])
                tprof.append(df_gt[target])

        except Exception as e:
            print(f"Error in _generate_subduction_profiles(): {e}")
            return None, None, None, None

        return filename, Pprof, tprof, labels

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _generate_mantle_profiles(self, target, geotherm_type):
        """
        """
        try:
            Pprof, tprof, labels = [], [], []
            filename = (f"{self.sid}-{self.perplex_db}-{target.replace('_', '-')}-"
                        f"depth-profile-{geotherm_type}.png")

            for j, pot in enumerate(self.pot_Ts):
                if geotherm_type == "mor":
                    gt = self._get_mantle_geotherm(
                        pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                        crust_thickness=7e3, litho_thickness=1e3)
                elif geotherm_type == "craton":
                    gt = self._get_mantle_geotherm(pot)

                df_gt = self._extract_target_along_geotherm(target, gt)

                labels.append(pot)
                Pprof.append(df_gt["P"])
                tprof.append(df_gt[target])

        except Exception as e:
            print(f"Error in _generate_mantle_profiles(): {e}")
            return None, None, None, None

        return filename, Pprof, tprof, labels

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_depth_profiles(self, geotherm_type="sub", slab_position="slabtop",
                                  figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        try:
            plt.rcParams["font.size"] = fontsize
            colormap = plt.colormaps["tab10"]

            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")
            if not self.results:
                raise Exception("No results! Call get_results() first ...")
            if self.target_array is None or self.target_array.size == 0:
                raise Exception("No target array! Call get_target_array() first ...")
            if geotherm_type not in ["sub", "craton", "mor"]:
                raise Exception("Unrecognized geotherm_type argument!")

            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir, exist_ok=True)

            # Filter targets
            targets_exclude = [
                "phase_assemblage", "assebmlage_index", "phase_assemblage_variance"]
            trg_vis = [t for t in self.targets_to_visualize if t not in targets_exclude]

            for i, target in enumerate(self.targets):
                if target not in trg_vis:
                    continue

                fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

                if geotherm_type == "sub":
                    filename, Pprof, tprof, labels = self._generate_subduction_profiles(
                        target, slab_position)
                elif geotherm_type in ["craton", "mor"]:
                    filename, Pprof, tprof, labels = self._generate_mantle_profiles(
                        target, geotherm_type)

                # Plot depth profiles along geotherm
                for j, (Pp, tp, lab) in enumerate(zip(Pprof, tprof, labels)):
                    ax1.plot(tp, Pp, "-", linewidth=2, color=colormap(j), label=lab)

                # Finalize plot
                target_label = self.target_labels_map[target]
                ax1.set_xlabel(f"{target_label} ({self.target_units_map[target]})")
                ax1.set_ylabel("P (GPa)")
                ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
                ax1.xaxis.set_major_formatter(
                    ticker.FormatStrFormatter(self.target_digits_map[target]))
                text_margin_x = 0.04
                text_margin_y = 0.15
                text_spacing_y = 0.1
                depth_conversion = lambda P: P * 30
                depth_values = depth_conversion(np.linspace(
                    self.P_min, self.P_max, len(Pp)))
                ax2 = ax1.secondary_yaxis(
                    "right", functions=(depth_conversion, depth_conversion))
                ax2.set_ylabel("Depth (km)")
                plt.legend(loc="upper left", columnspacing=0, handletextpad=0.2,
                           fontsize=fontsize * 0.833)
                plt.title("Depth Profile")

                # Save fig
                plt.savefig(f"{self.fig_dir}/{filename}")
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print(f"Error in _visualize_depth_profiles(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_werami_geotherm_paths(self, geotherm_type):
        """
        """
        try:
            if geotherm_type == "sub":
                for seg in self.segs:
                    if self.P_min < 6:
                        path_top = f"{self.model_out_dir}/slabmoho-{seg}.tab"
                        path_moho = f"{self.model_out_dir}/slabtop-{seg}.tab"
                        if not os.path.exists(path_top):
                            raise Exception(f"No werami data found at {path_top}!")
                        if not os.path.exists(path_moho):
                            raise Exception(f"No werami data found at {path_moho}!")
                    else:
                        print("  P_min too high to plot subduction geotherms!")
                        return None

            if geotherm_type == "craton":
                for pot in self.pot_Ts:
                    path = f"{self.model_out_dir}/craton-{pot}.tab"
                    if not os.path.exists(path):
                        raise Exception(f"No werami data found at {path}!")

            if geotherm_type == "mor":
                for pot in self.pot_Ts:
                    path = f"{self.model_out_dir}/mor-{pot}.tab"
                    if not os.path.exists(path):
                        raise Exception(f"No werami data found at {path}!")

        except Exception as e:
            print(f"Error in _check_werami_geotherm_paths(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _prepare_geotherm_paths_for_assemblages(self, geotherm_type, slab_position):
        """
        """
        try:
            tabfiles, filenames, gts, labels = [], [], [], []

            if geotherm_type == "sub":
                for seg in self.segs:
                    if self.P_min < 6:
                        labels.append(seg.replace("_", "-").lower())
                        tabfiles.append(f"{self.model_out_dir}/{slab_position}-{seg}.tab")
                        filenames.append(f"{self.sid}-{self.perplex_db}-{slab_position}-"
                                         f"{labels[-1]}-assemblages.png")
                        gts.append(self._get_subduction_geotherm(
                            seg, slab_position=slab_position))

            elif geotherm_type in ["craton", "mor"]:
                for pot in self.pot_Ts:
                    labels.append(pot)
                    tabfiles.append(f"{self.model_out_dir}/{geotherm_type}-{pot}.tab")
                    filenames.append(f"{self.sid}-{self.perplex_db}-{geotherm_type}-"
                                     f"{pot}-assemblages.png")
                    if geotherm_type == "craton":
                        gts.append(self._get_mantle_geotherm(pot))
                    elif geotherm_type == "mor":
                        gts.append(self._get_mantle_geotherm(
                            pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                            crust_thickness=7e3, litho_thickness=1e3))

        except Exception as e:
            print(f"Error in _prepare_geotherm_paths_for_assemblages(): {e}")
            return None, None, None, None

        return tabfiles, filenames, gts, labels

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_phase_names_for_assemblages(self, modal_thresh):
        """
        """
        try:
            all_phases = []
            if "hp" in self.perplex_db:
                file_patterns = [
                    os.path.join("gfems", "*hp*", "mor*.tab"),
                    os.path.join("gfems", "*hp*", "craton*.tab"),
                    os.path.join("gfems", "*hp*", "slab*.tab"),
                ]
            elif self.perplex_db == "stx21":
                file_patterns = [
                    os.path.join("gfems", "*stx*", "mor*.tab"),
                    os.path.join("gfems", "*stx*", "craton*.tab"),
                    os.path.join("gfems", "*stx*", "slab*.tab"),
                ]

            file_paths = []
            for pattern in file_patterns:
                file_paths.extend(glob.glob(pattern, recursive=True))
            for file_path in file_paths:
                df = pd.read_csv(file_path, sep="\\s+", skiprows=8)
                df = df.dropna(axis=1, how="all")
                df = df.fillna(0)
                df = df.drop(["T(K)", "P(bar)"], axis=1)
                normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
                duplicate_columns = normalized_columns[
                    normalized_columns.duplicated()].unique()
                for base_name in duplicate_columns:
                    cols_to_combine = df.loc[:, normalized_columns == base_name]
                    combined_col = cols_to_combine.sum(axis=1)
                    df[base_name] = combined_col
                    df = df.drop(cols_to_combine.columns[1:], axis=1)
                    normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
                df = df.drop(columns=[col for col in df.columns if
                                      (df[col] < modal_thresh).all()])
                all_phases.extend(df.columns)

            phase_names = sorted(set(all_phases))
            num_colors = len(phase_names)
            colormap = plt.colormaps["tab20"]
            colors = [colormap(i) for i in range(num_colors)]
            color_map = {col_name: colors[idx] for idx, col_name in enumerate(phase_names)}

        except Exception as e:
            print(f"Error in _get_phase_names_for_assemblages(): {e}")
            return None, None

        return phase_names, color_map

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_geotherm_assemblage_from_tabfile(self, tabfile, modal_thresh):
        """
        """
        try:
            df = pd.read_csv(tabfile, sep="\\s+", skiprows=8)
            df = df.dropna(axis=1, how="all")
            df = df.fillna(0)

            normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
            duplicate_columns = normalized_columns[normalized_columns.duplicated()].unique()

            for base_name in duplicate_columns:
                cols_to_combine = df.loc[:, normalized_columns == base_name]
                combined_col = cols_to_combine.sum(axis=1)
                df[base_name] = combined_col
                df = df.drop(cols_to_combine.columns[1:], axis=1)
                normalized_columns = df.columns.str.replace(r"\.\d+$", "", regex=True)

            df = df.drop(columns=[
                col for col in df.columns if (df[col] < modal_thresh).all()])

        except Exception as e:
            print(f"Error in _get_geotherm_assemblage_from_tabfile(): {e}")
            return None

        return df

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_geotherm_assemblages(self, geotherm_type="sub", slab_position="slabtop",
                                        modal_thresh=5, figwidth=6.3, figheight=4.725,
                                        fontsize=22):
        """
        """
        try:
            plt.rcParams["font.size"] = fontsize

            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")
            if not self.results:
                raise Exception("No results! Call get_results() first ...")
            if self.target_array is None or self.target_array.size == 0:
                raise Exception("No target array! Call get_target_array() first ...")
            if geotherm_type not in ["sub", "craton", "mor"]:
                raise Exception("Unrecognized geotherm_type argument!")

            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir, exist_ok=True)

            # Check for werami output along geotherm path
            self._check_werami_geotherm_paths(geotherm_type)

            # Handle geotherms, tabfile paths, and labels
            tabfiles, filenames, gts, labels = self._prepare_geotherm_paths_for_assemblages(
                geotherm_type, slab_position)

            # Get unique phase names and colormap
            phase_names, color_map = self._get_phase_names_for_assemblages(modal_thresh)

            for tabfile, filename, gt, lab in zip(tabfiles, filenames, gts, labels):
                # Get assemblages data and assign colors
                df = self._get_geotherm_assemblage_from_tabfile(tabfile, modal_thresh)
                cumulative = np.cumsum(df.drop(["T(K)", "P(bar)"], axis=1).values, axis=1)
                colors_plot = [
                    color_map[col] for col in df.drop(["T(K)", "P(bar)"], axis=1).columns]

                # Get geotherm density and H2O profiles
                df_gt = self._extract_target_along_geotherm("density", gt)
                Pg, Tg, rhog = df_gt["P"], df_gt["T"], df_gt["density"]
                df_gt = self._extract_target_along_geotherm("H2O", gt)
                H2Og = df_gt["H2O"]

                # Plot stable assemblage stackplot
                fig, axes = plt.subplots(
                    nrows=2, ncols=1, figsize=(figwidth * 2, figheight * 2))

                ax_stack = axes[0]
                ax_stack.stackplot(
                    df["P(bar)"].values / 1e4, df.drop(["T(K)", "P(bar)"], axis=1).values.T,
                    labels=df.drop(["T(K)", "P(bar)"], axis=1).columns, colors=colors_plot)

                for col, color in enumerate(colors_plot):
                    ax_stack.plot(
                        df["P(bar)"].values / 1e4, cumulative[:, col], color="black", lw=0.8)

                # Finalize stackplot
                ax_stack.set_ylim(0, 100)
                ax_stack.set_xlabel("")
                ax_stack.set_xticks([])
                ax_stack.set_ylabel("Cumulative %")
                if self.xi and self.h2o:
                    ax_stack.set_title(
                        f"Composition: ({self.xi:.2f} $\\xi$, {self.h2o:.2f} wt.% H$_2$O)")
                else:
                    ax_stack.set_title(
                        f"Sample: {self.sid}")

                # Plot density and H2O profiles
                ax_line = axes[1]
                ax_line.plot(Pg, rhog, color="black", linewidth=2, label=f"GFEM $\\rho$")

                # Finalize profile plot
                ax_line.set_xlabel("Pressure (GPa)")
                ax_line.set_ylabel("Density (g/cm$^3$)")
                lines1, labels1 = ax_line.get_legend_handles_labels()
                ax_line_sec = ax_line.twinx()
                ax_line_sec.plot(Pg, H2Og, color="blue", linewidth=2, label="GFEM H$_2$O")
                ax_line_sec.set_ylabel("H$_2$O (wt.%)")
                ax_line_sec.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                if self.perplex_db == "stx21" or np.all(H2Og == 0):
                    ax_line_sec.set_ylim(-0.04, 1)
                    ax_line_sec.set_yticks([0])
                lines2, labels2 = ax_line_sec.get_legend_handles_labels()
                ax_line.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                ax_line.set_title(lab)

                # Add legend
                handles, labels = ax_stack.get_legend_handles_labels()
                sorted_handles_labels = sorted(zip(handles, labels),
                                               key=lambda x: phase_names.index(x[1]))
                handles, labels = zip(*sorted_handles_labels)
                labels = [label.split("(")[0].strip() for label in labels]
                fig.legend(handles=handles, labels=labels, loc="upper left",
                           bbox_to_anchor=(0.9, 0.95), ncol=2, title="Stable phases")

                # Save fig
                plt.savefig(f"{self.fig_dir}/{filename}")
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print(f"Error in _visualize_geotherm_assemblages(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize(self):
        """
        """
        try:
            if not self.model_built:
                raise Exception("No GFEM model! Call build() first ...")

            self._get_results()
            self._get_target_array()
            self._get_pt_array()

            if not self._check_model_array_images(geotherm_type="mor", gradient=False):
                self._visualize_array_image(geotherm_type="mor", gradient=False)
            if not self._check_model_array_images(geotherm_type="sub", gradient=False):
                self._visualize_array_image(geotherm_type="sub", gradient=False)
            if not self._check_model_array_images(geotherm_type="craton", gradient=False):
                self._visualize_array_image(geotherm_type="craton", gradient=False)
            if not self._check_model_array_images(geotherm_type="mor", gradient=True):
                self._visualize_array_image(geotherm_type="mor", gradient=True)
            if not self._check_model_array_images(geotherm_type="sub", gradient=True):
                self._visualize_array_image(geotherm_type="sub", gradient=True)
            if not self._check_model_array_images(geotherm_type="craton", gradient=True):
                self._visualize_array_image(geotherm_type="craton", gradient=True)
            if not self._check_model_array_surfs():
                self._visualize_array_surf()

            if not self._check_model_depth_profile_images():
                self._visualize_depth_profiles(geotherm_type="mor")
                self._visualize_depth_profiles(geotherm_type="craton")
                self._visualize_depth_profiles(geotherm_type="sub", slab_position="slabtop")
                self._visualize_depth_profiles(geotherm_type="sub", slab_position="slabmoho")

            if not self._check_model_gt_assemblages_images():
                self._visualize_geotherm_assemblages(geotherm_type="mor")
                self._visualize_geotherm_assemblages(geotherm_type="craton")
                self._visualize_geotherm_assemblages(
                    geotherm_type="sub", slab_position="slabtop")
                self._visualize_geotherm_assemblages(
                    geotherm_type="sub", slab_position="slabmoho")

        except Exception as e:
            print(f"Error in visualize(): {e}")

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.4.           Build GFEM Models             !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def build(self):
        """
        """
        max_retries = 3
        for retry in range(max_retries):
            if self.model_built:
                break

            try:
                self._build_perplex_model()
                self._process_perplex_results()
                self._get_results()
                self._get_target_array()
                self._get_pt_array()
                break

            except Exception as e:
                print(f"Error in build(): {e}")
                traceback.print_exc()

                if retry < max_retries - 1:
                    print(f"Retrying in 5 seconds ...")
                    time.sleep(5)
                else:
                    self.model_built = False

#######################################################
## .2.   Build GFEM for RocMLM training data     !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sample_ids(filepath, batch="all", n_batches=8):
    """
    """
    try:
        if not os.path.exists(filepath):
            raise Exception("Sample data source does not exist!")

        df = pd.read_csv(filepath)
        sids = df["SAMPLEID"].values

    except Exception as e:
        print(f"Error in get_sample_ids(): {e}")
        return None

    return sids

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gfem_itr(args):
    """
    """
    try:
        sid, config_yaml = args
        iteration = GFEMModel(sid, config_yaml)

        if not iteration.model_built:
            iteration.build()

    except Exception as e:
        print(f"Error in gfem_itr(): {e}")
        return None

    return iteration

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_gfem_models(config_yaml, nprocs=os.cpu_count() - 2, verbose=1):
    """
    """
    try:
        if not os.path.exists(config_yaml):
            raise Exception(f"No config_yaml found at {config_yaml}!")

        with open(config_yaml, "r") as file:
            config_data = yaml.safe_load(file)

        perplex_options = config_data["perplex_options"]
        perplex_db = perplex_options["perplex_db"]
        source = perplex_options["source"]

        sids = get_sample_ids(source)
        nprocs = min(nprocs, os.cpu_count() - 2, len(sids))
        run_args = [(sid, config_yaml) for sid in sids]

        print(f"Building {perplex_db} GFEM models for {len(sids)} samples ...")

        models = []
        with cf.ProcessPoolExecutor(max_workers=nprocs) as executor:
            futures = [executor.submit(gfem_itr, args) for args in run_args]
            for future in tqdm(cf.as_completed(futures), total=len(futures)):
                iteration = future.result()
                models.append(iteration)

        gfems = [m for m in models if m.model_built]
        error_count = len([m for m in models if not m.model_built])

        if error_count > 0:
            print(f"Total GFEM models with errors: {error_count}")
        else:
            print("All GFEM models built successfully!")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    except Exception as e:
        print(f"Error in build_gfem_models(): {e}")
        return None

    return gfems

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    try:
        gfems = {}
        configs = ["assets/config_yamls/gfem-hydrated-shallow-upper-mantle-hp02m.yaml",
                   "assets/config_yamls/gfem-hydrated-shallow-upper-mantle-hp02r.yaml",
                   "assets/config_yamls/gfem-dry-deep-mantle-stx21m.yaml",
                   "assets/config_yamls/gfem-dry-deep-mantle-stx21r.yaml"]

        for yaml in configs:
            build_gfem_models(yaml)

    except Exception as e:
        print(f"Error in main(): {e}")

if __name__ == "__main__":
    main()
