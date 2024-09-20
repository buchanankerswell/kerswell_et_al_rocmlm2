#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import io
import os
import re
import time
import glob
import shutil
import textwrap
import traceback
import subprocess
import numpy as np
import pandas as pd
from hymatz import HyMaTZ
import multiprocessing as mp
from contextlib import redirect_stdout
np.set_printoptions(precision=3, suppress=True)
from sklearn.metrics import r2_score, mean_squared_error

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw, ImageFont
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
                 T_max=2273, geotherms="sub", seed=42, verbose=1):
        """
        """
        # Input
        self.res = res
        self.sid = sid
        self.seed = seed
        self.P_min = P_min
        self.P_max = P_max
        self.T_min = T_min
        self.T_max = T_max
        self.source = source
        self.verbose = verbose
        self.geotherms = geotherms

        # Syracuse 2010 subduction segments for creating subduction depth profiles
        self.segs = ["Central_Cascadia", "Kamchatka"]

        # Mantle potential temps for creating mantle depth profiles
        self.pot_Ts = [1173, 1573, 1773]

        # Set geotherm threshold for depth profiles
        if res <= 8:
            self.geothresh = 40
        elif res <= 16:
            self.geothresh = 20
        elif res <= 32:
            self.geothresh = 10
        elif res <= 64:
            self.geothresh = 5
        elif res <= 128:
            self.geothresh = 2.5
        else:
            self.geothresh = 1.25

        self.digits = 3
        self.features = ["XI_FRAC", "LOI"]
        self.target_units = ["g/cm$^3$", "km/s", "km/s", "vol.%", "wt.%", "", ""]
        self.targets = ["rho", "Vp", "Vs", "melt", "h2o", "assemblage", "variance"]

        # Check perplex db
        if perplex_db not in ["hp02", "hp11", "hp622", "hp633", "stx21"]:
            print("Unrecognized thermodynamic dataset! Defaulting to hp02 ...")
            self.perplex_db = "hp02"
        else:
            self.perplex_db = perplex_db

        # Perple_X chemical system
        self.ox_gfem = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "NA2O", "LOI"]

        # Melting limit (< K)
        if perplex_db in ["hp02", "hp11"]:
            self.T_melt = 1100
        else:
            self.T_melt = 273

        # Include fluid in computation of rock properties
        self.fluid_properties = "N"
        self.fluid_assemblages = "Y"

        # Treat melts as fluids ? (T: rock properties exclude melt fraction)
        self.melt_is_fluid = "T"

        # Perple_X solution models and endmembers to exclude
        if perplex_db == "hp02":
            self.melt_mod = "melt(HGPH)"
            self.em_exclude = sorted(["anL", "enL", "foL", "fo8L", "foHL", "diL", "woGL",
                                      "liz", "ak", "pswo", "wo"])
            self.sl_include = sorted(["O(HGP)", "Cpx(HGP)", "Omph(GHP)", "Opx(HGP)",
                                      "Sp(HP)", "Gt(HGP)", "Maj", "feldspar", "cAmph(G)",
                                      "Chl(W)", "Atg(PN)", "A-phase", "B", "T",
                                      f"{self.melt_mod}"])
        elif perplex_db in ["hp11", "hp622", "hp633"]:
            self.melt_mod = "melt(HGPH)"
            self.em_exclude = sorted(["foWL", "fojL", "foL", "fa8L", "faTL", "foTL", "perL",
                                      "neL", "fo8L", "diL", "dijL", "abL", "jdjL", "enL",
                                      "naph", "prl", "liz", "ne", "anl", "tap", "cg", "hen",
                                      "cen", "glt", "cgh", "dsp", "fctd"])
            self.sl_include = sorted(["O(HGP)", "Ring", "Wus", "Cpx(HGP)", "Omph(GHP)",
                                      "Opx(HGP)", "Sp(HGP)", "Gt(HGP)", "Maj", "feldspar",
                                      "cAmph(G)", "Chl(W)", "Atg(PN)", "A-phase", "B", "T",
                                      "Anth", f"{self.melt_mod}"])
        elif perplex_db == "stx21":
            self.melt_mod = None
            self.em_exclude = sorted(["ca-pv"])
            self.sl_include = sorted(["C2/c", "Wus", "Pv", "Pl", "Sp", "O", "Wad", "Ring",
                                      "Opx", "Cpx", "Aki", "Gt", "Ppv", "CF", "NaAl"])

        # Perplex dirs and filepaths
        self.model_out_dir = f"gfems/{self.sid}_{self.perplex_db}_{self.res}"

        # Output file paths
        self.data_dir = "assets"
        self.fig_dir = f"figs/{self.model_out_dir}"
        self.log_file = f"{self.model_out_dir}/log-{self.sid}"

        # Results
        self.loi = None
        self.results = {}
        self.model_built = False
        self.sample_features = []
        self.norm_sample_comp = []
        self.fertility_index = None
        self.pt_array = np.array([])
        self.target_array = np.array([])

        # Errors
        self.model_error = None
        self.timeout = (res**2) * 3
        self.model_build_error = False

        # Check for existing model build
        self._check_existing_model()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # print gfem info !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_gfem_info(self):
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
        targets = self.targets
        ox_gfem = self.ox_gfem
        features = self.features
        perplex_db = self.perplex_db
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"GFEM model: {sid} {perplex_db}")
        print("---------------------------------------------")
        print(f"  PT resolution:  {res}")
        print(f"  P range:        {P_min:.1f} - {P_max:.1f} GPa")
        print(f"  T range:        {T_min:.0f} - {T_max:.0f} K")
        print(f"  Sampleid:       {sid}")
        print(f"  Source:         {source}")
        print(f"  GFEM sys.:      {ox_gfem}")
        print(f"  Targets:        {targets}")
        print(f"  Features:       {features}")
        print(f"  Thermo. data:   {perplex_db}")
        print(f"  Model out dir:  {model_out_dir}")
        print("  --------------------")

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
                if verbose >= 1:
                    print(f"  Found {perplex_db} GFEM model for sample {sid} !")
                try:
                    self.model_built = True
                    self._get_normalized_sample_comp()
                    self._get_sample_features()
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
                return None
            else:
                shutil.rmtree(model_out_dir)
                os.makedirs(model_out_dir, exist_ok=True)
        else:
            os.makedirs(model_out_dir, exist_ok=True)
        return None

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
        perplex_db = self.perplex_db

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

        # Check for wrong sampleid
        if subset_df.empty:
            raise ValueError("Sample name not found in the dataset !")

        # Get Fertility Index and LOI
        self.loi = float(subset_df["LOI"].values[0])
        self.fertility_index = float(subset_df["XI_FRAC"].values[0])

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
        norm_comp = [round(((c / tot_comp) * 100 if c != 0 and o != "LOI" else c), digits)
                     for c, o in zip(sub_comp, ox_gfem)]

        # Check input
        if len(norm_comp) != len(ox_gfem):
            raise Exception("Normalized sample has incorrect number of oxide components !")

        self.norm_sample_comp = norm_comp

        return norm_comp

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get sample features !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_sample_features(self):
        """
        """
        # Get self attributes
        sid = self.sid
        source = self.source
        features = self.features

        # Read the data file
        df = pd.read_csv(source)

        # Subset the DataFrame based on the sample name
        subset_df = df[df["SAMPLEID"] == sid]

        if subset_df.empty:
            raise ValueError("Sample name not found in the dataset !")

        # Get features for selected sample
        sample_features = subset_df[features].values.flatten().tolist()

        self.sample_features = sample_features

        return sample_features

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # replace in file !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _replace_in_file(self, filepath, replacements):
        """
        """
        with open(filepath, "r") as file:
            file_data = file.read()

            for key, value in replacements.items():
                file_data = file_data.replace(key, value)

        with open(filepath, "w") as file:
            file.write(file_data)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # encode assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _encode_assemblages(self, assemblages):
        """
        """
        unique_assemblages = {}
        encoded_assemblages = []

        # Encoding unique phase assemblages
        for assemblage in assemblages:
            assemblage_tuple = tuple(sorted(assemblage))

            if assemblage_tuple and not any(np.isnan(item) for item in assemblage_tuple if
                                            isinstance(item, (int, float))):
                if assemblage_tuple not in unique_assemblages:
                    unique_assemblages[assemblage_tuple] = len(unique_assemblages) + 1

        # Create dataframe
        df = pd.DataFrame(list(unique_assemblages.items()), columns=["assemblage", "index"])

        # Put spaces between phases
        df["assemblage"] = df["assemblage"].apply(" ".join)

        # Reorder columns
        df = df[["index", "assemblage"]]

        # Save to csv
        assemblages_csv = f"{self.model_out_dir}/assemblages.csv"
        df.to_csv(assemblages_csv, index=False)

        # Encoding phase assemblage numbers
        for assemblage in assemblages:
            if assemblage == "":
                encoded_assemblages.append(np.nan)

            else:
                encoded_assemblage = unique_assemblages[tuple(sorted(assemblage))]
                encoded_assemblages.append(encoded_assemblage)

        return encoded_assemblages

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_array(self, array, n_neighbors=1, threshold=5):
        """
        """
        # Create a copy of the input array to avoid modifying the original array
        result_array = array.copy()

        # Iterate through each element of the array
        for i in range(len(result_array)):
            for j in range(len(result_array[i])):
                # Define the neighborhood indices
                neighbors = []

                for x in range(i - n_neighbors, i + n_neighbors + 1):
                    for y in range(j - n_neighbors, j + n_neighbors + 1):
                        if (0 <= x < len(result_array) and
                            0 <= y < len(result_array[i]) and
                            (x != i or y != j)):
                            neighbors.append((x, y))

                # Get neighborhood values
                surrounding_values = [result_array[x, y] for x, y in neighbors if not
                                      np.isnan(result_array[x, y])]

                # Define anomalies
                if surrounding_values:
                    mean_neighbors = np.mean(surrounding_values)
                    std_neighbors = np.std(surrounding_values)
                    anom_threshold = threshold * std_neighbors

                    # Impute anomalies
                    if abs(result_array[i, j] - mean_neighbors) > anom_threshold:
                        result_array[i, j] = np.mean(surrounding_values)

                    # Impute nans
                    elif np.isnan(result_array[i, j]):
                        nan_count = sum(1 for x, y in neighbors if
                                        0 <= x < len(result_array) and
                                        0 <= y < len(result_array[i]) and
                                        np.isnan(result_array[x, y]))

                        if nan_count >= int((((((n_neighbors * 2) + 1)**2) - 1) / 3)):
                            result_array[i, j] = 0
                        else:
                            result_array[i, j] = np.mean(surrounding_values)
                else:
                    result_array[i, j] = 0

        return result_array.flatten()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get 1d reference models !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_1d_reference_models(self, ref_model="prem"):
        """
        """
        # Get self attributes
        data_dir = self.data_dir

        # Check for data dir
        if not os.path.exists(data_dir):
            raise Exception(f"Data not found at {data_dir} !")

        # Check for ref models
        if ref_model not in ["prem", "stw105"]:
            raise Exception(f"Unrecognized reference model {ref_model} !")

        # Reference model paths and headers
        if ref_model == "prem":
            ref_path = f"{data_dir}/PREM_1s.csv"
            ref_cols = ["radius", "depth", "rho", "Vp", "Vph", "Vs", "Vsh", "eta", "Q_mu",
                        "Q_kappa"]
        else:
            ref_path = f"{data_dir}/STW105.csv"
            ref_cols = ["radius", "rho", "Vp", "Vs", "unk1", "unk2", "Vph", "Vsh", "eta"]

        columns_to_keep = ["depth", "P", "rho", "Vp", "Vs"]

        # Load reference model
        if not os.path.exists(ref_path):
            raise Exception(f"Refernce model {ref_model} not found at {ref_path} !")

        # Read reference model
        model = pd.read_csv(ref_path, header=None, names=ref_cols)

        # Transform units
        if ref_model == "stw105":
            model["depth"] = (model["radius"].max() - model["radius"]) / 1e3
            model["rho"] = model["rho"] / 1e3
            model["Vp"] = model["Vp"] / 1e3
            model["Vs"] = model["Vs"] / 1e3
            model.sort_values(by=["depth"], inplace=True)

        def calculate_pressure(row):
            z = row["depth"]
            depths = model[model["depth"] <= z]["depth"] * 1e3
            rhos = model[model["depth"] <= z]["rho"] * 1e3
            rho_integral = np.trapz(rhos, x=depths)
            pressure = 9.81 * rho_integral / 1e9
            return pressure

        model["P"] = model.apply(calculate_pressure, axis=1)

        # Clean up df
        model = model[columns_to_keep]
        model = model.round(3)

        # Invert STW105
        if ref_model == "stw105":
            model = model.sort_values(by="depth", ascending=True)

        return model

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get 1d subduction geotherm !!
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
        litho_P_gradient = 35 # Lithostatic P gradient (km/GPa)

        # Check for data dir
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
        T_min = self.T_min
        T_max = self.T_max

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

        P_geotherm = np.round(z / litho_P_gradient, 1)
        T_geotherm = np.round(T_geotherm, 2)

        geotherm = pd.DataFrame(
            {"P": P_geotherm, "T": T_geotherm}).sort_values(by=["P", "T"])

        return geotherm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get hymatz profile !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_hymatz_profile(self, target=None, mantle_potential=1573,
                            composition="Pyrolite", water_capacity=100):
        """
        """
        # Get self attributes
        P_min = self.P_min
        P_max = self.P_max

        # Check HyMaTZ inputs
        if composition not in ["Pyrolite"]:
            raise Exception(f"Unrecognized HyMaTZ composition '{composition}' !")
        if water_capacity not in np.linspace(0, 100, 11, dtype=int):
            raise Exception(f"HyMaTZ water capacity must be between 0 and 100 !")
        if target is not None and target not in ["rho", "Vp", "Vs", "h2o"]:
            raise Exception(f"Unrecognized HyMaTZ target '{target}'!")
        model = HyMaTZ(mantle_potential, hymatz_input[0], hymatz_input[1])
        results = model.results
        P_values, T_values, targets = results["P"], results["T"], results[target]

        # Cropping profile to same length as GFEM
        mask = (P_values >= P_min) & (P_values <= P_max)
        P_values, T_values, targets = P_values[mask], T_values[mask], targets[mask]

        return P_values, T_values, targets

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # extract target along 1d geotherm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _extract_target_along_1d_geotherm(self, target, geotherm):
        """
        """
        # Get self attributes
        results = self.results
        geothresh = self.geothresh

        # Define PT (target)
        P = results["P"]
        T = results["T"]
        trg = results[target]
        df = pd.DataFrame({"P": P, "T": T, target: trg}).sort_values(by=["P", "T"])

        # Create a DataFrame to hold the closest geotherm points for each result PT
        geotherm_P = geotherm["P"].values
        geotherm_T = geotherm["T"].values

        # Initialize lists for storing matched values
        P_values = []
        T_values = []
        target_values = []

        # Loop through each point in the geotherm DataFrame
        for idx, row in geotherm.iterrows():
            geo_P, geo_T = row["P"], row["T"]
            distances = np.sqrt((P - geo_P)**2 + (T - geo_T)**2)
            closest_idx = distances.argmin()
            if distances[closest_idx] < geothresh:
                P_values.append(P[closest_idx])
                T_values.append(T[closest_idx])
                target_values.append(trg[closest_idx])

        return P_values, T_values, target_values

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # crop depth profile !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _crop_depth_profile(self, P_gfem, target_gfem, P_ref, target_ref):
        """
        """
        # Get self attributes
        P_min = self.P_min
        P_max = self.P_max
        try:
            # New x values for interpolation
            x_new = np.linspace(P_min, P_max, len(P_gfem))

            # Interpolate profile
            P_ref, target_ref = x_new, np.interp(x_new, P_ref, target_ref)

            # Create cropping masks
            mask_ref = (P_ref >= P_min) & (P_ref <= P_max)
            mask_gfem = (P_gfem >= P_min) & (P_gfem <= P_max)

            # Crop profiles
            P_ref, target_ref = P_ref[mask_ref], target_ref[mask_ref]
            P_gfem, target_gfem = P_gfem[mask_gfem], target_gfem[mask_gfem]

            # Create nan and inf masks
            mask = (np.isnan(target_gfem) | np.isnan(target_ref) |
                    np.isinf(target_gfem) | np.isinf(target_ref))

            # Remove nans and inf
            P_ref, target_ref = P_ref[~mask], target_ref[~mask]
            P_gfem, target_gfem = P_gfem[~mask], target_gfem[~mask]

            # Calculate rmse and r2 along profiles
            r2 = np.round(r2_score(target_ref, target_gfem), 3)
            rmse = np.round(np.sqrt(mean_squared_error(target_ref, target_gfem)), 3)

            return P_gfem, target_gfem, P_ref, target_ref, rmse, r2

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in crop_depth_profile() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

            return P_gfem, target_gfem, P_ref, target_ref, None, None

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
        T_melt = self.T_melt
        pot_Ts = self.pot_Ts
        ox_gfem = self.ox_gfem
        melt_mod = self.melt_mod
        data_dir = self.data_dir
        geotherms = self.geotherms
        perplex_db = self.perplex_db
        melt_is_fluid = self.melt_is_fluid
        model_out_dir = self.model_out_dir
        T_min, T_max = self.T_min, self.T_max
        sl_include = "\n".join(self.sl_include)
        em_exclude = "\n".join(self.em_exclude)
        P_min, P_max = self.P_min * 1e4, self.P_max * 1e4
        norm_sample_comp = self._get_normalized_sample_comp()
        norm_sample_comp = " ".join(map(str, norm_sample_comp))
        fluid_properties = self.fluid_properties
        fluid_assemblages = self.fluid_assemblages

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
            # Werami targets
            w = (f"{sid}\n"           # Proj name
                 f"2\n"               # Operational mode (2: properties on 2D grid)
                 f"2\n"               # Select a property (2: Density kg/m3)
                 f"N\n"               # Calculate individual phase properties ?
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"13\n"              # Select a property (13: P-wave velocity m/s)
                 f"N\n"               # Calculate individual phase properties ?
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"14\n"              # Select a property (14: P-wave velocity m/s)
                 f"N\n"               # Calculate individual phase properties ?
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"24\n"              # Select a property (24: Assemblage index)
                 f"7\n"               # Select a property (7: Mode % of phase)
                 f"{melt_mod}\n"      # Enter solution or compound
                 f"6\n"               # Select a property (6: Composition of system)
                 f"1\n"               # Enter a component (1: H2O)
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"0\n"               # Zero to finish
                 f"N\n"               # Change default variable range ?
                 f"\n"                # Select the grid resolution (enter to continue)
                 f"5\n"               # Dummy
                 f"0\n"               # Zero to exit
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
            # Werami targets
            w = (f"{sid}\n"           # Proj name
                 f"2\n"               # Operational mode (2: properties on 2D grid)
                 f"2\n"               # Select a property (2: Density kg/m3)
                 f"N\n"               # Calculate individual phase properties ?
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"13\n"              # Select a property (13: P-wave velocity m/s)
                 f"N\n"               # Calculate individual phase properties ?
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"14\n"              # Select a property (14: P-wave velocity m/s)
                 f"N\n"               # Calculate individual phase properties ?
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"24\n"              # Select a property (24: Assemblage index)
                 f"7\n"               # Select a property (7: Mode % of phase)
                 f"{melt_mod}\n"      # Enter solution or compound
                 f"6\n"               # Select a property (6: Composition of system)
                 f"1\n"               # Enter a component (1: H2O)
                 f"{fluid_properties}\n" # Include fluid in computation of properties ?
                 f"0\n"               # Zero to finish
                 f"N\n"               # Change default variable range ?
                 f"\n"                # Select the grid resolution (enter to continue)
                 f"5\n"               # Dummy
                 f"0\n"               # Zero to exit
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
            # Werami targets
            w = (f"{sid}\n" # Proj name
                 f"2\n"     # Operational mode (2: properties on 2D grid)
                 f"2\n"     # Select a property (2: Density kg/m3)
                 f"N\n"     # Calculate individual phase properties ?
                 f"13\n"    # Select a property (13: P-wave velocity m/s)
                 f"N\n"     # Calculate individual phase properties ?
                 f"14\n"    # Select a property (14: S-wave velocity m/s)
                 f"N\n"     # Calculate individual phase properties ?
                 f"24\n"    # Select a property (24: Assemblage index)
                 f"0\n"     # Zero to finish
                 f"N\n"     # Change default variable range ?
                 f"\n"      # Select grid resolution (enter to continue)
                 f"0\n"     # Zero to exit
                 )
            # Copy thermodynamic data
            shutil.copy(f"{data_dir}/stx21-td", f"{model_out_dir}/td-data")
            shutil.copy(f"{data_dir}/stx21-sl", f"{model_out_dir}/solution-models")
        else:
            raise Exception("Unrecognized thermodynamic dataset !")

        # Write build config
        with open(f"{model_out_dir}/build-config", "w") as file:
            file.write(b)

        # Write werami targets
        with open(f"{model_out_dir}/werami-targets", "w") as file:
            file.write(w)

        # Write subduction geotherms to tsv files
        if geotherms == "sub":
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
        elif geotherms == "craton":
            for pot in pot_Ts:
                geotherm = self._get_mantle_geotherm(pot)
                geotherm["P"] = geotherm["P"] * 1e4
                geotherm = geotherm[["T", "P"]]
                geotherm.to_csv(f"{model_out_dir}/gt-craton-{pot}", sep="\t",
                                index=False, header=False, float_format="%.6E")
        elif geotherms == "mor":
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
            f = (f"{sid}\n"           # Proj name
                 f"2\n"               # Operational mode (2: properties on 2D grid)
                 f"25\n"              # Select a property (25: Modes of all phases)
                 f"N\n"               # Output cumulative modes ?
                 f"{fluid_assemblages}\n" # Include fluid in computation of properties ?
                 f"N\n"               # Change default variable range ?
                 f"\n"                # Select grid resolution (enter to continue)
                 f"0\n"               # Zero to exit
                 )
            # Werami geotherm
            if geotherms == "sub":
                for seg in segs:
                    if os.path.exists(f"{model_out_dir}/gt-slabtop-{seg}"):
                        g = (f"{sid}\n"            # Proj name
                             f"4\n"                # Op mode (4: properties along a 1d path)
                             f"2\n"                # Path (2: a file with T-P points)
                             f"gt-slabtop-{seg}\n" # Enter filename
                             f"1\n"                # How many nth points to plot ?
                             f"25\n"               # Select a property (25: Modes of all)
                             f"N\n"                # Output cumulative modes ?
                             f"{fluid_assemblages}\n"  # Include fluid in computation ?
                             f"0\n"                # Zero to exit
                             )
                        werami_geotherms_top.append(g)
                    if os.path.exists(f"{model_out_dir}/gt-slabmoho-{seg}"):
                        g = (f"{sid}\n"            # Proj name
                             f"4\n"                # Op mode (4: properties along a 1d path)
                             f"2\n"                # Path (2: a file with T-P points)
                             f"gt-slabmoho-{seg}\n"# Enter filename
                             f"1\n"                # How many nth points to plot ?
                             f"25\n"               # Select a property (25: Modes of all)
                             f"N\n"                # Output cumulative modes ?
                             f"{fluid_assemblages}\n"  # Include fluid in computation ?
                             f"0\n"                # Zero to exit
                             )
                        werami_geotherms_moho.append(g)
            elif geotherms == "craton":
                for pot in pot_Ts:
                    if os.path.exists(f"{model_out_dir}/gt-craton-{pot}"):
                        g = (f"{sid}\n"          # Proj name
                             f"4\n"              # Op mode (4: properties along a 1d path)
                             f"2\n"              # Path (2: a file with T-P points)
                             f"gt-craton-{pot}\n"# Enter filename
                             f"1\n"              # How many nth points to plot ?
                             f"25\n"             # Select a property (25: Modes of all)
                             f"N\n"              # Output cumulative modes ?
                             f"{fluid_assemblages}\n"# Include fluid in computation ?
                             f"0\n"              # Zero to exit
                             )
                        werami_geotherms_craton.append(g)
            elif geotherms == "mor":
                for pot in pot_Ts:
                    if os.path.exists(f"{model_out_dir}/gt-mor-{pot}"):
                        g = (f"{sid}\n"          # Proj name
                             f"4\n"              # Op mode (4: properties along a 1d path)
                             f"2\n"              # Path (2: a file with T-P points)
                             f"gt-mor-{pot}\n"   # Enter filename
                             f"1\n"              # How many nth points to plot ?
                             f"25\n"             # Select a property (25: Modes of all)
                             f"N\n"              # Output cumulative modes ?
                             f"{fluid_assemblages}\n"# Include fluid in computation ?
                             f"0\n"              # Zero to exit
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
            if geotherms == "sub":
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
                        g = (f"{sid}\n"            # Proj name
                             f"4\n"                # Op mode (4: properties along a 1d path)
                             f"2\n"                # Path (2: a file with T-P points)
                             f"gt-slabmoho-{seg}\n"# Enter filename
                             f"1\n"                # How many nth points to plot ?
                             f"25\n"               # Select a property (25: Modes of all)
                             f"N\n"                # Output cumulative modes ?
                             f"0\n"                # Zero to exit
                             )
                        werami_geotherms_moho.append(g)
            elif geotherms == "craton":
                for pot in pot_Ts:
                    if os.path.exists(f"{model_out_dir}/gt-craton-{pot}"):
                        g = (f"{sid}\n"          # Proj name
                             f"4\n"              # Op mode (4: properties along a 1d path)
                             f"2\n"              # Path (2: a file with T-P points)
                             f"gt-craton-{pot}\n"# Enter filename
                             f"1\n"              # How many nth points to plot ?
                             f"25\n"             # Select a property (25: Modes of all)
                             f"N\n"              # Output cumulative modes ?
                             f"0\n"              # Zero to exit
                             )
                        werami_geotherms_craton.append(g)
            elif geotherms == "mor":
                for pot in pot_Ts:
                    if os.path.exists(f"{model_out_dir}/gt-mor-{pot}"):
                        g = (f"{sid}\n"          # Proj name
                             f"4\n"              # Op mode (4: properties along a 1d path)
                             f"2\n"              # Path (2: a file with T-P points)
                             f"gt-mor-{pot}\n"   # Enter filename
                             f"1\n"              # How many nth points to plot ?
                             f"25\n"             # Select a property (25: Modes of all)
                             f"N\n"              # Output cumulative modes ?
                             f"0\n"              # Zero to exit
                             )
                        werami_geotherms_mor.append(g)
        else:
            raise Exception("Unrecognized thermodynamic dataset !")

        # Write werami phase
        with open(f"{model_out_dir}/werami-phase", "w") as file:
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
        res = self.res
        sid = self.sid
        model_out_dir = self.model_out_dir

        # Check for existing incomplete model
        if (os.path.exists(f"{model_out_dir}/results.csv") and
            os.path.exists(f"{model_out_dir}/assemblages.csv")):
            # Make new model if completed results not found
            shutil.rmtree(model_out_dir)
            os.makedirs(model_out_dir, exist_ok=True)

        # Write perplex configuration files
        self._write_perplex_config()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex(self):
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
        geotherms = self.geotherms
        em_exclude = self.em_exclude
        sl_include = self.sl_include
        perplex_db = self.perplex_db
        model_out_dir = self.model_out_dir
        norm_sample_comp = self.norm_sample_comp

        # Check for input perplex input files
        if not os.path.exists(f"{model_out_dir}/build-config"):
            raise Exception("No Perple_X input! Call _configure_perplex_model() first ...")

        if verbose >= 1:
            print(f"  Running Perple_X with {perplex_db} database and composition (wt.%):")
            max_oxide_width = max(len(oxide) for oxide in ox_gfem)
            max_comp_width = max(len(str(comp)) for comp in norm_sample_comp)
            max_width = max(max_oxide_width, max_comp_width)
            print(" ".join([f"  {oxide:<{max_width}}" for oxide in ox_gfem]))
            print(" ".join([f"  {comp:<{max_width}}" for comp in norm_sample_comp]))
            print("  --------------------")
            emwrp = textwrap.fill(", ".join(em_exclude), width=80, subsequent_indent="    ")
            slwrp = textwrap.fill(", ".join(sl_include), width=80, subsequent_indent="    ")
            print(f"  Excluded endmembers:")
            print(f"    {emwrp}")
            print("  --------------------")
            print(f"  Included solution models:")
            print(f"    {slwrp}")
            print("  --------------------")

        # Run programs with corresponding configuration files
        for program in ["build", "vertex", "werami", "pssect"]:
            # Get config files
            config_files = []

            if program == "build":
                config_files.append(f"{model_out_dir}/build-config")

            elif program == "vertex":
                config_files.append(f"{model_out_dir}/vertex-minimize")

            elif program == "werami":
                config_files.append(f"{model_out_dir}/werami-targets")
                config_files.append(f"{model_out_dir}/werami-phase")
                if geotherms == "sub":
                    for seg in segs:
                        config_files.append(f"{model_out_dir}/werami-gt-slabtop-{seg}")
                        config_files.append(f"{model_out_dir}/werami-gt-slabmoho-{seg}")
                elif geotherms == "craton":
                    for pot in pot_Ts:
                        config_files.append(f"{model_out_dir}/werami-gt-craton-{pot}")
                elif geotherms == "mor":
                    for pot in pot_Ts:
                        config_files.append(f"{model_out_dir}/werami-gt-mor-{pot}")

                self._replace_in_file(f"{model_out_dir}/build-options",
                                      {"Anderson-Gruneisen     F":
                                       "Anderson-Gruneisen     T"})

            elif program == "pssect":
                config_files.append(f"{model_out_dir}/pssect-draw")

            # Get program path
            program_path = f"Perple_X/{program}"
            relative_program_path = f"../../{program_path}"

            for i, config in enumerate(config_files):
                try:
                    # Set permissions
                    os.chmod(program_path, 0o755)

                    # Open the subprocess and redirect input from the input file
                    with open(config, "rb") as input_stream:
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
                        raise RuntimeError(f"Error executing perplex program '{program}'!")

                    elif verbose >= 2:
                        print(f"{program} output:")
                        print(f"{stdout.decode()}")

                    if program == "werami" and i == 0:
                        # Copy werami pseudosection output
                        shutil.copy(f"{model_out_dir}/{sid}_1.tab",
                                    f"{model_out_dir}/target-array.tab")

                        # Remove old output
                        os.remove(f"{model_out_dir}/{sid}_1.tab")

                    elif program == "werami" and i == 1:
                        # Copy werami mineral assemblage output
                        shutil.copy(f"{model_out_dir}/{sid}_1.tab",
                                    f"{model_out_dir}/phases.tab")

                        # Remove old output
                        os.remove(f"{model_out_dir}/{sid}_1.tab")

                    elif program == "werami" and i >= 2:
                        if geotherms == "sub":
                            seg_index = (i - 2) // 2
                            if (i - 2) % 2 == 0:
                                shutil.copy(
                                    f"{model_out_dir}/{sid}_1.tab",
                                    f"{model_out_dir}/gt-slabtop-{segs[seg_index]}.tab")
                            else:
                                shutil.copy(
                                    f"{model_out_dir}/{sid}_1.tab",
                                    f"{model_out_dir}/gt-slabmoho-{segs[seg_index]}.tab")
                        elif geotherms == "craton":
                            shutil.copy(
                                f"{model_out_dir}/{sid}_1.tab",
                                f"{model_out_dir}/gt-craton-{pot_Ts[i - 2]}.tab")
                        elif geotherms == "mor":
                            shutil.copy(
                                f"{model_out_dir}/{sid}_1.tab",
                                f"{model_out_dir}/gt-mor-{pot_Ts[i - 2]}.tab")

                        os.remove(f"{model_out_dir}/{sid}_1.tab")

                    elif program == "pssect":
                        # Copy pssect assemblages output
                        shutil.copy(f"{model_out_dir}/"
                                    f"{sid}_assemblages.txt",
                                    f"{model_out_dir}/assemblages.txt")

                        # Remove old output
                        os.remove(f"{model_out_dir}/{sid}_assemblages.txt")

                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read perplex targets !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_targets(self):
        """
        """
        # Get self attributes
        perplex_db = self.perplex_db
        model_out_dir = self.model_out_dir
        perplex_targets = f"{model_out_dir}/target-array.tab"

        # Initialize results
        if "hp" in perplex_db:
            results = {"T": [], "P": [], "rho": [], "Vp": [], "Vs": [],
                       "assemblage_index": [], "melt": [], "h2o": [],
                       "assemblage": [], "variance": []}
        elif perplex_db == "stx21":
            results = {"T": [], "P": [], "rho": [], "Vp": [], "Vs": [],
                       "assemblage_index": [], "assemblage": [], "variance": []}
        else:
            raise Exception("Unrecognized thermodynamic dataset !")

        # Open file
        with open(perplex_targets, "r") as file:
            # Skip lines until column headers are found
            for line in file:
                if "T(K)" in line and "P(bar)" in line:
                    break

            # Read the data
            for line in file:
                # Split line on whitespace
                vals = line.split()

                # Read the table of P, T, rho etc.
                try:
                    for i in range(len(results) - 2):
                        # Make vals floats or assign nan
                        value = float(vals[i]) if not np.isnan(float(vals[i])) else np.nan

                        # Convert from bar to GPa
                        if i == 1: value /= 1e4

                        # Convert from kg/m3 to g/cm3
                        if i == 2: value /= 1e3

                        # Convert assemblage index to an integer
                        if i == 5: value = int(value) if not np.isnan(value) else np.nan

                        # Make H2O nan if close to 0 wt.%
                        if i == 6 or i == 7: value = np.nan if value <= 1e-4 else value

                        # Append results
                        results[list(results.keys())[i]].append(value)

                except ValueError:
                    continue

        if all(len(values) == 0 for values in results.values()):
            raise ValueError("Results are empty. No data was read from the file.")

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

        if model_built:
            df = pd.read_csv(f"{model_out_dir}/assemblages.csv")
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

        return assemblage_dict

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process perplex results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_perplex_results(self):
        """
        """
        # Get self attributes
        sid = self.sid
        targets = self.targets
        verbose = self.verbose
        perplex_db = self.perplex_db
        model_out_dir = self.model_out_dir
        perplex_targets = f"{model_out_dir}/target-array.tab"
        perplex_assemblages = f"{model_out_dir}/assemblages.txt"

        # Check for targets
        if not os.path.exists(perplex_targets):
            raise Exception("No Perple_X files to process! Call _run_perplex() first ...")

        # Check for assemblages
        if not os.path.exists(perplex_assemblages):
            raise Exception("No Perple_X files to process! Call _run_perplex() first ...")

        if verbose >= 2:
            print(f"Reading Perple_X output: {model_out_dir} ...")

        # Read results
        results = self._read_perplex_targets()

        # Get assemblages from file
        assemblages = self._read_perplex_assemblages()

        # Parse assemblages by index
        for index in results.get("assemblage_index"):
            if np.isnan(index):
                results["assemblage"].append("")
            else:
                phases = assemblages[index]
                results["assemblage"].append(phases)

        # Count unique phases (assemblage variance)
        for assemblage in results.get("assemblage"):
            if assemblage is None:
                results["variance"].append(np.nan)
            else:
                unique_phases = set(assemblage)
                count = len(unique_phases)

                results["variance"].append(count)

        # Remove assemblage index
        results.pop("assemblage_index")

        # Encode assemblage
        encoded_assemblages = self._encode_assemblages(results["assemblage"])

        # Replace assemblage with encoded assemblages
        results["assemblage"] = encoded_assemblages

        # Point results that can be converted to numpy arrays
        point_params = ["T", "P"] + targets

        # Convert numeric point results into numpy arrays
        for key in point_params:
            if key in results and results[key]:
                results[key] = np.array(results[key])
            else:
                results[key] = np.full(len(results["T"]), np.nan)

        # Save as pandas df
        df = pd.DataFrame.from_dict(results)

        if verbose >= 2:
            print(f"Writing Perple_X results: {sid} ...")

        # Write to csv file
        df.to_csv(f"{model_out_dir}/results.csv", index=False)

        self.model_built = True

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.        Post-process GFEM Models         !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_results(self):
        """
        """
        # Get self attributes
        sid = self.sid
        verbose = self.verbose
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Get filepaths for gfem output
        filepath = f"{model_out_dir}/results.csv"

        if not os.path.exists(filepath):
            raise Exception("No results to read!")

        if verbose >= 2:
            print(f"Reading results: {filepath} ...")

        # Read results
        df = pd.read_csv(filepath)

        # Convert to dict of np arrays
        for column in df.columns:
            self.results[column] = df[column].values

        # Check for all nans
        any_array_all_nans = False

        for key, array in self.results.items():
            if key not in ["melt", "h2o"]:
                if np.all(np.isnan(array)):
                    any_array_all_nans = True

        if any_array_all_nans:
            self.results = {}
            self.model_build_error = True
            self.model_error = f"GFEM model {sid} produced all nans!"

            raise Exception(self.model_error)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get pt array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_pt_array(self):
        """
        """
        # Get self attributes
        results = self.results
        model_built = self.model_built

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Get P T arrays
        P, T = results["P"].copy(), results["T"].copy()

        # Stack PT arrays
        self.pt_array = np.stack((P, T), axis=-1).copy()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get target array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_target_array(self):
        """
        """
        # Get self attributes
        res = self.res
        results = self.results
        targets = self.targets
        verbose = self.verbose
        perplex_db = self.perplex_db
        model_built = self.model_built

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Initialize empty list for target arrays
        target_array_list = []

        # Rearrange results to match targets
        results_rearranged = {key: results[key] for key in targets}

        # Impute missing values
        for key, value in results_rearranged.items():
            if key in targets and key not in ["assemblage", "variance"]:
#                target_array_list.append(
#                    self._process_array(value.reshape(res + 1, res + 1)).flatten())
                target_array_list.append(value)
            else:
                target_array_list.append(value)

        # Stack target arrays
        self.target_array = np.stack(target_array_list, axis=-1).copy()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.3.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model array images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_array_images(self, gradient=False):
        """
        """
        # Get model data
        sid = self.sid
        targets = self.targets
        fig_dir = self.fig_dir
        perplex_db = self.perplex_db

        # Filter targets
        t_ind = [i for i, t in enumerate(targets) if t in ["rho", "Vp", "Vs"]]
        targets = [targets[i] for i in t_ind]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            if gradient:
                path = f"{fig_dir}/{sid}-{target}-grad-{perplex_db}.png"
            else:
                path = f"{fig_dir}/{sid}-{target}-{perplex_db}.png"

            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if len(existing_figs) == len(targets):
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
        targets = self.targets
        fig_dir = self.fig_dir
        perplex_db = self.perplex_db

        # Filter targets
        t_ind = [i for i, t in enumerate(targets) if t in ["rho", "Vp", "Vs"]]
        targets = [targets[i] for i in t_ind]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            path = f"{fig_dir}/{sid}-{target}-surf-{perplex_db}.png"

            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if len(existing_figs) == len(targets):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model depth profile images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_depth_profile_images(self, slab_position="slabtop"):
        """
        """
        # Get model data
        sid = self.sid
        targets = self.targets
        fig_dir = self.fig_dir
        geotherms = self.geotherms
        perplex_db = self.perplex_db

        # Filter targets
        t_ind = [i for i, t in enumerate(targets) if t not in ["assemblage", "variance"]]
        targets = [targets[i] for i in t_ind]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            if geotherms == "sub":
                path = (f"{fig_dir}/{sid}-{target}-depth-profile-sub-{slab_position}-"
                        f"{perplex_db}.png")
                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

            elif geotherms == "craton":
                path = f"{fig_dir}/{sid}-{target}-depth-profile-craton-{perplex_db}.png"
                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

            elif geotherms == "mor":
                path = f"{fig_dir}/{sid}-{target}-depth-profile-mor-{perplex_db}.png"
                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

        if len(existing_figs) == len(targets):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model gt assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_gt_assemblages_images(self, slab_position="slabtop"):
        """
        """
        # Get model data
        sid = self.sid
        segs = self.segs
        pot_Ts = self.pot_Ts
        fig_dir = self.fig_dir
        geotherms = self.geotherms
        perplex_db = self.perplex_db

        # Check for existing plots
        existing_figs = []
        if geotherms == "sub":
            for i, seg in enumerate(segs):
                seg_lab = seg.replace("_", "-").lower()
                path = (f"{fig_dir}/{sid}-gt-{slab_position}-{seg_lab}-assemblages-"
                        f"{perplex_db}.png")

                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

            if len(existing_figs) == len(segs):
                return True
            else:
                return False

        elif geotherms == "craton":
            for i, pot in enumerate(pot_Ts):
                path = f"{fig_dir}/{sid}-gt-craton-{pot}-assemblages-{perplex_db}.png"

                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

            if len(existing_figs) == len(pot_Ts):
                return True
            else:
                return False

        elif geotherms == "mor":
            for i, pot in enumerate(pot_Ts):
                path = f"{fig_dir}/{sid}-gt-mor-{pot}-assemblages-{perplex_db}.png"

                check = os.path.exists(path)

                if check:
                    existing_figs.append(check)

            if len(existing_figs) == len(pot_Ts):
                return True
            else:
                return False

        else:
            raise Exception("Unrecognized type argument !")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize array image  !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_image(self, palette="bone", gradient=False, figwidth=6.3,
                               figheight=4.725, fontsize=22):
        """
        """
        # Get self attributes
        sid = self.sid
        res = self.res
        segs = self.segs
        pot_Ts = self.pot_Ts
        targets = self.targets
        results = self.results
        fig_dir = self.fig_dir
        geotherms = self.geotherms
        geothresh = self.geothresh
        perplex_db = self.perplex_db
        model_built = self.model_built
        target_array = self.target_array
        target_units = self.target_units
        P, T = results["P"], results["T"]
        extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Check for targets
        if target_array is None or target_array.size == 0:
            raise Exception("No GFEM model target array! Call get_target_array() first ...")

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Filter targets for gradient images
        if gradient:
            grad_targets = ["rho", "Vp", "Vs", "melt", "h2o"]
            t_ind = [i for i, t in enumerate(targets) if t in grad_targets]
            targets = [targets[i] for i in t_ind]

        # Get subduction geotherms
        if geotherms == "sub":
            sub_gtt = {}
            sub_gtm = {}
            for seg in segs:
                sub_gtt[seg] = self._get_subduction_geotherm(seg, slab_position="slabtop")
                sub_gtm[seg] = self._get_subduction_geotherm(seg, slab_position="slabmoho")
        # Get mantle geotherms
        elif geotherms == "craton":
            ad_gt = {}
            for pot in pot_Ts:
                ad_gt[pot] = self._get_mantle_geotherm(pot)
        elif geotherms == "mor":
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
            # Target labels
            if target == "rho":
                target_label = "Density"
            elif target == "h2o":
                target_label = "H$_2$O"
            elif target == "melt":
                target_label = "Melt"
            else:
                target_label = target

            # Set filename
            filename = f"{sid}-{target}-{perplex_db}.png"
            if target not in ["assemblage", "variance"]:
                title = f"{target_label} ({target_units[i]})"
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

                filename = f"{sid}-{target}-grad-{perplex_db}.png"
                title = f"{target_label} Gradient"

            # Use discrete colorscale
            if target in ["assemblage", "variance"]:
                color_discrete = True
            else:
                color_discrete = False

            # Reverse color scale
            if palette in ["grey"]:
                if target in ["variance"]:
                    color_reverse = True
                else:
                    color_reverse = False
            else:
                if target in ["variance"]:
                    color_reverse = True
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
                if geotherms == "sub":
                    for seg, gt in sub_gtt.items():
                        ax.plot(gt["T"], gt["P"], linestyle="-", color="black",
                                linewidth=2, label=seg)
                    for seg, gt in sub_gtm.items():
                        ax.plot(gt["T"], gt["P"], linestyle="--", color="black",
                                linewidth=2, label=seg)
                elif geotherms == "craton":
                    for i, (pot, gt) in enumerate(ad_gt.items()):
                        ax.plot(gt["T"], gt["P"], linestyle=linestyles[i], color="black",
                                linewidth=2, label=pot)
                elif geotherms == "mor":
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
                if geotherms == "sub":
                    for seg, gt in sub_gtt.items():
                        ax.plot(gt["T"], gt["P"], linestyle="-", color="black",
                                linewidth=2, label=seg)
                    for seg, gt in sub_gtm.items():
                        ax.plot(gt["T"], gt["P"], linestyle="--", color="black",
                                linewidth=2, label=seg)
                elif geotherms == "craton":
                    for i, (pot, gt) in enumerate(ad_gt.items()):
                        ax.plot(gt["T"], gt["P"], linestyle=linestyles[i], color="black",
                                linewidth=2, label=pot)
                elif geotherms == "mor":
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
                if target == "rho":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "Vp":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "Vs":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "melt":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                elif target == "h2o":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "assemblage":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                elif target == "variance":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

            plt.title(title)

            # Vertical text spacing
            text_margin_x = 0.04
            text_margin_y = 0.15
            text_spacing_y = 0.1

            # Save the plot to a file
            plt.savefig(f"{fig_dir}/{filename}")

            # Close device
            plt.close()
            print(f"  Figure saved to: {fig_dir}/{filename} ...")

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
        targets = self.targets
        results = self.results
        fig_dir = self.fig_dir
        geothresh = self.geothresh
        perplex_db = self.perplex_db
        model_built = self.model_built
        target_array = self.target_array
        target_units = self.target_units
        P = results["P"].reshape(res + 1, res + 1)
        T = results["T"].reshape(res + 1, res + 1)

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Check for targets
        if target_array is None or target_array.size == 0:
            raise Exception("No GFEM model target array! Call get_target_array() first ...")

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

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
            # Target labels
            if target == "rho":
                target_label = "Density"
            elif target == "h2o":
                target_label = "H$_2$O"
            elif target == "melt":
                target_label = "Melt"
            else:
                target_label = target

            # Set filename
            filename = f"{sid}-{target}-surf-{perplex_db}.png"
            if target not in ["assemblage", "variance"]:
                title = f"{target_label} ({target_units[i]})"
            else:
                title = f"{target_label}"

            # Reshape targets into square array
            square_target = target_array[:, i].reshape(res + 1, res + 1)

            # Use discrete colorscale
            if target in ["assemblage", "variance"]:
                color_discrete = True
            else:
                color_discrete = False

            # Reverse color scale
            if palette in ["grey"]:
                if target in ["variance"]:
                    color_reverse = True
                else:
                    color_reverse = False
            else:
                if target in ["variance"]:
                    color_reverse = True
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
                if target == "rho":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "Vp":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "Vs":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "melt":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                elif target == "h2o":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                elif target == "assemblage":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                elif target == "variance":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                    ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

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
    def _visualize_depth_profiles(self, slab_position="slabtop", figwidth=6.3,
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
        targets = self.targets
        fig_dir = self.fig_dir
        data_dir = self.data_dir
        geotherms = self.geotherms
        geothresh = self.geothresh
        perplex_db = self.perplex_db
        model_built = self.model_built
        target_units = self.target_units
        model_out_dir = self.model_out_dir

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for data dir
        if not os.path.exists(data_dir):
            raise Exception(f"Data not found at {data_dir} !")

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

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
            if target not in ["assemblage", "variance"]:
                # Plotting
                fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

                Pprof, tprof, labels = [], [], []

                # Plot assemblages and rock properties along geotherms
                if geotherms == "sub":
                    for j, seg in enumerate(segs):
                        if slab_position == "slabtop":
                            filename = (f"{sid}-{target}-depth-profile-sub-slabtop-"
                                        f"{perplex_db}.png")
                            gt = self._get_subduction_geotherm(seg, slab_position="slabtop")
                        elif slab_position == "slabmoho":
                            filename = (f"{sid}-{target}-depth-profile-sub-slabmoho-"
                                        f"{perplex_db}.png")
                            gt = self._get_subduction_geotherm(seg, slab_position="slabmoho")
                        Pp, _, tp = (self._extract_target_along_1d_geotherm(target, gt))
                        seg_lab = seg.replace("_", " ").lower()
                        labels.append(seg_lab)
                        Pprof.append(Pp)
                        tprof.append(tp)
                elif geotherms == "craton":
                    filename = f"{sid}-{target}-depth-profile-craton-{perplex_db}.png"
                    for j, pot in enumerate(pot_Ts):
                        gt = self._get_mantle_geotherm(pot)
                        Pp, _, tp = (self._extract_target_along_1d_geotherm(target, gt))
                        labels.append(pot)
                        Pprof.append(Pp)
                        tprof.append(tp)
                elif geotherms == "mor":
                    filename = f"{sid}-{target}-depth-profile-mor-{perplex_db}.png"
                    for j, pot in enumerate(pot_Ts):
                        gt = self._get_mantle_geotherm(
                            pot, Qs=750e-3, A1=2.2e-8, k1=3.0,
                            crust_thickness=7e3, litho_thickness=1e3)
                        Pp, _, tp = (self._extract_target_along_1d_geotherm(target, gt))
                        labels.append(pot)
                        Pprof.append(Pp)
                        tprof.append(tp)

                for j, (Pp, tp, lab) in enumerate(zip(Pprof, tprof, labels)):
                    # Plot GFEM model profiles
                    ax1.plot(tp, Pp, "-", linewidth=2, color=colormap(j), label=lab)

                if target == "rho":
                    target_label = "Density"
                elif target == "h2o":
                    target_label = "H$_2$O"
                elif target == "melt":
                    target_label = "Melt"
                else:
                    target_label = target

                ax1.set_xlabel(f"{target_label} ({target_units[i]})")
                ax1.set_ylabel("P (GPa)")
                ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
                ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

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
                print(f"  Figure saved to: {fig_dir}/{filename} ...")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize geotherm assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_geotherm_assemblages(self, slab_position="slabtop", modal_thresh=5,
                                        figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get self attributes
        sid = self.sid
        res = self.res
        loi = self.loi
        segs = self.segs
        pot_Ts = self.pot_Ts
        fig_dir = self.fig_dir
        xi = self.fertility_index
        geotherms = self.geotherms
        perplex_db = self.perplex_db
        model_built = self.model_built
        model_out_dir = self.model_out_dir

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Check for werami files
        if geotherms not in ["sub", "craton", "mor"]:
            raise Exception("Unrecognized geotherms argument !")

        if geotherms == "sub":
            for seg in segs:
                path_top = f"{model_out_dir}/gt-slabmoho-{seg}.tab"
                path_moho = f"{model_out_dir}/gt-slabtop-{seg}.tab"
                if not os.path.exists(path_top):
                    raise Exception(f"No werami data found at {path_top} !")
                if not os.path.exists(path_moho):
                    raise Exception(f"No werami data found at {path_moho} !")

        if geotherms == "craton":
            for pot in pot_Ts:
                path = f"{model_out_dir}/gt-craton-{pot}.tab"
                if not os.path.exists(path):
                    raise Exception(f"No werami data found at {path} !")

        if geotherms == "mor":
            for pot in pot_Ts:
                path = f"{model_out_dir}/gt-mor-{pot}.tab"
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
            file_pattern = os.path.join("gfems", "*hp*", "gt*.tab")
        elif perplex_db == "stx21":
            file_pattern = os.path.join("gfems", "*stx*", "gt*.tab")
        else:
            raise Exception("Unrecognized thermodynamic dataset !")

        for file_path in glob.glob(file_pattern, recursive=True):
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
        hues = np.linspace(0, 1, num_colors, endpoint=False)
        colors = [hsv_to_rgb([hue, 0.8, 0.75]) for hue in hues]
        colormap = plt.colormaps["tab20"]
        colors = [colormap(i) for i in range(num_colors)]
        color_map = {col_name: colors[idx] for idx, col_name in enumerate(phase_names)}

        tabfiles, filenames, gts, labels = [], [], [], []

        # Plot assemblages and rock properties along geotherms
        if geotherms == "sub":
            for seg in segs:
                seg_lab = seg.replace("_", "-").lower()
                labels.append(seg_lab)
                if slab_position == "slabtop":
                    tabfiles.append(f"{model_out_dir}/gt-slabtop-{seg}.tab")
                    filenames.append(f"{sid}-gt-slabtop-{seg_lab}-assemblages-"
                                     f"{perplex_db}.png")
                    gt = self._get_subduction_geotherm(seg, slab_position="slabtop")
                    gts.append(gt)
                elif slab_position == "slabmoho":
                    tabfiles.append(f"{model_out_dir}/gt-slabmoho-{seg}.tab")
                    filenames.append(f"{sid}-gt-slabmoho-{seg_lab}-assemblages-"
                                     f"{perplex_db}.png")
                    gt = self._get_subduction_geotherm(seg, slab_position="slabmoho")
                    gts.append(gt)
        elif geotherms == "craton":
            for pot in pot_Ts:
                labels.append(pot)
                tabfiles.append(f"{model_out_dir}/gt-craton-{pot}.tab")
                filenames.append(f"{sid}-gt-craton-{pot}-assemblages-{perplex_db}.png")
                gt = self._get_mantle_geotherm(pot)
                gts.append(gt)
        elif geotherms == "mor":
            for pot in pot_Ts:
                labels.append(pot)
                tabfiles.append(f"{model_out_dir}/gt-mor-{pot}.tab")
                filenames.append(f"{sid}-gt-mor-{pot}-assemblages-{perplex_db}.png")
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
            Pg, Tg, rhog = self._extract_target_along_1d_geotherm("rho", gt)

            # Get water profile
            if perplex_db != "stx21":
                _, _, h2og = self._extract_target_along_1d_geotherm("h2o", gt)
            else:
                h2og = np.zeros(len(Pg))

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
            ax_line_sec.plot(Pg, h2og, color="blue", linewidth=2, label="GFEM H$_2$O")
            ax_line_sec.set_ylabel("H$_2$O (wt.%)")
            ax_line_sec.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            if perplex_db == "stx21" or np.all(h2og == 0):
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
            print(f"  Figure saved to: {fig_dir}/{filename} ...")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_model(self):
        """
        """
        try:
            if not self._check_model_array_images(gradient=False):
                self._visualize_array_image(gradient=False)
            if not self._check_model_array_images(gradient=True):
                self._visualize_array_image(gradient=True)
            if not self._check_model_array_surfs():
                self._visualize_target_surf()
            if self.geotherms == "sub":
                if not self._check_model_depth_profile_images(slab_position="slabtop"):
                    self._visualize_depth_profiles(slab_position="slabtop")
                if not self._check_model_depth_profile_images(slab_position="slabmoho"):
                    self._visualize_depth_profiles(slab_position="slabmoho")
                if not self._check_model_gt_assemblages_images(slab_position="slabtop"):
                    self._visualize_geotherm_assemblages(slab_position="slabtop")
                if not self._check_model_gt_assemblages_images(slab_position="slabmoho"):
                    self._visualize_geotherm_assemblages(slab_position="slabmoho")
            elif self.geotherms in ["craton", "mor"]:
                if not self._check_model_depth_profile_images():
                    self._visualize_depth_profiles()
                if not self._check_model_gt_assemblages_images():
                    self._visualize_geotherm_assemblages()
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in visualize_model() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.4.           Build GFEM Models             !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # build model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def build_model(self):
        """
        """
        max_retries = 3
        for retry in range(max_retries):
            # Check for built model
            if self.model_built:
                break
            try:
                self._print_gfem_info()
                self._configure_perplex_model()
                self._run_perplex()
                self._process_perplex_results()
                self._get_sample_features()
                self._get_results()
                self._get_target_array()
                self._get_pt_array()
                break
            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"!!! ERROR in build_model() !!!")
                print(f"{e}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                traceback.print_exc()
                if retry < max_retries - 1:
                    print(f"Retrying in 5 seconds ...")
                    time.sleep(5)
                else:
                    self.model_build_error = True
                    self.model_error = e
                    return None

        return None

#######################################################
## .2.                 Visualize                 !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine plots horizontally !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def combine_plots_horizontally(image1_path, image2_path, output_path, caption1, caption2,
                               font_size=150, caption_margin=25, dpi=300):
    """
    """
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum height between the two images
    max_height = max(image1.height, image2.height)

    # Create a new image with twice the width and the maximum height
    combined_width = image1.width + image2.width
    combined_image = Image.new("RGB", (combined_width, max_height), (255, 255, 255))

    # Set the DPI metadata
    combined_image.info["dpi"] = (dpi, dpi)

    # Paste the first image on the left
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right
    combined_image.paste(image2, (image1.width, 0))

    # Add captions
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("Arial", font_size)
    caption_margin = caption_margin

    # Add caption
    draw.text((caption_margin, caption_margin), caption1, font=font, fill="black")

    # Add caption "b"
    draw.text((image1.width + caption_margin, caption_margin), caption2, font=font,
              fill="black")

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine plots vertically !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def combine_plots_vertically(image1_path, image2_path, output_path, caption1, caption2,
                             font_size=150, caption_margin=25, dpi=300):
    """
    """
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum width between the two images
    max_width = max(image1.width, image2.width)

    # Create a new image with the maximum width and the sum of the heights
    combined_height = image1.height + image2.height
    combined_image = Image.new("RGB", (max_width, combined_height), (255, 255, 255))

    # Paste the first image on the top
    combined_image.paste(image1, (0, 0))

    # Paste the second image below the first
    combined_image.paste(image2, (0, image1.height))

    # Add captions
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("Arial", font_size)
    caption_margin = caption_margin

    # Add caption
    draw.text((caption_margin, caption_margin), caption1, font=font, fill="black")

    # Add caption "b"
    draw.text((caption_margin, image1.height + caption_margin), caption2, font=font,
              fill="black")

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose itr !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_itr(gfem_model, clean=False):
    """
    """
    # Get model data
    sid = gfem_model.sid
    res = gfem_model.res
    targets = gfem_model.targets
    fig_dir = gfem_model.fig_dir
    verbose = gfem_model.verbose

    # Check for existing plots
    existing_comps = []
    for target in targets:
        fig_1 = f"{fig_dir}/image2-{sid}-{target}.png"
        fig_2 = f"{fig_dir}/image3-{sid}-{target}.png"
        fig_3 = f"{fig_dir}/image4-{sid}-{target}.png"
        fig_4 = f"{fig_dir}/image9-{sid}.png"
        fig_5 = f"{fig_dir}/image12-{sid}.png"
        fig_6 = f"{fig_dir}/image15-{sid}.png"

        check_comps = ((os.path.exists(fig_3) and os.path.exists(fig_4)) |
                       (os.path.exists(fig_1) and os.path.exists(fig_2)) |
                       (os.path.exists(fig_1) and os.path.exists(fig_2) and
                        os.path.exists(fig_4) and os.path.exists(fig_5)) |
                       (os.path.exists(fig_1) and os.path.exists(fig_2) and
                        os.path.exists(fig_4) and os.path.exists(fig_6)))

        if check_comps:
            existing_comps.append(check_comps)

    if existing_comps:
        return None

    for target in targets:
        # Check for existing plots
        fig = f"{fig_dir}/{sid}-{target}.png"
        check_fig = os.path.exists(fig)

        if not check_fig:
            print(f"  No {target} plot found. Skipping composition ...")
            continue

        if target not in ["assemblage", "variance"]:
            combine_plots_horizontally(
                f"{fig_dir}/{sid}-{target}.png",
                f"{fig_dir}/{sid}-{target}-grad.png",
                f"{fig_dir}/image2-{sid}-{target}.png",
                caption1="a)",
                caption2="b)"
            )

            print(f"  Figure saved to: {fig_dir}/image2-{sid}-{target}.png ...")

            if target in ["rho", "Vp", "Vs", "melt", "h2o"]:
                combine_plots_horizontally(
                    f"{fig_dir}/{sid}-{target}.png",
                    f"{fig_dir}/{sid}-{target}-grad.png",
                    f"{fig_dir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                combine_plots_horizontally(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/{sid}-{target}-depth-profile.png",
                    f"{fig_dir}/image3-{sid}-{target}.png",
                    caption1="",
                    caption2="c)"
                )

                print(f"  Figure saved to: {fig_dir}/image3-{sid}-{target}.png ...")

    if all(item in targets for item in ["rho", "melt", "h2o"]):
        captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
        trgs = ["rho", "melt", "h2o"]

        for i, target in enumerate(trgs):
            # Check for existing plots
            fig = f"{fig_dir}/{sid}-{target}.png"
            check_fig = os.path.exists(fig)

            if not check_fig:
                print(f"  No {target} plot found. Skipping composition ...")
                continue

            combine_plots_horizontally(
                f"{fig_dir}/{sid}-{target}.png",
                f"{fig_dir}/{sid}-{target}-grad.png",
                f"{fig_dir}/temp1.png",
                caption1=captions[i][0],
                caption2=captions[i][1]
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{sid}-{target}-depth-profile.png",
                f"{fig_dir}/temp-{target}.png",
                caption1="",
                caption2=captions[i][2]
            )

        # Check for existing plots
        fig = f"{fig_dir}/{sid}-h2o.png"
        check_fig_h2o = os.path.exists(fig)
        fig = f"{fig_dir}/{sid}-melt.png"
        check_fig_melt = os.path.exists(fig)

        if not check_fig_h2o or not check_fig_melt:
            print(f"  No h2o or melt plot found. Skipping composition ...")
        else:
            combine_plots_vertically(
                f"{fig_dir}/temp-rho.png",
                f"{fig_dir}/temp-melt.png",
                f"{fig_dir}/temp1.png",
                caption1="",
                caption2=""
            )

            combine_plots_vertically(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/temp-h2o.png",
                f"{fig_dir}/image9-{sid}.png",
                caption1="",
                caption2=""
            )

            print(f"  Figure saved to: {fig_dir}/image9-{sid}.png ...")

    if all(item in targets for item in ["rho", "Vp", "Vs", "melt", "h2o"]):
        captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)"),
                    ("j)", "k)", "l)"), ("m)", "n)", "o)")]
        trgs = ["rho", "Vp", "Vs", "melt", "h2o"]

        for i, target in enumerate(trgs):
            # Check for existing plots
            fig = f"{fig_dir}/{sid}-{target}.png"
            check_fig = os.path.exists(fig)

            if not check_fig:
                print(f"  No {target} plot found. Skipping composition ...")
                continue

            combine_plots_horizontally(
                f"{fig_dir}/{sid}-{target}.png",
                f"{fig_dir}/{sid}-{target}-grad.png",
                f"{fig_dir}/temp1.png",
                caption1=captions[i][0],
                caption2=captions[i][1]
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{sid}-{target}-depth-profile.png",
                f"{fig_dir}/temp-{target}.png",
                caption1="",
                caption2=captions[i][2]
            )

        combine_plots_vertically(
            f"{fig_dir}/temp-rho.png",
            f"{fig_dir}/temp-Vp.png",
            f"{fig_dir}/temp1.png",
            caption1="",
            caption2=""
        )

        combine_plots_vertically(
            f"{fig_dir}/temp1.png",
            f"{fig_dir}/temp-Vs.png",
            f"{fig_dir}/temp2.png",
            caption1="",
            caption2=""
        )

        # Check for existing plots
        fig = f"{fig_dir}/{sid}-h2o.png"
        check_fig_h2o = os.path.exists(fig)
        fig = f"{fig_dir}/{sid}-melt.png"
        check_fig_melt = os.path.exists(fig)

        if not check_fig_h2o or not check_fig_melt:
            print(f"  No h2o or melt plot found. Skipping composition ...")
        else:
            combine_plots_vertically(
                f"{fig_dir}/temp2.png",
                f"{fig_dir}/temp-melt.png",
                f"{fig_dir}/temp3.png",
                caption1="",
                caption2=""
            )

            combine_plots_vertically(
                f"{fig_dir}/temp3.png",
                f"{fig_dir}/temp-h2o.png",
                f"{fig_dir}/image15-{sid}.png",
                caption1="",
                caption2=""
            )

            print(f"  Figure saved to: {fig_dir}/image15-{sid}.png ...")

    tmp_files = glob.glob(f"{fig_dir}/temp*.png")
    for file in tmp_files:
        os.remove(file)

    if clean:
        # Clean up directory
        depth_profile_files = glob.glob(f"{fig_dir}/*depth-profile.png")
        grad_files = glob.glob(f"{fig_dir}/*grad.png")

        for file in depth_profile_files + grad_files:
            os.remove(file)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose gfem plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_gfem_plots(gfem_models, clean=False, nprocs=os.cpu_count() - 2):
    """
    """
    # Iterate through all models
    print("Composing GFEM plots ...")
    with mp.Pool(processes=nprocs) as pool:
        results = [pool.apply_async(compose_itr, args=(m, clean)) for m in gfem_models]
        for result in results:
            try:
                result.get()
            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"!!! ERROR in compose_gfem_plots() !!!")
                print(f"{e}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                traceback.print_exc()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize depth profiles comps !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_depth_profiles_comps(gfem_models, figwidth=6.3, figheight=5.5, fontsize=28):
    """
    """
    # Data asset dir
    data_dir = "assets"
    fig_dir = "figs/other"

    # Check for data dir
    if not os.path.exists(data_dir):
        raise Exception(f"Data not found at {data_dir} !")

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Get correct Depletion column
    XI_col = "XI_FRAC"

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

    # Colormap
    colormap = plt.colormaps["tab10"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figwidth * 2, figheight * 2))
    axes = axes.flatten()

    for j, model in enumerate(gfem_models):
        # Get gfem model data
        sid = model.sid
        res = model.res
        P_min = model.P_min
        P_max = model.P_max
        targets = model.targets
        results = model.results
        xi = model.fertility_index
        geothresh = model.geothresh
        perplex_db = model.perplex_db
        target_units = model.target_units

        # Get dry synthetic endmember compositions
        df_mids = pd.read_csv("assets/synth-mids.csv")
        df_dry = df_mids[df_mids["LOI"] == 0]
        sids = [df_dry["SAMPLEID"].head(1).values[0], df_dry["SAMPLEID"].tail(1).values[0]]
        df_synth_bench = df_mids[df_mids["SAMPLEID"].isin(sids) & (df_mids["LOI"] == 0)]

        # Mixing array endmembers
        bend = df_synth_bench["SAMPLEID"].iloc[0]
        tend = df_synth_bench["SAMPLEID"].iloc[-1]

        for i, target in enumerate(targets):
            if target not in ["assemblage", "variance"]:
                # Get GFEM model profile
                P_gfem, _, target_gfem = model._get_1d_profile(
                    target, Qs=55e-3, A1=1e-6, k1=2.3, litho_thickness=150)

                # Create colorbar
                pal = plt.get_cmap("magma_r")
                norm = plt.Normalize(df_synth_bench[XI_col].min(),
                                     df_synth_bench[XI_col].max())
                sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
                sm.set_array([])

                ax = axes[i]

                # Plot GFEM profiles
                ax.plot(target_gfem, P_gfem, "-", linewidth=1,
                        color=sm.to_rgba(xi), alpha=0.2)

                # Target labels
                if target == "rho":
                    target_label = "Density"
                elif target == "h2o":
                    target_label = "H$_2$O"
                elif target == "melt":
                    target_label = "Melt"
                else:
                    target_label = target

                if (i == 1) or (i == 3):
                    ax.set_ylabel("")
                    ax.set_yticks([])
                else:
                    ax.set_ylabel("P (GPa)")

                ax.set_xlabel(f"{target_label} ({target_units[i]})")

                if target == "rho":
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                elif target == "Vp":
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                elif target == "Vs":
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                elif target == "melt":
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
                elif target == "h2o":
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

                # Convert the primary y-axis data (pressure) to depth
                depth_conversion = lambda P: P * 30
                depth_values = depth_conversion(np.linspace(P_min, P_max, len(P_gfem)))

                # Create the secondary y-axis and plot depth on it
                if (i == 1) or (i == 3):
                    ax2 = ax.secondary_yaxis(
                        "right", functions=(depth_conversion, depth_conversion))
                    ax2.set_yticks([410, 670])
                    ax2.set_ylabel("Depth (km)")

                if i == 0:
                    cbaxes = inset_axes(ax, width="40%", height="3%", loc=2)
                    colorbar = plt.colorbar(sm, ax=ax, cax=cbaxes, label="$\\xi$",
                                            orientation="horizontal")
                    ax.legend(loc="lower right", columnspacing=0, handletextpad=0.2,
                              fontsize=fontsize * 0.833)

        fig.text(0.08, 0.98, "a)", fontsize=fontsize * 1.4)
        fig.text(0.49, 0.98, "b)", fontsize=fontsize * 1.4)
        fig.text(0.08, 0.50, "c)", fontsize=fontsize * 1.4)
        fig.text(0.49, 0.50, "d)", fontsize=fontsize * 1.4)

        # Save the plot to a file
        filename = f"depth-profile-comps-{perplex_db}.png"
        plt.savefig(f"{fig_dir}/{filename}")

        # Close device
        plt.close()

    return None

#######################################################
## .3.   Build GFEM for RocMLM training data     !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get sampleids !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sampleids(filepath, batch="all", n_batches=8):
    """
    """
    # Check for file
    if not os.path.exists(filepath):
        raise Exception("Sample data source does not exist!")

    # Read data
    df = pd.read_csv(filepath)

    if "benchmark" in filepath or batch == "all":
        return df["SAMPLEID"].values

    # Calculate the batch size
    total_samples = len(df)
    batch_size = int(total_samples // n_batches)

    # Check if batch number is within valid range
    if batch < 0 or batch >= n_batches:
        print("Invalid batch number! Sampling from the first 0th batch ...")
        batch = 0

    # Calculate the start and end index for the specified batch
    start = batch * batch_size
    end = min((batch + 1) * batch_size, total_samples)

    return df[start:end]["SAMPLEID"].values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gfem_iteration !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gfem_iteration(args, queue):
    """
    """
    stdout_buffer = io.StringIO()
    perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax, geotherms = args

    with redirect_stdout(stdout_buffer):
        iteration = GFEMModel(perplex_db, sampleid, source, res,
                              Pmin, Pmax, Tmin, Tmax, geotherms)
        queue.put(stdout_buffer.getvalue())

        if not iteration.model_built:
            iteration.build_model()
            queue.put(stdout_buffer.getvalue())

    return iteration

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build gfem models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_gfem_models(source, sampleids=None, perplex_db="hp602", res=128,
                      Pmin=1, Pmax=28, Tmin=773, Tmax=2273, geotherms="sub",
                      nprocs=os.cpu_count() - 2, verbose=1):
    """
    """
    # Check sampleids
    if os.path.exists(source) and sampleids is None:
        sampleids = sorted(get_sampleids(source))
    elif os.path.exists(source) and sampleids is not None:
        sids = get_sampleids(source)
        if not set(sampleids).issubset(sids):
            raise Exception(f"Sampleids {sampleids} not in source: {source}!")
    else:
        raise Exception(f"Source {source} does not exist!")

    models = []
    smpwrp = textwrap.fill(", ".join(sampleids), width=80, subsequent_indent="  ")
    print(f"Building GFEM models for {len(sampleids)} samples:")
    print(f"  {smpwrp}")

    # Define number of processors
    if nprocs is None or nprocs > os.cpu_count():
        nprocs = os.cpu_count()

    # Make sure nprocs is not greater than sampleids
    if nprocs > len(sampleids):
        nprocs = len(sampleids)

    # Create list of args for mp pooling
    run_args = [(perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax, geotherms) for
                sampleid in sampleids]

    # Create a multiprocessing manager and queue
    with mp.Manager() as manager:
        queue = manager.Queue()

        # Update the arguments to include the queue
        run_args_with_queue = [(args, queue) for args in run_args]

        # Chunk the arguments into batches of size nprocs
        batches = [run_args_with_queue[i:i + nprocs] for
                   i in range(0, len(run_args_with_queue), nprocs)]

        # Create a multiprocessing pool for each batch
        for batch in batches:
            with mp.Pool(processes=len(batch)) as pool:
                models.append(pool.starmap(gfem_iteration, batch))
                pool.close()
                pool.join()

            # Collect and print the results in order
            while not queue.empty():
                print(queue.get(), end="")

    # Get successful models
    gfems = [m for mds in models for m in mds if not m.model_build_error]

    # Check for errors in the models
    error_count = 0

    for model in gfems:
        if model.model_build_error:
            error_count += 1

    if error_count > 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Total GFEM models with errors: {error_count}")
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("All GFEM models built successfully !")

    print(":::::::::::::::::::::::::::::::::::::::::::::")

    return gfems

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    try:
        gfems = {}
        sources = {"b": "assets/bench-pca.csv",
                   "m": "assets/synth-mids.csv",
                   "r": "assets/synth-rnds.csv"}

        # Build GFEM models
        for name, source in sources.items():
            sids = get_sampleids(source)
            gfems[name] = build_gfem_models(source, sids, res=64)

        # Visualize models
        for name, models in gfems.items():
            for m in models:
                m.visualize_model()

        # Compose plots
        for name, models in gfems.items():
            compose_gfem_plots(models)

        # Visualize depth profiles
        visualize_depth_profiles_comps(gfems["m"] + gfems["r"])

    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in main() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("GFEM models built and visualized !")
    print("=============================================")

    return None

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
