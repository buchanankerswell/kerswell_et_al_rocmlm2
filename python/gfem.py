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
import random
import shutil
import warnings
import traceback
import itertools
import subprocess
from datetime import datetime
from contextlib import redirect_stdout

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parallel computing !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing as mp

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.impute import KNNImputer
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, Normalize, SymLogNorm

#######################################################
## .1.              GFEMModel class              !!! ##
#######################################################
class GFEMModel:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, perplex_db, sid, source, res=128, P_min=1, P_max=28, T_min=773,
                 T_max=2273, seed=42, verbose=1):
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
        self.targets = ["rho", "Vp", "Vs", "melt", "h2o", "assemblage", "variance"]
        self.target_units = ["g/cm$^3$", "km/s", "km/s", "vol.%", "wt.%", "", ""]

        # Check perplex db
        if perplex_db not in ["hp02", "hp633"]:
            self.perplex_db = "hp02"
        else:
            self.perplex_db = perplex_db

        # System oxide components
        self.ox_exclude = ["FE2O3", "P2O5", "NIO", "MNO", "H2O", "CO2"]
        self.ox_gfem = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2",
                        "CR2O3", "LOI"]
        if self.perplex_db == "hp02":
            self.ox_gfem = [ox for ox in self.ox_gfem if ox != "CR2O3"]

        # Perplex dirs and filepaths
        self.model_out_dir = f"gfems/{self.sid}_{self.perplex_db}"
        self.perplex_assemblages = f"{self.model_out_dir}/assemblages.txt"
        self.perplex_targets = f"{self.model_out_dir}/target-array.tab"

        # Output file paths
        self.data_dir = "assets/data"
        self.fig_dir = f"figs/{self.model_out_dir}"
        self.log_file = f"{self.model_out_dir}/log-{self.sid}"

        # Results
        self.loi = None
        self.results = {}
        self.comp_time = None
        self.sample_comp = []
        self.model_built = False
        self.sample_features = []
        self.norm_sample_comp = []
        self.fertility_index = None
        self.model_visualized = False
        self.target_array = np.array([])
        self.feature_array = np.array([])

        # Visualizations
        self.pt_range_image = True
        self.model_target_images = False
        self.model_gradient_images = False

        # Errors
        self.model_error = None
        self.timeout = (res**2) * 3
        self.model_build_error = False

        # Check for existing model build
        if os.path.exists(self.model_out_dir):
            if (os.path.exists(f"{self.model_out_dir}/results.csv") and
                os.path.exists(f"{self.model_out_dir}/assemblages.csv")):
                if verbose >= 1:
                    print(f"  Found GFEM model for sample {self.sid} !")

                try:
                    self.model_built = True
                    self._get_sample_comp()
                    self._measure_gfem_model_accuracy_vs_prem()
                except Exception as e:
                    print(f"!!! ERROR in GFEMModel() !!!")
                    print(f"{e}")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    traceback.print_exc()
                    return None

                self.visualize_model()
                return None
            else:
                shutil.rmtree(self.model_out_dir)
                os.makedirs(self.model_out_dir, exist_ok=True)
        else:
            os.makedirs(self.model_out_dir, exist_ok=True)

        # Set np array printing option
        np.set_printoptions(precision=3, suppress=True)

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # print gfem model info !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_gfem_model_info(self):
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
        print(f"GFEM model: {sid}")
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
    # get sample composition !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_sample_comp(self):
        """
        """
        # Get self attributes
        sid = self.sid
        source = self.source
        ox_gfem = self.ox_gfem

        # Read the data file
        df = pd.read_csv(source)

        # Subset the DataFrame based on the sample name
        subset_df = df[df["SAMPLEID"] == sid]

        if subset_df.empty:
            raise ValueError("Sample name not found in the dataset !")

        # Get Fertility Index and LOI
        self.loi = float(subset_df["LOI"].values[0])
        self.fertility_index = float(subset_df["XI_FRAC"].values[0])

        # Get the oxide compositions for the selected sample
        composition = []

        for oxide in ox_gfem:
            if oxide in subset_df.columns and pd.notnull(subset_df[oxide].iloc[0]):
                composition.append(float(subset_df[oxide].iloc[0]))

            else:
                composition.append(0)

        self.sample_comp = composition

        return composition

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # normalize sample composition !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _normalize_sample_comp(self):
        """
        """
        # Get self attributes
        digits = self.digits
        ox_gfem = self.ox_gfem
        ox_exclude = self.ox_exclude
        sample_comp = self.sample_comp
        ox_sub = [oxide for oxide in ox_gfem if oxide not in ox_exclude]

        # Check for sample composition
        if not sample_comp:
            raise Exception("No sample found! Call _get_sample_comp() first ...")

        # No normalizing for all components
        if not ox_exclude:
            self.norm_sample_comp = sample_comp
            return sample_comp

        # Check input
        if len(sample_comp) != len(ox_gfem):
            error_message = (f"The input sample list must have exactly {len(ox_gfem)} "
                             f"components!\n{ox_gfem}")

            raise ValueError(error_message)

        # Filter components
        subset_sample = [
            comp for comp, oxide in zip(sample_comp, ox_gfem) if oxide in ox_sub]

        # Set negative compositions to zero
        subset_sample = [comp if comp >= 0 else 0 for comp in subset_sample]

        # Get total oxides
        total_subset_concentration = sum([comp for comp in subset_sample if comp != 0])

        # Normalize
        normalized_concentrations = [
            round(((comp / total_subset_concentration) * 100 if comp != 0 else 0), digits)
            for comp, oxide in zip(subset_sample, ox_sub)]

        self.norm_sample_comp = normalized_concentrations

        return normalized_concentrations

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
        norm_sample_comp = self.norm_sample_comp

        # Check for sample composition
        if not norm_sample_comp:
            raise Exception("Sample composition not normalized yet! "
                            "Call _normalize_sample_comp() first ...")

        # Read the data file
        df = pd.read_csv(source)

        # Subset the DataFrame based on the sample name
        subset_df = df[df["SAMPLEID"] == sid]

        if subset_df.empty:
            raise ValueError("Sample name not found in the dataset !")

        # Get features for selected sample
        feature_values = subset_df[features].values.flatten().tolist()
        sample_features = norm_sample_comp + feature_values

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
    # get comp time !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_comp_time(self):
        """
        """
        # Get self attributes
        sid = self.sid
        res = self.res
        ox_gfem = self.ox_gfem
        log_file = self.log_file
        perplex_db = self.perplex_db

        if os.path.exists(log_file):
            with open(log_file, "r") as log:
                lines = log.readlines()

            for line in reversed(lines):
                if "Total elapsed time" in line:
                    match = re.search(r"\s+([\d.]+)", line)
                    if match:
                        time_m = float(match.group(1))
                        time_s = time_m * 60
                    break

            self.comp_time = round(time_s, 3)

        return round(time_s, 3)

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
                        if surrounding_values:
                            result_array[i, j] = np.mean(surrounding_values)
                        else:
                            result_array[i, j] = 0

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

        return result_array.flatten()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get 1d reference models !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_1d_reference_models(self):
        """
        """
        # Get self attributes
        data_dir = self.data_dir

        # Check for data dir
        if not os.path.exists(data_dir):
            raise Exception(f"Data not found at {data_dir} !")

        # Reference model paths
        ref_paths = {"prem": f"{data_dir}/PREM_1s.csv", "stw105": f"{data_dir}/STW105.csv"}

        # Define column headers
        prem_cols = ["radius", "depth", "rho", "Vp", "Vph", "Vs", "Vsh", "eta", "Q_mu",
                     "Q_kappa"]
        stw105_cols = ["radius", "rho", "Vp", "Vs", "unk1", "unk2", "Vph", "Vsh", "eta"]

        ref_cols = {"prem": prem_cols, "stw105": stw105_cols}
        columns_to_keep = ["depth", "P", "rho", "Vp", "Vs"]

        # Initialize reference models
        ref_models = {}

        # Load reference models
        for name, path in ref_paths.items():
            if not os.path.exists(path):
                raise Exception(f"Refernce model {name} not found at {path} !")

            # Read reference model
            model = pd.read_csv(path, header=None, names=ref_cols[name])

            # Transform units
            if name == "stw105":
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

            # Save model
            ref_models[name] = model

        # Invert STW105
        ref_models["stw105"] = ref_models["stw105"].sort_values(by="depth", ascending=True)

        return ref_models

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get 1d profile !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_1d_profile(self, target, mantle_potential=1573, Qs=250e-3, Ts=273, A1=2.2e-8,
                        A2=2.2e-8, k1=3.0, k2=3.0, crust_thickness=35, litho_thickness=1):
        """
        """
        # Get model attributes
        results = self.results
        geothresh = self.geothresh
        model_built = self.model_built

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Get P results
        P = results["P"]

        # Get PT and target values and transform units
        df = pd.DataFrame({"P": results["P"], "T": results["T"], target: results[target]}
                          ).sort_values(by="P")

        # Geotherm Parameters
        Z_min = np.min(P) * 35e3
        Z_max = np.max(P) * 35e3
        z = np.linspace(Z_min, Z_max, len(P))
        T_geotherm = np.zeros(len(P))

        # Layer1 (crust)
        # A1 Radiogenic heat production (W/m^3)
        # k1 Thermal conductivity (W/mK)
        D1 = crust_thickness * 1e3 # Thickness (m)

        # Layer2 (lithospheric mantle)
        # A2 Radiogenic heat production (W/m^3)
        # k2 Thermal conductivity (W/mK)
        D2 = litho_thickness * 1e3

        # Calculate heat flow at the top of each layer
        Qt2 = Qs - (A1 * D1)
        Qt1 = Qs

        # Calculate T at the top of each layer
        Tt1 = Ts
        Tt2 = Tt1 + (Qt1 * D1 / k1) - (A1 / 2 / k1 * D1**2)
        Tt3 = Tt2 + (Qt2 * D2 / k2) - (A2 / 2 / k2 * D2**2)

        # Calculate T within each layer
        for j in range(len(P)):
            potential_temp = mantle_potential + 0.5e-3 * z[j]
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
                T_geotherm[j] = Tt3 + 0.5e-3 * (z[j] - D1 - D2)
                if T_geotherm[j] >= potential_temp:
                    T_geotherm[j] = potential_temp

        P_geotherm = np.round(z / 35e3, 1)
        T_geotherm = np.round(T_geotherm, 2)

        df["geotherm_P"] = P_geotherm
        df["geotherm_T"] = T_geotherm

        # Subset df along geotherm
        df = df[abs(df["T"] - df["geotherm_T"]) < geothresh]

        # Extract the three vectors
        P_values = df["P"].values
        T_values = df["T"].values
        targets = df[target].values

        return P_values, T_values, targets

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # crop 1d profile !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _crop_1d_profile(self, P_gfem, target_gfem, P_ref, target_ref):
        """
        """
        try:
            # Initialize interpolators
            interp_ref = interp1d(P_ref, target_ref, fill_value="extrapolate")

            # Get min and max P
            P_min, P_max = np.nanmin(P_gfem), np.nanmax(P_gfem)

            # New x values for interpolation
            x_new = np.linspace(P_min, P_max, len(P_gfem))

            # Interpolate profile
            P_ref, target_ref = x_new, interp_ref(x_new)

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
            print(f"!!! ERROR in crop_1d_profile() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

            return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.1.          Perple_X Functions             !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # configure perplex model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_perplex_model(self):
        """
        """
        # Get self attributes
        res = self.res
        sid = self.sid
        digits = self.digits
        perplex_db = self.perplex_db
        model_out_dir = self.model_out_dir

        # Check for existing incomplete model
        if (os.path.exists(f"{model_out_dir}/results.csv") and
            os.path.exists(f"{model_out_dir}/assemblages.csv")):
            # Make new model if completed results not found
            shutil.rmtree(model_out_dir)
            os.makedirs(model_out_dir, exist_ok=True)

        # Transform units to bar
        T_min, T_max = self.T_min, self.T_max
        P_min, P_max = self.P_min * 1e4, self.P_max * 1e4

        # Get sample composition
        sample_comp = self._get_sample_comp()
        norm_comp = self._normalize_sample_comp()

        # Config dir
        config_dir = f"assets/config_{perplex_db}"

        # Configuration files
        draw = "pssect-draw"
        thermodb = "td-data"
        plot = "plot-options"
        build = "build-config"
        phase = "werami-phase"
        options = "build-options"
        targets = "werami-targets"
        minimize = "vertex-minimize"
        solutions = "solution-models"

        # Copy original configuration files to the perplex directory
        shutil.copy(f"{config_dir}/{draw}", f"{model_out_dir}/{draw}")
        shutil.copy(f"{config_dir}/{build}", f"{model_out_dir}/{build}")
        shutil.copy(f"{config_dir}/{phase}", f"{model_out_dir}/{phase}")
        shutil.copy(f"{config_dir}/{targets}", f"{model_out_dir}/{targets}")
        shutil.copy(f"{config_dir}/{options}", f"{model_out_dir}/{options}")
        shutil.copy(f"{config_dir}/{thermodb}", f"{model_out_dir}/{thermodb}")
        shutil.copy(f"{config_dir}/{minimize}", f"{model_out_dir}/{minimize}")
        shutil.copy(f"{config_dir}/{solutions}", f"{model_out_dir}/{solutions}")
        shutil.copy(f"{config_dir}/{plot}", f"{model_out_dir}/perplex_plot_option.dat")

        # Modify the copied configuration files within the perplex directory
        self._replace_in_file(f"{model_out_dir}/{build}",
                              {"{SAMPLEID}": f"{sid}",
                               "{TMIN}": str(T_min), "{TMAX}": str(T_max),
                               "{PMIN}": str(P_min), "{PMAX}": str(P_max),
                               "{SAMPLECOMP}": " ".join(map(str, norm_comp))})
        self._replace_in_file(f"{model_out_dir}/{minimize}",
                              {"{SAMPLEID}": f"{sid}"})
        self._replace_in_file(f"{model_out_dir}/{targets}",
                              {"{SAMPLEID}": f"{sid}"})
        self._replace_in_file(f"{model_out_dir}/{phase}",
                              {"{SAMPLEID}": f"{sid}"})
        self._replace_in_file(f"{model_out_dir}/{options}",
                              {"{XNODES}": f"{int(res / 4)} {res + 1}",
                               "{YNODES}": f"{int(res / 4)} {res + 1}"})
        self._replace_in_file(f"{model_out_dir}/{draw}",
                              {"{SAMPLEID}": f"{sid}"})

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run perplex !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_perplex(self):
        """
        """
        # Get self attributes
        sid = self.sid
        ox_gfem = self.ox_gfem
        timeout = self.timeout
        verbose = self.verbose
        log_file = self.log_file
        perplex_db = self.perplex_db
        ox_exclude = self.ox_exclude
        model_out_dir = self.model_out_dir
        norm_sample_comp = self.norm_sample_comp
        ox_sub = [oxide for oxide in ox_gfem if oxide not in ox_exclude]

        # Check for input perplex input files
        if not os.path.exists(f"{model_out_dir}/build-config"):
            raise Exception("No Perple_X input! Call _configure_perplex_model() first ...")

        if verbose >= 1:
            print(f"  Running Perple_X with {perplex_db} database and composition (wt.%):")
            max_oxide_width = max(len(oxide) for oxide in ox_sub)
            max_comp_width = max(len(str(comp)) for comp in norm_sample_comp)
            max_width = max(max_oxide_width, max_comp_width)
            print(" ".join([f"  {oxide:<{max_width}}" for oxide in ox_sub]))
            print(" ".join([f"  {comp:<{max_width}}" for comp in norm_sample_comp]))
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

                    elif program == "pssect":
                        # Copy pssect assemblages output
                        shutil.copy(f"{model_out_dir}/"
                                    f"{sid}_assemblages.txt",
                                    f"{model_out_dir}/assemblages.txt")

                        # Copy pssect auto refine output
                        shutil.copy(f"{model_out_dir}/"
                                    f"{sid}_auto_refine.txt",
                                    f"{model_out_dir}/auto_refine.txt")

                        # Copy pssect seismic data output
                        shutil.copy(f"{model_out_dir}/"
                                    f"{sid}_seismic_data.txt",
                                    f"{model_out_dir}/seismic_data.txt")

                        # Remove old output
                        os.remove(f"{model_out_dir}/{sid}_assemblages.txt")
                        os.remove(f"{model_out_dir}/{sid}_auto_refine.txt")
                        os.remove(f"{model_out_dir}/{sid}_seismic_data.txt")

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
        perplex_targets = self.perplex_targets

        # Initialize results
        results = {"T": [], "P": [], "rho": [], "Vp": [], "Vs": [], "assemblage_index": [],
                   "melt": [], "h2o": [], "assemblage": [], "variance": []}

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

        return results

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read perplex assemblages !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_perplex_assemblages(self):
        """
        """
        # Initialize dictionary to store assemblage info
        assemblage_dict = {}

        # Open assemblage file
        with open(self.perplex_assemblages, "r") as file:
            for i, line in enumerate(file, start=1):
                assemblages = line.split("-")[1].strip().split()

                # Make string formatting consistent
                cleaned_assemblages = [assemblage.split("(")[0].lower()
                                       for assemblage in assemblages]

                # Add assemblage to dict
                assemblage_dict[i] = cleaned_assemblages

        return assemblage_dict

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process perplex results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_perplex_results(self):
        """
        """
        # Get self attributes
        sid = self.sid
        verbose = self.verbose
        perplex_db = self.perplex_db
        model_out_dir = self.model_out_dir
        perplex_targets = self.perplex_targets
        perplex_assemblages = self.perplex_assemblages

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
        point_params = ["T", "P", "rho", "Vp", "Vs", "melt", "h2o", "assemblage", "variance"]

        # Convert numeric point results into numpy arrays
        for key, value in results.items():
            if key in point_params:
                results[key] = np.array(value)

        # Save as pandas df
        df = pd.DataFrame.from_dict(results)

        if verbose >= 2:
            print(f"Writing Perple_X results: {sid} ...")

        # Write to csv file
        df.to_csv(f"{model_out_dir}/results.csv", index=False)

        # Clean up output directory
        files_to_keep = ["assemblages.csv", "results.csv", f"log-{sid}"]

        # Create dir to store model files
        model_out_files_dir = f"{model_out_dir}/model"
        os.makedirs(model_out_files_dir, exist_ok=True)

        try:
            # List all files in the directory
            all_files = os.listdir(model_out_dir)

            # Iterate through the files and delete those not in the exclusion list
            for filename in all_files:
                file_path = os.path.join(model_out_dir, filename)
                destination_path = os.path.join(model_out_files_dir, filename)

                if os.path.isfile(file_path) and filename not in files_to_keep:
                    shutil.move(file_path, destination_path)

        except Exception as e:
            print(f"Error: {e}")

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
    # get feature array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_feature_array(self):
        """
        """
        # Get self attributes
        results = self.results
        targets = self.targets
        verbose = self.verbose
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
        self.feature_array = np.stack((P, T), axis=-1).copy()

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

        # Set n_neighbors for KNN imputer
        if res <= 8:
            n_neighbors = 1
        elif res <= 16:
            n_neighbors = 2
        elif res <= 32:
            n_neighbors = 3
        elif res <= 64:
            n_neighbors = 4
        elif res <= 128:
            n_neighbors = 5

        # Initialize empty list for target arrays
        target_array_list = []

        # Rearrange results to match targets
        results_rearranged = {key: results[key] for key in targets}

        # Impute missing values
        for key, value in results_rearranged.items():
            if key in targets:
                if key in ["melt", "h2o"]:
                    target_array_list.append(
                        self._process_array(value.reshape(res + 1, res + 1)).flatten())

                else:
                    # Impute target array with KNN
                    imputer = KNNImputer(n_neighbors = n_neighbors, weights="distance")
                    imputed_array = imputer.fit_transform(value.reshape(res + 1, res + 1))

                    # Process imputed array
                    target_array_list.append(self._process_array(imputed_array).flatten())

        # Stack target arrays
        self.target_array = np.stack(target_array_list, axis=-1).copy()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # measure gfem model accuracy vs prem !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _measure_gfem_model_accuracy_vs_prem(self):
        """
        """
        # Get model attributes
        sid = self.sid
        res = self.res
        P_min = self.P_min
        P_max = self.P_max
        ox_gfem = self.ox_gfem
        results = self.results
        verbose = self.verbose
        data_dir = self.data_dir
        perplex_db = self.perplex_db
        targets = ["rho", "Vp", "Vs"]
        model_built = self.model_built

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Post-process results
        try:
            self._get_results()
            self._get_target_array()
            self._get_feature_array()
            comp_time = self._get_comp_time()
        except Exception as e:
            print(f"!!! ERROR in _measure_gfem_model_accuracy_vs_prem() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Check for data dir
        if not os.path.exists(data_dir):
            raise Exception(f"Data not found at {data_dir} !")

        # Get 1D reference models
        ref_models = self._get_1d_reference_models()

        # Initialize metrics lists
        smpid, db, ox, sz, trgt, tm = [], [], [], [], [], []
        rmse_prem_profile, r2_prem_profile = [], []
        rmse_stw105_profile, r2_stw105_profile = [], []

        for target in targets:
            # Get 1D reference model profile
            P_prem, target_prem = ref_models["prem"]["P"], ref_models["prem"][target]

            # Get GFEM model profile
            P_gfem, _, target_gfem = self._get_1d_profile(
                target, Qs=55e-3, A1=1e-6, k1=2.3, litho_thickness=150)

            # Crop GFEM profile and compare with PREM
            _, _, _, _, rmse_prem, r2_prem = (
                self._crop_1d_profile(P_gfem, target_gfem, P_prem, target_prem))

            # Save other info
            smpid.append(sid)
            sz.append(res**2)
            ox.append(len(ox_gfem))
            trgt.append(target)
            tm.append(comp_time)
            db.append(perplex_db)

        # Create dataframe
        df = pd.DataFrame(
            {"SAMPLEID": smpid, "DATABASE": db, "N_OX_SYS": ox, "SIZE": sz,
             "COMP_TIME": tm, "TARGET": trgt, "RMSE_PREM": rmse_prem, "R2_PREM": r2_prem})

        # Write csv
        filename = "assets/data/gfem-model-results-summary.csv"

        if os.path.exists(filename) and os.stat(filename).st_size > 0:
            try:
                df_existing = pd.read_csv(filename)
                if df_existing.empty:
                    df.to_csv(filename, index=False)
                else:
                    df_combined = pd.concat([df_existing, df], ignore_index=True)
                    df_combined = df_combined.drop_duplicates(
                        subset=["SAMPLEID", "DATABASE", "SIZE"], keep="first")
                    df_combined = df_combined.sort_values(
                        by=["SAMPLEID", "TARGET"], ignore_index=True)
                    df_combined.to_csv(filename, index=False)
            except pd.errors.EmptyDataError:
                df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.3.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check pt range image !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_pt_range_image(self):
        """
        """
        if os.path.exists(f"figs/other/training-dataset-design.png"):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model target images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_target_images(self, gradient=False):
        """
        """
        # Get model data
        sid = self.sid
        targets = self.targets
        fig_dir = self.fig_dir

        # Filter targets for gradient images
        if gradient:
            targets = ["rho", "Vp", "Vs", "melt", "h2o"]

        # Filter targets for dry samples
        if sid in ["DMM", "PUM", "PYR"]:
            targets = [t for t in targets if t != "h2o"]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            if gradient:
                path = f"{fig_dir}/{sid}-{target}-grad.png"
            else:
                path = f"{fig_dir}/{sid}-{target}.png"

            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if len(existing_figs) == len(targets):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model prem images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_prem_images(self):
        """
        """
        # Get model data
        sid = self.sid
        targets = self.targets
        fig_dir = self.fig_dir
        target_units = self.target_units

        # Filter targets for PREM
        t_ind = [i for i, t in enumerate(targets) if t in ["rho", "Vp", "Vs"]]
        target_units = [target_units[i] for i in t_ind]
        targets = [targets[i] for i in t_ind]

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            path = f"{fig_dir}/{sid}-{target}-prem.png"
            check = os.path.exists(path)

            if check:
                existing_figs.append(check)

        if len(existing_figs) == len(targets):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize gfem pt range !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_gfem_pt_range(self, fontsize=12, figwidth=6.3, figheight=3.3):
        """
        """
        # Get model data
        sid = self.sid
        res = self.res
        targets = self.targets
        results = self.results
        fig_dir = "figs/other"
        geothresh = self.geothresh
        model_built = self.model_built
        target_array = self.target_array

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

        # Get min max PT
        P_gt, T_gt, rho_gt = results["P"], results["T"], results["rho"]
        P_min, P_max, T_min, T_max = np.min(P_gt), np.max(P_gt), np.min(T_gt), np.max(T_gt)
        T = np.arange(0, T_max + 728)

        # Olivine --> Ringwoodite Clapeyron slopes
        references_410 = {"ol$\\rightarrow$wad (Akaogi89)": [0.001, 0.002],
                          "ol$\\rightarrow$wad (Katsura89)": [0.0025],
                          "ol$\\rightarrow$wad (Morishima94)": [0.0034, 0.0038]}

        # Ringwoodite --> Bridgmanite + Ferropericlase Clapeyron slopes
        references_670 = {"ring$\\rightarrow$brg (Ito82)": [-0.002],
                          "ring$\\rightarrow$brg (Ito89 & Hirose02)": [-0.0028],
                          "ring$\\rightarrow$brg (Ito90)": [-0.002, -0.006],
                          "ring$\\rightarrow$brg (Katsura03)": [-0.0004, -0.002],
                          "ring$\\rightarrow$brg (Akaogi07)": [-0.0024, -0.0028]}

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

        # Legend colors
        colormap = plt.colormaps["tab10"]
        colors = [colormap(i) for i in range(9)]

        # Calculate phase boundaries:
        # Olivine --> Ringwoodite
        lines_410 = []
        labels_410 = set()

        for i, (ref, c_values) in enumerate(references_410.items()):
            ref_lines = []

            for j, c in enumerate(c_values):
                P = (T - 1758) * c + 13.4
                ref_lines.append(P)
                label = f"{ref}"
                labels_410.add(label)

            lines_410.append(ref_lines)

        # Ringwoodite --> Bridgmanite + Ferropericlase
        lines_670 = []
        labels_670 = set()

        for i, (ref, c_values) in enumerate(references_670.items()):
            ref_lines = []

            for j, c in enumerate(c_values):
                P = (T - 1883) * c + 23.0
                ref_lines.append(P)
                label = f"{ref}"
                labels_670.add(label)

            lines_670.append(ref_lines)

        # Plotting
        plt.figure()

        # Map labels to colors
        label_color_mapping = {}

        # Olivine --> Ringwoodite
        for i, (ref, ref_lines) in enumerate(zip(references_410.keys(), lines_410)):
            color = colors[i % len(colors)]

            for j, line in enumerate(ref_lines):
                label = f"{ref}" if j == 0 else None
                plt.plot(T[(T >= 1200) & (T <= 2000)], line[(T >= 1200) & (T <= 2000)],
                         color=color, label=label)

                if label not in label_color_mapping:
                    label_color_mapping[label] = color

        # Ringwoodite --> Bridgmanite + Ferropericlase
        for j, (ref, ref_lines) in enumerate(zip(references_670.keys(), lines_670)):
            color = colors[j + i + 1 % len(colors)]

            for j, line in enumerate(ref_lines):
                label = f"{ref}" if j == 0 else None
                plt.plot(T[(T >= 1250) & (T <= 2100)], line[(T >= 1250) & (T <= 2100)],
                         color=color, label=label)

                if label not in label_color_mapping:
                    label_color_mapping[label] = color

        # Plot shaded rectangle for PT range of training dataset
        fill = plt.fill_between(T, P_min, P_max, where=(T >= T_min) & (T <= T_max),
                                hatch="++", facecolor="none", alpha=0.1)

        # Mantle adiabats
        T_mantle1, T_mantle2, grad_mantle1, grad_mantle2 = 673, 1773, 0.5, 0.5

        # Calculate mantle geotherms
        geotherm1 = (T - T_mantle1) / (grad_mantle1 * 35)
        geotherm2 = (T - T_mantle2) / (grad_mantle2 * 35)

        # Find boundaries
        T1_Pmax = (P_max * grad_mantle1 * 35) + T_mantle1
        T2_Pmin = (P_min * grad_mantle2 * 35) + T_mantle2
        T2_Pmax = (P_max * grad_mantle2 * 35) + T_mantle2
        P1_Tmin = (T_min - T_mantle1) / (grad_mantle1 * 35)

        # Crop geotherms
        geotherm2_cropped = geotherm2[geotherm2 >= P_min]
        geotherm1_cropped = geotherm1[geotherm1 >= P1_Tmin]
        geotherm2_cropped = geotherm2_cropped[geotherm2_cropped <= P_max]
        geotherm1_cropped = geotherm1_cropped[geotherm1_cropped <= P_max]

        # Crop T vectors
        T_cropped_geotherm1= T[T >= T_min]
        T_cropped_geotherm2= T[T >= T2_Pmin]
        T_cropped_geotherm1 = T_cropped_geotherm1[T_cropped_geotherm1 <= T1_Pmax]
        T_cropped_geotherm2 = T_cropped_geotherm2[T_cropped_geotherm2 <= T2_Pmax]

        # Get geotherm (non-adiabatic)
        results = pd.DataFrame({"P": P_gt, "T": T_gt, "rho": rho_gt})
        P_geotherm, T_geotherm, _ = self._get_1d_profile("rho")

        # Plot mantle geotherms
        plt.plot(T_cropped_geotherm1, geotherm1_cropped, ":", color="black")
        plt.plot(T_cropped_geotherm2, geotherm2_cropped, ":", color="black")
        plt.plot(T_geotherm, P_geotherm, linestyle="--", color="black")

        # Interpolate the geotherms to have the same length as temperature vectors
        geotherm1_interp = np.interp(T_cropped_geotherm1, T, geotherm1)
        geotherm2_interp = np.interp(T_cropped_geotherm2, T, geotherm2)

        # Define the vertices for the polygon
        vertices = np.vstack(
            (
                np.vstack((T_cropped_geotherm1, geotherm1_interp)).T,
                (T_cropped_geotherm2[-1], geotherm2_interp[-1]),
                np.vstack((T_cropped_geotherm2[::-1], geotherm2_interp[::-1])).T,
                np.array([T_min, P_min]),
                (T_cropped_geotherm1[0], geotherm1_interp[0])
            )
        )

        # Fill the area within the polygon
        plt.fill(vertices[:, 0], vertices[:, 1], facecolor="white", edgecolor=None)
        plt.fill(vertices[:, 0], vertices[:, 1], facecolor="none", edgecolor="whitesmoke",
                 hatch="++")
        plt.fill(vertices[:, 0], vertices[:, 1], facecolor="white", edgecolor=None,
                 alpha=0.2)
        plt.fill_between(T, P_min, P_max, where=(T >= T_min) & (T <= T_max),
                         facecolor="none", edgecolor="black", linewidth=1.5)

        # Geotherm legend handles
        geotherm_handle = mlines.Line2D(
            [], [], linestyle="--", color="black", label="Avg. Mid-Ocean Ridge")

        # Phase boundaries legend handles
        ref_line_handles = [
            mlines.Line2D([], [], color=color, label=label)
            for label, color in label_color_mapping.items() if label]

        # Add geotherms to legend handles
        ref_line_handles.extend([geotherm_handle])

        db_data_handle = mpatches.Patch(facecolor="white", edgecolor="black", alpha=0.8,
                                        hatch="++", label="RocMLM Training Data")

        labels_670.add("RocMLM Training Data")
        label_color_mapping["RocMLM Training Data"] = "black"

        training_data_handle = mpatches.Patch(
            facecolor="white", edgecolor="black", linestyle=":",
            label="Hypothetical Mantle PTs")

        labels_670.add("Hypothetical Mantle PTs")
        label_color_mapping["Hypothetical Mantle PTs"] = "gray"

        # Define the desired order of the legend items
        desired_order = ["RocMLM Training Data",
                         "Hypothetical Mantle PTs",
                         "Avg. Mid-Ocean Ridge",
                         "ol$\\rightarrow$wad (Akaogi89)",
                         "ol$\\rightarrow$wad (Katsura89)",
                         "ol$\\rightarrow$wad (Morishima94)",
                         "ring$\\rightarrow$brg (Ito82)",
                         "ring$\\rightarrow$brg (Ito89 & Hirose02)",
                         "ring$\\rightarrow$brg (Ito90)",
                         "ring$\\rightarrow$brg (Katsura03)",
                         "ring$\\rightarrow$brg (Akaogi07)"]

        # Sort the legend handles based on the desired order
        legend_handles = sorted(ref_line_handles + [db_data_handle, training_data_handle],
                                key=lambda x: desired_order.index(x.get_label()))

        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure (GPa)")
        plt.xlim(T_min - 100, T_max + 100)
        plt.ylim(P_min - 1, P_max + 1)

        # Move the legend outside the plot to the right
        plt.legend(title="", handles=legend_handles, loc="center left", handleheight=1.2,
                   bbox_to_anchor=(1.02, 0.5))

        # Adjust the figure size
        fig = plt.gcf()
        fig.set_size_inches(figwidth, figheight)

        # Save the plot to a file
        plt.savefig(f"{fig_dir}/training-dataset-design.png")

        # Close device
        plt.close()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize targets  !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_target_array(self, rmse=None, r2=None, palette="bone", geotherms=True,
                                gradient=False, figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get model data
        sid = self.sid
        res = self.res
        targets = self.targets
        results = self.results
        fig_dir = self.fig_dir
        geothresh = self.geothresh
        model_built = self.model_built
        target_array = self.target_array
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
            targets = ["rho", "Vp", "Vs", "melt", "h2o"]

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
            # Set filename
            filename = f"{sid}-{target}.png"

            # Reshape targets into square array
            square_target = target_array[:, i].reshape(res + 1, res + 1)

            # Check for all nans
            if np.all(np.isnan(square_target)):
                print(f"  All nans in {target} array. Skipping plot ...")
                continue

            # Sobel filter for gradient images
            if gradient:
                original_image = square_target.copy()

                # Apply Sobel edge detection
                edges_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
                edges_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate the magnitude of the gradient
                square_target = np.sqrt(edges_x**2 + edges_y**2) / np.nanmax(original_image)

                filename = f"{sid}-{target}-grad.png"

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
                    color_reverse = False
                else:
                    color_reverse = True

            # Set colorbar limits for better comparisons
            if not color_discrete:
                vmin = np.min(square_target[np.logical_not(np.isnan(square_target))])
                vmax = np.max(square_target[np.logical_not(np.isnan(square_target))])

            else:
                vmin = int(np.nanmin(np.unique(square_target)))
                vmax = int(np.nanmax(np.unique(square_target)))

            # Make results df
            results = pd.DataFrame({"P": P, "T": T, target: square_target.flatten()})

            # Get geotherm
            P_geotherm, T_geotherm, _ = self._get_1d_profile(target, 1173)
            P_geotherm2, T_geotherm2, _ = self._get_1d_profile(target, 1573)
            P_geotherm3, T_geotherm3, _ = self._get_1d_profile(target, 1773)

            if color_discrete:
                # Discrete color palette
                num_colors = vmax - vmin + 1
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
                cmap.set_bad(color="white")

                # Plot as a raster using imshow
                fig, ax = plt.subplots(figsize=(figwidth, figheight))

                im = ax.imshow(square_target, extent=extent, aspect="auto", cmap=cmap,
                               origin="lower", vmin=vmin, vmax=vmax)
                if geotherms:
                    ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white",
                            linewidth=3)
                    ax.plot(T_geotherm2, P_geotherm2, linestyle="--", color="white",
                            linewidth=3)
                    ax.plot(T_geotherm3, P_geotherm3, linestyle="-.", color="white",
                            linewidth=3)
                    plt.text(1163 + (6 * 0.5 * 35), 6, "1173", fontsize=fontsize * 0.833,
                             horizontalalignment="center", verticalalignment="bottom",
                             rotation=67, color="white")
                    plt.text(1563 + (6 * 0.5 * 35), 6, "1573", fontsize=fontsize * 0.833,
                             horizontalalignment="center", verticalalignment="bottom",
                             rotation=67, color="white")
                    plt.text(1763 + (6 * 0.5 * 35), 6, "1773", fontsize=fontsize * 0.833,
                             horizontalalignment="center", verticalalignment="bottom",
                             rotation=67, color="white")
                ax.set_xlabel("T (K)")
                ax.set_ylabel("P (GPa)")
                plt.colorbar(im, ax=ax, label="", ticks=np.arange(vmin, vmax, num_colors))

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

                    # Set melt fraction to 0–100 vol.%
                    if target == "melt": vmin, vmax = 0, 100

                    # Set h2o fraction to 0–100 wt.%
                    if target == "h2o": vmin, vmax = 0, 20

                # Set nan color
                cmap = plt.colormaps[cmap]
                cmap.set_bad(color="white")

                # Plot as a raster using imshow
                fig, ax = plt.subplots()

                im = ax.imshow(square_target, extent=extent, aspect="auto", cmap=cmap,
                               origin="lower", vmin=vmin, vmax=vmax)
                if geotherms:
                    ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white",
                            linewidth=3)
                    ax.plot(T_geotherm2, P_geotherm2, linestyle="--", color="white",
                            linewidth=3)
                    ax.plot(T_geotherm3, P_geotherm3, linestyle="-.", color="white",
                            linewidth=3)
                    plt.text(1163 + (6 * 0.5 * 35), 6, "1173", fontsize=fontsize * 0.833,
                             horizontalalignment="center", verticalalignment="bottom",
                             rotation=67, color="white")
                    plt.text(1563 + (6 * 0.5 * 35), 6, "1573", fontsize=fontsize * 0.833,
                             horizontalalignment="center", verticalalignment="bottom",
                             rotation=67, color="white")
                    plt.text(1763 + (6 * 0.5 * 35), 6, "1773", fontsize=fontsize * 0.833,
                             horizontalalignment="center", verticalalignment="bottom",
                             rotation=67, color="white")
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
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                elif target == "assemblage":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
                elif target == "variance":
                    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

            plt.title("GFEM")

            # Vertical text spacing
            text_margin_x = 0.04
            text_margin_y = 0.15
            text_spacing_y = 0.1

            # Add rmse and r2
            if rmse is not None and r2 is not None:
                bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=1.5,
                                  alpha=0.3)
                plt.text(text_margin_x, text_margin_y - (text_spacing_y * 0),
                         f"R$^2$: {r2:.3f}", transform=plt.gca().transAxes,
                         fontsize=fontsize * 0.833, horizontalalignment="left",
                         verticalalignment="bottom", bbox=bbox_props)
                plt.text(text_margin_x, text_margin_y - (text_spacing_y * 1),
                         f"RMSE: {rmse:.3f}", transform=plt.gca().transAxes,
                         fontsize=fontsize * 0.833, horizontalalignment="left",
                         verticalalignment="bottom", bbox=bbox_props)

            # Save the plot to a file
            plt.savefig(f"{fig_dir}/{filename}")

            # Close device
            plt.close()
            print(f"  Figure saved to: {fig_dir}/{filename} ...")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize prem !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_prem(self, geotherms=["low", "mid", "high"], figwidth=6.3,
                        figheight=4.725, fontsize=22):
        """
        """
        # Get model data
        sid = self.sid
        res = self.res
        P_min = self.P_min
        P_max = self.P_max
        T_min = self.T_min
        T_max = self.T_max
        targets = self.targets
        results = self.results
        fig_dir = self.fig_dir
        data_dir = self.data_dir
        geothresh = self.geothresh
        perplex_db = self.perplex_db
        model_built = self.model_built
        target_units = self.target_units

        # Check for model
        if not model_built:
            raise Exception("No GFEM model! Call build_model() first ...")

        # Check for results
        if not results:
            raise Exception("No GFEM model results! Call get_results() first ...")

        # Check for data dir
        if not os.path.exists(data_dir):
            raise Exception(f"Data not found at {data_dir} !")

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Check for average geotherm
        if "mid" not in geotherms:
            geotherms = geotherms + ["mid"]

        # Filter targets for PREM
        t_ind = [i for i, t in enumerate(targets) if t in ["rho", "Vp", "Vs"]]
        target_units = [target_units[i] for i in t_ind]
        targets = [targets[i] for i in t_ind]

        # Get 1D reference models
        ref_models = self._get_1d_reference_models()

        for i, target in enumerate(targets):
            # Get 1D reference model profiles
            P_prem, target_prem = ref_models["prem"]["P"], ref_models["prem"][target]
            P_stw105, target_stw105 = ref_models["stw105"]["P"], ref_models["stw105"][target]

            # Process GFEM model profiles
            if "low" in geotherms:
                P_gfem, _, target_gfem = self._get_1d_profile(target, 1173)
                P_gfem, target_gfem, _, _, _, _ = self._crop_1d_profile(
                    P_gfem, target_gfem, P_prem, target_prem)
            if "mid" in geotherms:
                P_gfem2, _, target_gfem2 = self._get_1d_profile(target, 1573)
                P_gfem2, target_gfem2, _, _, rmse, r2 = self._crop_1d_profile(
                    P_gfem2, target_gfem2, P_prem, target_prem)
            if "high" in geotherms:
                P_gfem3, _, target_gfem3 = self._get_1d_profile(target, 1773)
                P_gfem3, target_gfem3, _, _, _, _ = self._crop_1d_profile(
                    P_gfem3, target_gfem3, P_prem, target_prem)

            # Change endmember sampleids
            if sid == "sm000":
                sid_lab = "DSUM"
            elif sid == "sm129":
                sid_lab = "PSUM"
            else:
                sid_lab = sid

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

            # Plotting
            fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

            # Plot GFEM model profiles
            if "low" in geotherms:
                ax1.plot(target_gfem, P_gfem, "-", linewidth=3, color=colormap(0),
                         label=f"{sid_lab}-1173")
                ax1.fill_betweenx(P_gfem, target_gfem * (1 - 0.05), target_gfem * (1 + 0.05),
                                  color=colormap(0), alpha=0.2)
            if "mid" in geotherms:
                ax1.plot(target_gfem2, P_gfem2, "-", linewidth=3, color=colormap(2),
                         label=f"{sid_lab}-1573")
                ax1.fill_betweenx(P_gfem2, target_gfem2 * (1 - 0.05),
                                  target_gfem2 * (1 + 0.05), color=colormap(2), alpha=0.2)
            if "high" in geotherms:
                ax1.plot(target_gfem3, P_gfem3, "-", linewidth=3, color=colormap(1),
                         label=f"{sid_lab}-1773")
                ax1.fill_betweenx(P_gfem3, target_gfem3 * (1 - 0.05),
                                  target_gfem3 * (1 + 0.05), color=colormap(1), alpha=0.3)

            # Plot reference models
            ax1.plot(target_prem, P_prem, "-", linewidth=2, color="black")
            ax1.plot(target_stw105, P_stw105, ":", linewidth=2, color="black")

            if target == "rho":
                target_label = "Density"
            else:
                target_label = target

            ax1.set_xlabel(f"{target_label} ({target_units[i]})")
            ax1.set_ylabel("P (GPa)")

            if target in ["Vp", "Vs", "rho"]:
                ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

            # Vertical text spacing
            text_margin_x = 0.04
            text_margin_y = 0.15
            text_spacing_y = 0.1

            # Add R-squared and RMSE values as text annotations in the plot
            plt.text(text_margin_x, 1 - (text_margin_y - (text_spacing_y * 0)),
                     f"R$^2$: {r2:.3f}", transform=plt.gca().transAxes,
                     fontsize=fontsize * 0.833, horizontalalignment="left",
                     verticalalignment="top")
            plt.text(text_margin_x, 1 - (text_margin_y - (text_spacing_y * 1)),
                     f"RMSE: {rmse:.3f}", transform=plt.gca().transAxes,
                     fontsize=fontsize * 0.833, horizontalalignment="left",
                     verticalalignment="top")

            # Convert the primary y-axis data (pressure) to depth
            depth_conversion = lambda P: P * 30
            depth_values = depth_conversion(P_prem)

            # Create the secondary y-axis and plot depth on it
            ax2 = ax1.secondary_yaxis(
                "right", functions=(depth_conversion, depth_conversion))
            ax2.set_yticks([410, 670])
            ax2.set_ylabel("Depth (km)")

            plt.legend(loc="lower right", columnspacing=0, handletextpad=0.2,
                       fontsize=fontsize * 0.833)

            plt.title("Depth Profile")

            # Save the plot to a file
            filename = f"{sid}-{target}-prem.png"
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
            if not self._check_pt_range_image():
                self._visualize_gfem_pt_range()
            if not self._check_model_target_images():
                self._visualize_target_array()
            if not self._check_model_target_images(gradient=True):
                self._visualize_target_array(gradient=True)
            if not self._check_model_prem_images():
                self._visualize_prem()
        except Exception as e:
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
        self._print_gfem_model_info()
        max_retries = 3
        for retry in range(max_retries):
            # Check for built model
            if self.model_built:
                break
            try:
                self._configure_perplex_model()
                self._run_perplex()
                self._get_comp_time()
                self._process_perplex_results()
                break
            except Exception as e:
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
        try:
            self._measure_gfem_model_accuracy_vs_prem()
            self.visualize_model()
        except Exception as e:
            print(f"!!! ERROR in build_model() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

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
# compose gfem plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_gfem_plots(gfem_models, clean=False):
    """
    """
    # Iterate through all models
    print("Composing GFEM plots ...")
    for gfem_model in gfem_models:
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

            check_comps = ((os.path.exists(fig_3) and os.path.exists(fig_4)) |
                           (os.path.exists(fig_1) and os.path.exists(fig_2)) |
                           (os.path.exists(fig_1) and os.path.exists(fig_2) and
                            os.path.exists(fig_4)))

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

                if target in ["rho", "Vp", "Vs"]:
                    combine_plots_horizontally(
                        f"{fig_dir}/{sid}-{target}.png",
                        f"{fig_dir}/{sid}-{target}-grad.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{sid}-{target}-prem.png",
                        f"{fig_dir}/image3-{sid}-{target}.png",
                        caption1="",
                        caption2="c)"
                    )

                    print(f"  Figure saved to: {fig_dir}/image3-{sid}-{target}.png ...")

        if all(item in targets for item in ["rho", "Vp", "Vs"]):
            captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
            targets = ["rho", "Vp", "Vs"]

            for i, target in enumerate(targets):
                combine_plots_horizontally(
                    f"{fig_dir}/{sid}-{target}.png",
                    f"{fig_dir}/{sid}-{target}-grad.png",
                    f"{fig_dir}/temp1.png",
                    caption1=captions[i][0],
                    caption2=captions[i][1]
                )

                combine_plots_horizontally(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/{sid}-{target}-prem.png",
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
                f"{fig_dir}/image9-{sid}.png",
                caption1="",
                caption2=""
            )

            print(f"  Figure saved to: {fig_dir}/image9-{sid}.png ...")

        tmp_files = glob.glob(f"{fig_dir}/temp*.png")
        for file in tmp_files:
            os.remove(file)

        if clean:
            # Clean up directory
            prem_files = glob.glob(f"{fig_dir}/*prem.png")
            grad_files = glob.glob(f"{fig_dir}/*grad.png")

            for file in prem_files + grad_files:
                os.remove(file)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize prem comps !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_prem_comps(gfem_models, fig_dir="figs/other", filename="prem-comps.png",
                         figwidth=6.3, figheight=5.8, fontsize=28):
    """
    """
    warnings.simplefilter("ignore", category=UserWarning)

    # Data asset dir
    data_dir = "assets/data"

    # Check for data dir
    if not os.path.exists(data_dir):
        raise Exception(f"Data not found at {data_dir} !")

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Get correct Depletion column
    XI_col = "XI_FRAC"

    # Check for benchmark samples
    df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"

    # Read benchmark samples
    if os.path.exists(df_synth_bench_path):
        df_synth_bench = pd.read_csv(df_synth_bench_path)

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

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(figwidth * 3, figheight))

    for j, model in enumerate(gfem_models):
        # Get gfem model data
        sid = model.sid
        res = model.res
        targets = model.targets
        results = model.results
        xi = model.fertility_index
        geothresh = model.geothresh
        target_units = model.target_units

        # Filter targets for PREM
        t_ind = [i for i, t in enumerate(targets) if t in ["rho", "Vp", "Vs"]]
        target_units = [target_units[i] for i in t_ind]
        targets = [targets[i] for i in t_ind]

        # Get 1D reference models
        ref_models = model._get_1d_reference_models()

        for i, target in enumerate(targets):
            # Get 1D reference model profiles
            P_prem, target_prem = ref_models["prem"]["P"], ref_models["prem"][target]
            P_stw105, target_stw105 = ref_models["stw105"]["P"], ref_models["stw105"][target]

            # Get GFEM model profile
            P_gfem, _, target_gfem = gfem._get_1d_profile(
                target, Qs=55e-3, A1=1e-6, k1=2.3, litho_thickness=150)

            # Crop GFEM profile and compare with PREM
            P_gfem, target_gfem, P_prem, target_prem, _, _ = (
                self._crop_1d_profile(P_gfem, target_gfem, P_prem, target_prem))

            # Create colorbar
            pal = sns.color_palette("magma", as_cmap=True).reversed()
            norm = plt.Normalize(df_synth_bench[XI_col].min(), df_synth_bench[XI_col].max())
            sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
            sm.set_array([])

            ax = axes[i]

            if j == 0:
                # Plot reference models
                ax.plot(target_prem, P_prem, "-", linewidth=4.5, color="forestgreen",
                        label="PREM", zorder=7)
                ax.plot(target_stw105, P_stw105, ":", linewidth=4.5, color="forestgreen",
                        label="STW105", zorder=7)

            if sid == "sm129":
                ax.plot(target_gfem, P_gfem, "-", linewidth=6.5, color=sm.to_rgba(xi),
                        label="PSUM", zorder=6)
            if sid == "sm000":
                ax.plot(target_gfem, P_gfem, "-", linewidth=6.5, color=sm.to_rgba(xi),
                        label="DSUM", zorder=6)

            # Plot GFEM and RocMLM profiles
            ax.plot(target_gfem, P_gfem, "-", linewidth=1, color=sm.to_rgba(xi), alpha=0.1)

            if target == "rho":
                target_label = "Density"
            else:
                target_label = target

            if i != 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("P (GPa)")

            ax.set_xlabel(f"{target_label} ({target_units[i]})")

            if target in ["Vp", "Vs", "rho"]:
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

            # Convert the primary y-axis data (pressure) to depth
            depth_conversion = lambda P: P * 30
            depth_values = depth_conversion(P_prem)

            if i == 2:
                # Create the secondary y-axis and plot depth on it
                ax2 = ax.secondary_yaxis(
                    "right", functions=(depth_conversion, depth_conversion))
                ax2.set_yticks([410, 670])
                ax2.set_ylabel("Depth (km)")
                cbaxes = inset_axes(ax, width="40%", height="3%", loc=2)
                colorbar = plt.colorbar(sm, ax=ax, cax=cbaxes, label="$\\xi$",
                                        orientation="horizontal")

                ax.legend(loc="lower right", columnspacing=0, handletextpad=0.2,
                           fontsize=fontsize * 0.833)

    fig.text(0.00, 0.98, "a)", fontsize=fontsize * 1.4)
    fig.text(0.33, 0.98, "b)", fontsize=fontsize * 1.4)
    fig.text(0.65, 0.98, "c)", fontsize=fontsize * 1.4)

    # Save the plot to a file
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
    perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax = args

    with redirect_stdout(stdout_buffer):
        iteration = GFEMModel(perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax)

        if not iteration.model_built:
            iteration.build_model()
            queue.put(stdout_buffer.getvalue())

    return iteration

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build gfem models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_gfem_models(source, sampleids=None, perplex_db="hp02", res=128, Pmin=1,
                      Pmax=28, Tmin=773, Tmax=2273, nprocs=os.cpu_count() - 2, verbose=1):
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
    print("Building GFEM models for samples:")
    print(sampleids)

    # Define number of processors
    if nprocs is None or nprocs > os.cpu_count():
        nprocs = os.cpu_count()

    # Make sure nprocs is not greater than sampleids
    if nprocs > len(sampleids):
        nprocs = len(sampleids)

    # Create list of args for mp pooling
    run_args = [(perplex_db, sampleid, source, res, Pmin, Pmax, Tmin, Tmax) for
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
        print(f"Total models with errors: {error_count}")
    else:
        print("All GFEM models successfully built!")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return gfems

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    try:
        # Build GFEM models
        gfems = {}
        sources = {"benchmark": "assets/data/benchmark-samples-pca.csv",
                   "middle": "assets/data/synthetic-samples-mixing-middle.csv",
                   "random": "assets/data/synthetic-samples-mixing-random.csv"}

        for name, source in sources.items():
            sids = get_sampleids(source)
            gfems[name] = build_gfem_models(source, sids, "hp02", 64)

        # Compose plots
        for name, models in gfems.items():
            compose_gfem_plots(models)

        # Visualize profile compositions vs. PREM
        visualize_prem_comps(gfems["middle"] + gfems["random"])

    except Exception as e:
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
