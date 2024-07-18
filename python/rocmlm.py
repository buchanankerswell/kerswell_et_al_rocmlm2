#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import glob
import time
import joblib
import warnings
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
from gfem import get_sampleids, build_gfem_models

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parallel computing !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing as mp

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from scipy.interpolate import interp1d
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import ListedColormap, Normalize, SymLogNorm

#######################################################
## .1.               RocMLM Class                !!! ##
#######################################################
class RocMLM:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, gfem_models, ml_model, X_step=1, PT_step=1,
                 training_features=["XI_FRAC", "LOI"],
                 training_targets=["rho", "h2o", "melt"], tune=True,
                 epochs=100, batchprop=0.2, kfolds=5, nprocs=os.cpu_count() - 2, seed=42,
                 verbose=1):
        """
        """
        # Input
        self.seed = seed
        self.tune = tune
        self.epochs = epochs
        self.kfolds = kfolds
        self.X_step = X_step
        self.PT_step = PT_step
        self.verbose = verbose
        self.batchprop = batchprop
        self.gfem_models = gfem_models
        self.training_targets = training_targets
        self.training_features = training_features

        # Parallel processing
        if nprocs is None or nprocs > os.cpu_count():
            self.nprocs = os.cpu_count()
        else:
            self.nprocs = nprocs

        # Model labels
        if ml_model == "KN":
            ml_model_label_full = "K Neighbors"
        elif ml_model == "RF":
            ml_model_label_full = "Random Forest"
        elif ml_model == "DT":
            ml_model_label_full = "Decision Tree"
        elif ml_model == "NN1":
            ml_model_label_full = "Neural Net 1L"
        elif ml_model == "NN2":
            ml_model_label_full = "Neural Net 2L"
        elif ml_model == "NN3":
            ml_model_label_full = "Neural Net 3L"
        self.ml_model_label = ml_model
        self.ml_model_label_full = ml_model_label_full

        # Feature and target arrays
        self.sids = []
        self.res = None
        self.P_min = None
        self.P_max = None
        self.T_min = None
        self.T_max = None
        self.targets = None
        self.features = None
        self.geothresh = None
        self.target_units = None
        self.shape_target = None
        self.target_train = None
        self.feature_train = None
        self.shape_feature = None
        self.shape_target_square = None
        self.shape_feature_square = None

        # Get gfem model metadata
        self._get_gfem_model_metadata()
        self._process_training_data()

        # Output filepaths
        self.data_dir = "assets/data"
        self.model_out_dir = f"rocmlms"
        if any(sample in self.sids for sample in ["PUM", "DMM", "PYR"]):
            self.model_prefix = (f"benchmark-{self.ml_model_label}-"
                                 f"S{self.shape_feature_square[0]}-"
                                 f"W{self.shape_feature_square[1]}")
            self.fig_dir = f"figs/rocmlm/benchmark_{self.ml_model_label}"
        elif any("sm" in sample or "sr" in sample for sample in self.sids):
            self.model_prefix = (f"synthetic-{self.ml_model_label}-"
                                 f"S{self.shape_feature_square[0]}-"
                                 f"W{self.shape_feature_square[1]}")
            self.fig_dir = (f"figs/rocmlm/synthetic_{self.ml_model_label}")
        self.rocmlm_path = f"{self.model_out_dir}/{self.model_prefix}.pkl"
        self.ml_model_only_path = f"{self.model_out_dir}/{self.model_prefix}-model-only.pkl"
        self.ml_model_scaler_X_path = (f"{self.model_out_dir}/{self.model_prefix}-"
                                       f"scaler_X.pkl")
        self.ml_model_scaler_y_path = (f"{self.model_out_dir}/{self.model_prefix}-"
                                       f"scaler_y.pkl")

        # Check for figs directory
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir, exist_ok=True)

        # ML model definition and tuning
        self.ml_model = None
        self.ml_model_tuned = False
        self.model_hyperparams = None

        # Cross validation performance metrics
        self.cv_info = {}
        self.ml_model_cross_validated = False

        # Square arrays for visualizations
        self.target_square = np.array([])
        self.feature_square = np.array([])
        self.prediction_square = np.array([])

        # Trained model
        self.ml_model_only = None
        self.ml_model_error = None
        self.ml_model_trained = False
        self.ml_model_scaler_X = None
        self.ml_model_scaler_y = None
        self.ml_model_training_error = False

        # Set np array printing option
        np.set_printoptions(precision=3, suppress=True)

        # Check for existing pretrained model
        self._check_pretrained_model()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # print rocmlm info !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_rocmlm_info(self):
        """
        """
        # Get self attributes
        kfolds = self.kfolds
        epochs = self.epochs
        feat_train = self.feature_train
        target_train = self.target_train
        ml_model_label = self.ml_model_label
        training_targets = self.training_targets
        training_features = self.training_features
        model_hyperparams = self.model_hyperparams
        ml_model_label_full = self.ml_model_label_full

        # Print rocmlm config
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("RocMLM model defined as:")
        print(f"    model: {ml_model_label_full}")
        if "NN" in ml_model_label:
            print(f"    epochs: {epochs}")
        print(f"    k folds: {kfolds}")
        print(f"    features: {training_features}")
        print(f"    targets: {training_targets}")
        print(f"    features array shape: {feat_train.shape}")
        print(f"    targets array shape: {target_train.shape}")
        print(f"    hyperparameters:")
        for key, value in model_hyperparams.items():
            print(f"        {key}: {value}")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Running ({kfolds}) kfold cross validation ...")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check pretrained model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_pretrained_model(self):
        """
        """
        # Check for existing model build
        if os.path.exists(self.model_out_dir):
            if os.path.exists(self.rocmlm_path):
                self.ml_model_trained = True
                if self.verbose >= 1:
                    print(f"Found pretrained model {self.rocmlm_path}!")
        else:
            os.makedirs(self.model_out_dir, exist_ok=True)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get unique value !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_unique_value(self, input_list):
        """
        """
        unique_value = input_list[0]
        for item in input_list[1:]:
            if item != unique_value:
                raise ValueError("Not all values are the same!")

        return unique_value

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get gfem model metadata !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_gfem_model_metadata(self):
        """
        """
        # Get self attributes
        gfem_models = self.gfem_models

        # Check for gfem models
        if not gfem_models:
            raise Exception("No GFEM models to compile!")

        try:
            # Get model metadata
            sids = [m.sid for m in gfem_models]
            res = self._get_unique_value([m.res for m in gfem_models])
            P_min = self._get_unique_value([m.P_min for m in gfem_models])
            P_max = self._get_unique_value([m.P_max for m in gfem_models])
            T_min = self._get_unique_value([m.T_min for m in gfem_models])
            T_max = self._get_unique_value([m.T_max for m in gfem_models])
            targets = self._get_unique_value([m.targets for m in gfem_models])
            ox_gfem = self._get_unique_value([m.ox_gfem for m in gfem_models])
            fts_list = self._get_unique_value([m.features for m in gfem_models])
            ox_exclude = self._get_unique_value([m.ox_exclude for m in gfem_models])
            target_units = self._get_unique_value([m.target_units for m in gfem_models])

            # Combine training features
            subset_oxides = [oxide for oxide in ox_gfem if oxide not in ox_exclude]
            features = subset_oxides + [ft for ft in fts_list if ft not in subset_oxides]

            # Update self attributes
            self.res = res
            self.sids = sids
            self.P_min = P_min
            self.P_max = P_max
            self.T_min = T_min
            self.T_max = T_max
            self.targets = targets
            self.features = features
            self.target_units = target_units

            # Set geotherm threshold for depth profiles
            res_step = int(res / self.PT_step)
            if res_step <= 8:
                self.geothresh = 40
            elif res_step <= 16:
                self.geothresh = 20
            elif res_step <= 32:
                self.geothresh = 10
            elif res_step <= 64:
                self.geothresh = 5
            elif res_step <= 128:
                self.geothresh = 2.5
            else:
                self.geothresh = 1.25

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_gfem_model_metadata() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process training data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_training_data(self):
        """
        """
        # Get self attributes
        res = self.res
        sids = self.sids
        X_step = self.X_step
        PT_step = self.PT_step
        targets = self.targets
        features = self.features
        gfem_models = self.gfem_models
        training_targets = self.training_targets
        training_features = self.training_features

        try:
            # Get all PT arrays
            pt_train = np.stack([m.feature_array for m in gfem_models])

            # Select features
            ft_ind = [features.index(ft) for ft in training_features if ft in features]
            training_features = [features[i] for i in ft_ind]

            # Get sample features
            feat_train = []

            for m in gfem_models:
                selected_features = [m.sample_features[i] for i in ft_ind]
                feat_train.append(selected_features)

            feat_train = np.array(feat_train)

            # Tile features to match PT array shape
            feat_train = np.tile(feat_train[:, np.newaxis, :], (1, pt_train.shape[1], 1))

            # Combine features
            combined_train = np.concatenate((feat_train, pt_train), axis=2)

            # Flatten features
            feature_train = combined_train.reshape(-1, combined_train.shape[-1])

            # Define target indices
            t_ind = [targets.index(t) for t in training_targets if t in targets]
            training_targets = [targets[i] for i in t_ind]

            # Get target arrays
            target_train = np.stack([m.target_array for m in gfem_models])

            # Flatten targets
            target_train = target_train.reshape(-1, target_train.shape[-1])

            # Select training targets
            target_train = target_train[:, t_ind]

            # Define array shapes
            M = int(len(gfem_models))
            W = int((res + 1) ** 2)
            w = int(np.sqrt(W))
            F = int(len(training_features) + 2)
            T = int(len(training_targets))
            shape_target = (M, W, T)
            shape_feature = (M, W, F)
            shape_target_square = (M, w, w, T)
            shape_feature_square = (M, w, w, F)

            if not X_step and not PT_step:
                self.shape_target = shape_target
                self.target_train = target_train
                self.shape_feature = shape_feature
                self.feature_train = feature_train
                self.shape_target_square = shape_target_square
                self.shape_feature_square = shape_feature_square

            else:
                # Reshape arrays
                new_feature_train = feature_train.reshape((shape_feature_square))
                new_target_train = target_train.reshape((shape_target_square))

                # Subset arrays
                new_feature_train = new_feature_train[::X_step, ::PT_step, ::PT_step, :]
                new_target_train = new_target_train[::X_step, ::PT_step, ::PT_step, :]

                # Redefine array shapes
                new_M = np.ceil((len(gfem_models) / X_step)).astype(int)
                new_W = int(((res / PT_step) + 1) ** 2)
                new_w = int(np.sqrt(new_W))
                new_shape_target = (new_M, new_W, T)
                new_shape_feature = (new_M, new_W, F)
                new_shape_target_square = (new_M, new_w, new_w, T)
                new_shape_feature_square = (new_M, new_w, new_w, F)

                # Flatten arrays
                new_feature_train = new_feature_train.reshape(
                    -1, new_feature_train.shape[-1])
                new_target_train = new_target_train.reshape(
                    -1, new_target_train.shape[-1])

                self.target_train = new_target_train
                self.shape_target = new_shape_target
                self.shape_feature = new_shape_feature
                self.feature_train = new_feature_train
                self.shape_target_square = new_shape_target_square
                self.shape_feature_square = new_shape_feature_square

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _process_training_data() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

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
    def _get_1d_profile(self, target=None, results=None, mantle_potential=1573, Qs=250e-3,
                        Ts=273, A1=2.2e-8, A2=2.2e-8, k1=3.0, k2=3.0, crust_thickness=35,
                        litho_thickness=1):
        """
        """
        # Get model attributes
        res = self.res
        P_min = self.P_min
        P_max = self.P_max
        T_min = self.T_min
        T_max = self.T_max
        geothresh = self.geothresh
        ml_model_trained = self.ml_model_trained

        # Check for model
        if not ml_model_trained:
            raise Exception("No RocMLM model! Call train_rocmlm() first ...")

        # Define PT (target)
        if not target:
            P_array = np.linspace(P_min, P_max, res + 1)
            T_array = np.linspace(T_min, T_max, res + 1)
            P_grid, T_grid = np.meshgrid(P_array, T_array)
            P, T = P_grid.flatten(), T_grid.flatten()
            df = pd.DataFrame({"P": P, "T": T}).sort_values(by=["P", "T"])
        else:
            P = results["P"]
            T = results["T"]
            trg = results[target]
            df = pd.DataFrame({"P": P, "T": T, target: trg}).sort_values(by=["P", "T"])

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

        # Get PT values
        P_values = np.nan_to_num(df["P"].values)
        T_values = np.nan_to_num(df["T"].values)

        # Get target values
        if target:
            targets = np.nan_to_num(df[target].values)
            return P_values, T_values, targets
        else:
            return P_values, T_values

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
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in crop_1d_profile() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

            return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # scale arrays !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scale_arrays(self, feature_array, target_array):
        """
        """
        # Reshape the features and targets arrays
        X = feature_array
        y = target_array

        # Replace inf with nan
        X[~np.isfinite(X)] = np.nan
        y[~np.isfinite(y)] = np.nan

        # Create nan mask
        mask = np.any(np.isnan(y), axis=1)

        # Remove nans
#        X, y = X[~mask,:], y[~mask,:]
        X, y = np.nan_to_num(X), np.nan_to_num(y)

        # Check for infinity in input data
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Input data contains NaN or infinity values.")

        # Initialize scalers
        scaler_X, scaler_y = StandardScaler(), StandardScaler()

        # Scale features array
        X_scaled = scaler_X.fit_transform(X)

        # Scale the target array
        y_scaled = scaler_y.fit_transform(y)

        return X, y, scaler_X, scaler_y, X_scaled, y_scaled

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.             Lookup Tables               !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # append to lut csv !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _append_to_lut_csv(self, data_dict):
        """
        """
        # CSV filepath
        filepath = f"assets/data/lut-efficiency.csv"

        # Check if the CSV file already exists
        if not pd.io.common.file_exists(filepath):
            df = pd.DataFrame(data_dict)

        else:
            df = pd.read_csv(filepath)

            # Append the new data dictionary to the DataFrame
            new_data = pd.DataFrame(data_dict)

            df = pd.concat([df, new_data], ignore_index=True)

        # Sort df
        df = df.sort_values(by=["sample", "size"])

        # Save the updated DataFrame back to the CSV file
        df.to_csv(filepath, index=False)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # evaluate lut performance !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _evaluate_lut_performance(self):
        """
        """
        # Get self attributes
        sids = self.sids
        X_step = self.X_step
        PT_step = self.PT_step
        gfem_models = self.gfem_models
        target_train = self.target_train

        # Check benchmark models
        if any(sample in sids for sample in ["PUM", "DMM", "PYR"]):
            return None

        # Check for gfem models
        if not gfem_models:
            raise Exception("No GFEM models to compile!")

        try:
            # Get feature (PTX) arrays
            P = np.unique(gfem_models[0].feature_array[:, 0])
            T = np.unique(gfem_models[0].feature_array[:, 1])
            X = np.array([m.fertility_index for m in gfem_models])

            # Make X increase monotonically
            X = np.linspace(np.min(X), np.max(X), X.shape[0])

            # Get central PTX points
            P_point = (np.max(P) - np.min(P)) / 2
            T_point = (np.max(T) - np.min(T)) / 2
            X_point = (np.max(X) - np.min(X)) / 2

            # Get target arrays
            target_train = np.stack([m.target_array for m in gfem_models])

            # Initialize df columns
            sample, model_label, size, eval_time, model_size_mb = [], [], [], [], []

            # Subset grid
            X_sub = X[::X_step]
            P_sub = P[::PT_step]
            T_sub = T[::PT_step]

            # Initialize eval times
            eval_times = []

            # Clock evaluation time for each target
            for i in range(target_train.shape[-1] - 1):
                # Select a single target array
                Z = target_train[:, :, i]

                # Reshape into 3d rectilinear grid
                Z = np.reshape(Z, (X.shape[0], P.shape[0], T.shape[0]))

                # Subset grid
                Z_sub = Z[::X_step, ::PT_step, ::PT_step]

                # Handle nans
                Z_sub = np.nan_to_num(Z_sub)

                # Initialize interpolator
                I = RegularGridInterpolator((X_sub, P_sub, T_sub), Z_sub, method="cubic",
                                            bounds_error=False)

                # Time lookup table evaluation at central PT point
                start_time = time.time()
                point_eval = I(np.array([X_point, P_point, T_point]))
                end_time = time.time()

                # Calculate evaluation time
                elapsed_time = (end_time - start_time)

                # Save elapsed time
                eval_times.append(elapsed_time)

            # Store df info
            sample.append(f"SYNTH{X_sub.shape[0]}")

            model_label.append("lut")
            size.append((P_sub.shape[0] - 1) ** 2)
            eval_time.append(round(sum(eval_times), 5))

            # Save ml model only
            lut_path = f"rocmlms/lut-S{X_sub.shape[0]}-W{P_sub.shape[0]}.pkl"
            with open(lut_path, "wb") as file:
                joblib.dump(I, file)

            # Add lut model size (Mb)
            model_size = os.path.getsize(lut_path) * (target_train.shape[-1] - 1)
            model_size_mb.append(round(model_size / (1024 ** 2), 5))

            # Create df
            evals = {"sample": sample, "model": model_label, "size": size,
                     "time": eval_time, "model_size_mb": model_size_mb}

            # Write csv
            self._append_to_lut_csv(evals)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _evaluate_lut_performance() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.3.               RocMLMs                   !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # configure rocmlm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_rocmlm(self):
        """
        """
        # Get self attributes
        tune = self.tune
        seed = self.seed
        epochs = self.epochs
        nprocs = self.nprocs
        verbose = self.verbose
        batchprop = self.batchprop
        model_prefix = self.model_prefix
        target_train = self.target_train
        model_label = self.ml_model_label
        feature_train = self.feature_train

        # Check for training features
        if feature_train.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_train.size == 0:
            raise Exception("No training targets!")

        # Scale training dataset
        X_train, y_train, scaler_X_train, scaler_y_train, X_scaled_train, y_scaled_train = \
            self._scale_arrays(feature_train, target_train)

        # Define NN layer sizes
        nn_L1, nn_L2, nn_L3 = int(8), int(16), int(32)

        print(f"Configuring model {model_prefix} ...")

        if tune:
            # Define ML model and grid search param space for hyperparameter tuning
            print(f"Tuning model {model_prefix} ...")

            if model_label == "KN":
                model = KNeighborsRegressor()

                param_grid = dict(n_neighbors=[2, 4, 8], weights=["uniform", "distance"])

            elif model_label == "RF":
                model = RandomForestRegressor(random_state=seed)

                param_grid = dict(n_estimators=[400, 800, 1200],
                                  max_features=[1, 2, 3],
                                  min_samples_leaf=[1, 2, 3],
                                  min_samples_split=[2, 4, 6])

            elif model_label == "DT":
                model = DecisionTreeRegressor(random_state=seed)

                param_grid = dict(splitter=["best", "random"],
                                  max_features=[1, 2, 3],
                                  min_samples_leaf=[1, 2, 3],
                                  min_samples_split=[2, 4, 6])

            elif model_label == "NN1":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     batch_size=max(int(len(y_scaled_train) * batchprop), 8))

                param_grid = dict(hidden_layer_sizes=[(nn_L1), (nn_L2), (nn_L3)],
                                  learning_rate_init=[0.0001, 0.0005, 0.001])

            elif model_label == "NN2":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     batch_size=max(int(len(y_scaled_train) * batchprop), 8))

                param_grid = dict(hidden_layer_sizes=[(nn_L2, nn_L2),
                                                      (nn_L3, nn_L2),
                                                      (nn_L3, nn_L3)],
                                  learning_rate_init=[0.0001, 0.0005, 0.001])

            elif model_label == "NN3":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     batch_size=max(int(len(y_scaled_train) * batchprop), 8))

                param_grid = dict(hidden_layer_sizes=[(nn_L3, nn_L2, nn_L2),
                                                      (nn_L3, nn_L3, nn_L2),
                                                      (nn_L3, nn_L3, nn_L3)],
                                  learning_rate_init=[0.0001, 0.0005, 0.001])

            # K-fold cross validation
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)

            # Perform grid search hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf,
                                       scoring="neg_root_mean_squared_error",
                                       n_jobs=nprocs, verbose=verbose)
            grid_search.fit(X_scaled_train, y_scaled_train)

            # Define ML model with tuned hyperparameters
            if model_label == "KN":
                model = KNeighborsRegressor(
                    n_neighbors=grid_search.best_params_["n_neighbors"],
                    weights=grid_search.best_params_["weights"]
                )

            elif model_label == "RF":
                model = RandomForestRegressor(
                    random_state=seed,
                    n_estimators=grid_search.best_params_["n_estimators"],
                    max_features=grid_search.best_params_["max_features"],
                    min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                    min_samples_split=grid_search.best_params_["min_samples_split"]
                )

            elif model_label == "DT":
                model = DecisionTreeRegressor(
                    random_state=seed,
                    splitter=grid_search.best_params_["splitter"],
                    max_features=grid_search.best_params_["max_features"],
                    min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                    min_samples_split=grid_search.best_params_["min_samples_split"]
                )

            elif model_label in ["NN1", "NN2", "NN3"]:
                model = MLPRegressor(
                    random_state=seed,
                    learning_rate_init=grid_search.best_params_["learning_rate_init"],
                    hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"]
                )

            self.ml_model_tuned = True

        else:
            # Define ML models without tuning
            if model_label == "KN":
                model = KNeighborsRegressor(n_neighbors=4, weights="distance")

            elif model_label == "RF":
                model = RandomForestRegressor(random_state=seed, n_estimators=400,
                                              max_features=2, min_samples_leaf=1,
                                              min_samples_split=2)

            elif model_label == "DT":
                model = DecisionTreeRegressor(random_state=seed, splitter="best",
                                              max_features=2, min_samples_leaf=1,
                                              min_samples_split=2)

            elif model_label == "NN1":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     learning_rate_init=0.001,
                                     hidden_layer_sizes=(nn_L3))

            elif model_label == "NN2":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     learning_rate_init=0.0001,
                                     hidden_layer_sizes=(nn_L3, nn_L3))

            elif model_label == "NN3":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     learning_rate_init=0.0001,
                                     hidden_layer_sizes=(nn_L3, nn_L3, nn_L3))

        # Get trained model
        self.ml_model = model

        # Get hyperparameters
        self.model_hyperparams = model.get_params()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # iterate kfold !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _iterate_kfold(self, fold_args):
        """
        """
        # Get self attributes
        epochs = self.epochs
        model = self.ml_model
        verbose = self.verbose
        batchprop = self.batchprop
        target_train = self.target_train
        model_label = self.ml_model_label
        feature_train = self.feature_train

        # Check for training features
        if feature_train.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_train.size == 0:
            raise Exception("No training targets!")

        # Scale training dataset
        X_train, y_train, scaler_X_train, scaler_y_train, X_scaled_train, y_scaled_train = \
            self._scale_arrays(feature_train, target_train)

        # Get fold indices
        (train_index, test_index) = fold_args

        # Split the data into training and testing sets
        X_train, X_test = X_scaled_train[train_index], X_scaled_train[test_index]
        y_train, y_test = y_scaled_train[train_index], y_scaled_train[test_index]

        if "NN" in model_label:
            # Initialize lists to store loss values
            epoch_, train_loss_, test_loss_ = [], [], []

            # Set batch size as a proportion of the training dataset size
            batch_size = int(len(y_train) * batchprop)

            # Ensure a minimum batch size
            batch_size = max(batch_size, 8)

            # Start training timer
            training_start_time = time.time()

            # Partial training
            with tqdm(total=epochs, desc="Training NN", position=0) as pbar:
                for epoch in range(epochs):
                    # Shuffle the training data for each epoch
                    indices = np.arange(len(y_train))
                    np.random.shuffle(indices)

                    for start_idx in range(0, len(indices), batch_size):
                        end_idx = start_idx + batch_size

                        # Ensure that the batch size doesn't exceed the dataset size
                        end_idx = min(end_idx, len(indices))

                        # Subset training data
                        batch_indices = indices[start_idx:end_idx]
                        X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                        # Train NN model on batch
                        model.partial_fit(X_batch, y_batch)

                    # Calculate and store training loss
                    train_loss = model.loss_
                    train_loss_.append(train_loss)

                    # Calculate and store test loss
                    test_loss = mean_squared_error(y_test, model.predict(X_test))
                    test_loss_.append(test_loss)

                    # Store epoch
                    epoch_.append(epoch + 1)

                    # Update progress bar
                    pbar.update(1)

            # End training timer
            training_end_time = time.time()

            # Create loss curve dict
            loss_curve = {"epoch": epoch_, "train_loss": train_loss_,
                          "test_loss": test_loss_}

        else:
            # Start training timer
            training_start_time = time.time()

            # Train ML model
            model.fit(X_train, y_train)

            # End training timer
            training_end_time = time.time()

            # Empty loss curve
            loss_curve = None

        # Calculate training time
        training_time = training_end_time - training_start_time

        # Make predictions on the test dataset
        y_pred_scaled = model.predict(X_test)

        # Test inference time on single random PT datapoint from the test dataset
        rand_PT_point = X_test[np.random.choice(X_test.shape[0], 1, replace=False)]

        inference_start_time = time.time()
        single_PT_pred = model.predict(rand_PT_point)
        inference_end_time = time.time()

        inference_time = inference_end_time - inference_start_time

        # Inverse transform predictions
        y_pred_original = scaler_y_train.inverse_transform(y_pred_scaled)

        # Inverse transform test dataset
        y_test_original = scaler_y_train.inverse_transform(y_test)

        # Calculate performance metrics to evaluate the model
        rmse_test = np.sqrt(mean_squared_error(y_test_original, y_pred_original,
                                               multioutput="raw_values"))

        r2_test = r2_score(y_test_original, y_pred_original, multioutput="raw_values")

        return (loss_curve, rmse_test, r2_test, training_time, inference_time)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process_kfold_results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_kfold_results(self, results):
        """
        """
        # Get self attributes
        sids = self.sids
        kfolds = self.kfolds
        verbose = self.verbose
        M = self.shape_feature_square[0]
        w = self.shape_feature_square[1]
        model_label = self.ml_model_label
        training_targets = self.training_targets
        model_label_full = self.ml_model_label_full

        # Initialize empty lists for storing performance metrics
        loss_curves = []
        rmse_test_scores, r2_test_scores = [], []
        training_times, inference_times = [], []

        # Unpack results
        for (loss_curve, rmse_test, r2_test, training_time, inference_time) in results:
            loss_curves.append(loss_curve)
            rmse_test_scores.append(rmse_test)
            r2_test_scores.append(r2_test)
            training_times.append(training_time)
            inference_times.append(inference_time)

        if "NN" in model_label:
            self._visualize_loss_curve(loss_curves)

        # Stack arrays
        rmse_test_scores = np.stack(rmse_test_scores)
        r2_test_scores = np.stack(r2_test_scores)

        # Calculate performance values with uncertainties
        rmse_test_mean = np.mean(rmse_test_scores, axis=0)
        rmse_test_std = np.std(rmse_test_scores, axis=0)
        r2_test_mean = np.mean(r2_test_scores, axis=0)
        r2_test_std = np.std(r2_test_scores, axis=0)
        training_time_mean = np.mean(training_times)
        training_time_std = np.std(training_times)
        inference_time_mean = np.mean(inference_times)
        inference_time_std = np.std(inference_times)

        # Get sample label
        if any(sample in sids for sample in ["PUM", "DMM", "PYR"]):
            sample_label = "benchmark"
        elif any("sm" in sample or "sr" in sample for sample in sids):
            sample_label = f"SYNTH{M}"
        elif any("st" in sample for sample in sids):
            sample_label = f"SMAT{M}"
        elif any("sm" in sample for sample in sids):
            sample_label = f"SMAM{M}"
        elif any("sb" in sample for sample in sids):
            sample_label = f"SMAB{M}"
        elif any("sr" in sample for sample in sids):
            sample_label = f"SMAR{M}"

        # Config and performance info
        cv_info = {
            "model": [model_label],
            "sample": [sample_label],
            "size": [(w - 1) ** 2],
            "n_targets": [len(training_targets)],
            "k_folds": [kfolds],
            "training_time_mean": [round(training_time_mean, 5)],
            "training_time_std": [round(training_time_std, 5)],
            "inference_time_mean": [round(inference_time_mean, 5)],
            "inference_time_std": [round(inference_time_std, 5)]
        }

        # Add performance metrics for each parameter to the dictionary
        for i, target in enumerate(training_targets):
            cv_info[f"rmse_test_mean_{target}"] = round(rmse_test_mean[i], 5)
            cv_info[f"rmse_test_std_{target}"] = round(rmse_test_std[i], 5)
            cv_info[f"r2_test_mean_{target}"] = round(r2_test_mean[i], 5)
            cv_info[f"r2_test_std_{target}"] = round(r2_test_std[i], 5)

        if verbose >= 1:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # Print performance
            print(f"{model_label_full} performance:")
            print(f"    training time: {training_time_mean:.5f} ± {training_time_std:.5f}")
            print(f"    inference time: {inference_time_mean:.5f} ± "
                  f"{inference_time_std:.5f}")
            print(f"    rmse test:")
            for r, e, p in zip(rmse_test_mean, rmse_test_std, training_targets):
                print(f"        {p}: {r:.5f} ± {e:.5f}")
            print(f"    r2 test:")
            for r, e, p in zip(r2_test_mean, r2_test_std, training_targets):
                print(f"        {p}: {r:.5f} ± {e:.5f}")
            print(f"    rmse test:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")

        self.cv_info = cv_info

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # kfold cv !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _kfold_cv(self):
        """
        """
        # Get self attributes
        seed = self.seed
        kfolds = self.kfolds
        nprocs = self.nprocs
        target_train = self.target_train
        feature_train = self.feature_train

        # Check for training features
        if feature_train.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_train.size == 0:
            raise Exception("No training targets!")

        # Scale training dataset
        X_train, y_train, scaler_X_train, scaler_y_train, X_scaled_train, y_scaled_train = \
            self._scale_arrays(feature_train, target_train)

        # Check for ml model
        if self.ml_model is None:
            raise Exception("No ML model! Call _configure_rocmlm() first ...")

        # K-fold cross validation
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

        # Create list of args for mp pooling
        fold_args = [(train_index, test_index) for _, (train_index, test_index) in
                     enumerate(kf.split(X_train))]

        # Create a multiprocessing pool
        with mp.Pool(processes=nprocs) as pool:
            results = pool.map(self._iterate_kfold, fold_args)

            # Wait for all processes
            pool.close()
            pool.join()

        self.ml_model_cross_validated = True

        # Process cross validation results
        self._process_kfold_results(results)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # retrain !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _retrain(self):
        """
        """
        # Get self attributes
        seed = self.seed
        epochs = self.epochs
        model = self.ml_model
        targets = self.targets
        verbose = self.verbose
        batchprop = self.batchprop
        model_prefix = self.model_prefix
        model_label = self.ml_model_label
        model_out_dir = self.model_out_dir
        target_array = self.target_train.copy()
        feature_array = self.feature_train.copy()
        shape_target_square = self.shape_target_square
        shape_feature_square = self.shape_feature_square

        # Check for ml model
        if self.ml_model is None:
            raise Exception("No ML model! Call _configure_rocmlm() first ...")

        # Check for ml model cross validated
        if not self.ml_model_cross_validated:
            raise Exception("ML model not cross validated! Call _kfold_cv() first ...")

        # Check for training features
        if feature_array.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_array.size == 0:
            raise Exception("No training targets!")

        print(f"Retraining model {model_prefix} ...")

        # Scale arrays
        X, y, scaler_X, scaler_y, X_scaled, y_scaled = \
            self._scale_arrays(feature_array, target_array)

        # Train model on entire training dataset
        X_train, X_test, y_train, y_test = \
            train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)

        # Train ML model
        if "NN" in model_label:
            # Set batch size as a proportion of the training dataset size
            batch_size = int(len(y_train) * batchprop)

            # Ensure a minimum batch size
            batch_size = max(batch_size, 8)

            # Partial training
            with tqdm(total=epochs, desc="Retraining NN", position=0) as pbar:
                for epoch in range(epochs):
                    # Shuffle the training data for each epoch
                    indices = np.arange(len(y_train))
                    np.random.shuffle(indices)

                    for start_idx in range(0, len(indices), batch_size):
                        end_idx = start_idx + batch_size

                        # Ensure that the batch size doesn't exceed the dataset size
                        end_idx = min(end_idx, len(indices))

                        # Subset training data
                        batch_indices = indices[start_idx:end_idx]
                        X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                        # Train NN model on batch
                        model.partial_fit(X_batch, y_batch)

                    # Update progress bar
                    pbar.update(1)

        else:
            model.fit(X_train, y_train)

        print(":::::::::::::::::::::::::::::::::::::::::::::")
        self.ml_model_only = model
        self.ml_model_trained = True
        self.ml_model_scaler_X = scaler_X
        self.ml_model_scaler_y = scaler_y

        # Copy feature and target arrays
        X = feature_array.copy()
        y = target_array.copy()

        # Scale features array
        X_scaled = scaler_X.transform(X)

        # Make predictions on features
        pred_scaled = model.predict(X_scaled)

        # Inverse transform predictions
        pred_original = scaler_y.inverse_transform(pred_scaled)

        # Reshape arrays into squares for visualization
        target_square = y.reshape(shape_target_square)
        feature_square = X.reshape(shape_feature_square)
        pred_square = pred_original.reshape(shape_target_square)

        # Update arrays
        self.target_square = target_square
        self.prediction_square = pred_square
        self.feature_square = feature_square

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # append to csv !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _append_to_csv(self):
        """
        """
        # Get self attributes
        data_dict = self.cv_info

        # Check for cross validation results
        if not self.cv_info:
            raise Exception("No cross validation! Call _kfold_cv() first ...")

        # CSV filepath
        filepath = f"assets/data/rocmlm-performance.csv"

        # Check if the CSV file already exists
        if not pd.io.common.file_exists(filepath):
            df = pd.DataFrame(data_dict)
        else:
            df = pd.read_csv(filepath)
            new_data = pd.DataFrame(data_dict)
            df = pd.concat([df, new_data], ignore_index=True)

        df = df.sort_values(by=["model", "sample", "size"])
        df.to_csv(filepath, index=False)

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.4.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check model array images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_model_array_images(self, type="targets"):
        """
        """
        # Get ml model attributes
        sids = self.sids
        fig_dir = self.fig_dir
        model_prefix = self.model_prefix
        training_targets = self.training_targets

        # Check for existing plots
        existing_figs = []
        for sid in sids:
            for i, target in enumerate(training_targets):
                target_rename = target.replace("_", "-")
                if type == "targets":
                    path = f"{fig_dir}/{model_prefix}-{sid}-{target_rename}-targets.png"
                elif type == "predictions":
                    path = f"{fig_dir}/{model_prefix}-{sid}-{target_rename}-prediction.png"
                elif type == "diff":
                    path = f"{fig_dir}/{model_prefix}-{sid}-{target_rename}-diff.png"
                else:
                    raise ValueError("Unrecognized array image type!")
                check = os.path.exists(path)
                if check:
                    existing_figs.append(check)

        if len(existing_figs) == len(training_targets) * len(sids):
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
        sids = self.sids
        fig_dir = self.fig_dir
        training_targets = self.training_targets

        # Filter targets for PREM
        t_ind = [i for i, t in enumerate(training_targets) if t in ["rho", "Vp", "Vs"]]
        targets = [training_targets[i] for i in t_ind]

        # Check for existing plots
        existing_figs = []
        for sid in sids:
            for i, target in enumerate(training_targets):
                path = f"{fig_dir}/{sid}-{target}-prem.png"
                check = os.path.exists(path)
                if check:
                    existing_figs.append(check)

        if len(existing_figs) == len(training_targets) * len(sids):
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize loss curve !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_loss_curve(self, loss_curves, figwidth=6.3, figheight=3.54, fontsize=14):
        """
        """
        # Get self attributes
        fig_dir = self.fig_dir
        model_label = self.ml_model_label
        model_label_full = self.ml_model_label_full

        # Initialize empty dict for combined loss curves
        merged_curves = {}

        # Merge loss curves
        for curve in loss_curves:
            for key, value in curve.items():
                if key in merged_curves:
                    if isinstance(merged_curves[key], list):
                        merged_curves[key].extend(value)
                    else:
                        merged_curves[key] = [merged_curves[key], value]
                        merged_curves[key].extend(value)
                else:
                    merged_curves[key] = value

        # Make dict into pandas df
        df = pd.DataFrame.from_dict(merged_curves, orient="index").transpose()
        df.sort_values(by="epoch", inplace=True)

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

        # Plot loss curve
        fig = plt.figure(figsize=(figwidth, figheight))

        # Colormap
        colormap = plt.get_cmap("tab10")

        plt.plot(df["epoch"], df["train_loss"], label="train loss", color=colormap(0))
        plt.plot(df["epoch"], df["test_loss"], label="test loss", color=colormap(1))
        plt.xlabel("Epoch")
        plt.ylabel(f"Loss")

        plt.title(f"{model_label_full} Loss Curve")
        plt.legend()

        # Save the plot to a file
        os.makedirs(f"{fig_dir}", exist_ok=True)
        plt.savefig(f"{fig_dir}/{model_label}-loss-curve.png")

        # Close plot
        plt.close()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize array image  !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_image(self, palette="bone", geotherms=True, type="targets",
                               figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get ml model attributes
        sids = self.sids
        cv_info = self.cv_info
        fig_dir = self.fig_dir
        verbose = self.verbose
        target_units = self.target_units
        model_prefix = self.model_prefix
        target_arrays = self.target_square
        feature_arrays = self.feature_square
        pred_arrays = self.prediction_square
        training_targets = self.training_targets
        ml_model_trained = self.ml_model_trained

        # Get number of training features
        n_feats = feature_arrays.shape[-1] - 2

        # Check for model
        if not ml_model_trained:
            raise Exception("No RocMLM model! Call train_rocmlm() first ...")

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
        plt.rcParams["figure.constrained_layout.use"] = "True"

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        try:
            for s, sid in enumerate(sids):
                if verbose >= 1: print(f"Visualizing {model_prefix}-{sid} ...")

                # Slice arrays
                feature_array = feature_arrays[s, :, :, :]
                target_array = target_arrays[s, :, :, :]
                pred_array = pred_arrays[s, :, :, :]

                for i, target in enumerate(training_targets):
                    # Rename target
                    target_rename = target.replace("_", "-")

                    # Get 2d arrays
                    P = feature_array[:, :, 0 + n_feats]
                    T = feature_array[:, :, 1 + n_feats]
                    t = target_array[:, :, i]
                    p = pred_array[:, :, i]
                    extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

                    if type == "targets":
                        square_array = t
                        filename = f"{model_prefix}-{sid}-{target_rename}-targets.png"
                        rmse, r2 = None, None
                    elif type == "predictions":
                        square_array = p
                        filename = f"{model_prefix}-{sid}-{target_rename}-predictions.png"
                        rmse, r2 = None, None
                    elif type == "diff":
                        mask = np.isnan(t)
                        p[mask] = np.nan
                        square_array = t - p
                        square_array[mask] = np.nan
                        rmse = cv_info[f"rmse_test_mean_{target}"]
                        r2 = cv_info[f"r2_test_mean_{target}"]
                        palette = "seismic"
                        filename = f"{model_prefix}-{sid}-{target_rename}-diff.png"
                    else:
                        raise ValueError("Unrecognized array image type!")

                    # Target labels
                    if target == "rho":
                        target_label = "Density"
                    elif target == "h2o":
                        target_label = "H$_2$O"
                    elif target == "melt":
                        target_label = "Melt"
                    else:
                        target_label = target

                    # Set title
                    if target not in ["assemblage", "variance"]:
                        title = f"{target_label} ({target_units[i]})"
                    else:
                        title = f"{target_label}"

                    # Check for all nans
                    if np.all(np.isnan(square_array)):
                        square_array = np.nan_to_num(square_array)

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
                        non_nan_values = square_array[np.logical_not(np.isnan(square_array))]
                        if non_nan_values.size > 0:
                            vmin = np.min(non_nan_values)
                            vmax = np.max(non_nan_values)
                        else:
                            vmin = 0
                            vmax = 0
                    else:
                        vmin = int(np.nanmin(np.unique(square_array)))
                        vmax = int(np.nanmax(np.unique(square_array)))

                    # Make results df
                    results = pd.DataFrame({"P": P.flatten(), "T": T.flatten(),
                                            target: square_array.flatten()})

                    # Get geotherm
                    P_geotherm, T_geotherm, _ = self._get_1d_profile(target, results, 1173)
                    P_geotherm2, T_geotherm2, _ = self._get_1d_profile(target, results, 1573)
                    P_geotherm3, T_geotherm3, _ = self._get_1d_profile(target, results, 1773)

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

                        im = ax.imshow(square_array, extent=extent, aspect="auto", cmap=cmap,
                                       origin="lower", vmin=vmin, vmax=vmax)
                        if geotherms:
                            ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white",
                                    linewidth=3)
                            ax.plot(T_geotherm2, P_geotherm2, linestyle="--", color="white",
                                    linewidth=3)
                            ax.plot(T_geotherm3, P_geotherm3, linestyle="-.", color="white",
                                    linewidth=3)
                            plt.text(1163 + (6 * 0.5 * 35), 6, "1173 K",
                                     fontsize=fontsize * 0.833,
                                     horizontalalignment="center",
                                     verticalalignment="bottom",
                                     rotation=67, color="white")
                            plt.text(1563 + (6 * 0.5 * 35), 6, "1573 K",
                                     fontsize=fontsize * 0.833,
                                     horizontalalignment="center",
                                     verticalalignment="bottom",
                                     rotation=67, color="white")
                            plt.text(1763 + (6 * 0.5 * 35), 6, "1773 K",
                                     fontsize=fontsize * 0.833,
                                     horizontalalignment="center",
                                     verticalalignment="bottom",
                                     rotation=67, color="white")
                        ax.set_xlabel("T (K)")
                        ax.set_ylabel("P (GPa)")
                        plt.colorbar(im, ax=ax, label="",
                                     ticks=np.arange(vmin, vmax, num_colors))

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
                                np.abs(square_array[np.logical_not(np.isnan(square_array))]))
                            vmax = np.max(
                                np.abs(square_array[np.logical_not(np.isnan(square_array))]))
                        else:
                            vmin, vmax = vmin, vmax

                            # Adjust vmin close to zero
                            if vmin <= 1e-4: vmin = 0

                            # Set melt fraction to 0–100 vol.%
                            if target == "melt": vmin, vmax = 0, 100

                            # Set h2o fraction to 0–100 wt.%
                            if target == "h2o": vmin, vmax = 0, 5

                        # Set nan color
                        cmap = plt.colormaps[cmap]
                        cmap.set_bad(color="white")

                        # Plot as a raster using imshow
                        fig, ax = plt.subplots()

                        im = ax.imshow(square_array, extent=extent, aspect="auto", cmap=cmap,
                                       origin="lower", vmin=vmin, vmax=vmax)
                        if geotherms:
                            ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white",
                                    linewidth=3)
                            ax.plot(T_geotherm2, P_geotherm2, linestyle="--", color="white",
                                    linewidth=3)
                            ax.plot(T_geotherm3, P_geotherm3, linestyle="-.", color="white",
                                    linewidth=3)
                            plt.text(1163 + (6 * 0.5 * 35), 6, "1173 K",
                                     fontsize=fontsize * 0.833,
                                     horizontalalignment="center",
                                     verticalalignment="bottom",
                                     rotation=67, color="white")
                            plt.text(1563 + (6 * 0.5 * 35), 6, "1573 K",
                                     fontsize=fontsize * 0.833,
                                     horizontalalignment="center",
                                     verticalalignment="bottom",
                                     rotation=67, color="white")
                            plt.text(1763 + (6 * 0.5 * 35), 6, "1773 K",
                                     fontsize=fontsize * 0.833,
                                     horizontalalignment="center",
                                     verticalalignment="bottom",
                                     rotation=67, color="white")
                        ax.set_xlabel("T (K)")
                        ax.set_ylabel("P (GPa)")

                        # Diverging colorbar
                        if palette == "seismic":
                            cbar = plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], label="")

                        # Continuous colorbar
                        else:
                            cbar = plt.colorbar(im, ax=ax, label="",
                                                ticks=np.linspace(vmin, vmax, num=4))

                        # Set colorbar limits and number formatting
                        if target == "rho":
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

                    # Add rmse and r2
                    if rmse is not None and r2 is not None:
                        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white",
                                          lw=1.5, alpha=0.3)
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

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _visualize_array_image() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize prem !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_prem(self, type="targets", geotherms=["low", "mid", "high"],
                        figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get model data
        res = self.res
        sids = self.sids
        P_min = self.P_min
        P_max = self.P_max
        fig_dir = self.fig_dir
        verbose = self.verbose
        data_dir = self.data_dir
        geothresh = self.geothresh
        target_units = self.target_units
        model_prefix = self.model_prefix
        target_arrays = self.target_square
        feature_arrays = self.feature_square
        pred_arrays = self.prediction_square
        training_targets = self.training_targets
        ml_model_trained = self.ml_model_trained

        # Get number of training features
        n_feats = feature_arrays.shape[-1] - 2

        # Check for model
        if not ml_model_trained:
            raise Exception("No RocMLM model! Call train_rocmlm() first ...")

        # Check for data dir
        if not os.path.exists(data_dir):
            raise Exception(f"Data not found at {data_dir} !")

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Check for average geotherm
        if "mid" not in geotherms:
            geotherms = geotherms + ["mid"]

        # Get 1D reference models
        ref_models = self._get_1d_reference_models()

        # Get synthetic endmember compositions
        sids_end = ["sm000-loi000", f"sm{str(res).zfill(3)}-loi000"]
        df_mids = pd.read_csv("assets/data/synth-mids.csv")
        df_synth_bench = df_mids[df_mids["SAMPLEID"].isin(sids_end) & (df_mids["LOI"] == 0)]

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

        try:
            for s, sid in enumerate(sids):
                if verbose >= 1: print(f"Visualizing {model_prefix}-{sid} ...")

                # Slice arrays
                feature_array = feature_arrays[s, :, :, :]
                target_array = target_arrays[s, :, :, :]
                pred_array = pred_arrays[s, :, :, :]

                for i, target in enumerate(training_targets):
                    # Rename target
                    target_rename = target.replace("_", "-")

                    # Get 2d arrays
                    P = feature_array[:, :, 0 + n_feats]
                    T = feature_array[:, :, 1 + n_feats]
                    t = target_array[:, :, i]
                    p = pred_array[:, :, i]

                    if type == "targets":
                        square_array = t
                    elif type == "predictions":
                        square_array = p
                    else:
                        raise ValueError("Unrecognized array image type!")

                    # Check for all nans
                    if np.all(np.isnan(square_array)):
                        square_array = np.nan_to_num(square_array)

                    filename = f"{model_prefix}-{sid}-{target_rename}-prem.png"

                    if target in ["rho", "Vp", "Vs"]:
                        # Get 1D reference model profiles
                        P_prem = ref_models["prem"]["P"]
                        target_prem = ref_models["prem"][target]
                        P_stw105, target_stw105 = (ref_models["stw105"]["P"],
                                                   ref_models["stw105"][target])

                    # Make results df
                    results = pd.DataFrame({"P": P.flatten(), "T": T.flatten(),
                                            target: square_array.flatten()})

                    # Process GFEM model profiles
                    if "low" in geotherms:
                        P_gfem, _, target_gfem = (
                            self._get_1d_profile(target, results, 1173))
                        if target in ["rho", "Vp", "Vs"]:
                            P_gfem, target_gfem, _, _, _, _ = self._crop_1d_profile(
                                P_gfem, target_gfem, P_prem, target_prem)
                    if "mid" in geotherms:
                        P_gfem2, _, target_gfem2 = (
                            self._get_1d_profile(target, results, 1573))
                        if target in ["rho", "Vp", "Vs"]:
                            P_gfem2, target_gfem2, _, _, rmse, r2 = self._crop_1d_profile(
                                P_gfem2, target_gfem2, P_prem, target_prem)
                    if "high" in geotherms:
                        P_gfem3, _, target_gfem3 = (
                            self._get_1d_profile(target, results, 1773))
                        if target in ["rho", "Vp", "Vs"]:
                            P_gfem3, target_gfem3, _, _, _, _ = self._crop_1d_profile(
                                P_gfem3, target_gfem3, P_prem, target_prem)

                    if target in ["rho", "Vp", "Vs"]:
                        _, _, P_prem, target_prem, _, _ = self._crop_1d_profile(
                            P_gfem2, target_gfem2, P_prem, target_prem)
                        _, _, P_stw105, target_stw105, _, _ = self._crop_1d_profile(
                            P_gfem2, target_gfem2, P_stw105, target_stw105)

                    # Change endmember sampleids
                    if sid == tend:
                        sid_lab = "DSUM"
                    elif sid == bend:
                        sid_lab = "PSUM"
                    else:
                        sid_lab = sid

                    # Colormap
                    colormap = plt.colormaps["tab10"]

                    # Plotting
                    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

                    # Plot GFEM model profiles
                    if "low" in geotherms:
                        ax1.plot(target_gfem, P_gfem, "-", linewidth=3, color=colormap(0),
                                 label=f"1173 K")
                        ax1.fill_betweenx(
                            P_gfem, target_gfem * (1 - 0.05), target_gfem * (1 + 0.05),
                            color=colormap(0), alpha=0.2)
                    if "mid" in geotherms:
                        ax1.plot(target_gfem2, P_gfem2, "-", linewidth=3, color=colormap(2),
                                 label=f"1573 K")
                        ax1.fill_betweenx(
                            P_gfem2, target_gfem2 * (1 - 0.05), target_gfem2 * (1 + 0.05),
                            color=colormap(2), alpha=0.2)
                    if "high" in geotherms:
                        ax1.plot(target_gfem3, P_gfem3, "-", linewidth=3, color=colormap(1),
                                 label=f"1773 K")
                        ax1.fill_betweenx(
                            P_gfem3, target_gfem3 * (1 - 0.05), target_gfem3 * (1 + 0.05),
                            color=colormap(1), alpha=0.3)

                    # Plot reference models
                    if target in ["rho", "Vp", "Vs"]:
                        ax1.plot(target_prem, P_prem, "-", linewidth=2, color="black")
                        ax1.plot(target_stw105, P_stw105, ":", linewidth=2, color="black")

                    if target == "rho":
                        target_label = "Density"
                    elif target == "h2o":
                        target_label = "H$_2$O"
                    elif target == "melt":
                        target_label = "Melt"
                    else:
                        target_label = target

                    if target not in ["assemblage", "variance"]:
                        ax1.set_xlabel(f"{target_label} ({target_units[i]})")
                    else:
                        ax1.set_xlabel(f"{target_label}")
                    ax1.set_ylabel("P (GPa)")
                    if target == "h2o":
                        ax1.set_xlim(None, 5)
                    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
                    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

                    # Vertical text spacing
                    text_margin_x = 0.04
                    text_margin_y = 0.15
                    text_spacing_y = 0.1

                    # Add R-squared and RMSE values as text annotations in the plot
                    if target in ["rho", "Vp", "Vs"]:
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
                    depth_values = depth_conversion(np.linspace(P_min, P_max, len(P_gfem)))

                    # Create the secondary y-axis and plot depth on it
                    ax2 = ax1.secondary_yaxis(
                        "right", functions=(depth_conversion, depth_conversion))
                    ax2.set_yticks([410, 670])
                    ax2.set_ylabel("Depth (km)")

                    plt.legend(loc="lower right", columnspacing=0, handletextpad=0.2,
                               fontsize=fontsize * 0.833)

                    plt.title("Depth Profile")

                    # Save the plot to a file
                    plt.savefig(f"{fig_dir}/{filename}")

                    # Close device
                    plt.close()
                    print(f"  Figure saved to: {fig_dir}/{filename} ...")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _visualize_prem() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_model(self):
        """
        """
        try:
            if not self._check_model_array_images(type="targets"):
                self._visualize_array_image(type="targets")
            if not self._check_model_array_images(type="predictions"):
                self._visualize_array_image(type="predictions")
            if not self._check_model_array_images(type="diff"):
                self._visualize_array_image(type="diff")
            if not self._check_model_prem_images():
                self._visualize_prem()
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in visualize_model() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.5.             Train RocMLMs               !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # train rocmlm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def train_rocmlm(self):
        """
        """
        max_retries = 3
        for retry in range(max_retries):
            # Check for pretrained model
            if self.ml_model_trained:
                break
            try:
                self._configure_rocmlm()
                self._print_rocmlm_info()
                self._kfold_cv()
                self._retrain()
                self._evaluate_lut_performance()
                with open(self.ml_model_only_path, "wb") as file:
                    joblib.dump(self.ml_model_only, file)
                with open(self.ml_model_scaler_X_path, "wb") as file:
                    joblib.dump(self.ml_model_scaler_X, file)
                with open(self.ml_model_scaler_y_path, "wb") as file:
                    joblib.dump(self.ml_model_scaler_y, file)
                model_size = os.path.getsize(self.ml_model_only_path)
                model_size_mb = round(model_size / (1024 ** 2), 5)
                self.cv_info["model_size_mb"] = model_size_mb
                self._append_to_csv()

            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"!!! ERROR in train_rocmlm() !!!")
                print(f"{e}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                traceback.print_exc()
                if retry < max_retries - 1:
                    print(f"Retrying in 5 seconds ...")
                    time.sleep(5)
                else:
                    self.ml_model_training_error = True
                    self.ml_model_error = e

                    return None
        try:
            self.visualize_model()
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in train_rocmlm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

#######################################################
## .2.                Visualize                  !!! ##
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
# compose rocmlm plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocmlm_plots(rocmlm):
    """
    """
    # Get ml model attributes
    res = rocmlm.res
    sids = rocmlm.sids
    fig_dir = rocmlm.fig_dir
    verbose = rocmlm.verbose
    fig_dir_perf = "figs/rocmlm"
    model_prefix = rocmlm.model_prefix
    ml_model_label = rocmlm.ml_model_label
    training_targets = rocmlm.training_targets

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in training_targets]

    # Check for existing plots
    existing_figs = []
    for target in targets_rename:
        for sid in sids:
            fig_1 = f"{fig_dir}/image-{sid}-{ml_model_label}-{target}-diff.png"
            fig_2 = f"{fig_dir}/image-{sid}-{ml_model_label}-{target}-profile.png"
            fig_3 = f"{fig_dir}/image9-{sid}-{ml_model_label}-diff.png"
            fig_4 = f"{fig_dir}/image9-{sid}-{ml_model_label}-profile.png"
            check = (os.path.exists(fig_1) and os.path.exists(fig_2) and
                     os.path.exists(fig_3) and os.path.exists(fig_4))
            if check: existing_figs.append(check)
    if existing_figs: return None

    for sid in sids:
        for target in targets_rename:
            if verbose >= 1:
                print(f"Composing {model_prefix}-{sid}-{target} ...")

            combine_plots_horizontally(
                f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png",
                f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png",
                f"{fig_dir}/image-{sid}-{ml_model_label}-{target}-diff.png",
                caption1="",
                caption2="c)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png",
                f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png",
                f"{fig_dir}/image-{sid}-{ml_model_label}-{target}-profile.png",
                caption1="",
                caption2="c)"
            )

            if all(item in targets_rename for item in ["rho", "melt", "h2o"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "melt", "h2o"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png",
                        f"{fig_dir}/temp-{target}.png",
                        caption1="",
                        caption2=captions[i][2]
                    )

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
                    f"{fig_dir}/image9-{sid}-{ml_model_label}-diff.png",
                    caption1="",
                    caption2=""
                )

            if all(item in targets_rename for item in ["rho", "melt", "h2o"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "melt", "h2o"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png",
                        f"{fig_dir}/temp-{target}.png",
                        caption1="",
                        caption2=captions[i][2]
                    )

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
                    f"{fig_dir}/image9-{sid}-{ml_model_label}-profile.png",
                    caption1="",
                    caption2=""
                )

    # Clean up directory
    rocmlm_files = glob.glob(f"{fig_dir}/{model_prefix}*.png")
    tmp_files = glob.glob(f"{fig_dir}/temp*.png")

    for file in rocmlm_files + tmp_files:
        os.remove(file)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocmlm performance !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocmlm_performance(fig_dir="figs/other", filename="rocmlm-performance.png",
                                 figwidth=6.3, figheight=2.5, fontsize=12):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read gfem efficiency data
    data_gfem = pd.read_csv(f"{data_dir}/gfem-efficiency.csv")
    data_gfem["time"] = data_gfem["time"] / data_gfem["size"]

    # Get Perple_X program size
    def get_directory_size(path="."):
        total_size = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total_size += entry.stat().st_size
                elif entry.is_dir():
                    total_size += get_directory_size(entry.path)
        return total_size
    perplex_size = get_directory_size("Perple_X")

    # Calculate efficiency in milliseconds/Megabyte
    data_gfem["model_size_mb"] = round(perplex_size / (1024 ** 2), 5)
    data_gfem["model_efficiency"] = data_gfem["time"] * 1e3 * data_gfem["model_size_mb"]
    data_gfem = data_gfem[data_gfem["program"] == "perplex"]

    # Read lookup table efficiency data
    data_lut = pd.read_csv(f"{data_dir}/lut-efficiency.csv")

    # Add RMSE
    data_lut["rmse_val_mean_rho"] = 0
    data_lut["rmse_val_mean_Vp"] = 0
    data_lut["rmse_val_mean_Vs"] = 0

    # Calculate efficiency in milliseconds/Megabyte
    data_lut["model_efficiency"] = data_lut["time"] * 1e3 * data_lut["model_size_mb"]

    # Read rocmlm efficiency data
    data_rocmlm = pd.read_csv(f"{data_dir}/rocmlm-performance.csv")

    # Process rocmlm df for merging
    data_rocmlm.drop([col for col in data_rocmlm.columns if "training" in col],
                     axis=1, inplace=True)
    data_rocmlm.drop([col for col in data_rocmlm.columns if "rmse_test" in col],
                     axis=1, inplace=True)
    data_rocmlm.drop([col for col in data_rocmlm.columns if "rmse_val_std" in col],
                     axis=1, inplace=True)
    data_rocmlm.drop([col for col in data_rocmlm.columns if "r2" in col],
                     axis=1, inplace=True)

    # Combine model with rocmlm
    def label_rocmlm_model(row):
        return f"RocMLM ({row["model"]})"

    data_rocmlm["program"] = data_rocmlm.apply(label_rocmlm_model, axis=1)
    data_rocmlm.drop(["n_targets", "k_folds", "inference_time_std"], axis=1, inplace=True)
    data_rocmlm.rename(columns={"inference_time_mean": "time"}, inplace=True)

    # Calculate efficiency in milliseconds/Megabyte
    data_rocmlm["model_efficiency"] = (data_rocmlm["time"] * 1e3 *
                                       data_rocmlm["model_size_mb"])

    # Select columns
    data_rocmlm = data_rocmlm[["sample", "program", "size", "time",
                               "model_size_mb", "model_efficiency", "rmse_val_mean_rho",
                               "rmse_val_mean_Vp", "rmse_val_mean_Vs"]]

    # Combine data
    data = pd.concat([data_lut, data_rocmlm], axis=0, ignore_index=True)

    # Relabel programs
    def label_programs(row):
        if row["program"] == "perplex":
            return "Perple_X"
        elif row["program"] == "lut":
            return "Lookup Table"
        else:
            return row["program"]

    data["program"] = data.apply(label_programs, axis=1)

    # Filter samples and programs
    data = data[data["sample"].isin(["SYNTH129", "SYNTH65", "SYNTH33", "SYNTH11"])]
    data = data[data["program"].isin(["Lookup Table", "RocMLM (DT)", "RocMLM (KN)",
                                      "RocMLM (NN1)", "RocMLM (NN3)"])]

    # Get X resolution
    def get_x_res(row):
        if row["sample"].startswith("SYNTH") and row["sample"][5:].isdigit():
            x_res = int(row["sample"][5:])
            if x_res < 257:
                return int(x_res - 1)
            else:
                return int(x_res - 2)
        else:
            return None

    data["x_res"] = data.apply(get_x_res, axis=1)
    data["size"] = np.log2(data["size"] * data["x_res"]).astype(int)

    # Arrange data by resolution and sample
    data.sort_values(by=["size", "sample", "program"], inplace=True)

    # Group by size and select min time
    grouped_data = data.groupby(["program", "size"])
    min_time_indices = grouped_data["time"].idxmin()
    data = data.loc[min_time_indices]
    data.reset_index(drop=True, inplace=True)

    # Compute summary statistics
    summary_stats = data.groupby(["program"]).agg({
        "time": ["mean", "std", "min", "max"],
        "rmse_val_mean_rho": ["mean", "std", "min", "max"],
        "rmse_val_mean_Vp": ["mean", "std", "min", "max"],
        "rmse_val_mean_Vs": ["mean", "std", "min", "max"],
        "model_efficiency": ["mean", "std", "min", "max"]
    })
    summary_stats_gfem = data_gfem.agg({
        "time": ["mean", "std", "min", "max"],
        "model_efficiency": ["mean", "std", "min", "max"]
    })

    # Print summary statistics
    print("Inference time:")
    print(summary_stats["time"] * 1e3)
    print("....................................")
    print("Inference time GFEM:")
    print(summary_stats_gfem["time"] * 1e3)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Model efficiency:")
    print(summary_stats["model_efficiency"])
    print("....................................")
    print("Model efficiency GFEM:")
    print(summary_stats_gfem["model_efficiency"])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

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

    # Define marker types for each program
    marker_dict = {"Lookup Table": "o", "RocMLM (DT)": "d", "RocMLM (KN)": "s",
                   "RocMLM (NN1)": "X", "RocMLM (NN3)": "^"}

    # Create a colormap
    palette = sns.color_palette("mako_r", data["rmse_val_mean_rho"].nunique())
    cmap = plt.get_cmap("mako_r")

    # Create a ScalarMappable to map rmse to colors
    norm = Normalize(data["rmse_val_mean_rho"].min(), data["rmse_val_mean_rho"].max())
    sm = plt.ScalarMappable(cmap=cmap, norm=norm)

    # Map X resolution to colors
    color_dict = dict(zip(sorted(data["rmse_val_mean_rho"].unique()),
                          cmap(norm(sorted(data["rmse_val_mean_rho"].unique())))))

    fig = plt.figure(figsize=(figwidth * 2, figheight))
    ax = fig.add_subplot(121)

    # Plot gfem efficiency
    ax.fill_between(data["size"], data_gfem["time"].min() * 1e3,
                    data_gfem["time"].max() * 1e3, facecolor="white", edgecolor="black")

    plt.text(data["size"].min(), data_gfem["time"].min() * 1e3, "Perple_X",
             fontsize=fontsize * 0.833, horizontalalignment="left",
             verticalalignment="bottom")
    plt.text(data["size"].min(), data_gfem["time"].min() * 1e3 * 1.2,
             " [stx21, NCFMAS, 15]", fontsize=fontsize * 0.833,
             horizontalalignment="left", verticalalignment="top")
    plt.text(data["size"].min(), data_gfem["time"].max() * 1e3 * 0.80,
             " [hp633, KNCFMASTCr, 21]", fontsize=fontsize * 0.833,
             horizontalalignment="left", verticalalignment="bottom")

    # Plot rocmlm efficiency
    for program, group in data.groupby("program"):
        for rmse, sub_group in group.groupby("rmse_val_mean_rho"):
            if program in ["RocMLM (DT)", "RocMLM (KN)", "RocMLM (NN1)", "RocMLM (NN3)"]:
                ax.scatter(x=sub_group["size"], y=(sub_group["time"] * 1e3),
                           marker=marker_dict.get(program, "o"), s=65,
                           color=color_dict[rmse], edgecolor="black", zorder=2)
            else:
                ax.scatter(x=sub_group["size"], y=(sub_group["time"] * 1e3),
                           marker=marker_dict.get(program, "o"), s=65,
                           color="pink", edgecolor="black", zorder=2)

    # Set labels and title
    plt.xlabel("Log2 Capacity")
    plt.ylabel("Elapsed Time (ms)")
    plt.title("Execution Speed")
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.xticks(np.arange(11, 22, 2))

    ax2 = fig.add_subplot(122)

    # Plot gfem efficiency
    ax2.fill_between(data["size"], data_gfem["model_efficiency"].min(),
                     data_gfem["model_efficiency"].max(), facecolor="white",
                     edgecolor="black")

    # Plot lut and rocmlm efficiency
    for program, group in data.groupby("program"):
        for rmse, sub_group in group.groupby("rmse_val_mean_rho"):
            if program in ["RocMLM (DT)", "RocMLM (KN)", "RocMLM (NN1)", "RocMLM (NN3)"]:
                ax2.scatter(x=sub_group["size"], y=sub_group["model_efficiency"],
                            marker=marker_dict.get(program, "o"), s=65,
                            color=color_dict[rmse], edgecolor="black", zorder=2)
            else:
                ax2.scatter(x=sub_group["size"], y=sub_group["model_efficiency"],
                            marker=marker_dict.get(program, "o"), s=65,
                            color="pink", edgecolor="black", zorder=2)

    # Create the legend
    legend_elements = []
    for program, marker in marker_dict.items():
        if program in ["RocMLM (DT)", "RocMLM (KN)", "RocMLM (NN1)", "RocMLM (NN3)"]:
            legend_elements.append(
                mlines.Line2D(
                    [0], [0], marker=marker, color="none", label=program,
                    markerfacecolor="black", markersize=10)
            )
        else:
            legend_elements.append(
                mlines.Line2D(
                    [0], [0], marker=marker, color="none", markeredgecolor="black",
                    label=program, markerfacecolor="pink", markersize=10)
            )

    fig.legend(handles=legend_elements, title="Method", loc="center right", ncol=1,
               bbox_to_anchor=(1.22, 0.6), columnspacing=0.2, handletextpad=-0.1,
               fontsize=fontsize * 0.833)

    # Create a colorbar
    cbar = plt.colorbar(sm, label="RMSE (g/cm$^3$)", ax=(ax, ax2), format="%0.1g",
                        orientation="horizontal", anchor=(1.50, 0.5), shrink=0.25, aspect=10)
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_ticks([sm.get_clim()[0], sm.get_clim()[1]])

    # Set labels and title
    plt.xlabel("Log2 Capacity")
    plt.ylabel("Inefficiency (ms$\\cdot$Mb)")
    plt.title("Model Efficiency")
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.xticks(np.arange(11, 22, 2))

    # Add captions
    fig.text(0.025, 0.92, "a)", fontsize=fontsize * 1.3)
    fig.text(0.525, 0.92, "b)", fontsize=fontsize * 1.3)

    # Adjust the figure size
    fig = plt.gcf()
    fig.set_size_inches(figwidth, figheight)

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/{filename}")

    # Close device
    plt.close()

    return None

#######################################################
## .3.              Train RocMLMs                !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train rocmlms !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_rocmlms(gfem_models, ml_models=["DT", "KN", "NN1", "NN2", "NN3"],
                  PT_steps=[16, 8, 4, 2, 1], X_steps=[16, 8, 4, 2, 1]):
    """
    """
    # Check for gfem models
    if not gfem_models:
        raise Exception("No GFEM models to compile!")

    # Single X step for benchmark models
    sids = [m.sid for m in gfem_models]
    if any(sample in sids for sample in ["PUM", "DMM", "PYR"]):
        X_steps = [1]

    # Train rocmlm models at various PTX grid resolution levels
    rocmlms = []
    for X_step in X_steps:
        for PT_step in PT_steps:
            mlms = []
            for model in ml_models:
                rocmlm = RocMLM(gfem_models, model, X_step, PT_step)
                if rocmlm.ml_model_trained:
                    pretrained_rocmlm = joblib.load(rocmlm.rocmlm_path)
                    mlms.append(pretrained_rocmlm)
                else:
                    rocmlm.train_rocmlm()
                    mlms.append(rocmlm)
                    with open(rocmlm.rocmlm_path, "wb") as file:
                        joblib.dump(rocmlm, file)

            # Compile rocmlms
            rocmlms.extend(mlms)

    # Check for errors in the models
    error_count = 0

    for model in rocmlms:
        if model.ml_model_training_error:
            error_count += 1

    if error_count > 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Total RocMLMs with errors: {error_count}")
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("All RocMLMs built successfully !")

    print(":::::::::::::::::::::::::::::::::::::::::::::")

    return rocmlms

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    try:
        # Build GFEM models
        gfems = {}
        sources = {"b": "assets/data/bench-pca.csv",
                   "m": "assets/data/synth-mids.csv",
                   "r": "assets/data/synth-rnds.csv"}

        for name, source in sources.items():
            sids = get_sampleids(source, "all")
            gfems[name] = build_gfem_models(source, sids)

        # Combine synthetic models for RocMLM training
        rocmlms = {}
        training_data = {"b": gfems["b"], "s": gfems["m"] + gfems["r"]}

        # Train RocMLMs
        for name, models in training_data.items():
            rocmlms[name] = train_rocmlms(models)

        # Visualize RocMLMs
        visualize_rocmlm_performance()

        for name, models in rocmlms.items():
            if name == "b":
                for model in models:
                    compose_rocmlm_plots(model)

    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in main() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("RocMLMs trained and visualized !")
    print("=============================================")

    return None

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()