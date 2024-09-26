#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import glob
import time
import joblib
import warnings
import textwrap
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures as cf

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.pyplot as plt
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
    def __init__(self, gfem_models, ml_algo="DT", tune=True, verbose=1):
        """
        """
        # Input
        self.tune = tune
        self.verbose = verbose
        self.ml_algo = ml_algo
        self.gfem_models = gfem_models

        # Random seed
        self.seed = 42

        # Parallel processing
        self.nprocs = os.cpu_count() - 2

        # Syracuse 2010 subduction segments for subduction depth profiles
        self.segs = ["Central_Cascadia", "Kamchatka"]

        # Mantle potential temps for mantle depth profiles
        self.pot_Ts = [1173, 1573, 1773]

        # GFEM model metadata
        self.gfem_sids = []
        self.gfem_db = None
        self.gfem_res = None
        self.gfem_P_min = None
        self.gfem_P_max = None
        self.gfem_T_min = None
        self.gfem_T_max = None
        self.gfem_targets = None
        self.gfem_features = None
        self.gfem_target_units_map = None
        self.gfem_target_labels_map = None
        self.gfem_target_digits_map = None
        self._get_gfem_model_metadata()

        # RocMLM and training arrays
        self.rocmlm = None
        self.rocmlm_scaler_X = None
        self.rocmlm_scaler_y = None
        self.rocmlm_target_array = None
        self.rocmlm_feature_array = None
        self.rocmlm_target_array_shape = None
        self.rocmlm_feature_array_shape = None
        self.rocmlm_target_array_shape_square = None
        self.rocmlm_feature_array_shape_square = None
        self.rocmlm_features = ["XI_FRAC", "LOI"]
        self.rocmlm_targets = ["density", "Vp", "Vs"]
#        self.rocmlm_targets = ["density", "Vp", "Vs", "melt_fraction", "H2O"]
        self._get_rocmlm_training_data()

        # RocMLM options
        self.kfolds = 5
        self.epochs = int(2e2)
        mask = np.any(np.isnan(self.rocmlm_target_array), axis=1)
        self.batch_size = int(len(self.rocmlm_target_array[~mask,:]) * 0.1)
        self.batch_size = max(self.batch_size, 8)

        # NN architecture
        L8, L16, L32 = int(8), int(16), int(32)
        nn_arch = (L8, L8, L8)
#        nn_arch = (L8, L8, L8, L8, L8, L8)

        # RocMLM labels
        self.rocmlm_label_map = {
            "KN": "K Neighbors",
            "DT": "Decision Tree",
            "NN": "Neural Network"
        }
        self.default_rocmlm_map = {
            "KN": KNeighborsRegressor(weights="distance"),
            "DT": DecisionTreeRegressor(random_state=self.seed),
            "NN": MLPRegressor(nn_arch, max_iter=self.epochs, random_state=self.seed,
                               batch_size=self.batch_size),
        }
        self.rocmlm_gridsearch_map = {
            "KN": dict(n_neighbors=[3, 5, 7]),
            "DT": dict(max_features=[1, 2, 3], min_samples_leaf=[1, 2, 3],
                        min_samples_split=[2, 4, 6]),
            "NN": dict(learning_rate_init=[0.0001, 0.0005, 0.001])
        }

        # Output filepaths
        self.model_out_dir = f"rocmlms"
        self.model_prefix = (f"{self.ml_algo}-"
                             f"S{self.rocmlm_feature_array_shape_square[0]}-"
                             f"W{self.rocmlm_feature_array_shape_square[1]}-"
                             f"F{self.rocmlm_feature_array_shape_square[3]}-"
                             f"{self.gfem_db}")
        self.fig_dir = (f"figs/rocmlm/{self.model_prefix}")
        self.rocmlm_path = f"{self.model_out_dir}/{self.model_prefix}-pretrained.pkl"
        self.scaler_X_path = f"{self.model_out_dir}/{self.model_prefix}-scaler_X.pkl"
        self.scaler_y_path = f"{self.model_out_dir}/{self.model_prefix}-scaler_y.pkl"

        # Cross validation metrics
        self.cv_info = {}

        # Square arrays for visualizations
        self.gfem_target_array_square = np.array([])
        self.gfem_feature_array_square = np.array([])
        self.rocmlm_prediction_array_square = np.array([])

        # Rocmlm training checks
        self.model_error = None
        self.model_tuned = False
        self.model_trained = False
        self.model_hyperparams = None
        self.model_training_error = False
        self.model_cross_validated = False

        # Set np array printing option
        np.set_printoptions(precision=3, suppress=True)

        # Check for existing model
        self._check_existing_model()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get unique value !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_unique_value(self, input_list):
        """
        """
        try:
            unique_value = input_list[0]
            for item in input_list[1:]:
                if item != unique_value:
                    raise Exception("Not all values are the same !")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_unique_value() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return unique_value

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get gfem model metadata !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_gfem_model_metadata(self):
        """
        """
        # Get self attributes
        gfem_models = self.gfem_models

        try:
            if not gfem_models:
                raise Exception("No GFEM models to compile !")

            # Get model metadata
            gfem_sids = [m.sid for m in gfem_models]
            gfem_res = self._get_unique_value([m.res for m in gfem_models])
            ox_gfem = self._get_unique_value([m.ox_gfem for m in gfem_models])
            gfem_P_min = self._get_unique_value([m.P_min for m in gfem_models])
            gfem_P_max = self._get_unique_value([m.P_max for m in gfem_models])
            gfem_T_min = self._get_unique_value([m.T_min for m in gfem_models])
            gfem_T_max = self._get_unique_value([m.T_max for m in gfem_models])
            gfem_db = self._get_unique_value([m.perplex_db for m in gfem_models])
            gfem_targets = self._get_unique_value([m.targets for m in gfem_models])
            gfem_features = self._get_unique_value([m.features for m in gfem_models])
            target_units_map = self._get_unique_value(
                [m.target_units_map for m in gfem_models])
            target_labels_map = self._get_unique_value(
                [m.target_labels_map for m in gfem_models])
            target_digits_map = self._get_unique_value(
                [m.target_digits_map for m in gfem_models])

            # Update self attributes
            self.gfem_db = gfem_db
            self.gfem_res = gfem_res
            self.gfem_sids = gfem_sids
            self.gfem_P_min = gfem_P_min
            self.gfem_P_max = gfem_P_max
            self.gfem_T_min = gfem_T_min
            self.gfem_T_max = gfem_T_max
            self.gfem_targets = gfem_targets
            self.gfem_features = gfem_features
            self.target_units_map = target_units_map
            self.target_labels_map = target_labels_map
            self.target_digits_map = target_digits_map

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_gfem_model_metadata() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get rocmlm training data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_rocmlm_training_data(self):
        """
        """
        # Get self attributes
        gfem_res = self.gfem_res
        gfem_sids = self.gfem_sids
        gfem_models = self.gfem_models
        gfem_targets = self.gfem_targets
        gfem_features = self.gfem_features
        rocmlm_targets = self.rocmlm_targets
        rocmlm_features = self.rocmlm_features

        try:
            if not gfem_models:
                raise Exception("No GFEM models to compile !")

            # Get PT array (M, W, 2)
            pt_train = np.stack([m.pt_array for m in gfem_models])

            # Get rocmlm training features (M, W, rocmlm f)
            ft_idx = [gfem_features.index(f) for f in rocmlm_features if f in gfem_features]
            rocmlm_features = [gfem_features[i] for i in ft_idx]

            feat_train = []
            for m in gfem_models:
                selected_features = [m.sample_features[i] for i in ft_idx]
                feat_train.append(selected_features)

            feat_train = np.array(feat_train)

            # Tile features to match PT array shape (M, W, rocmlm f)
            feat_train = np.tile(feat_train[:, np.newaxis, :], (1, pt_train.shape[1], 1))

            # Combine PT and rocmlm training feature arrays (M, W, rocmlm f + 2)
            combined_train = np.concatenate((feat_train, pt_train), axis=2)

            # Flatten features (M * W, rocmlm f + 2)
            rocmlm_feature_array = combined_train.reshape(-1, combined_train.shape[-1])

            # Get rocmlm training targets
            t_idx = [gfem_targets.index(t) for t in rocmlm_targets if t in gfem_targets]
            rocmlm_targets = [gfem_targets[i] for i in t_idx]

            # Get target arrays (M, W, gfem T)
            rocmlm_target_array = np.stack([m.target_array for m in gfem_models])

            # Flatten targets (M * W, gfem T)
            rocmlm_target_array = rocmlm_target_array.reshape(
                -1, rocmlm_target_array.shape[-1])

            # Select training targets (M * W, rocmlm T)
            rocmlm_target_array = rocmlm_target_array[:, t_idx]

            # Define array shapes
            M = int(len(gfem_models))         # Number of samples
            W = int((gfem_res + 1) ** 2)      # PT grid size
            w = int(np.sqrt(W))               # P or T array size
            f = int(len(rocmlm_features))     # Number of rocmlm features
            F = int(f + 2)                    # Number of rocmlm features + PT
            T = int(len(rocmlm_targets))      # Number of rocmlm targets
            rocmlm_target_array_shape = (M, W, T)
            rocmlm_feature_array_shape = (M, W, F)
            rocmlm_target_array_shape_square = (M, w, w, T)
            rocmlm_feature_array_shape_square = (M, w, w, F)

            self.rocmlm_target_array = rocmlm_target_array
            self.rocmlm_feature_array = rocmlm_feature_array
            self.rocmlm_target_array_shape = rocmlm_target_array_shape
            self.rocmlm_feature_array_shape = rocmlm_feature_array_shape
            self.rocmlm_target_array_shape_square = rocmlm_target_array_shape_square
            self.rocmlm_feature_array_shape_square = rocmlm_feature_array_shape_square

            gfem_target_array_square = \
                rocmlm_target_array.reshape(rocmlm_target_array_shape_square)
            gfem_feature_array_square = \
                rocmlm_feature_array.reshape(rocmlm_feature_array_shape_square)
            self.gfem_target_array_square = gfem_target_array_square
            self.gfem_feature_array_square = gfem_feature_array_square

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_rocmlm_training_data() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check existing model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_existing_model(self):
        """
        """
        # Get self attributes
        verbose = self.verbose
        rocmlm_path = self.rocmlm_path
        model_out_dir = self.model_out_dir

        model = None

        if os.path.exists(rocmlm_path):
            try:
                if verbose >= 1: print(f"  Found pretrained model {rocmlm_path} !")
                self.model_trained = True
                self.load_pretrained_model(f"{rocmlm_path}")
            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"!!! ERROR in _check_existing_model() !!!")
                print(f"{e}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                traceback.print_exc()
                return None
        else:
            os.makedirs(model_out_dir, exist_ok=True)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load pretrained model
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_pretrained_model(self, file_path):
        """
        """
        # Get self attributes
        verbose = self.verbose
        rocmlm_path = self.rocmlm_path
        scaler_X_path = self.scaler_X_path
        scaler_y_path = self.scaler_y_path

        if os.path.exists(file_path):
            try:
                if verbose >= 1: print(f"  Loading RocMLM object from {file_path} ...")
                rocmlm = joblib.load(rocmlm_path)
                scaler_X = joblib.load(scaler_X_path)
                scaler_y = joblib.load(scaler_y_path)
                self._get_gfem_model_metadata()
                self._get_rocmlm_training_data()
                self.rocmlm = rocmlm

                target_array = self.rocmlm_target_array.copy()
                feature_array = self.rocmlm_feature_array.copy()
                rocmlm_target_array_shape_square = self.rocmlm_target_array_shape_square

                X, y = feature_array.copy(), target_array.copy()

                X_scaled = scaler_X.transform(X)

                pred_scaled = rocmlm.predict(X_scaled)

                pred_original = scaler_y.inverse_transform(pred_scaled)

                rocmlm_prediction_array_square = pred_original.reshape(
                    rocmlm_target_array_shape_square)

                self.rocmlm_prediction_array_square = rocmlm_prediction_array_square

            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"!!! ERROR in load_pretrained_model() !!!")
                print(f"{e}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                traceback.print_exc()
                return None
        else:
            print(f"File {file_path} does not exist !")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # scale arrays !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scale_arrays(self, feature_array, target_array):
        """
        """
        try:
            X = feature_array
            y = target_array

            X[~np.isfinite(X)] = np.nan
            y[~np.isfinite(y)] = np.nan

            mask = np.any(np.isnan(y), axis=1)
            X, y = X[~mask,:], y[~mask,:]
#            X, y = np.nan_to_num(X), np.nan_to_num(y)

            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise Exception("Input data contains NaN or infinity values !")

            scaler_X, scaler_y = StandardScaler(), StandardScaler()

            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _scale_arrays() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None, None, None, None, None, None

        return X, y, scaler_X, scaler_y, X_scaled, y_scaled

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.               RocMLMs                   !!! ++
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
        nprocs = self.nprocs
        verbose = self.verbose
        ml_algo = self.ml_algo
        model_prefix = self.model_prefix
        default_rocmlm_map = self.default_rocmlm_map
        rocmlm_target_array = self.rocmlm_target_array
        rocmlm_feature_array = self.rocmlm_feature_array
        rocmlm_gridsearch_map = self.rocmlm_gridsearch_map

        try:
            if rocmlm_feature_array.size == 0:
                raise Exception("No training features !")

            if rocmlm_target_array.size == 0:
                raise Exception("No training targets !")

            _, _, _, _, X_scaled, y_scaled = self._scale_arrays(
                rocmlm_feature_array, rocmlm_target_array)

            if verbose >= 1:
                print(f"  Configuring model {model_prefix} ...")

            if tune:
                if verbose >= 1:
                    print(f"  Tuning model {model_prefix} ...")

                model = default_rocmlm_map[ml_algo]
                search = rocmlm_gridsearch_map[ml_algo]
                kf = KFold(n_splits=6, shuffle=True, random_state=seed)
                grid_search = GridSearchCV(model, search, cv=kf, n_jobs=nprocs,
                                           scoring="neg_root_mean_squared_error")
                grid_search.fit(X_scaled, y_scaled)

                if ml_algo == "KN":
                    model =  KNeighborsRegressor(
                        n_neighbors=grid_search.best_params_["n_neighbors"])
                elif ml_algo == "DT":
                    model = DecisionTreeRegressor(
                        random_state=seed,
                        max_features=grid_search.best_params_["max_features"],
                        min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                        min_samples_split=grid_search.best_params_["min_samples_split"])
                elif "NN" in ml_algo:
                    model = MLPRegressor(
                        nn_arch, random_state=seed,
                        learning_rate_init=grid_search.best_params_["learning_rate_init"])
                self.model_tuned = True
            else:
                model = default_rocmlm_map[ml_algo]

            self.rocmlm = model
            self.model_hyperparams = model.get_params()

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _configure_rocmlm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # print rocmlm info !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_rocmlm_info(self):
        """
        """
        # Get self attributes
        kfolds = self.kfolds
        epochs = self.epochs
        ml_algo = self.ml_algo
        model_prefix = self.model_prefix
        rocmlm_targets = self.rocmlm_targets
        rocmlm_features = self.rocmlm_features
        rocmlm_label_map = self.rocmlm_label_map
        model_hyperparams = self.model_hyperparams
        rocmlm_target_array = self.rocmlm_target_array
        rocmlm_feature_array = self.rocmlm_feature_array

        tgwrp = textwrap.fill(", ".join(rocmlm_targets), width=80,
                              subsequent_indent="                  ")
        ftwrp = textwrap.fill(", ".join(rocmlm_features), width=80)

        # Print rocmlm configuration
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"  RocMLM model: {model_prefix}")
        print("---------------------------------------------")
        print(f"    model: {rocmlm_label_map[ml_algo]}")
        print(f"    k folds: {kfolds}")
        print(f"    features: {ftwrp}")
        print(f"    targets: {tgwrp}")
        print(f"    features array shape: {rocmlm_feature_array.shape}")
        print(f"    targets array shape: {rocmlm_target_array.shape}")
        print(f"    hyperparameters:")
        for key, value in model_hyperparams.items():
            print(f"        {key}: {value}")
        print("---------------------------------------------")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # kfold itr !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _kfold_itr(self, fold_args):
        """
        """
        # Get self attributes
        rocmlm = self.rocmlm
        epochs = self.epochs
        ml_algo = self.ml_algo
        batch_size = self.batch_size
        rocmlm_target_array = self.rocmlm_target_array
        rocmlm_feature_array = self.rocmlm_feature_array

        try:
            # Check for training features
            if rocmlm_feature_array.size == 0:
                raise Exception("No training features !")

            # Check for training targets
            if rocmlm_target_array.size == 0:
                raise Exception("No training targets !")

            # Scale training dataset
            X, y, _, scaler_y, X_scaled, y_scaled = \
                self._scale_arrays(rocmlm_feature_array, rocmlm_target_array)

            # Get fold indices
            (train_index, test_idxex) = fold_args

            # Split the data into training and testing sets
            X, X_test = X_scaled[train_index], X_scaled[test_idxex]
            y, y_test = y_scaled[train_index], y_scaled[test_idxex]

            if "NN" in ml_algo:
                # Initialize lists to store loss values
                epoch_, train_loss_, test_loss_ = [], [], []

                # Start training timer
                training_start_time = time.time()

                # Partial training
                with tqdm(total=epochs, desc="Training NN", position=0) as pbar:
                    for epoch in range(epochs):
                        # Shuffle the training data for each epoch
                        indices = np.arange(len(y))
                        np.random.shuffle(indices)

                        for start_idx in range(0, len(indices), batch_size):
                            end_idx = start_idx + batch_size

                            # Ensure that the batch size doesn't exceed the dataset size
                            end_idx = min(end_idx, len(indices))

                            # Subset training data
                            batch_indices = indices[start_idx:end_idx]
                            X_batch, y_batch = X[batch_indices], y[batch_indices]

                            # Train NN model on batch
                            rocmlm.partial_fit(X_batch, y_batch)

                        # Calculate and store training loss
                        train_loss = rocmlm.loss_
                        train_loss_.append(train_loss)

                        # Calculate and store test loss
                        test_loss = mean_squared_error(
                            y_test, rocmlm.predict(X_test))
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
                rocmlm.fit(X, y)

                # End training timer
                training_end_time = time.time()

                # Empty loss curve
                loss_curve = None

            # Calculate training time
            training_time = training_end_time - training_start_time

            # Make predictions on the test dataset
            y_pred_scaled = rocmlm.predict(X_test)

            # Inverse transform predictions
            y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

            # Inverse transform test dataset
            y_test_original = scaler_y.inverse_transform(y_test)

            # Calculate performance metrics to evaluate the model
            rmse_test = np.sqrt(mean_squared_error(y_test_original, y_pred_original,
                                                   multioutput="raw_values"))

            r2_test = r2_score(y_test_original, y_pred_original, multioutput="raw_values")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _kfold_itr() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return (None, None, None, None)

        return (loss_curve, rmse_test, r2_test, training_time)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process kfold results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_kfold_results(self, results):
        """
        """
        # Get self attributes
        kfolds = self.kfolds
        verbose = self.verbose
        ml_algo = self.ml_algo
        gfem_sids = self.gfem_sids
        rocmlm_targets = self.rocmlm_targets
        rocmlm_label_map = self.rocmlm_label_map
        M = self.rocmlm_feature_array_shape_square[0]
        w = self.rocmlm_feature_array_shape_square[1]

        try:
            # Initialize empty lists for storing performance metrics
            loss_curves = []
            training_times = []
            rmse_test_scores, r2_test_scores = [], []

            # Unpack results
            for (loss_curve, rmse_test, r2_test, training_time) in results:
                loss_curves.append(loss_curve)
                rmse_test_scores.append(rmse_test)
                r2_test_scores.append(r2_test)
                training_times.append(training_time)

            if "NN" in ml_algo:
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

            # Get sample label
            if any("sm" in sample or "sr" in sample for sample in gfem_sids):
                sample_label = f"SMA{M}"
            elif any("st" in sample for sample in gfem_sids):
                sample_label = f"SMAT{M}"
            elif any("sm" in sample for sample in gfem_sids):
                sample_label = f"SMAM{M}"
            elif any("sb" in sample for sample in gfem_sids):
                sample_label = f"SMAB{M}"
            elif any("sr" in sample for sample in gfem_sids):
                sample_label = f"SMAR{M}"

            # Config and performance info
            cv_info = {
                "model": [ml_algo],
                "sample": [sample_label],
                "size": [(w - 1) ** 2],
                "n_targets": [len(rocmlm_targets)],
                "k_folds": [kfolds],
                "training_time_mean": [round(training_time_mean, 5)],
                "training_time_std": [round(training_time_std, 5)]
            }

            # Add performance metrics for each parameter to the dictionary
            for i, target in enumerate(rocmlm_targets):
                cv_info[f"rmse_test_mean_{target}"] = round(rmse_test_mean[i], 5)
                cv_info[f"rmse_test_std_{target}"] = round(rmse_test_std[i], 5)
                cv_info[f"r2_test_mean_{target}"] = round(r2_test_mean[i], 5)
                cv_info[f"r2_test_std_{target}"] = round(r2_test_std[i], 5)

            if verbose >= 1:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                # Print performance
                print(f"{rocmlm_label_map[ml_algo]} performance:")
                print(f"    training time: {training_time_mean:.5f} ± "
                      f"{training_time_std:.5f}")
                print(f"    rmse test:")
                for r, e, p in zip(rmse_test_mean, rmse_test_std, rocmlm_targets):
                    print(f"        {p}: {r:.5f} ± {e:.5f}")
                print(f"    r2 test:")
                for r, e, p in zip(r2_test_mean, r2_test_std, rocmlm_targets):
                    print(f"        {p}: {r:.5f} ± {e:.5f}")
                print("+++++++++++++++++++++++++++++++++++++++++++++")

            self.cv_info = cv_info

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _process_kfold_results() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

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
        rocmlm_target_array = self.rocmlm_target_array
        rocmlm_feature_array = self.rocmlm_feature_array

        try:
            if rocmlm_feature_array.size == 0:
                raise Exception("No training features !")

            if rocmlm_target_array.size == 0:
                raise Exception("No training targets !")

            X, _, _, _, _, _ = self._scale_arrays(rocmlm_feature_array, rocmlm_target_array)

            kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

            fold_args = [(train_idx, test_idx) for _, (train_idx, test_idx) in
                         enumerate(kf.split(X))]

            with cf.ProcessPoolExecutor(max_workers=nprocs) as executor:
                results = list(tqdm(executor.map(self._kfold_itr, fold_args),
                                    total=len(fold_args), desc="Processing folds ..."))


            self.model_cross_validated = True
            self._process_kfold_results(results)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _kfold_cv() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # fit training data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _fit_training_data(self):
        """
        """
        # Get self attributes
        seed = self.seed
        epochs = self.epochs
        rocmlm = self.rocmlm
        ml_algo = self.ml_algo
        batch_size = self.batch_size
        rocmlm_path = self.rocmlm_path
        model_prefix = self.model_prefix
        scaler_X_path = self.scaler_X_path
        scaler_y_path = self.scaler_y_path
        target_array = self.rocmlm_target_array.copy()
        feature_array = self.rocmlm_feature_array.copy()
        rocmlm_target_array_shape_square = self.rocmlm_target_array_shape_square

        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            if not self.model_cross_validated:
                raise Exception("ML model not cross validated! Call _kfold_cv() first ...")

            if feature_array.size == 0:
                raise Exception("No training features !")

            if target_array.size == 0:
                raise Exception("No training targets !")

            # Scale arrays
            _, _, scaler_X, scaler_y, X_scaled, y_scaled = \
                self._scale_arrays(feature_array, target_array)

            # Train model on entire training dataset
            X_train, X_test, y_train, y_test = \
                train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)

            # Train ML model
            if "NN" in ml_algo:
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
                            rocmlm.partial_fit(X_batch, y_batch)

                        # Update progress bar
                        pbar.update(1)

            else:
                print(f"Training model {model_prefix} ...")
                rocmlm.fit(X_train, y_train)

            print(":::::::::::::::::::::::::::::::::::::::::::::")
            self.model_trained = True
            self.rocmlm_scaler_X = scaler_X
            self.rocmlm_scaler_y = scaler_y

            # Save pretrained rocmlm and scalers
            with open(rocmlm_path, "wb") as file:
                joblib.dump(rocmlm, file)
            with open(scaler_X_path, "wb") as file:
                joblib.dump(scaler_X, file)
            with open(scaler_y_path, "wb") as file:
                joblib.dump(scaler_y, file)

            # Copy feature and target arrays
            X, y = feature_array.copy(), target_array.copy()

            # Scale features array
            X_scaled = scaler_X.transform(X)

            # Make predictions on features
            pred_scaled = rocmlm.predict(X_scaled)

            # Inverse transform predictions
            pred_original = scaler_y.inverse_transform(pred_scaled)

            # Reshape array into square for visualization
            rocmlm_prediction_array_square = pred_original.reshape(
                rocmlm_target_array_shape_square)

            # Update array
            self.rocmlm_prediction_array_square = rocmlm_prediction_array_square

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _fit_training_data() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # save rocmlm cv info !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _save_rocmlm_cv_info(self):
        """
        """
        # Get self attributes
        data_dict = self.cv_info

        try:
            if not self.cv_info:
                raise Exception("No cross validation! Call _kfold_cv() first ...")

            # CSV filepath
            filepath = f"assets/rocmlm-performance.csv"

            # Check if the CSV file already exists
            if not pd.io.common.file_exists(filepath):
                df = pd.DataFrame(data_dict)
            else:
                df = pd.read_csv(filepath)
                new_data = pd.DataFrame(data_dict)
                df = pd.concat([df, new_data], ignore_index=True)

            df = df.sort_values(by=["model", "sample", "size"])
            df.to_csv(filepath, index=False)

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _save_rocmlm_cv_info() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.4.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check rocmlm images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_rocmlm_images(self, sid, type="predictions"):
        """
        """
        # Get self attributes
        fig_dir = self.fig_dir
        model_prefix = self.model_prefix
        rocmlm_targets = self.rocmlm_targets

        try:
            if type not in {"targets", "predictions", "diff"}:
                raise Exception("Unrecognized array image type !")

            # Check for existing plots
            check = True
            for target in rocmlm_targets:
                path = (f"{fig_dir}/{model_prefix}-{sid}-{target.replace("_", "-")}-"
                        f"{type}.png")
                if not os.path.exists(path):
                    check = False
                    break

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _check_rocmlm_images() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return check

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize loss curve !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_loss_curve(self, loss_curves, figwidth=6.3, figheight=3.54, fontsize=14):
        """
        """
        # Get self attributes
        fig_dir = self.fig_dir
        ml_algo = self.ml_algo
        rocmlm_label_map = self.rocmlm_label_map

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

        plt.title(f"{rocmlm_label_map[ml_algo]} Loss Curve")
        plt.legend()

        # Save the plot to a file
        os.makedirs(f"{fig_dir}", exist_ok=True)
        plt.savefig(f"{fig_dir}/{ml_algo}-loss-curve.png")

        # Close plot
        plt.close()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize array image  !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_image(self, sid, sid_idx, palette="bone", type="targets",
                               figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get self attributes
        fig_dir = self.fig_dir
        verbose = self.verbose
        gfem_db = self.gfem_db
        model_prefix = self.model_prefix
        model_trained = self.model_trained
        rocmlm_targets = self.rocmlm_targets
        target_units_map = self.target_units_map
        target_labels_map = self.target_labels_map
        target_digits_map = self.target_digits_map
        gfem_target_array_square = self.gfem_target_array_square
        gfem_feature_array_square = self.gfem_feature_array_square
        rocmlm_prediction_array_square = self.rocmlm_prediction_array_square

        if type not in {"targets", "predictions", "diff"}:
            raise Exception("Unrecognized array image type !")

        if not model_trained:
            raise Exception("No RocMLM model! Call train_rocmlm() first ...")

        if gfem_target_array_square is None or gfem_target_array_square.size == 0:
            raise Exception("No GFEM model target array !")

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

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
        plt.rcParams["figure.constrained_layout.use"] = "True"

        try:
            # Get number of training features
            n_feats = gfem_feature_array_square.shape[-1] - 2

            # Slice arrays
            target_array = gfem_target_array_square[sid_idx, :, :, :]
            feature_array = gfem_feature_array_square[sid_idx, :, :, :]
            pred_array = rocmlm_prediction_array_square[sid_idx, :, :, :]

            for i, target in enumerate(rocmlm_targets):
                rmse, r2 = None, None

                # Get 2d arrays
                p = pred_array[:, :, i]
                t = target_array[:, :, i]
                P = feature_array[:, :, 0 + n_feats]
                T = feature_array[:, :, 1 + n_feats]
                extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

                if type == "predictions":
                    square_array = p
                    filename = (f"{model_prefix}-{sid}-"
                                f"{target.replace("_", "-")}-predictions.png")
                elif type == "targets":
                    square_array = t
                    filename = (f"{model_prefix}-{sid}-"
                                f"{target.replace("_", "-")}-targets.png")
                elif type == "diff":
                    palette = "seismic"
                    mask = np.isnan(t)
                    p[mask] = np.nan
                    square_array = 100 * (t - p) / np.abs(t)
                    r2 = r2_score(t, p)
                    rmse = np.sqrt(mean_squared_error(t, p))
                    filename = (f"{model_prefix}-{sid}-"
                                f"{target.replace("_", "-")}-diff.png")

                # Target labels
                target_units = target_units_map[target]
                target_label = target_labels_map[target]

                # Set title
                if target not in ["assemblage_index", "phase_assemblage_variance"]:
                    title = f"{target_label} ({target_units})"
                else:
                    title = f"{target_label}"

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
                    cmap.set_bad(color="0.9")

                    # Plot as a raster using imshow
                    fig, ax = plt.subplots(figsize=(figwidth, figheight))

                    im = ax.imshow(square_array, extent=extent, aspect="auto", cmap=cmap,
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
                            np.abs(square_array[np.logical_not(np.isnan(square_array))]))
                        vmax = np.max(
                            np.abs(square_array[np.logical_not(np.isnan(square_array))]))
                    else:
                        vmin, vmax = vmin, vmax

                        # Adjust vmin close to zero
                        if vmin <= 1e-4: vmin = 0

                    # Set nan color
                    cmap = plt.colormaps[cmap]
                    cmap.set_bad(color="0.9")

                    # Plot as a raster using imshow
                    fig, ax = plt.subplots()

                    im = ax.imshow(square_array, extent=extent, aspect="auto", cmap=cmap,
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
                        cbar = plt.colorbar(im, ax=ax, label="",
                                            ticks=np.linspace(vmin, vmax, num=4))

                    # Set colorbar limits and number formatting
                    cbar.ax.yaxis.set_major_formatter(
                        plt.FormatStrFormatter(target_digits_map[target]))

                plt.title(title)

                # Vertical text spacing
                text_margin_x = 0.04
                text_margin_y = 0.15
                text_spacing_y = 0.1

                # Add rmse and r2
                if rmse and r2:
                    plt.text(text_margin_x, text_margin_y - (text_spacing_y * 0),
                             f"R$^2$: {r2:.3f}", transform=plt.gca().transAxes,
                             fontsize=fontsize * 0.833, horizontalalignment="left",
                             verticalalignment="bottom")
                    plt.text(text_margin_x, text_margin_y - (text_spacing_y * 1),
                             f"RMSE: {rmse:.3f}", transform=plt.gca().transAxes,
                             fontsize=fontsize * 0.833, horizontalalignment="left",
                             verticalalignment="bottom")

                # Save the plot to a file
                plt.savefig(f"{fig_dir}/{filename}")

                # Close device
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _visualize_array_image() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize(self, skip=8):
        """
        """
        # Get self attributes
        gfem_sids = self.gfem_sids

        try:
            for sid_idx, sid in enumerate(gfem_sids[::skip]):
                if not self._check_rocmlm_images(sid, type="diff"):
                    self._visualize_array_image(sid, sid_idx, type="diff")
                if not self._check_rocmlm_images(sid, type="targets"):
                    self._visualize_array_image(sid, sid_idx, type="targets")
                if not self._check_rocmlm_images(sid, type="predictions"):
                    self._visualize_array_image(sid, sid_idx, type="predictions")
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in visualize_rocmlm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.5.             Train RocMLMs               !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # rocmlm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def train(self):
        """
        """
        max_retries = 3
        for retry in range(max_retries):
            if self.model_trained:
                break
            try:
                self._configure_rocmlm()
                self._print_rocmlm_info()
                self._kfold_cv()
                self._fit_training_data()
                self._save_rocmlm_cv_info()
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
                    self.model_training_error = True
                    self.model_error = e
                    return None
        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.6.           RocMLM Inference              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # predict !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def predict(self, xi, h2o, P, T):
        """
        """
        # Get self attributes
        model = self.ml_algo
        scaler_X = self.rocmlm_scaler_X
        scaler_y = self.rocmlm_scaler_y
        model_trained = self.model_trained

        try:
            if not model_trained:
                raise Exception("No RocMLM model! Call train_rocmlm() first ...")

            for var in [xi, h2o, P, T]:
                if not isinstance(var, (list, np.ndarray)):
                    raise TypeError(f"Inputs must be lists or numpy arrays !")

            # Create feature array
            if all(var is not None for var in [xi, h2o, P, T]):
                xi_array = np.asarray(xi)
                h2o_array = np.asarray(h2o)
                P_array = np.asarray(P)
                T_array = np.asarray(T)

                if (xi_array.shape[0] == h2o_array.shape[0] == P_array.shape[0] ==
                    T_array.shape[0]):
                    xi_array = xi_array.reshape(-1, 1)
                    h2o_array = h2o_array.reshape(-1, 1)
                    P_array = P_array.reshape(-1, 1)
                    T_array = T_array.reshape(-1, 1)

                    X = np.concatenate((xi_array, h2o_array, P_array, T_array), axis=1)
                else:
                    raise Exception("The arrays xi, h2o, P, and T are not the same size !")
            else:
                raise Exception("One or more of xi, h2o, P, T does not exist !")

            # Scale features array
            X_scaled = scaler_X.transform(X)

            # Make predictions on features
            inference_start_time = time.time()
            pred_scaled = model.predict(X_scaled)
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            print(inference_time * 1e3)

            # Inverse transform predictions
            pred_original = scaler_y.inverse_transform(pred_scaled)

            return pred_original

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in predict() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

            return None

#######################################################
## .2.              Train RocMLMs                !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train rocmlms !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_rocmlms(gfem_models, ml_algos=["DT", "KN", "NN"]):
    """
    """
    try:
        if not gfem_models:
            raise Exception("No GFEM models to compile !")

        models = []
        for algo in ml_algos:
            model = RocMLM(gfem_models, algo)
            if model.model_trained:
                continue
            else:
                model.train_rocmlm()
            models.extend(model)

    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in train_rocmlms() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()
        return None

    rocmlms = [model for model in models if not model.model_training_error]
    error_count = len([model for model in models if model.model_training_error])

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
    from gfem import build_gfem_models
    try:
        # Build GFEM models
        gfems = {}
        sources = {"m": "assets/synth-mids.csv", "r": "assets/synth-rnds.csv"}
        db, res, P_min, P_max, T_min, T_max = "hp02", 64, 0.1, 8.1, 273, 1973

        for name, source in sources.items():
            gfems[name] = build_gfem_models(source, db, res, P_min, P_max, T_min, T_max)

        training_data = gfems["m"] + gfems["r"]
        rocmlms = train_rocmlms(models)

        for model in rocmlms:
            model.visualize()

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
    main()