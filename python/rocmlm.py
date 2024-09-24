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
import multiprocessing as mp

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
from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

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
    def __init__(self, gfem_models, ml_algo="DT", verbose=1):
        """
        """
        # Input
        self.verbose = verbose
        self.ml_algo = ml_algo
        self.gfem_models = gfem_models

        # Random seed
        self.seed = 42

        # ML options
        self.kfolds = 5
        self.tune = True
        self.nn_L1 = int(8)
        self.nn_L2 = int(16)
        self.nn_L3 = int(32)
        self.batchprop = 0.2
        self.epochs = int(1e3)
        self.rocmlm_features = ["XI_FRAC", "LOI"]
        self.rocmlm_targets = ["density", "Vp", "Vs", "H2O", "melt_fraction",
                               "molar_entropy", "molar_heat_capacity"]

        # Parallel processing
        self.nprocs = os.cpu_count() - 2

        # Model labels
        self.ml_label_map = {
            "KN": "K Neighbors",
            "RF": "Random Forest",
            "DT": "Decision Tree",
            "NN1": "Neural Net 1L",
            "NN2": "Neural Net 2L",
            "NN3": "Neural Net 3L"
        }
        self.ml_algo_label = self.ml_label_map[ml_algo]

        # GFEM model metadata
        self.sids = []
        self.res = None
        self.P_min = None
        self.P_max = None
        self.T_min = None
        self.T_max = None
        self.targets = None
        self.features = None
        self.shape_target = None
        self.target_train = None
        self.feature_train = None
        self.shape_feature = None
        self.target_units_map = None
        self.shape_target_square = None
        self.shape_feature_square = None

        # Get GFEM model metadata
        self._get_gfem_model_metadata()
        self._process_training_data()

        # Output filepaths
        self.data_dir = "assets"
        self.model_out_dir = f"rocmlms"
        self.model_prefix = (f"{self.ml_algo}-"
                             f"S{self.shape_feature_square[0]}-"
                             f"W{self.shape_feature_square[1]}-"
                             f"F{self.shape_feature_square[2]}")
        self.fig_dir = (f"figs/rocmlm/{self.model_prefix}")
        self.rocmlm_path = f"{self.model_out_dir}/{self.model_prefix}.pkl"
        self.rocmlm_only_path = f"{self.model_out_dir}/{self.model_prefix}-model-only.pkl"
        self.scaler_X_path = f"{self.model_out_dir}/{self.model_prefix}-scaler_X.pkl"
        self.scaler_y_path = f"{self.model_out_dir}/{self.model_prefix}-scaler_y.pkl"

        # Check for figs directory
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir, exist_ok=True)

        # ML model definition and tuning
        self.ml_model_tuned = False
        self.ml_model_hyperparams = None

        # Cross validation performance metrics
        self.cv_info = {}
        self.ml_model_cross_validated = False

        # Square arrays for visualizations
        self.target_square = np.array([])
        self.feature_square = np.array([])
        self.prediction_square = np.array([])

        # Trained model
        self.rocmlm = None
        self.ml_model_error = None
        self.rocmlm_scaler_X = None
        self.rocmlm_scaler_y = None
        self.ml_model_trained = False
        self.ml_model_training_error = False

        # Set np array printing option
        np.set_printoptions(precision=3, suppress=True)

        # Check for existing model
        self._check_existing_model()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load pretrained model
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_pretrained_model(file_path):
        """
        """
        if os.path.exists(file_path):
            try:
                print(f"  Loading RocMLM object from {file_path} ...")
                model = joblib.load(file_path)
                return model
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
    # print rocmlm info !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_rocmlm_info(self):
        """
        """
        # Get self attributes
        kfolds = self.kfolds
        epochs = self.epochs
        ml_algo = self.ml_algo
        feat_train = self.feature_train
        target_train = self.target_train
        ml_algo_label = self.ml_algo_label
        rocmlm_targets = self.rocmlm_targets
        rocmlm_features = self.rocmlm_features
        ml_model_hyperparams = self.ml_model_hyperparams

        tgwrp = textwrap.fill(", ".join(rocmlm_targets), width=80,
                              subsequent_indent="                  ")
        ftwrp = textwrap.fill(", ".join(rocmlm_features), width=80)

        # Print rocmlm config
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("  RocMLM model defined as:")
        print(f"    model: {ml_algo_label}")
        if "NN" in ml_algo:
            print(f"    epochs: {epochs}")
        print(f"    k folds: {kfolds}")
        print(f"    features: {ftwrp}")
        print(f"    targets: {tgwrp}")
        print(f"    features array shape: {feat_train.shape}")
        print(f"    targets array shape: {target_train.shape}")
        print(f"    hyperparameters:")
        for key, value in ml_model_hyperparams.items():
            print(f"        {key}: {value}")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Running ({kfolds}) kfold cross validation ...")

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

        # Check for existing model build
        if os.path.exists(model_out_dir):
            if os.path.exists(rocmlm_path):
                if verbose >= 1:
                    print(f"Found pretrained model {rocmlm_path} !")
                try:
                    self.ml_model_trained = True
                    model = joblib.load(rocmlm_path)
                except Exception as e:
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"!!! ERROR in _check_existing_model() !!!")
                    print(f"{e}")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    traceback.print_exc()
                    return None
            else:
                os.makedirs(model_out_dir, exist_ok=True)
        else:
            os.makedirs(model_out_dir, exist_ok=True)

        return None

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
            sids = [m.sid for m in gfem_models]
            res = self._get_unique_value([m.res for m in gfem_models])
            P_min = self._get_unique_value([m.P_min for m in gfem_models])
            P_max = self._get_unique_value([m.P_max for m in gfem_models])
            T_min = self._get_unique_value([m.T_min for m in gfem_models])
            T_max = self._get_unique_value([m.T_max for m in gfem_models])
            ox_gfem = self._get_unique_value([m.ox_gfem for m in gfem_models])
            gfem_targets = self._get_unique_value([m.targets for m in gfem_models])
            gfem_features = self._get_unique_value([m.features for m in gfem_models])
            target_units_map = self._get_unique_value(
                [m.target_units_map for m in gfem_models])

            # Update self attributes
            self.res = res
            self.sids = sids
            self.P_min = P_min
            self.P_max = P_max
            self.T_min = T_min
            self.T_max = T_max
            self.gfem_targets = gfem_targets
            self.gfem_features = gfem_features
            self.target_units_map = target_units_map

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _get_gfem_model_metadata() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

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
            feature_train = combined_train.reshape(-1, combined_train.shape[-1])

            # Get rocmlm training targets
            t_idx = [gfem_targets.index(t) for t in rocmlm_targets if t in gfem_targets]
            rocmlm_targets = [gfem_targets[i] for i in t_idx]

            # Get target arrays (M, W, gfem T)
            target_train = np.stack([m.target_array for m in gfem_models])

            # Flatten targets (M * W, gfem T)
            target_train = target_train.reshape(-1, target_train.shape[-1])

            # Select training targets (M * W, rocmlm T)
            target_train = target_train[:, t_idx]

            # Define array shapes
            M = int(len(gfem_models))         # Number of samples
            W = int((res + 1) ** 2)           # PT grid size
            w = int(np.sqrt(W))               # P or T array size
            f = int(len(rocmlm_features))     # Number of rocmlm features
            F = int(f + 2)                    # Number of rocmlm features + PT
            T = int(len(rocmlm_targets))      # Number of rocmlm targets
            shape_target = (M, W, T)
            shape_feature = (M, W, F)
            shape_target_square = (M, w, w, T)
            shape_feature_square = (M, w, w, F)

            self.shape_target = shape_target
            self.target_train = target_train
            self.shape_feature = shape_feature
            self.feature_train = feature_train
            self.shape_target_square = shape_target_square
            self.shape_feature_square = shape_feature_square

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _process_training_data() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # scale arrays !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scale_arrays(self, feature_array, target_array):
        """
        """
        try:
            # Reshape the features and targets arrays
            X = feature_array
            y = target_array

            # Replace inf with nan
            X[~np.isfinite(X)] = np.nan
            y[~np.isfinite(y)] = np.nan

            # Create nan mask
            mask = np.any(np.isnan(y), axis=1)

            # Remove nans
            X, y = X[~mask,:], y[~mask,:]
#            X, y = np.nan_to_num(X), np.nan_to_num(y)

            # Check for infinity in input data
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise Exception("Input data contains NaN or infinity values !")

            # Initialize scalers
            scaler_X, scaler_y = StandardScaler(), StandardScaler()

            # Scale features array
            X_scaled = scaler_X.fit_transform(X)

            # Scale the target array
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
        nn_L1 = self.nn_L1
        nn_L2 = self.nn_L2
        nn_L3 = self.nn_L3
        epochs = self.epochs
        nprocs = self.nprocs
        verbose = self.verbose
        ml_algo = self.ml_algo
        batchprop = self.batchprop
        model_prefix = self.model_prefix
        target_train = self.target_train
        feature_train = self.feature_train

        try:
            if feature_train.size == 0:
                raise Exception("No training features !")

            if target_train.size == 0:
                raise Exception("No training targets !")

            # Scale training dataset
            _, _, _, _, X_scaled, y_scaled = self._scale_arrays(feature_train, target_train)

            # Define NN batch size
            bs = max(int(len(y_scaled) * batchprop), 8)

            if verbose >= 1:
                print(f"  Configuring model {model_prefix} ...")

            if tune:
                if verbose >= 1:
                    print(f"  Tuning model {model_prefix} ...")

                # Define ML model and grid search param space for hyperparameter tuning
                if ml_algo == "KN":
                    model = KNeighborsRegressor()
                    param_grid = dict(n_neighbors=[2, 4, 8], weights=["uniform", "distance"])
                elif ml_algo == "RF":
                    model = RandomForestRegressor(random_state=seed)
                    param_grid = dict(n_estimators=[400, 800, 1200],
                                      max_features=[1, 2, 3],
                                      min_samples_leaf=[1, 2, 3],
                                      min_samples_split=[2, 4, 6])
                elif ml_algo == "DT":
                    model = DecisionTreeRegressor(random_state=seed)
                    param_grid = dict(splitter=["best", "random"],
                                      max_features=[1, 2, 3],
                                      min_samples_leaf=[1, 2, 3],
                                      min_samples_split=[2, 4, 6])
                elif ml_algo == "NN1":
                    model = MLPRegressor(random_state=seed, max_iter=epochs, batch_size=bs)
                    param_grid = dict(hidden_layer_sizes=[(nn_L1), (nn_L2), (nn_L3)],
                                      learning_rate_init=[0.0001, 0.0005, 0.001])
                elif ml_algo == "NN2":
                    model = MLPRegressor(random_state=seed, max_iter=epochs, batch_size=bs)
                    param_grid = dict(hidden_layer_sizes=[(nn_L2, nn_L2),
                                                          (nn_L3, nn_L2),
                                                          (nn_L3, nn_L3)],
                                      learning_rate_init=[0.0001, 0.0005, 0.001])
                elif ml_algo == "NN3":
                    model = MLPRegressor(random_state=seed, max_iter=epochs, batch_size=bs)
                    param_grid = dict(hidden_layer_sizes=[(nn_L3, nn_L2, nn_L2),
                                                          (nn_L3, nn_L3, nn_L2),
                                                          (nn_L3, nn_L3, nn_L3)],
                                      learning_rate_init=[0.0001, 0.0005, 0.001])

                # K-fold cross validation
                kf = KFold(n_splits=6, shuffle=True, random_state=seed)

                # Perform grid search hyperparameter tuning
                grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf,
                                           scoring="neg_root_mean_squared_error",
                                           n_jobs=nprocs)
                grid_search.fit(X_scaled, y_scaled)

                # Define ML model with tuned hyperparameters
                if ml_algo == "KN":
                    model = KNeighborsRegressor(
                        n_neighbors=grid_search.best_params_["n_neighbors"],
                        weights=grid_search.best_params_["weights"]
                    )

                elif ml_algo == "RF":
                    model = RandomForestRegressor(
                        random_state=seed,
                        n_estimators=grid_search.best_params_["n_estimators"],
                        max_features=grid_search.best_params_["max_features"],
                        min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                        min_samples_split=grid_search.best_params_["min_samples_split"]
                    )

                elif ml_algo == "DT":
                    model = DecisionTreeRegressor(
                        random_state=seed,
                        splitter=grid_search.best_params_["splitter"],
                        max_features=grid_search.best_params_["max_features"],
                        min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                        min_samples_split=grid_search.best_params_["min_samples_split"]
                    )

                elif ml_algo in ["NN1", "NN2", "NN3"]:
                    model = MLPRegressor(
                        random_state=seed,
                        learning_rate_init=grid_search.best_params_["learning_rate_init"],
                        hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"]
                    )

                self.ml_model_tuned = True

            else:
                # Define ML models without tuning
                if ml_algo == "KN":
                    model = KNeighborsRegressor(n_neighbors=4, weights="distance")
                elif ml_algo == "RF":
                    model = RandomForestRegressor(random_state=seed, n_estimators=400,
                                                  max_features=2, min_samples_leaf=1,
                                                  min_samples_split=2)
                elif ml_algo == "DT":
                    model = DecisionTreeRegressor(random_state=seed, splitter="best",
                                                  max_features=2, min_samples_leaf=1,
                                                  min_samples_split=2)
                elif ml_algo == "NN1":
                    model = MLPRegressor(random_state=seed, max_iter=epochs,
                                         learning_rate_init=0.001,
                                         hidden_layer_sizes=(nn_L3))
                elif ml_algo == "NN2":
                    model = MLPRegressor(random_state=seed, max_iter=epochs,
                                         learning_rate_init=0.0001,
                                         hidden_layer_sizes=(nn_L3, nn_L3))
                elif ml_algo == "NN3":
                    model = MLPRegressor(random_state=seed, max_iter=epochs,
                                         learning_rate_init=0.0001,
                                         hidden_layer_sizes=(nn_L3, nn_L3, nn_L3))

            # Get trained model
            self.rocmlm = model

            # Get hyperparameters
            self.ml_model_hyperparams = model.get_params()

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _configure_rocmlm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
            return None

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
        batchprop = self.batchprop
        target_train = self.target_train
        feature_train = self.feature_train

        try:
            # Check for training features
            if feature_train.size == 0:
                raise Exception("No training features !")

            # Check for training targets
            if target_train.size == 0:
                raise Exception("No training targets !")

            # Scale training dataset
            X, y, _, scaler_y, X_scaled, y_scaled = \
                self._scale_arrays(feature_train, target_train)

            # Get fold indices
            (train_index, test_idxex) = fold_args

            # Split the data into training and testing sets
            X, X_test = X_scaled[train_index], X_scaled[test_idxex]
            y, y_test = y_scaled[train_index], y_scaled[test_idxex]

            if "NN" in ml_algo:
                # Initialize lists to store loss values
                epoch_, train_loss_, test_loss_ = [], [], []

                # Set batch size as a proportion of the training dataset size
                bs = max(int(len(y) * batchprop), 8)

                # Start training timer
                training_start_time = time.time()

                # Partial training
                with tqdm(total=epochs, desc="Training NN", position=0) as pbar:
                    for epoch in range(epochs):
                        # Shuffle the training data for each epoch
                        indices = np.arange(len(y))
                        np.random.shuffle(indices)

                        for start_idx in range(0, len(indices), bs):
                            end_idx = start_idx + bs

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
        sids = self.sids
        kfolds = self.kfolds
        verbose = self.verbose
        ml_algo = self.ml_algo
        M = self.shape_feature_square[0]
        w = self.shape_feature_square[1]
        ml_algo_label = self.ml_algo_label
        rocmlm_targets = self.rocmlm_targets

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
            if any("sm" in sample or "sr" in sample for sample in sids):
                sample_label = f"SMA{M}"
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
                print(f"{ml_algo_label} performance:")
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
        target_train = self.target_train
        feature_train = self.feature_train

        try:
            # Check for training features
            if feature_train.size == 0:
                raise Exception("No training features !")

            # Check for training targets
            if target_train.size == 0:
                raise Exception("No training targets !")

            # Scale training dataset
            X, _, _, _, _, _ = self._scale_arrays(feature_train, target_train)

            # Check for ml model
            if self.ml_algo is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            # K-fold cross validation
            kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

            # Create list of args for mp pooling
            fold_args = [(train_idx, test_idx) for _, (train_idx, test_idx) in
                         enumerate(kf.split(X))]

            # Create a multiprocessing pool
            with mp.Pool(processes=nprocs) as pool:
                results = pool.map(self._kfold_itr, fold_args)

                # Wait for all processes
                pool.close()
                pool.join()

            self.ml_model_cross_validated = True

            # Process cross validation results
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
        batchprop = self.batchprop
        model_prefix = self.model_prefix
        scaler_X_path = self.scaler_X_path
        scaler_y_path = self.scaler_y_path
        target_array = self.target_train.copy()
        feature_array = self.feature_train.copy()
        rocmlm_only_path = self.rocmlm_only_path
        shape_target_square = self.shape_target_square
        shape_feature_square = self.shape_feature_square

        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            if not self.ml_model_cross_validated:
                raise Exception("ML model not cross validated! Call _kfold_cv() first ...")

            if feature_array.size == 0:
                raise Exception("No training features !")

            if target_array.size == 0:
                raise Exception("No training targets !")

            print(f"Retraining model {model_prefix} ...")

            # Scale arrays
            _, _, scaler_X, scaler_y, X_scaled, y_scaled = \
                self._scale_arrays(feature_array, target_array)

            # Train model on entire training dataset
            X_train, X_test, y_train, y_test = \
                train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)

            # Train ML model
            if "NN" in ml_algo:
                # Set batch size as a proportion of the training dataset size
                bs = max(int(len(y_train) * batchprop), 8)

                # Partial training
                with tqdm(total=epochs, desc="Retraining NN", position=0) as pbar:
                    for epoch in range(epochs):
                        # Shuffle the training data for each epoch
                        indices = np.arange(len(y_train))
                        np.random.shuffle(indices)

                        for start_idx in range(0, len(indices), bs):
                            end_idx = start_idx + bs

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
                rocmlm.fit(X_train, y_train)

            print(":::::::::::::::::::::::::::::::::::::::::::::")
            self.ml_model_trained = True
            self.rocmlm_scaler_X = scaler_X
            self.rocmlm_scaler_y = scaler_y

            # Save pretrained rocmlm and scalers
            with open(rocmlm_only_path, "wb") as file:
                joblib.dump(rocmlm, file)
            with open(scaler_X_path, "wb") as file:
                joblib.dump(scaler_X, file)
            with open(scaler_y_path, "wb") as file:
                joblib.dump(scaler_y, file)

            # Add model size to cv_info
            model_size = os.path.getsize(rocmlm_only_path)
            model_size_mb = round(model_size / (1024 ** 2), 5)
            self.cv_info["model_size_mb"] = model_size_mb

            # Copy feature and target arrays
            X, y = feature_array.copy(), target_array.copy()

            # Scale features array
            X_scaled = scaler_X.transform(X)

            # Make predictions on features
            pred_scaled = rocmlm.predict(X_scaled)

            # Inverse transform predictions
            pred_original = scaler_y.inverse_transform(pred_scaled)

            # Reshape arrays into squares for visualization
            target_square = y.reshape(shape_target_square)
            feature_square = X.reshape(shape_feature_square)
            pred_square = pred_original.reshape(shape_target_square)

            # Update arrays
            self.target_square = target_square
            self.feature_square = feature_square
            self.prediction_square = pred_square

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
    # check rocmlm images !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_rocmlm_images(self, type="targets"):
        """
        """
        # Get self attributes
        sids = self.sids
        fig_dir = self.fig_dir
        model_prefix = self.model_prefix
        rocmlm_targets = self.rocmlm_targets

        if type not in {"targets", "predictions", "diff", "prem"}:
            raise Exception("Unrecognized array image type !")

        # Only visualize dry and max hydrated samples
        if any(s in sids for s in ["PUM", "DMM", "PYR"]):
            pass
        else:
            sids = [sid for sid in sids if
                    re.search(r"sm.*loi001", sid) or
                    re.search(r"sm.*loi007", sid)]

        missing_sids = []

        # Check for existing plots
        for sid in sids:
            all_files_exist = True
            for target in rocmlm_targets:
                path = f"{fig_dir}/{model_prefix}-{sid}-{target}-{type}.png"
                if not os.path.exists(path):
                    all_files_exist = False
                    break
            if not all_files_exist:
                missing_sids.append(sid)

        return missing_sids

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize loss curve !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_loss_curve(self, loss_curves, figwidth=6.3, figheight=3.54, fontsize=14):
        """
        """
        # Get self attributes
        fig_dir = self.fig_dir
        ml_algo = self.ml_algo
        model_label_full = self.ml_algo_label

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
        plt.savefig(f"{fig_dir}/{ml_algo}-loss-curve.png")

        # Close plot
        plt.close()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize array image  !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_image(self, sids, palette="bone", geotherms=True, type="targets",
                               figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get self attributes
        cv_info = self.cv_info
        fig_dir = self.fig_dir
        verbose = self.verbose
        target_units = self.target_units
        model_prefix = self.model_prefix
        target_arrays = self.target_square
        feature_arrays = self.feature_square
        pred_arrays = self.prediction_square
        rocmlm_targets = self.rocmlm_targets
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
                if verbose >= 1: print(f"  Visualizing {model_prefix}-{sid} ...")

                # Slice arrays
                feature_array = feature_arrays[s, :, :, :]
                target_array = target_arrays[s, :, :, :]
                pred_array = pred_arrays[s, :, :, :]

                for i, target in enumerate(rocmlm_targets):
                    # Get 2d arrays
                    P = feature_array[:, :, 0 + n_feats]
                    T = feature_array[:, :, 1 + n_feats]
                    t = target_array[:, :, i]
                    p = pred_array[:, :, i]
                    extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

                    if type == "targets":
                        square_array = t
                        filename = f"{model_prefix}-{sid}-{target}-targets.png"
                        rmse, r2 = None, None
                    elif type == "predictions":
                        square_array = p
                        filename = f"{model_prefix}-{sid}-{target}-predictions.png"
                        rmse, r2 = None, None
                    elif type == "diff":
                        mask = np.isnan(t)
                        p[mask] = np.nan
                        square_array = t - p
                        square_array[mask] = np.nan
                        rmse = cv_info[f"rmse_test_mean_{target}"]
                        r2 = cv_info[f"r2_test_mean_{target}"]
                        palette = "seismic"
                        filename = f"{model_prefix}-{sid}-{target}-diff.png"
                    else:
                        raise Exception("Unrecognized array image type !")

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

                            # Set rho to 1.8–4.6 g/cm3
                            if target == "rho": vmin, vmax = 1.8, 4.6

                            # Set h2o fraction to 0–100 wt.%
                            if target == "h2o": vmin, vmax = 0, 2

                            # Set melt fraction to 0–100 vol.%
                            if target == "melt": vmin, vmax = 0, 100

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
                        elif target == "h2o":
                            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                        elif target == "melt":
                            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
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
    def _visualize_prem(self, sids, type="targets", geotherms=["low", "mid", "high"],
                        figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        # Get model data
        res = self.res
        P_min = self.P_min
        P_max = self.P_max
        fig_dir = self.fig_dir
        verbose = self.verbose
        data_dir = self.data_dir
        target_units = self.target_units
        model_prefix = self.model_prefix
        target_arrays = self.target_square
        feature_arrays = self.feature_square
        pred_arrays = self.prediction_square
        rocmlm_targets = self.rocmlm_targets
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
        df_mids = pd.read_csv("assets/synth-mids.csv")
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
                if verbose >= 1: print(f"  Visualizing {model_prefix}-{sid} ...")

                # Slice arrays
                feature_array = feature_arrays[s, :, :, :]
                target_array = target_arrays[s, :, :, :]
                pred_array = pred_arrays[s, :, :, :]

                for i, target in enumerate(rocmlm_targets):
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
                        raise Exception("Unrecognized array image type !")

                    # Check for all nans
                    if np.all(np.isnan(square_array)):
                        square_array = np.nan_to_num(square_array)

                    filename = f"{model_prefix}-{sid}-{target}-prem.png"

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
                        ax1.set_xlim(None, 2)
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
    # visualize rocmlm !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_rocmlm(self):
        """
        """
        # Get self attributes
        sids = self.sids
        S = self.shape_feature_square[0]
        W = self.shape_feature_square[1]

        try:
            sids = self._check_rocmlm_images(type="targets")
            if sids: self._visualize_array_image(sids, type="targets")
            sids = self._check_rocmlm_images(type="predictions")
            if sids: self._visualize_array_image(sids, type="predictions")
            sids = self._check_rocmlm_images(type="diff")
            if sids: self._visualize_array_image(sids, type="diff")
            sids = self._check_rocmlm_images(type="prem")
            if sids: self._visualize_prem(sids, type="predictions")
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in visualize_rocmlm() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
        else:
            return None

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
            if self.ml_model_trained:
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
                    self.ml_model_training_error = True
                    self.ml_model_error = e
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
        scaler_X = self.ml_model_scaler_X
        scaler_y = self.ml_model_scaler_y
        ml_model_trained = self.ml_model_trained

        # Check for pretrained model
        if not ml_model_trained:
            raise Exception("No RocMLM model! Call train_rocmlm() first ...")

        try:
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

    def predict_old(self, xi, P, T):
        """
        """
        # Get self attributes
        model = self.ml_algo
        scaler_X = self.ml_model_scaler_X
        scaler_y = self.ml_model_scaler_y
        ml_model_trained = self.ml_model_trained

        # Check for pretrained model
        if not ml_model_trained:
            raise Exception("No RocMLM model! Call train_rocmlm() first ...")

        try:
            for var in [xi, P, T]:
                if not isinstance(var, (list, np.ndarray)):
                    raise TypeError(f"Inputs must be lists or numpy arrays !")

            # Create feature array
            if all(var is not None for var in [xi, P, T]):
                xi_array = np.asarray(xi)
                P_array = np.asarray(P)
                T_array = np.asarray(T)

                if (xi_array.shape[0] == P_array.shape[0] == T_array.shape[0]):
                    xi_array = xi_array.reshape(-1, 1)
                    P_array = P_array.reshape(-1, 1)
                    T_array = T_array.reshape(-1, 1)

                    X = np.concatenate((xi_array, P_array, T_array), axis=1)
                else:
                    raise Exception("The arrays xi, P, and T are not the same size !")
            else:
                raise Exception("One or more of xi, P, T does not exist !")

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
# compose rocmlm img3 itr !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocmlm_img3_itr(args):
    # Unpack args
    sid, target, rocmlm = args

    # Get rocmlm attributes
    fig_dir = rocmlm.fig_dir
    verbose = rocmlm.verbose
    model_prefix = rocmlm.model_prefix
    ml_algo = rocmlm.ml_algo
    rocmlm_targets = rocmlm.rocmlm_targets

    # Unique pid
    pid = os.getpid()

    # Define temp paths
    temp = f"{fig_dir}/temp-{pid}-{sid}-{target}.png"

    # Define fig paths
    fig_trg = f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png"
    fig_pred = f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png"
    fig_diff = f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png"
    fig_prem = f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png"

    # Define composition paths
    img3_diff = f"{fig_dir}/image3-{sid}-{ml_algo}-{target}-diff.png"
    img3_prof = f"{fig_dir}/image3-{sid}-{ml_algo}-{target}-profile.png"

    # Check for existing plots
    fig_1 = f"{fig_dir}/image-{sid}-{ml_algo}-{target}-diff.png"
    fig_2 = f"{fig_dir}/image-{sid}-{ml_algo}-{target}-profile.png"
    if (os.path.exists(fig_1) and os.path.exists(fig_2)):
        if os.path.exists(temp): os.remove(temp)
        return None

    if verbose >= 1: print(f"Composing {model_prefix}-{sid}-{target} ...")

    # Image3 diff
    combine_plots_horizontally(fig_trg, fig_pred, temp, caption1="a)", caption2="b)")
    combine_plots_horizontally(temp, fig_diff, img3_diff, caption1="", caption2="c)")

    # Image3 profile
    combine_plots_horizontally(fig_trg, fig_pred, temp, caption1="a)", caption2="b)")
    combine_plots_horizontally(temp, fig_prem, img3_prof, caption1="", caption2="c)")

    # Clean up temporary files created by this worker
    if os.path.exists(temp): os.remove(temp)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose rocmlm img9 itr !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocmlm_img9_itr(args):
    # Unpack args
    sid, rocmlm = args

    # Get rocmlm attributes
    fig_dir = rocmlm.fig_dir
    verbose = rocmlm.verbose
    model_prefix = rocmlm.model_prefix
    ml_algo = rocmlm.ml_algo

    # Unique pid
    pid = os.getpid()

    # Define temp paths
    temp = f"{fig_dir}/temp-{pid}-{sid}.png"
    temp_rho = f"{fig_dir}/temp-rho-{pid}-{sid}.png"
    temp_h2o = f"{fig_dir}/temp-h2o-{pid}-{sid}.png"
    temp_melt = f"{fig_dir}/temp-melt-{pid}-{sid}.png"

    # Define composition paths
    img9_diff = f"{fig_dir}/image9-{sid}-{ml_algo}-diff.png"
    img9_prof = f"{fig_dir}/image9-{sid}-{ml_algo}-profile.png"

    # Check for existing plots
    fig_1 = f"{fig_dir}/image9-{sid}-{ml_algo}-diff.png"
    fig_2 = f"{fig_dir}/image9-{sid}-{ml_algo}-profile.png"
    if (os.path.exists(fig_1) and os.path.exists(fig_2)):
        for file in [temp, temp_rho, temp_h2o, temp_melt]:
            if os.path.exists(file): os.remove(file)
        return None

    # Image9 diff
    captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
    for i, target in enumerate(["rho", "h2o", "melt"]):
        if verbose >= 1: print(f"Composing {model_prefix}-{sid}-{target} ...")
        fig_trg = f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png"
        fig_pred = f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png"
        fig_diff = f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png"
        fig_prem = f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png"
        temp1 = f"{fig_dir}/temp-{pid}-{sid}-{target}.png"
        temp2 = f"{fig_dir}/temp-{target}-{pid}-{sid}.png"
        combine_plots_horizontally(fig_trg, fig_pred, temp1, caption1=captions[i][0],
                                   caption2=captions[i][1])
        combine_plots_horizontally(temp1, fig_diff, temp2, caption1="",
                                   caption2=captions[i][2])

    combine_plots_vertically(temp_rho, temp_h2o, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_melt, img9_diff, caption1="", caption2="")

    for i, target in enumerate(["rho", "h2o", "melt"]):
        if verbose >= 1: print(f"Composing {model_prefix}-{sid}-{target} ...")
        fig_trg = f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png"
        fig_pred = f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png"
        fig_diff = f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png"
        fig_prem = f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png"
        temp1 = f"{fig_dir}/temp-{pid}-{sid}-{target}.png"
        temp2 = f"{fig_dir}/temp-{target}-{pid}-{sid}.png"
        combine_plots_horizontally(fig_trg, fig_pred, temp1, caption1=captions[i][0],
                                   caption2=captions[i][1])
        combine_plots_horizontally(temp1, fig_prem, temp2, caption1="",
                                   caption2=captions[i][2])

    combine_plots_vertically(temp_rho, temp_h2o, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_melt, img9_prof, caption1="", caption2="")

    # Clean up temporary files created by this worker
    for file in [temp, temp1, temp2, temp_rho, temp_h2o, temp_melt]:
        if os.path.exists(file): os.remove(file)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose rocmlm img15 itr !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocmlm_img15_itr(args):
    # Unpack args
    sid, rocmlm = args

    # Get rocmlm attributes
    fig_dir = rocmlm.fig_dir
    verbose = rocmlm.verbose
    model_prefix = rocmlm.model_prefix
    ml_algo = rocmlm.ml_algo

    # Unique pid
    pid = os.getpid()

    # Define temp paths
    temp = f"{fig_dir}/temp-{pid}-{sid}.png"
    temp_rho = f"{fig_dir}/temp-rho-{pid}-{sid}.png"
    temp_vp = f"{fig_dir}/temp-vp-{pid}-{sid}.png"
    temp_vs = f"{fig_dir}/temp-vs-{pid}-{sid}.png"
    temp_h2o = f"{fig_dir}/temp-h2o-{pid}-{sid}.png"
    temp_melt = f"{fig_dir}/temp-melt-{pid}-{sid}.png"

    # Define composition paths
    img15_diff = f"{fig_dir}/image15-{sid}-{ml_algo}-diff.png"
    img15_prof = f"{fig_dir}/image15-{sid}-{ml_algo}-profile.png"

    # Check for existing plots
    fig_1 = f"{fig_dir}/image15-{sid}-{ml_algo}-diff.png"
    fig_2 = f"{fig_dir}/image15-{sid}-{ml_algo}-profile.png"
    if (os.path.exists(fig_1) and os.path.exists(fig_2)):
        for file in [temp, temp_rho, temp_h2o, temp_melt]:
            if os.path.exists(file): os.remove(file)
        return None

    # Image15 diff
    captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)"),
                ("j)", "k)", "l)"), ("m)", "n)", "o)")]
    for i, target in enumerate(["rho", "vp", "vs", "h2o", "melt"]):
        if verbose >= 1: print(f"Composing {model_prefix}-{sid}-{target} ...")
        fig_trg = f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png"
        fig_pred = f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png"
        fig_diff = f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png"
        fig_prem = f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png"
        temp1 = f"{fig_dir}/temp-{pid}-{sid}-{target}.png"
        temp2 = f"{fig_dir}/temp-{target}-{pid}-{sid}.png"
        combine_plots_horizontally(fig_trg, fig_pred, temp1, caption1=captions[i][0],
                                   caption2=captions[i][1])
        combine_plots_horizontally(temp1, fig_diff, temp2, caption1="",
                                   caption2=captions[i][2])

    combine_plots_vertically(temp_rho, temp_vp, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_vs, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_h2o, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_melt, img15_diff, caption1="", caption2="")

    for i, target in enumerate(["rho", "vp", "vs", "h2o", "melt"]):
        if verbose >= 1: print(f"Composing {model_prefix}-{sid}-{target} ...")
        fig_trg = f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png"
        fig_pred = f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png"
        fig_diff = f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png"
        fig_prem = f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png"
        temp1 = f"{fig_dir}/temp-{pid}-{sid}-{target}.png"
        temp2 = f"{fig_dir}/temp-{target}-{pid}-{sid}.png"
        combine_plots_horizontally(fig_trg, fig_pred, temp1, caption1=captions[i][0],
                                   caption2=captions[i][1])
        combine_plots_horizontally(temp1, fig_prem, temp2, caption1="",
                                   caption2=captions[i][2])

    combine_plots_vertically(temp_rho, temp_vp, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_vs, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_h2o, temp, caption1="", caption2="")
    combine_plots_vertically(temp, temp_melt, img15_prof, caption1="", caption2="")

    # Clean up temporary files created by this worker
    for file in [temp, temp1, temp2, temp_rho, temp_vp, temp_vs, temp_h2o, temp_melt]:
        if os.path.exists(file): os.remove(file)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose rocmlm plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocmlm_plots(rocmlm):
    """
    """
    # Get rocmlm attributes
    sids = rocmlm.sids
    nprocs = rocmlm.nprocs
    fig_dir = rocmlm.fig_dir
    S = rocmlm.shape_feature_square[0]
    W = rocmlm.shape_feature_square[1]
    rocmlm_targets = rocmlm.rocmlm_targets

    # Only visualize largets models
#    if ((S == 323 and W == 129) or
#        (S == 3 and W == 129 and any(s in sids for s in ["PUM", "DMM", "PYR"]))):
    if ((S == 248 and W == 65) or
        (S == 3 and W == 65 and any(s in sids for s in ["PUM", "DMM", "PYR"]))):

        # Only visualize dry and max hydrated samples
        if any(s in sids for s in ["PUM", "DMM", "PYR"]):
            pass
        else:
            sids = [sid for sid in sids if
                    re.search(r"sm.*loi001", sid) or
                    re.search(r"sm.*loi007", sid)]

        # Create list of args for mp pooling
        args_img3 = [(sid, target, rocmlm) for sid in sids for target in rocmlm_targets]
        args_img9 = [(sid, rocmlm) for sid in sids]
        args_img15 = [(sid, rocmlm) for sid in sids]

        print(f"Composing RocMLM plots for {len(sids)} samples:")
        print(sids)

        # Create a multiprocessing pool
        with mp.Pool(processes=nprocs) as pool:
            pool.map(compose_rocmlm_img3_itr, args_img3)

            # Wait for all processes
            pool.close()
            pool.join()

        # Create a multiprocessing pool
        with mp.Pool(processes=nprocs) as pool:
            pool.map(compose_rocmlm_img9_itr, args_img9)

            # Wait for all processes
            pool.close()
            pool.join()

        # Create a multiprocessing pool
        with mp.Pool(processes=nprocs) as pool:
            pool.map(compose_rocmlm_img15_itr, args_img15)

            # Wait for all processes
            pool.close()
            pool.join()

        tmp_files = glob.glob(f"{fig_dir}/temp*.png")
        for file in tmp_files: os.remove(file)
    else:
        return None

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocmlm performance !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocmlm_performance(fig_dir="figs/other", filename="rocmlm-performance.png",
                                 figwidth=6.3, figheight=2.5, fontsize=12):
    """
    """
    # Data assets dir
    data_dir = "assets"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read gfem efficiency data
    gfem_summaries = os.listdir(f"{data_dir}/gfem_summaries")
    csv_files = [f for f in gfem_summaries if f.endswith(".csv")]

    gfem_dfs = []
    for file in csv_files:
        file_path = os.path.join(f"{data_dir}/gfem_summaries", file)
        df = pd.read_csv(file_path)
        gfem_dfs.append(df)

    data_gfem = pd.concat(gfem_dfs, ignore_index=True)
    data_gfem = data_gfem[data_gfem["DATABASE"] == "hp633"]

    data_gfem.rename(columns={"COMP_TIME": "time", "SIZE": "size"}, inplace=True)
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

    # Add RMSE
    data_gfem["rmse_test_mean_rho"] = 0
    data_gfem["rmse_test_mean_h2o"] = 0
    data_gfem["rmse_test_mean_melt"] = 0

    # Read lookup table efficiency data
    data_lut = pd.read_csv(f"{data_dir}/lut-efficiency.csv")

    # Add RMSE
    data_lut["rmse_test_mean_rho"] = 0
    data_lut["rmse_test_mean_h2o"] = 0
    data_lut["rmse_test_mean_melt"] = 0

    # Calculate efficiency in milliseconds/Megabyte
    data_lut["model_efficiency"] = data_lut["time"] * 1e3 * data_lut["model_size_mb"]

    # Read rocmlm efficiency data
    data_rocmlm = pd.read_csv(f"{data_dir}/rocmlm-performance.csv")

    # Process rocmlm df for merging
    data_rocmlm.drop([col for col in data_rocmlm.columns if "r2" in col],
                     axis=1, inplace=True)
    data_rocmlm.drop(["n_targets", "k_folds", "inference_time_std"], axis=1, inplace=True)

    # Combine model with rocmlm
    def label_rocmlm_model(row):
        return f"RocMLM ({row["model"]})"

    data_rocmlm["model"] = data_rocmlm.apply(label_rocmlm_model, axis=1)
    data_rocmlm.rename(columns={"inference_time_mean": "time"}, inplace=True)

    # Calculate efficiency in milliseconds/Megabyte
    data_rocmlm["model_efficiency"] = (data_rocmlm["time"] * 1e3 *
                                       data_rocmlm["model_size_mb"])

    # Select columns
    data_rocmlm = data_rocmlm[["sample", "model", "size", "time", "model_size_mb",
                               "model_efficiency", "rmse_test_mean_rho",
                               "rmse_test_mean_h2o", "rmse_test_mean_melt"]]

    # Combine data
    data = pd.concat([data_lut, data_rocmlm], axis=0, ignore_index=True)

    # Relabel models
    def label_models(row):
        if row["model"] == "gfem":
            return "Perple_X"
        elif row["model"] == "lut":
            return "Lookup Table"
        else:
            return row["model"]

    data["model"] = data.apply(label_models, axis=1)

    # Filter samples and models
    data = data[data["sample"].isin(["SYNTH248", "SYNTH124", "SYNTH62", "SYNTH31",
                                     "SYNTH16"])]
#    data = data[data["sample"].isin(["SYNTH323", "SYNTH162", "SYNTH81", "SYNTH41",
#                                     "SYNTH21"])]
    data = data[data["model"].isin(["Lookup Table", "RocMLM (DT)", "RocMLM (KN)",
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

    # Calculate model size (capacity)
    data["x_res"] = data.apply(get_x_res, axis=1)
    data["size"] = np.log2(data["size"] * data["x_res"]).astype(int)

    # Arrange data by resolution and sample
    data.sort_values(by=["size", "sample", "model"], inplace=True)

    # Group by size and select min time
    grouped_data = data.groupby(["model", "size"])
    min_time_indices = grouped_data["time"].idxmin()
    data = data.loc[min_time_indices]
    data.reset_idxex(drop=True, inplace=True)

    # Compute summary statistics
    summary_stats = data.groupby(["model"]).agg({
        "time": ["mean", "std", "min", "max"],
        "rmse_test_mean_rho": ["mean", "std", "min", "max"],
        "rmse_test_mean_h2o": ["mean", "std", "min", "max"],
        "rmse_test_mean_melt": ["mean", "std", "min", "max"],
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

    # Define marker types for each model
    marker_dict = {"Lookup Table": "o", "RocMLM (DT)": "d", "RocMLM (KN)": "s",
                   "RocMLM (NN1)": "X", "RocMLM (NN3)": "^"}

    # Create a colormap
    palette = sns.color_palette("mako_r", data["rmse_test_mean_rho"].nunique())
    cmap = plt.get_cmap("mako_r")

    # Create a ScalarMappable to map rmse to colors
    norm = Normalize(data["rmse_test_mean_rho"].min(), data["rmse_test_mean_rho"].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Map X resolution to colors
    color_dict = dict(zip(sorted(data["rmse_test_mean_rho"].unique()),
                          cmap(norm(sorted(data["rmse_test_mean_rho"].unique())))))

    fig = plt.figure(figsize=(figwidth * 2, figheight))
    ax = fig.add_subplot(121)

    # Plot gfem efficiency
    ax.fill_between(data["size"], data_gfem["time"].min() * 1e3,
                    data_gfem["time"].max() * 1e3, facecolor="white", edgecolor="black")

#    plt.text(data["size"].min(), data_gfem["time"].max() * 1e3 * 0.9,
#             " GFEM [Perple_X]", fontsize=fontsize * 0.833,
#             horizontalalignment="left", verticalalignment="top")

    # Plot rocmlm efficiency
    for model, group in data.groupby("model"):
        for rmse, sub_group in group.groupby("rmse_test_mean_rho"):
            if model in ["RocMLM (DT)", "RocMLM (KN)", "RocMLM (NN1)", "RocMLM (NN3)"]:
                ax.scatter(x=sub_group["size"], y=(sub_group["time"] * 1e3),
                           marker=marker_dict.get(model, "o"), s=65,
                           color=color_dict[rmse], edgecolor="black", zorder=2)
            else:
                ax.scatter(x=sub_group["size"], y=(sub_group["time"] * 1e3),
                           marker=marker_dict.get(model, "o"), s=65,
                           color="pink", edgecolor="black", zorder=99)

    # Set labels and title
    plt.xlabel("Log2 Capacity")
    plt.ylabel("Elapsed Time (ms)")
    plt.title("Execution Speed")
    plt.yscale("log")
    plt.xticks(np.arange(7, 19, 2))
#    plt.xticks(np.arange(10, 23, 2))

    ax2 = fig.add_subplot(122)

    # Plot gfem efficiency
    ax2.fill_between(data["size"], data_gfem["model_size_mb"].min(),
                     data_gfem["model_size_mb"].max(), facecolor="white",
                     edgecolor="black")

    # Plot lut and rocmlm efficiency
    for model, group in data.groupby("model"):
        for rmse, sub_group in group.groupby("rmse_test_mean_rho"):
            if model in ["RocMLM (DT)", "RocMLM (KN)", "RocMLM (NN1)", "RocMLM (NN3)"]:
                ax2.scatter(x=sub_group["size"], y=sub_group["model_size_mb"],
                            marker=marker_dict.get(model, "o"), s=65,
                            color=color_dict[rmse], edgecolor="black", zorder=2)
            else:
                ax2.scatter(x=sub_group["size"], y=sub_group["model_size_mb"],
                            marker=marker_dict.get(model, "o"), s=65,
                            color="pink", edgecolor="black", zorder=99)

    # Create the legend
    legend_elements = []
    for model, marker in marker_dict.items():
        if model in ["RocMLM (DT)", "RocMLM (KN)", "RocMLM (NN1)", "RocMLM (NN3)"]:
            legend_elements.append(
                mlines.Line2D(
                    [0], [0], marker=marker, color="none", label=model,
                    markerfacecolor="black", markersize=10)
            )
        else:
            legend_elements.append(
                mlines.Line2D(
                    [0], [0], marker=marker, color="none", markeredgecolor="black",
                    label=model, markerfacecolor="pink", markersize=10)
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
    plt.ylabel("Size (Mb)")
    plt.title("Model Size")
    plt.yscale("log")
    plt.xticks(np.arange(7, 19, 2))
#    plt.xticks(np.arange(10, 23, 2))

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
def train_rocmlms(gfem_models, ml_algos=["DT", "KN", "NN1", "NN2", "NN3"],
                  PT_steps=[16, 8, 4, 2, 1], X_steps=[16, 8, 4, 2, 1]):
    """
    """
    # Check for gfem models
    if not gfem_models:
        raise Exception("No GFEM models to compile !")

    # Single X step for benchmark models
    sids = [m.sid for m in gfem_models]
    if any(s in sids for s in ["PUM", "DMM", "PYR"]):
        X_steps = [1]

    # Train rocmlm models at various PTX grid resolution levels
    try:
        rocmlms = []
        for X_step in X_steps:
            for PT_step in PT_steps:
                mlms = []
                for model in ml_algos:
                    rocmlm = RocMLM(gfem_models, model, X_step, PT_step)
                    if rocmlm.ml_model_trained:
                        model = joblib.load(rocmlm.rocmlm_path)
                        mlms.append(model)
                    else:
                        rocmlm.train_rocmlm()
                        mlms.append(rocmlm)
                        with open(rocmlm.rocmlm_path, "wb") as file:
                            joblib.dump(rocmlm, file)

                # Compile rocmlms
                rocmlms.extend(mlms)

    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in train_rocmlms() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()

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
    from gfem import get_sampleids, build_gfem_models
    try:
        # Build GFEM models
        gfems = {}
        sources = {"b": "assets/bench-pca.csv",
                   "m": "assets/synth-mids.csv",
                   "r": "assets/synth-rnds.csv"}

        for name, source in sources.items():
            sids = get_sampleids(source, "all")
            gfems[name] = build_gfem_models(source, sids, res=64)

        # Combine synthetic models for RocMLM training
        training_data = {"b": gfems["b"], "s": gfems["m"] + gfems["r"]}

        # Train RocMLMs
        rocmlms = {}
        for name, models in training_data.items():
            rocmlms[name] = train_rocmlms(models)

        # Visualize RocMLMs
        visualize_rocmlm_performance()

        for name, models in rocmlms.items():
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