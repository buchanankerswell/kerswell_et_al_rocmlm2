#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import time
import shutil
import joblib
import traceback
import itertools
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
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

#######################################################
## .1.               RocMLM Class                !!! ##
#######################################################
class RocMLM:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, gfem_models, ml_model, training_features=["XI_FRAC"],
                 training_targets=["rho", "h2o"], X_step=1, PT_step=1, tune=True, epochs=100,
                 batchprop=0.2, kfolds=5, nprocs=os.cpu_count() - 2, seed=42, verbose=1):
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
        self.res = None
        self.targets = None
        self.features = None
        self.sample_ids = []
        self.shape_target = None
        self.target_train = None
        self.feature_train = None
        self.shape_feature = None
        self.shape_target_square = None
        self.shape_feature_square = None

        # Get gfem model metadata
        self._get_gfem_model_metadata()

        # Output filepaths
        self.model_out_dir = f"rocmlms"
        if any(sample in sample_ids for sample in ["PUM", "DMM", "PYR"]):
            self.model_prefix = (f"benchmark-{self.ml_model_label}-"
                                 f"S{self.shape_feature_square[0]}-"
                                 f"W{self.shape_feature_square[1]}")
            self.fig_dir = f"figs/rocmlm/benchmark_{self.ml_model_label}"
        elif any("sm" in sample or "sr" in sample for sample in sample_ids):
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
        targets = self.targets
        feat_train = self.feature_train
        target_train = self.target_train
        ml_model_label = self.ml_model_label
        model_hyperparams = self.model_hyperparams
        ml_model_label_full = self.ml_model_label_full

        # Print rocmlm config
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("RocMLM model defined as:")
        print(f"    model: {ml_model_label_full}")
        if "NN" in ml_model_label:
            print(f"    epochs: {epochs}")
        print(f"    k folds: {kfolds}")
        print(f"    targets: {targets}")
        print(f"    features array shape: {feat_train.shape}")
        print(f"    targets array shape: {target_train.shape}")
        print(f"    hyperparameters:")
        for key, value in model_hyperparams.items():
            print(f"        {key}: {value}")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Running kfold ({kfolds}) cross validation ...")

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
    def _get_unique_value(input_list):
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
            sample_ids = [m.sample_id for m in gfem_models]
            res = _get_unique_value([m.res for m in gfem_models])
            targets = _get_unique_value([m.targets for m in gfem_models])
            features = _get_unique_value([m.features for m in gfem_models])
            oxides = _get_unique_value([m.oxides_system for m in gfem_models])
            oxides_exclude = _get_unique_value([m.oxides_exclude for m in gfem_models])
            subset_oxides = [oxide for oxide in oxides if oxide not in oxides_exclude]
            feature_list = subset_oxides + features

            self.res = res
            self.targets = targets
            self.features = feature_list

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
        X_step = self.X_step
        PT_step = self.PT_step
        targets = self.targets
        features = self.features
        sample_ids = self.sample_ids
        gfem_models = self.gfem_models
        training_targets = self.training_targets
        training_features = self.training_features

        try:
            # Get all PT arrays
            pt_train = np.stack([m.feature_array for m in gfem_models])

            # Select features
            feature_indices = [i for i, ft in enumerate(features) if ft in training_features]

            # Get sample features
            feat_train = []

            for m in gfem_models:
                selected_features = [m.sample_features[i] for i in feature_indices]
                feat_train.append(selected_features)

            feat_train = np.array(feat_train)

            # Tile features to match PT array shape
            feat_train = np.tile(feat_train[:, np.newaxis, :], (1, pt_train.shape[1], 1))

            # Combine features
            combined_train = np.concatenate((feat_train, pt_train), axis=2)

            # Flatten features
            feature_train = combined_train.reshape(-1, combined_train.shape[-1])

            # Define target indices
            target_indices = [targets.index(target) for target in training_targets]
            targets = [target for target in training_targets]

            # Get target arrays
            target_train = np.stack([m.target_array for m in gfem_models])

            # Flatten targets
            target_train = target_train.reshape(-1, target_train.shape[-1])

            # Select training targets
            target_train = target_train[:, target_indices]

            # Define array shapes
            M = int(len(gfem_models))
            W = int((res + 1) ** 2)
            w = int(np.sqrt(W))
            F = int(len(training_features) + 2)
            T = int(len(targets))
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
        X, y = X[~mask,:], y[~mask,:]

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
        X_step = self.X_step
        PT_step = self.PT_step
        sample_ids = self.sample_ids
        gfem_models = self.gfem_models
        target_train = self.target_train

        # Check benchmark models
        if any(sample in sample_ids for sample in ["PUM", "DMM", "PYR"]):
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
            if name == "top":
                sample.append(f"SMAT{X_sub.shape[0]}")
            elif name == "middle":
                sample.append(f"SMAM{X_sub.shape[0]}")
            elif name == "bottom":
                sample.append(f"SMAB{X_sub.shape[0]}")
            elif name == "random":
                sample.append(f"SMAR{X_sub.shape[0]}")
            elif name == "synthetic":
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
            print("Tuning RocMLM model ...")

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

            print("Tuning successful!")

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

        print("Configuring successful!")

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
        kfolds = self.kfolds
        targets = self.targets
        fig_dir = self.fig_dir
        verbose = self.verbose
        sample_ids = self.sample_ids
        M = self.shape_feature_square[0]
        w = self.shape_feature_square[1]
        model_label = self.ml_model_label
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
            fig = plt.figure(figsize=(6.3, 3.54))

            # Colormap
            colormap = plt.cm.get_cmap("tab10")

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
        if any(sample in sample_ids for sample in ["PUM", "DMM", "PYR"]):
            sample_label = "benchmark"
        elif any("sm" in sample or "sr" in sample for sample in sample_ids):
            sample_label = f"SYNTH{M}"
        elif any("st" in sample for sample in sample_ids):
            sample_label = f"SMAT{M}"
        elif any("sm" in sample for sample in sample_ids):
            sample_label = f"SMAM{M}"
        elif any("sb" in sample for sample in sample_ids):
            sample_label = f"SMAB{M}"
        elif any("sr" in sample for sample in sample_ids):
            sample_label = f"SMAR{M}"

        # Config and performance info
        cv_info = {
            "model": [model_label],
            "sample": [sample_label],
            "size": [(w - 1) ** 2],
            "n_targets": [len(targets)],
            "k_folds": [kfolds],
            "training_time_mean": [round(training_time_mean, 5)],
            "training_time_std": [round(training_time_std, 5)],
            "inference_time_mean": [round(inference_time_mean, 5)],
            "inference_time_std": [round(inference_time_std, 5)]
        }

        # Add performance metrics for each parameter to the dictionary
        for i, target in enumerate(targets):
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
            for r, e, p in zip(rmse_test_mean, rmse_test_std, targets):
                print(f"        {p}: {r:.5f} ± {e:.5f}")
            print(f"    r2 test:")
            for r, e, p in zip(r2_test_mean, r2_test_std, targets):
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

        print("Kfold cross validation successful!")
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

        print("Retraining successful!")
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
    # visualize model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_model(self):
        """
        """
        try:
            if not self._check_image():
                self._visualize_image()
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
        self._print_rocmlm_info()
        max_retries = 3
        for retry in range(max_retries):
            # Check for built model
            if self.model_built:
                break
            try:
                self._process_training_data()
                self._evaluate_lut_performance()
                self._configure_rocmlm()
                self._kfold_cv()
                self._retrain()
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

        return None

#######################################################
## .2.                Visualize                  !!! ##
#######################################################

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

    # Train rocmlm models at various PTX grid resolution levels
    for X_step in X_steps:
        for PT_step in PT_steps:
            rocmlms = []
            for model in ml_models:
                rocmlm = RocMLM(gfem_models, model)
                rocmlm._check_pretrained_model()
                if rocmlm.ml_model_trained:
                    pretrained_rocmlm = joblib.load(rocmlm.rocmlm_path)
                    rocmlms.append(pretrained_rocmlm)
                else:
                    rocmlm.train_rocmlm()
                    rocmlms.append(rocmlm)
                    with open(rocmlm.rocmlm_path, "wb") as file:
                        joblib.dump(rocmlm, file)

            # Get successful models
            rocmlms = [m for m in rocmlms if not m.ml_model_training_error]

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
                    visualize_rocmlm(model)
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