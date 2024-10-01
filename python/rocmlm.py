#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import yaml
import time
import joblib
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
from matplotlib.colors import ListedColormap

#######################################################
## .1.               RocMLM Class                !!! ##
#######################################################
class RocMLM:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, gfem_models, ml_algo="DT", tune=False, config_yaml=None, verbose=1):
        """
        """
        self.tune = tune
        self.ml_algo = ml_algo
        self.verbose = verbose
        self.config_yaml = config_yaml
        self.gfem_models = sorted(gfem_models, key=lambda model: model.sid)

        self.model_trained = False
        self.model_cross_validated = False
        self.model_out_dir = "rocmlms"

        self._load_global_options()
        self._load_rocmlm_options()
        self._get_gfem_model_metadata()
        self._get_rocmlm_training_data()

        if self.ml_algo == "NN":
            self.model_prefix = (f"{self.ml_algo}{self.sum_nodes}-"
                                 f"S{self.rocmlm_feature_array_shape_square[0]}-"
                                 f"W{self.rocmlm_feature_array_shape_square[1]}-"
                                 f"F{self.rocmlm_feature_array_shape_square[3]}-"
                                 f"F{len(self.rocmlm_targets)}-"
                                 f"{self.gfem_db}")
        else:
            self.model_prefix = (f"{self.ml_algo}-"
                                 f"S{self.rocmlm_feature_array_shape_square[0]}-"
                                 f"W{self.rocmlm_feature_array_shape_square[1]}-"
                                 f"F{self.rocmlm_feature_array_shape_square[3]}-"
                                 f"F{len(self.rocmlm_targets)}-"
                                 f"{self.gfem_db}")
        self.fig_dir = (f"figs/rocmlm/{self.model_prefix}")
        self.rocmlm_path = f"{self.model_out_dir}/{self.model_prefix}-pretrained.pkl"
        self.scaler_X_path = f"{self.model_out_dir}/{self.model_prefix}-scaler_X.pkl"
        self.scaler_y_path = f"{self.model_out_dir}/{self.model_prefix}-scaler_y.pkl"

        self._check_existing_model()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_global_options(self):
        """
        """
        try:
            if self.config_yaml:
                if not os.path.exists(self.config_yaml):
                    raise Exception(f"No config_yaml found at {self.config_yaml}!")
                with open(self.config_yaml, "r") as file:
                    config_data = yaml.safe_load(file)
                global_options = config_data["global_options"]
            else:
                global_options = {
                    "seed": 42,
                    "digits": 3,
                    "nprocs": os.cpu_count() - 2,
                    "pot_Ts": [1173, 1573, 1773],
                    "segs": ["Central_Cascadia", "Kamchatka"]
                }

            # Assign values from global options
            self.seed = global_options["seed"]
            self.digits = global_options["digits"]
            self.nprocs = global_options["nprocs"]
            self.pot_Ts = global_options["pot_Ts"]
            self.segs = global_options["segs"]

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
            print(f"Error in _load_global_options():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_rocmlm_options(self):
        """
        """
        try:
            if self.config_yaml:
                if not os.path.exists(self.config_yaml):
                    raise Exception(f"No config_yaml found at {self.config_yaml}!")
                with open(self.config_yaml, "r") as file:
                    config_data = yaml.safe_load(file)
                rocmlm_options = config_data["rocmlm_options"]
                ml_algorithms = config_data["ml_algorithms"]
            else:
                rocmlm_options = {
                    "kfolds": 5,
                    "rocmlm_features": ["XI_FRAC", "H2O"],
                    "rocmlm_targets": ["density", "Vp", "Vs", "melt_fraction", "H2O"]
                }
                ml_algorithms = {
                    "KN": {
                        "label": "K Neighbors",
                        "hyperparams": {
                            "weights": "distance",
                            "n_neighbors": 5,
                        },
                        "grid_search": {
                            "n_neighbors": [3, 5, 7, 9, 11]
                        }
                    },
                    "DT": {
                        "label": "Decision Tree",
                        "hyperparams": {
                            "splitter": "best",
                            "max_features": 4,
                            "min_samples_leaf": int(1),
                            "min_samples_split": int(2)
                        },
                        "grid_search": {
                            "splitter": ["best", "random"],
                            "max_features": [int(1), int(2), int(3), int(4)],
                            "min_samples_leaf": [int(2), int(3), int(4), int(5), int(6)],
                            "min_samples_split": [int(2), int(3), int(4), int(5), int(6)]
                        }
                    },
                    "NN": {
                        "label": "Neural Network",
                        "hyperparams": {
                            "hidden_layer_sizes": [int(8), int(8), int(8), int(8), int(8)],
                            "max_iter": 200,
                            "learning_rate_init": 0.001
                        },
                        "grid_search": {
                            "hidden_layer_sizes": [[int(8), int(8), int(8), int(8), int(8)]],
                            "max_iter": [200],
                            "learning_rate_init": [0.0001, 0.0005, 0.001]
                        }
                    }
                }

            # Assign values from rocmlm_options and ml_algorithms
            self.kfolds = rocmlm_options["kfolds"]
            self.rocmlm_features = rocmlm_options["rocmlm_features"]
            self.rocmlm_targets = rocmlm_options["rocmlm_targets"]

            self.rocmlm_label = ml_algorithms[self.ml_algo]["label"]
            self.rocmlm_gridsearch = ml_algorithms[self.ml_algo]["grid_search"]
            default_hyperparams = ml_algorithms[self.ml_algo]["hyperparams"]

            if self.ml_algo == "KN":
                self.default_rocmlm = KNeighborsRegressor(
                    weights=default_hyperparams["weights"],
                    n_neighbors=default_hyperparams["n_neighbors"]
                )
            elif self.ml_algo == "DT":
                self.default_rocmlm = DecisionTreeRegressor(
                    splitter=default_hyperparams["splitter"],
                    max_features=default_hyperparams["max_features"],
                    min_samples_leaf=default_hyperparams["min_samples_leaf"],
                    min_samples_split=default_hyperparams["min_samples_split"],
                    random_state=self.seed
                )
            elif self.ml_algo == "NN":
                self.default_rocmlm = MLPRegressor(
                    hidden_layer_sizes=default_hyperparams["hidden_layer_sizes"],
                    max_iter=default_hyperparams["max_iter"],
                    learning_rate_init=default_hyperparams["learning_rate_init"],
                    random_state=self.seed
                )
                self.sum_nodes = sum(default_hyperparams["hidden_layer_sizes"])
                self.max_iter = default_hyperparams["max_iter"]

        except Exception as e:
            print(f"Error in _load_rocmlm_options():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_unique_value(self, input_list):
        """
        """
        try:
            unique_value = input_list[0]
            for item in input_list[1:]:
                if item != unique_value:
                    raise Exception("Not all values are the same!")
        except Exception as e:
            print(f"Error in _get_unique_value():\n  {e}")
            return None

        return unique_value

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_gfem_model_metadata(self):
        """
        """
        try:
            if not self.gfem_models:
                raise Exception("No GFEM models to compile !")

            gfem_sids = [m.sid for m in self.gfem_models]

            gfem_res = self._get_unique_value([m.res for m in self.gfem_models])
            ox_gfem = self._get_unique_value([m.ox_gfem for m in self.gfem_models])
            gfem_P_min = self._get_unique_value([m.P_min for m in self.gfem_models])
            gfem_P_max = self._get_unique_value([m.P_max for m in self.gfem_models])
            gfem_T_min = self._get_unique_value([m.T_min for m in self.gfem_models])
            gfem_T_max = self._get_unique_value([m.T_max for m in self.gfem_models])
            gfem_db = self._get_unique_value([m.perplex_db for m in self.gfem_models])
            gfem_targets = self._get_unique_value([m.targets for m in self.gfem_models])
            gfem_features = self._get_unique_value([m.features for m in self.gfem_models])
            target_units_map = self._get_unique_value(
                [m.target_units_map for m in self.gfem_models])
            target_labels_map = self._get_unique_value(
                [m.target_labels_map for m in self.gfem_models])
            target_digits_map = self._get_unique_value(
                [m.target_digits_map for m in self.gfem_models])

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
            print(f"Error in _get_gfem_model_metadata():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_rocmlm_training_data(self):
        """
        """
        try:
            if not self.gfem_models:
                raise Exception("No GFEM models to compile !")

            pt_train = np.stack([m.pt_array for m in self.gfem_models])

            ft_idx = [self.gfem_features.index(f)
                      for f in self.rocmlm_features if f in self.gfem_features]
            self.rocmlm_features = [self.gfem_features[i] for i in ft_idx]

            feat_train = []
            for m in self.gfem_models:
                selected_features = [m.sample_features[i] for i in ft_idx]
                feat_train.append(selected_features)

            feat_train = np.array(feat_train)
            feat_train = np.tile(feat_train[:, np.newaxis, :], (1, pt_train.shape[1], 1))
            combined_train = np.concatenate((feat_train, pt_train), axis=2)
            rocmlm_feature_array = combined_train.reshape(-1, combined_train.shape[-1])

            t_idx = [self.gfem_targets.index(t)
                     for t in self.rocmlm_targets if t in self.gfem_targets]
            self.rocmlm_targets = [self.gfem_targets[i] for i in t_idx]

            rocmlm_target_array = np.stack([m.target_array for m in self.gfem_models])
            rocmlm_target_array = rocmlm_target_array.reshape(
                -1, rocmlm_target_array.shape[-1])
            rocmlm_target_array = rocmlm_target_array[:, t_idx]

            M = int(len(self.gfem_models))     # Number of samples
            W = int((self.gfem_res + 1) ** 2)  # PT grid size
            w = int(np.sqrt(W))                # P or T array size
            f = int(len(self.rocmlm_features)) # Number of rocmlm features
            F = int(f + 2)                     # Number of rocmlm features + PT
            T = int(len(self.rocmlm_targets))  # Number of rocmlm targets

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
            print(f"Error in _get_rocmlm_training_data():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_existing_model(self):
        """
        """
        if os.path.exists(self.rocmlm_path):
            try:
                if self.verbose >= 1:
                    print(f"Found pretrained model {self.rocmlm_path} !")

                self.model_trained = True
                self.load_pretrained_model(f"{self.rocmlm_path}")

            except Exception as e:
                print(f"Error in _check_existing_model():\n  {e}")
        else:
            os.makedirs(self.model_out_dir, exist_ok=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_pretrained_model(self, file_path):
        """
        """
        if os.path.exists(file_path):
            try:
                if self.verbose >= 1:
                    print(f"Loading RocMLM object from {file_path} ...")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                rocmlm = joblib.load(self.rocmlm_path)
                scaler_X = joblib.load(self.scaler_X_path)
                scaler_y = joblib.load(self.scaler_y_path)

                self._get_gfem_model_metadata()
                self._get_rocmlm_training_data()

                self.rocmlm = rocmlm

                rocmlm_target_array = self.rocmlm_target_array.copy()
                rocmlm_feature_array = self.rocmlm_feature_array.copy()
                rocmlm_target_array_shape_square = self.rocmlm_target_array_shape_square

                X, y = rocmlm_feature_array.copy(), rocmlm_target_array.copy()
                X_scaled = scaler_X.transform(X)
                pred_scaled = rocmlm.predict(X_scaled)
                pred_original = scaler_y.inverse_transform(pred_scaled)
                rocmlm_prediction_array_square = pred_original.reshape(
                    rocmlm_target_array_shape_square)

                self.rocmlm_prediction_array_square = rocmlm_prediction_array_square

            except Exception as e:
                print(f"Error in load_pretrained_model():\n  {e}")
        else:
            print(f"File {file_path} does not exist !")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scale_arrays(self, feature_array, target_array):
        """
        """
        try:
            X, y = feature_array, target_array
            X[~np.isfinite(X)] = np.nan
            y[~np.isfinite(y)] = np.nan
            X, y = np.nan_to_num(X), np.nan_to_num(y)

            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise Exception("Input data contains NaN or infinity values!")

            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

        except Exception as e:
            print(f"Error in _scale_arrays():\n  {e}")
            return None, None, None, None, None, None

        return X, y, scaler_X, scaler_y, X_scaled, y_scaled

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.               RocMLMs                   !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_rocmlm(self):
        """
        """
        try:
            if self.rocmlm_feature_array.size == 0:
                raise Exception("No training features!")

            if self.rocmlm_target_array.size == 0:
                raise Exception("No training targets!")

            _, _, _, _, X_scaled, y_scaled = self._scale_arrays(
                self.rocmlm_feature_array, self.rocmlm_target_array)

            if self.verbose >= 1:
                print(f"Configuring model {self.model_prefix} ...")

            if self.tune:
                if self.verbose >= 1:
                    print(f"Tuning model {self.model_prefix} ...")

                default_rocmlm = self.default_rocmlm
                rocmlm_gridsearch = self.rocmlm_gridsearch

                kf = KFold(n_splits=6, shuffle=True, random_state=self.seed)

                grid_search = GridSearchCV(
                    default_rocmlm, rocmlm_gridsearch, cv=kf, n_jobs=self.nprocs,
                    scoring="neg_root_mean_squared_error"
                )
                grid_search.fit(X_scaled, y_scaled)

                if self.ml_algo == "KN":
                    self.default_rocmlm = KNeighborsRegressor(
                        weights=grid_search.best_params_["weights"],
                        n_neighbors=grid_search.best_params_["n_neighbors"]
                    )
                elif self.ml_algo == "DT":
                    self.default_rocmlm = DecisionTreeRegressor(
                        splitter=grid_search.best_params_["splitter"],
                        max_features=grid_search.best_params_["max_features"],
                        min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                        min_samples_split=grid_search.best_params_["min_samples_split"],
                        random_state=self.seed
                    )
                elif self.ml_algo == "NN":
                    self.default_rocmlm = MLPRegressor(
                        hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"],
                        max_iter=grid_search.best_params_["max_iter"],
                        learning_rate_init=grid_search.best_params_["learning_rate_init"],
                        random_state=self.seed
                    )
            else:
                default_rocmlm = self.default_rocmlm

            self.rocmlm = default_rocmlm
            self.rocmlm_hyperparams = self.rocmlm.get_params()

        except Exception as e:
            print(f"Error in _configure_rocmlm():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_rocmlm_info(self):
        """
        """
        try:
            tgwrp = textwrap.fill(", ".join(self.rocmlm_targets), width=80,
                                  subsequent_indent="                  ")
            ftwrp = textwrap.fill(", ".join(self.rocmlm_features), width=80)

            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print(f"RocMLM model: {self.model_prefix}")
            print("---------------------------------------------")
            print(f"    model: {self.rocmlm_label}")
            print(f"    k folds: {self.kfolds}")
            print(f"    features: {ftwrp}")
            print(f"    targets: {tgwrp}")
            print(f"    features array shape: {self.rocmlm_feature_array.shape}")
            print(f"    targets array shape: {self.rocmlm_target_array.shape}")
            print(f"    hyperparameters:")
            for key, value in self.rocmlm_hyperparams.items():
                print(f"        {key}: {value}")
            print("---------------------------------------------")

        except Exception as e:
            print(f"Error in _print_rocmlm_info():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _kfold_itr(self, fold_args):
        """
        """
        try:
            if self.rocmlm_feature_array.size == 0:
                raise Exception("No training features!")

            if self.rocmlm_target_array.size == 0:
                raise Exception("No training targets!")

            X, y, _, scaler_y, X_scaled, y_scaled = \
                self._scale_arrays(self.rocmlm_feature_array, self.rocmlm_target_array)
            (train_index, test_idxex) = fold_args
            X, X_test = X_scaled[train_index], X_scaled[test_idxex]
            y, y_test = y_scaled[train_index], y_scaled[test_idxex]

            if "NN" in self.ml_algo:
                num_samples = len(y)
                batch_size = min(max(int(num_samples * 0.1), 8), num_samples)

                epoch_, train_loss_, test_loss_ = [], [], []
                training_start_time = time.time()

                for epoch in range(self.max_iter):
                    indices = np.arange(len(y))
                    np.random.shuffle(indices)

                    for start_idx in range(0, len(indices), batch_size):
                        end_idx = start_idx + batch_size
                        end_idx = min(end_idx, len(indices))
                        batch_indices = indices[start_idx:end_idx]
                        X_batch, y_batch = X[batch_indices], y[batch_indices]
                        self.rocmlm.partial_fit(X_batch, y_batch)

                    train_loss = self.rocmlm.loss_
                    train_loss_.append(train_loss)
                    test_loss = mean_squared_error(y_test, self.rocmlm.predict(X_test))
                    test_loss_.append(test_loss)
                    epoch_.append(epoch + 1)

                training_end_time = time.time()
                loss_curve = {
                    "epoch": epoch_, "train_loss": train_loss_, "test_loss": test_loss_}
            else:
                training_start_time = time.time()
                self.rocmlm.fit(X, y)
                training_end_time = time.time()
                loss_curve = None

            training_time = training_end_time - training_start_time

            y_pred_scaled = self.rocmlm.predict(X_test)
            y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test)

            rmse_test = np.sqrt(mean_squared_error(
                y_test_original, y_pred_original, multioutput="raw_values"))
            r2_test = r2_score(y_test_original, y_pred_original, multioutput="raw_values")

        except Exception as e:
            print(f"Error in _kfold_itr():\n  {e}")
            return (None, None, None, None)

        return (loss_curve, rmse_test, r2_test, training_time)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_kfold_results(self, results):
        """
        """
        try:
            M = self.rocmlm_feature_array_shape_square[0]
            w = self.rocmlm_feature_array_shape_square[1]

            loss_curves = []
            training_times = []
            rmse_test_scores, r2_test_scores = [], []

            for (loss_curve, rmse_test, r2_test, training_time) in results:
                loss_curves.append(loss_curve)
                rmse_test_scores.append(rmse_test)
                r2_test_scores.append(r2_test)
                training_times.append(training_time)

            if "NN" in self.ml_algo:
                self._visualize_loss_curve(loss_curves)

            rmse_test_scores = np.stack(rmse_test_scores)
            r2_test_scores = np.stack(r2_test_scores)
            rmse_test_mean = np.mean(rmse_test_scores, axis=0)
            rmse_test_std = np.std(rmse_test_scores, axis=0)
            r2_test_mean = np.mean(r2_test_scores, axis=0)
            r2_test_std = np.std(r2_test_scores, axis=0)
            training_time_mean = np.mean(training_times)
            training_time_std = np.std(training_times)

            if any("sm" in sample or "sr" in sample for sample in self.gfem_sids):
                sample_label = f"SMA{M}"
            elif any("st" in sample for sample in self.gfem_sids):
                sample_label = f"SMAT{M}"
            elif any("sm" in sample for sample in self.gfem_sids):
                sample_label = f"SMAM{M}"
            elif any("sb" in sample for sample in self.gfem_sids):
                sample_label = f"SMAB{M}"
            elif any("sr" in sample for sample in self.gfem_sids):
                sample_label = f"SMAR{M}"

            cv_info = {
                "model": [self.ml_algo],
                "sample": [sample_label],
                "size": [(w - 1) ** 2],
                "n_targets": [len(self.rocmlm_targets)],
                "k_folds": [self.kfolds],
                "training_time_mean": [round(training_time_mean, 5)],
                "training_time_std": [round(training_time_std, 5)]
            }

            for i, target in enumerate(self.rocmlm_targets):
                cv_info[f"rmse_test_mean_{target}"] = round(rmse_test_mean[i], 5)
                cv_info[f"rmse_test_std_{target}"] = round(rmse_test_std[i], 5)
                cv_info[f"r2_test_mean_{target}"] = round(r2_test_mean[i], 5)
                cv_info[f"r2_test_std_{target}"] = round(r2_test_std[i], 5)

            if self.verbose >= 1:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"{self.rocmlm_label} performance:")
                print(f"    training time: {training_time_mean:.5f} ± "
                      f"{training_time_std:.5f}")
                print(f"    rmse test:")
                for r, e, p in zip(rmse_test_mean, rmse_test_std, self.rocmlm_targets):
                    print(f"        {p}: {r:.5f} ± {e:.5f}")
                print(f"    r2 test:")
                for r, e, p in zip(r2_test_mean, r2_test_std, self.rocmlm_targets):
                    print(f"        {p}: {r:.5f} ± {e:.5f}")
                print("+++++++++++++++++++++++++++++++++++++++++++++")

            self.cv_info = cv_info

        except Exception as e:
            print(f"Error in _process_kfold_results():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _kfold_cv(self):
        """
        """
        try:
            if self.rocmlm_feature_array.size == 0:
                raise Exception("No training features!")

            if self.rocmlm_target_array.size == 0:
                raise Exception("No training targets!")

            X, _, _, _, _, _ = self._scale_arrays(
                self.rocmlm_feature_array, self.rocmlm_target_array)

            kf = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)

            fold_args = [
                (train_idx, test_idx) for _, (train_idx, test_idx) in enumerate(kf.split(X))]

            with cf.ProcessPoolExecutor(max_workers=self.nprocs) as executor:
                results = list(tqdm(executor.map(self._kfold_itr, fold_args),
                                    total=len(fold_args), desc="Cross-validating model ..."))

            self.model_cross_validated = True
            self._process_kfold_results(results)

        except Exception as e:
            print(f"Error in _kfold_cv():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _fit_training_data(self):
        """
        """
        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            if not self.model_cross_validated:
                raise Exception("ML model not cross validated! Call _kfold_cv() first ...")

            if self.rocmlm_feature_array.size == 0:
                raise Exception("No training features!")

            if self.rocmlm_target_array.size == 0:
                raise Exception("No training targets!")

            _, _, scaler_X, scaler_y, X_scaled, y_scaled = \
                self._scale_arrays(self.rocmlm_feature_array, self.rocmlm_target_array)

            X, X_test, y, y_test = \
                train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=self.seed)

            if "NN" in self.ml_algo:
                num_samples = len(y)
                batch_size = min(max(int(num_samples * 0.1), 8), num_samples)

                with tqdm(total=self.max_iter, desc="Training NN", position=0) as pbar:
                    for epoch in range(self.max_iter):
                        indices = np.arange(len(y))
                        np.random.shuffle(indices)

                        for start_idx in range(0, len(indices), batch_size):
                            end_idx = start_idx + batch_size
                            end_idx = min(end_idx, len(indices))
                            batch_indices = indices[start_idx:end_idx]
                            X_batch, y_batch = X[batch_indices], y[batch_indices]
                            self.rocmlm.partial_fit(X_batch, y_batch)

                        pbar.update(1)
            else:
                print(f"Training model {self.model_prefix} ...")
                self.rocmlm.fit(X, y)

            print(":::::::::::::::::::::::::::::::::::::::::::::")

            self.model_trained = True
            self.rocmlm_scaler_X = scaler_X
            self.rocmlm_scaler_y = scaler_y

            with open(self.rocmlm_path, "wb") as file:
                joblib.dump(self.rocmlm, file)
            with open(self.scaler_X_path, "wb") as file:
                joblib.dump(scaler_X, file)
            with open(self.scaler_y_path, "wb") as file:
                joblib.dump(scaler_y, file)

            X, y = self.rocmlm_feature_array.copy(), self.rocmlm_target_array.copy()
            X_scaled = scaler_X.transform(X)

            pred_scaled = self.rocmlm.predict(X_scaled)
            pred_original = scaler_y.inverse_transform(pred_scaled)
            rocmlm_prediction_array_square = pred_original.reshape(
                self.rocmlm_target_array_shape_square)

            self.rocmlm_prediction_array_square = rocmlm_prediction_array_square

        except Exception as e:
            print(f"Error in _fit_training_data():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _save_rocmlm_cv_info(self):
        """
        """
        try:
            filepath = f"assets/rocmlm-performance.csv"

            if not self.cv_info:
                raise Exception("No cross validation! Call _kfold_cv() first ...")

            if not pd.io.common.file_exists(filepath):
                df = pd.DataFrame(self.cv_info)
            else:
                df = pd.read_csv(filepath)
                new_data = pd.DataFrame(self.cv_info)
                df = pd.concat([df, new_data], ignore_index=True)

            df = df.sort_values(by=["model", "sample", "size"])
            df.to_csv(filepath, index=False)

        except Exception as e:
            print(f"Error in _save_rocmlm_cv_info():\n  {e}")

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.4.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_subduction_geotherm(self, segment="Central_Cascadia", slab_position="slabmoho"):
        """
        Retrieves the subduction geotherm data for a specified segment and slab position.

        Parameters:
            segment (str): The name of the subduction segment (default is
                           "Central_Cascadia").
            slab_position (str): Position of the slab (either "slabmoho" or "slabtop";
                                 default is "slabmoho").

        Returns:
            pd.DataFrame: A DataFrame containing the pressure (P) and temperature (T) data
                          for the specified slab position and segment.

        Raises:
            Exception: If the segment data file does not exist or if the slab position
            is unrecognized.
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
            print(f"Error in _get_subduction_geotherm():\n  {e}")
            return None

        return geotherm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_mantle_geotherm(self, mantle_potential=1573, Qs=55e-3, Ts=273, A1=1.0e-6,
                             A2=2.2e-8, k1=2.3, k2=3.0, mantle_adiabat=0.5e-3,
                             crust_thickness=35e3, litho_thickness=150e3):
        """
        Calculates the geotherm for a 1D layered lithospheric cooling model.

        Parameters:
            Qs (float): Surface heat flux (W/m²).
            Ts (float): Surface temperature (K).
            A1 (float): Heat production in layer 1 (W/m³).
            A2 (float): Heat production in layer 2 (W/m³).
            k1 (float): Thermal conductivity in layer 1 (W/m·K).
            k2 (float): Thermal conductivity in layer 2 (W/m·K).
            mantle_potential (float): Mantle potential temperature (K).
            mantle_adiabat (float): Mantle adiabatic gradient (K/m).
            crust_thickness (float): Crustal thickness (m).
            litho_thickness (float): Lithospheric thickness (m).

        Returns:
            pd.DataFrame: A DataFrame containing pressure (P) and temperature (T) values
                          for the calculated geotherm.

        Raises:
            Exception: If any error occurs during geotherm calculation.
        """
        try:
            array_size = (self.res + 1) * 10
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
            print(f"Error in _get_mantle_geotherm():\n  {e}")
            return None

        return geotherm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_rocmlm_images(self, sid, type="predictions"):
        """
        """
        try:
            if type not in {"targets", "predictions", "diff"}:
                raise Exception("Unrecognized array image type !")

            check = True

            for target in self.rocmlm_targets:
                path = (f"{self.fig_dir}/{self.model_prefix}-{sid}-"
                        f"{target.replace("_", "-")}-{type}.png")
                if not os.path.exists(path):
                    check = False
                    break

        except Exception as e:
            print(f"Error in _check_rocmlm_images():\n  {e}")
            return None

        return check

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_loss_curve(self, loss_curves, figwidth=6.3, figheight=3.54, fontsize=14):
        """
        """
        try:
            merged_curves = {}
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
            df = pd.DataFrame.from_dict(merged_curves, orient="index").transpose()
            df.sort_values(by="epoch", inplace=True)

            plt.rcParams["font.size"] = fontsize

            fig = plt.figure(figsize=(figwidth, figheight))
            colormap = plt.get_cmap("tab10")
            plt.plot(df["epoch"], df["train_loss"], label="train loss", color=colormap(0))
            plt.plot(df["epoch"], df["test_loss"], label="test loss", color=colormap(1))
            plt.xlabel("Epoch")
            plt.ylabel(f"Loss")
            plt.title(f"{self.rocmlm_label} Loss Curve")
            plt.legend()
            os.makedirs(f"{self.fig_dir}", exist_ok=True)
            plt.savefig(f"{self.fig_dir}/{self.ml_algo}-loss-curve.png")
            plt.close()

        except Exception as e:
            print(f"Error in _visualize_loss_curve():\n  {e}")

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
            raise Exception("No RocMLM model! Call train() first ...")
        if gfem_target_array_square is None or gfem_target_array_square.size == 0:
            raise Exception("No GFEM model target array !")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)
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

        plt.rcParams["font.size"] = fontsize

        try:
            n_feats = gfem_feature_array_square.shape[-1] - 2
            target_array = gfem_target_array_square[sid_idx, :, :, :]
            feature_array = gfem_feature_array_square[sid_idx, :, :, :]
            pred_array = rocmlm_prediction_array_square[sid_idx, :, :, :]
            for i, target in enumerate(rocmlm_targets):
                rmse, r2 = None, None
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
                    np.seterr(divide="ignore", invalid="ignore")
                    square_array = 100 * (t - p) / np.abs(t)
                    valid_mask = ~np.isnan(t) & ~np.isnan(p)
                    t_valid = t[valid_mask]
                    p_valid = p[valid_mask]
                    if len(t_valid) == 0 or len(p_valid) == 0:
                        print("Warning: no valid data points to calculate metrics !")
                    else:
                        r2 = r2_score(t_valid, p_valid)
                        rmse = np.sqrt(mean_squared_error(t_valid, p_valid))

                    filename = (f"{model_prefix}-{sid}-"
                                f"{target.replace("_", "-")}-diff.png")
                target_units = target_units_map[target]
                target_label = target_labels_map[target]
                if target not in ["assemblage_index", "phase_assemblage_variance"]:
                    title = f"{target_label} ({target_units})"
                else:
                    title = f"{target_label}"
                if target in ["assemblage_index", "phase_assemblage_variance"]:
                    color_discrete = True
                else:
                    color_discrete = False
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
                    color_palette = pal(np.linspace(0, 1, num_colors))
                    cmap = ListedColormap(color_palette)
                    cmap.set_bad(color="0.9")
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
                    if palette == "seismic":
                        vmin = -np.max(np.abs(square_array))
                        vmax = np.max(np.abs(square_array))
                    else:
                        vmin, vmax = vmin, vmax
                        if vmin <= 1e-4: vmin = 0
                    cmap = plt.colormaps[cmap]
                    cmap.set_bad(color="0.9")
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
                    if palette == "seismic":
                        cbar = plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], label="")
                    else:
                        cbar = plt.colorbar(im, ax=ax, label="",
                                            ticks=np.linspace(vmin, vmax, num=4))
                    cbar.ax.yaxis.set_major_formatter(
                        plt.FormatStrFormatter(target_digits_map[target]))
                plt.title(title)
                text_margin_x = 0.04
                text_margin_y = 0.15
                text_spacing_y = 0.1
                if rmse and r2:
                    plt.text(text_margin_x, text_margin_y - (text_spacing_y * 0),
                             f"R$^2$: {r2:.3f}", transform=plt.gca().transAxes,
                             fontsize=fontsize * 0.833, horizontalalignment="left",
                             verticalalignment="bottom")
                    plt.text(text_margin_x, text_margin_y - (text_spacing_y * 1),
                             f"RMSE: {rmse:.3f}", transform=plt.gca().transAxes,
                             fontsize=fontsize * 0.833, horizontalalignment="left",
                             verticalalignment="bottom")

                plt.savefig(f"{fig_dir}/{filename}")
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print(f"Error in _visualize_array_image():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize(self, skip=20):
        """
        """
        try:
            for sid_idx, sid in enumerate(self.gfem_sids[::skip]):
                if not self._check_rocmlm_images(sid, type="diff"):
                    self._visualize_array_image(sid, sid_idx, type="diff")

                if not self._check_rocmlm_images(sid, type="targets"):
                    self._visualize_array_image(sid, sid_idx, type="targets")

                if not self._check_rocmlm_images(sid, type="predictions"):
                    self._visualize_array_image(sid, sid_idx, type="predictions")

        except Exception as e:
            print(f"Error in visualize_rocmlm():\n  {e}")

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.5.             Train RocMLMs               !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
                print(f"Error in train():\n  {e}")
                traceback.print_exc()

                if retry < max_retries - 1:
                    print(f"Retrying in 5 seconds ...")
                    time.sleep(5)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.6.           RocMLM Inference              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
                raise Exception("No RocMLM model! Call train() first ...")
            for var in [xi, h2o, P, T]:
                if not isinstance(var, (list, np.ndarray)):
                    raise TypeError(f"Inputs must be lists or numpy arrays !")
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
            # Make predictions on features
            X_scaled = scaler_X.transform(X)
            inference_start_time = time.time()
            pred_scaled = model.predict(X_scaled)
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            pred_original = scaler_y.inverse_transform(pred_scaled)

        except Exception as e:
            print(f"Error in predict():\n  {e}")
            return None

        return pred_original

#######################################################
## .2.              Train RocMLMs                !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train rocmlms !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_rocmlms(gfem_models, ml_algos=["DT", "KN", "NN"], config_yaml=None):
    """
    """
    try:
        if not gfem_models:
            raise Exception("No GFEM models to compile !")

        models = []
        for algo in ml_algos:
            tune = True if algo != "NN" else False
            model = RocMLM(gfem_models, algo, tune=tune, config_yaml=config_yaml)
            model.train()
            models.append(model)

        rocmlms = [m for m in models if not m.model_training_error]
        error_count = len([m for m in models if m.model_training_error])

        if error_count > 0:
            print(f"Total RocMLMs with errors: {error_count}")
        else:
            print("All RocMLMs built successfully !")

        print(":::::::::::::::::::::::::::::::::::::::::::::")

    except Exception as e:
        print(f"Error in train_rocmlms():\n  {e}")
        return None

    return rocmlms

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    from gfem import build_gfem_models
    try:
        gfems = []
        gfem_configs = {"stx21m": "assets/config_yamls/dry-deep-mantle-stx21m.yaml",
                        "stx21r": "assets/config_yamls/dry-deep-mantle-stx21r.yaml"}

        for name, yaml in gfem_configs.items():
            gfems.extend(build_gfem_models(config_yaml=yaml))

        rocmlm_config = "assets/config_yamls/rocmlm-test.yaml"
        rocmlms = train_rocmlms(gfems, config_yaml=rocmlm_config)

    except Exception as e:
        print(f"Error in main():\n  {e}")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("RocMLMs trained and visualized !")
    print("=============================================")

    return None

if __name__ == "__main__":
    main()