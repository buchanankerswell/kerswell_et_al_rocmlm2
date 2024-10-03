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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, ParameterGrid

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

#######################################################
## .1.             PyTorch NN Class              !!! ##
#######################################################
class SimpleNN(nn.Module):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation="relu"):
        """
        """
        super(SimpleNN, self).__init__()

        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes

        # Build the model layers
        layers = []
        prev_size = input_size

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if activation == "relu":
                layers.append(nn.ReLU())
            # Add more activation functions as needed here
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def predict(self, x):
        """
        """
        return self.model(x)

    def forward(self, x):
        """
        """
        return self.model(x)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params(self):
        """
        """
        params_info = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "activation": self.activation,
            "num_layers": len(self.hidden_layer_sizes) + 1,
            "hidden_layer_sizes": self.hidden_layer_sizes
        }

        return params_info


#######################################################
## .1.               RocMLM Class                !!! ##
#######################################################
class RocMLM:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, gfem_models, ml_algo="DT", config_yaml=None, verbose=1):
        """
        """
        self.ml_algo = ml_algo
        self.verbose = verbose
        self.config_yaml = config_yaml
        self.gfem_models = sorted(gfem_models, key=lambda model: model.sid)

        if self.ml_algo not in ["DT", "KN", "MLP", "NN"]:
            raise Exception("Unrecognized ml_algo! Must be 'DT', 'KN' or 'MLP' ...")

        # Determine torch device
        self.device = torch.device("mps" if torch.backends.mps.is_available() and
                                   self.ml_algo == "NN" else "cpu")
        print(f"Using device: {self.device}")

        self.model_trained = False
        self.model_cross_validated = False
        self.model_out_dir = "rocmlms"

        self._load_global_options()
        self._load_rocmlm_options()
        self._get_gfem_model_metadata()
        self._get_rocmlm_training_data()

        self.model_prefix = (f"{self.ml_algo}-"
                             f"S{self.rocmlm_feature_array_shape_square[0]}-"
                             f"R{self.rocmlm_feature_array_shape_square[1]}-"
                             f"F{self.rocmlm_feature_array_shape_square[3]}-"
                             f"T{len(self.rocmlm_targets)}-"
                             f"{self.gfem_db}")
        self.fig_dir = (f"figs/rocmlm/{self.model_prefix}")
        self.rocmlm_path = f"{self.model_out_dir}/{self.model_prefix}-pretrained.pkl"

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
                    "kfolds": 5,
                    "palette": "bone",
                    "pot_Ts": [1173, 1573, 1773],
                    "segs": ["Central_Cascadia", "Kamchatka"],
                    "rocmlm_features": ["XI_FRAC", "H2O"],
                    "rocmlm_targets": ["density", "Vp", "Vs", "melt_fraction", "H2O"]
                }

            # Assign values from global options
            self.segs = global_options["segs"]
            self.seed = global_options["seed"]
            self.digits = global_options["digits"]
            self.nprocs = global_options["nprocs"]
            self.pot_Ts = global_options["pot_Ts"]
            self.kfolds = global_options["kfolds"]
            self.palette = global_options["palette"]
            self.rocmlm_features = global_options["rocmlm_features"]
            self.rocmlm_targets = global_options["rocmlm_targets"]

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
            else:
                rocmlm_options = {
                    "KN": {
                        "tune": False,
                        "cross_validate": False,
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
                        "tune": False,
                        "cross_validate": False,
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
                    "MLP": {
                        "tune": False,
                        "cross_validate": False,
                        "label": "Scikit-learn Neural Network",
                        "hyperparams": {
                            "hidden_layer_sizes": [int(8), int(8), int(8)],
                            "max_iter": 200,
                            "learning_rate_init": 0.001
                        },
                        "grid_search": {
                            "hidden_layer_sizes": [[int(8), int(8), int(8)]],
                            "learning_rate_init": [0.0001, 0.0005, 0.001]
                        }
                    },
                    "NN": {
                        "tune": False,
                        "cross_validate": False,
                        "label": "PyTorch Neural Network",
                        "hyperparams": {
                            "hidden_layer_sizes": [int(8), int(8), int(8)],
                            "max_iter": 200,
                            "learning_rate_init": 0.001
                        },
                        "grid_search": {
                            "hidden_layer_sizes": [[int(8), int(8), int(8)]],
                            "learning_rate_init": [0.0001, 0.0005, 0.001]
                        }
                    }
                }

            # Assign values from rocmlm_options and rocmlm_options
            self.rocmlm_tune = rocmlm_options[self.ml_algo]["tune"]
            self.rocmlm_label = rocmlm_options[self.ml_algo]["label"]
            self.rocmlm_gridsearch = rocmlm_options[self.ml_algo]["grid_search"]
            self.default_hyperparams = rocmlm_options[self.ml_algo]["hyperparams"]
            self.rocmlm_cross_validate = rocmlm_options[self.ml_algo]["cross_validate"]

            if self.ml_algo == "KN":
                self.default_rocmlm = KNeighborsRegressor(
                    weights=self.default_hyperparams["weights"],
                    n_neighbors=self.default_hyperparams["n_neighbors"]
                )
            elif self.ml_algo == "DT":
                self.default_rocmlm = DecisionTreeRegressor(
                    splitter=self.default_hyperparams["splitter"],
                    max_features=self.default_hyperparams["max_features"],
                    min_samples_leaf=self.default_hyperparams["min_samples_leaf"],
                    min_samples_split=self.default_hyperparams["min_samples_split"],
                    random_state=self.seed
                )
            elif self.ml_algo == "MLP":
                self.default_rocmlm = MLPRegressor(
                    max_iter=self.default_hyperparams["max_iter"],
                    hidden_layer_sizes=self.default_hyperparams["hidden_layer_sizes"],
                    learning_rate_init=self.default_hyperparams["learning_rate_init"],
                    random_state=self.seed
                )
                self.max_iter = self.default_hyperparams["max_iter"]
            elif self.ml_algo == "NN":
                self.default_rocmlm = SimpleNN(
                    input_size=len(self.rocmlm_features) + 2,
                    hidden_layer_sizes=self.default_hyperparams["hidden_layer_sizes"],
                    output_size=len(self.rocmlm_targets)
                ).to(self.device)
                self.max_iter = self.default_hyperparams["max_iter"]
                self.default_lr = self.default_hyperparams["learning_rate_init"]

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
                raise Exception("No GFEM models to compile!")

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
                raise Exception("No GFEM models to compile!")

            pt_train = np.stack([m.pt_array for m in self.gfem_models])

            ft_idx = [self.gfem_features.index(f)
                      for f in self.rocmlm_features if f in self.gfem_features]
            self.rocmlm_features = [self.gfem_features[i] for i in ft_idx] + ["P", "T"]

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

            S = int(len(self.gfem_models))     # Number of samples
            R = int((self.gfem_res + 1) ** 2)  # PT grid size
            r = int(np.sqrt(R))                # PT grid resolution
            F = int(len(self.rocmlm_features)) # Number of rocmlm features + PT
            T = int(len(self.rocmlm_targets))  # Number of rocmlm targets

            rocmlm_target_array_shape = (S, R, T)
            rocmlm_feature_array_shape = (S, R, F)
            rocmlm_target_array_shape_square = (S, r, r, T)
            rocmlm_feature_array_shape_square = (S, r, r, F)

            self.rocmlm_target_array = rocmlm_target_array.astype(np.float32)
            self.rocmlm_target_array_shape = rocmlm_target_array_shape
            self.rocmlm_target_array_shape_square = rocmlm_target_array_shape_square

            self.rocmlm_feature_array = rocmlm_feature_array.astype(np.float32)
            self.rocmlm_feature_array_shape = rocmlm_feature_array_shape
            self.rocmlm_feature_array_shape_square = rocmlm_feature_array_shape_square

        except Exception as e:
            print(f"Error in _get_rocmlm_training_data():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_existing_model(self):
        """
        """
        try:
            if os.path.exists(self.rocmlm_path):
                if self.verbose >= 1:
                    print(f"Found pretrained model {self.model_prefix}!")
                self._load_pretrained_rocmlm(self.rocmlm_path)
            else:
                os.makedirs(self.model_out_dir, exist_ok=True)

        except Exception as e:
            print(f"Error in _check_existing_model():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_pretrained_rocmlm(self, rocmlm_path):
        """
        """
        try:
            if (os.path.exists(rocmlm_path)):
                if self.verbose >= 1:
                    print(f"Loading RocMLM object from {rocmlm_path} ...")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                # Load RocMLM object
                with open(rocmlm_path, "rb") as file:
                    loaded_rocmlm = joblib.load(file)

                # Update the current instance
                self.__dict__.update(loaded_rocmlm.__dict__)

                if self.rocmlm is None:
                    raise Exception("RocMLM model not loaded properly!")
            else:
                print(f"File {rocmlm_path} does not exist!")

        except Exception as e:
            print(f"Error in _load_pretrained_rocmlm():\n  {e}")

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
    def _tune_scikit_model(self, X, y):
        """
        """
        try:
            model = self.default_rocmlm
            rocmlm_gridsearch = self.rocmlm_gridsearch

            kf = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)

            grid_search = GridSearchCV(
                model, rocmlm_gridsearch, cv=kf, n_jobs=self.nprocs,
                scoring="neg_root_mean_squared_error"
            )
            grid_search.fit(X, y)

            if self.ml_algo == "KN":
                model = KNeighborsRegressor(
                    weights=grid_search.best_params_["weights"],
                    n_neighbors=grid_search.best_params_["n_neighbors"]
                )
            elif self.ml_algo == "DT":
                model = DecisionTreeRegressor(
                    splitter=grid_search.best_params_["splitter"],
                    max_features=grid_search.best_params_["max_features"],
                    min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                    min_samples_split=grid_search.best_params_["min_samples_split"],
                    random_state=self.seed
                )
            elif self.ml_algo == "MLP":
                model = MLPRegressor(
                    max_iter=self.max_iter,
                    hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"],
                    learning_rate_init=grid_search.best_params_["learning_rate_init"],
                    random_state=self.seed
                )

            self.rocmlm = model
            self.rocmlm_hyperparams = self.rocmlm.get_params()

        except Exception as e:
            print(f"Error in _tune_scikit_model():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _tune_torch_nn(self, X, y):
        """
        """
        try:
            best_params = None
            best_score = float("inf")
            rocmlm_gridsearch = self.rocmlm_gridsearch

            kf = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)

            for params in ParameterGrid(rocmlm_gridsearch):
                fold_scores = []

                for train_idx, test_idx in kf.split(X, y):
                    # Configure model
                    model = SimpleNN(
                        input_size=X.shape[1],
                        hidden_layer_sizes=params["hidden_layer_sizes"],
                        output_size=y.shape[1]
                    ).to(self.device)
                    optimizer = optim.Adam(
                        model.parameters(), lr=params["learning_rate_init"])
                    loss_fn = nn.MSELoss()

                    # Convert to torch tensors
                    X_train = torch.Tensor(X[train_idx]).to(self.device)
                    y_train = torch.Tensor(y[train_idx]).to(self.device)
                    X_test = torch.Tensor(X[test_idx]).to(self.device)
                    y_test = torch.Tensor(y[test_idx]).to(self.device)

                    # Train model on training set
                    for epoch in range(self.max_iter):
                        optimizer.zero_grad()
                        y_pred = model.predict(X_train)
                        loss = loss_fn(y_pred, y_train)
                        loss.backward()
                        optimizer.step()

                    # Evaluate on test set
                    with torch.no_grad():
                        y_test_pred = model.predict(X_test)
                        test_loss = loss_fn(y_test_pred, y_test).item()
                        fold_scores.append(test_loss)

                # Average score across folds
                avg_score = np.mean(fold_scores)

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params

            if best_params:
                best_model = SimpleNN(
                    input_size=X.shape[1],
                    hidden_layer_sizes=best_params["hidden_layer_sizes"],
                    output_size=y.shape[1]
                ).to(self.device)
                best_lr = best_params["learning_rate_init"]

                self.lr = best_lr
                self.rocmlm = best_model

            self.rocmlm_hyperparams = self.rocmlm.get_params()

        except Exception as e:
            print(f"Error in _tune_torch_nn():\n  {e}")

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

            if self.rocmlm_tune:
                if self.verbose >= 1:
                    print(f"Tuning model {self.model_prefix} ...")

                if self.ml_algo in ["DT", "KN", "MLP"]:
                    self._tune_scikit_model(X_scaled, y_scaled)
                elif self.ml_algo == "NN":
                    self._tune_torch_nn(X_scaled, y_scaled)
                else:
                    raise Exception("Unrecognized ml_algo!")
            else:
                if self.ml_algo == "NN":
                    self.lr = self.default_lr
                self.rocmlm = self.default_rocmlm
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
            print(f"    features square shape: {self.rocmlm_feature_array_shape_square}")
            print(f"    targets array shape: {self.rocmlm_target_array.shape}")
            print(f"    targets square shape: {self.rocmlm_target_array_shape_square}")
            print(f"    hyperparameters:")
            for key, value in self.rocmlm_hyperparams.items():
                print(f"        {key}: {value}")
            print("---------------------------------------------")

        except Exception as e:
            print(f"Error in _print_rocmlm_info():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _determine_batch_size(self, num_samples):
        """
        """
        try:
            if num_samples >= 5e5:
                batch_size = 8192
            elif num_samples < 5e5 and num_samples >= 1e5:
                batch_size = 4096
            elif num_samples < 1e5 and num_samples >= 5e4:
                batch_size = 2048
            elif num_samples < 5e4 and num_samples >= 1e4:
                batch_size = 1024
            elif num_samples < 1e4 and num_samples >= 5e3:
                batch_size = 512
            elif num_samples < 5e3 and num_samples >= 1e3:
                batch_size = 256
            else:
                batch_size = min(num_samples, 64)

            if self.device.type == "mps" and self.ml_algo == "NN":
                batch_size = min(num_samples, batch_size * 2**4)

        except Exception as e:
            print(f"Error in _determine_batch_size():\n  {e}")
            return None

        return batch_size

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _train_mlp(self, X, y, X_test, y_test):
        """
        """
        try:
            batch_size = self._determine_batch_size(y.shape[0])
            epoch_, train_loss_, test_loss_ = [], [], []

            with tqdm(total=self.max_iter,
                      desc=f"Training MLP with batch size {batch_size}",
                      position=0) as pbar:
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

                train_loss = self.rocmlm.loss_
                train_loss_.append(train_loss)
                test_loss = mean_squared_error(y_test, self.rocmlm.predict(X_test))
                test_loss_.append(test_loss)
                epoch_.append(epoch + 1)

            loss_curve = {
                "epoch": epoch_, "train_loss": train_loss_, "test_loss": test_loss_}

        except Exception as e:
            print(f"Error in _train_mlp():\n  {e}")
            return None

        return loss_curve

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _train_torch_nn(self, X, y, X_test, y_test):
        """
        """
        try:
            batch_size = self._determine_batch_size(y.shape[0])
            epoch_, train_loss_, test_loss_ = [], [], []

            # Set model to training mode
            self.rocmlm.to(self.device)
            self.rocmlm.train()
            optimizer = optim.Adam(self.rocmlm.parameters(), lr=self.lr)

            # Train model on training set
            with tqdm(total=self.max_iter,
                      desc=f"Training NN with batch size: {batch_size}",
                      position=0) as pbar:
                for epoch in range(self.max_iter):
                    indices = np.arange(len(y))
                    np.random.shuffle(indices)

                    for start_idx in range(0, len(indices), batch_size):
                        end_idx = start_idx + batch_size
                        end_idx = min(end_idx, len(indices))
                        batch_indices = indices[start_idx:end_idx]
                        X_batch, y_batch = X[batch_indices], y[batch_indices]

                        # Convert to torch tensors
                        X_train = torch.Tensor(X_batch).to(self.device)
                        y_train = torch.Tensor(y_batch).to(self.device)

                        # Training step
                        optimizer.zero_grad()
                        y_pred = self.rocmlm.predict(X_train)
                        loss_fn = nn.MSELoss()
                        loss = loss_fn(y_pred, y_train)
                        loss.backward()
                        optimizer.step()

                    pbar.update(1)

                    # Track training loss
                    train_loss = loss.item()
                    train_loss_.append(train_loss)

                    # Convert to torch tensors
                    X_tst = torch.Tensor(X_test).to(self.device)
                    y_tst = torch.Tensor(y_test).to(self.device)

                    # Evaluate on test set
                    with torch.no_grad():
                        y_test_tensor = torch.Tensor(X_tst)
                        y_test_pred = self.rocmlm.predict(y_test_tensor)
                        test_loss = loss_fn(y_test_pred, y_tst).item()
                        test_loss_.append(test_loss)

                    epoch_.append(epoch + 1)

            loss_curve = {
                "epoch": epoch_,
                "train_loss": train_loss_,
                "test_loss": test_loss_
            }

        except Exception as e:
            print(f"Error in _train_torch_nn():\n  {e}")
            return None

        return loss_curve

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

            training_start_time = time.time()

            if self.ml_algo == "MLP":
                loss_curve = self._train_mlp(X, y, X_test, y_test)
            elif self.ml_algo == "NN":
                loss_curve = self._train_torch_nn(X, y, X_test, y_test)
            else:
                self.rocmlm.fit(X, y)
                loss_curve = None

            training_end_time = time.time()

            training_time = training_end_time - training_start_time

            # For torch models with mps
            if self.ml_algo == "NN" and self.device.type == "mps":
                X_test = torch.Tensor(X_test).to(self.device)
                y_pred_scaled = self.rocmlm.predict(X_test)
                # Detach and convert tensor to NumPy
                y_pred_scaled = y_pred_scaled.detach().cpu().numpy()
            else:
                y_pred_scaled = self.rocmlm.predict(X_test)

            y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = scaler_y.inverse_transform(y_test)

            rmse_test = np.sqrt(mean_squared_error(
                y_test_original, y_pred_original, multioutput="raw_values"))
            r2_test = r2_score(y_test_original, y_pred_original, multioutput="raw_values")

            # Normalize RMSE: rmse / (max - min of y_test_original) * 100
            y_range = np.ptp(y_test_original, axis=0)
            normalized_rmse = (rmse_test / y_range) * 100

        except Exception as e:
            print(f"Error in _kfold_itr():\n  {e}")
            return (None, None, None, None)

        return (loss_curve, normalized_rmse, r2_test, training_time)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_kfold_results(self, results):
        """
        """
        try:
            S = self.rocmlm_feature_array_shape_square[0]
            r = self.rocmlm_feature_array_shape_square[1]

            loss_curves = []
            training_times = []
            rmse_test_scores, r2_test_scores = [], []

            for (loss_curve, rmse_test, r2_test, training_time) in results:
                loss_curves.append(loss_curve)
                rmse_test_scores.append(rmse_test)
                r2_test_scores.append(r2_test)
                training_times.append(training_time)

            if self.ml_algo == "MLP":
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
                sample_label = f"SMA{S}"
            elif any("st" in sample for sample in self.gfem_sids):
                sample_label = f"SMAT{S}"
            elif any("sm" in sample for sample in self.gfem_sids):
                sample_label = f"SMAM{S}"
            elif any("sb" in sample for sample in self.gfem_sids):
                sample_label = f"SMAB{S}"
            elif any("sr" in sample for sample in self.gfem_sids):
                sample_label = f"SMAR{S}"

            cv_info = {
                "model": [self.ml_algo],
                "sample": [sample_label],
                "size": [(r - 1) ** 2],
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
                print(f"    rmse test (normalized) %:")
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

            print("Cross-validating RocMLM ...")

            # If running on MPS, run K-fold in serial
            if self.device.type == "mps":
                results = [self._kfold_itr(fold) for fold in fold_args]
            else:
                # Otherwise, run K-fold in parallel
                with cf.ProcessPoolExecutor(max_workers=self.nprocs) as executor:
                    results = list(executor.map(self._kfold_itr, fold_args))

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
                print("Warning: ML model not cross validated. "
                      "Cannot provide performance metrics ...")

            if self.rocmlm_feature_array.size == 0:
                raise Exception("No training features!")

            if self.rocmlm_target_array.size == 0:
                raise Exception("No training targets!")

            _, _, scaler_X, scaler_y, X_scaled, y_scaled = \
                self._scale_arrays(self.rocmlm_feature_array, self.rocmlm_target_array)

            X, X_test, y, y_test = \
                train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=self.seed)

            if self.ml_algo == "MLP":
                self._train_mlp(X, y, X_test, y_test)
            elif self.ml_algo == "NN":
                self._train_torch_nn(X, y, X_test, y_test)
            else:
                print(f"Training model {self.model_prefix} ...")
                self.rocmlm.fit(X, y)

            self.model_trained = True
            self.rocmlm_scaler_X = scaler_X
            self.rocmlm_scaler_y = scaler_y

            # Delete gfem models to reduce disk space
            del self.gfem_models

            with open(self.rocmlm_path, "wb") as file:
                joblib.dump(self, file, compress=3)

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
                (geotherm["P"] >= self.gfem_P_min) & (geotherm["P"] <= self.gfem_P_max) &
                (geotherm["T"] >= self.gfem_T_min) & (geotherm["T"] <= self.gfem_T_max)
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
            Z_min = self.gfem_P_min * litho_P_gradient
            Z_max = self.gfem_P_max * litho_P_gradient
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
    def _check_rocmlm_images(self, sid, plot_type="predictions"):
        """
        """
        try:
            if plot_type not in {"targets", "predictions", "diff"}:
                raise Exception("Unrecognized array image plot_type!")

            check = True

            for target in self.rocmlm_targets:
                path = (f"{self.fig_dir}/{self.model_prefix}-{sid}-"
                        f"{target.replace("_", "-")}-{plot_type}.png")
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
    def _get_geotherms_for_array_image(self, geotherm_type):
        """
        """
        try:
            geotherms = {}
            if geotherm_type == "sub":
                for seg in self.segs:
                    if self.gfem_P_min < 6:
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
            print(f"Error in _get_geotherms_for_array_image():\n  {e}")
            return None

        return geotherms

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_square_predictions_for_array_image(self):
        """
        """
        try:
            X, y = self.rocmlm_feature_array.copy(), self.rocmlm_target_array.copy()
            X_scaled = self.rocmlm_scaler_X.transform(X)

            pred_scaled = self.rocmlm.predict(X_scaled)
            pred_original = self.rocmlm_scaler_y.inverse_transform(pred_scaled)

            return pred_original.reshape(self.rocmlm_target_array_shape_square)

        except Exception as e:
            print(f"Error in _get_square_predictions_for_array_image():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_square_target_for_array_image(self, sid, target, target_index,
                                           pred_array, target_array, plot_type):
        """
        """
        try:
            p = pred_array[:, :, target_index]
            t = target_array[:, :, target_index]
            rmse, r2 = None, None

            if plot_type == "predictions":
                square_target = p
                filename = (f"{self.model_prefix}-{sid}-"
                            f"{target.replace("_", "-")}-predictions.png")
            elif plot_type == "targets":
                square_target = t
                filename = (f"{self.model_prefix}-{sid}-"
                            f"{target.replace("_", "-")}-targets.png")
            elif plot_type == "diff":
                mask = np.isnan(t)
                p[mask] = np.nan
                np.seterr(divide="ignore", invalid="ignore")
                square_target = 100 * (t - p) / np.abs(t)
                valid_mask = ~np.isnan(t) & ~np.isnan(p)
                t_valid = t[valid_mask]
                p_valid = p[valid_mask]

                if len(t_valid) == 0 or len(p_valid) == 0:
                    print("Warning: no valid data points to calculate metrics!")
                else:
                    r2 = r2_score(t_valid, p_valid)
                    rmse = np.sqrt(mean_squared_error(t_valid, p_valid))
                    normalized_rmse = (rmse / np.ptp(t_valid)) * 100

                filename = (f"{self.model_prefix}-{sid}-"
                            f"{target.replace("_", "-")}-diff.png")

        except Exception as e:
            print(f"Error in _get_square_target_for_array_image():\n  {e}")
            return None

        return square_target, filename, r2, rmse

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
            print(f"Error in _get_colormap_for_array_image():\n  {e}")
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
            print(f"Error in _get_vmin_vmax_for_array_image():\n  {e}")
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
            print(f"Error in _plot_geotherms_on_array_image():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_array_image(self, sid, sid_idx, plot_type="targets", geotherm_type=None,
                               figwidth=6.3, figheight=4.725, fontsize=22):
        """
        """
        try:
            plt.rcParams["font.size"] = fontsize
            linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]

            if plot_type not in {"targets", "predictions", "diff"}:
                raise Exception("Unrecognized plot_type argument!")
            if geotherm_type not in ["sub", "craton", "mor", None]:
                raise Exception("Unrecognized geotherm_type argument!")
            if not self.model_trained:
                raise Exception("No RocMLM! Call train() first ...")

            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir, exist_ok=True)

            gfem_target_array_square = \
                self.rocmlm_target_array.reshape(self.rocmlm_target_array_shape_square)
            gfem_feature_array_square = \
                self.rocmlm_feature_array.reshape(self.rocmlm_feature_array_shape_square)
            rocmlm_prediction_array_square = self._get_square_predictions_for_array_image()

            n_feats = gfem_feature_array_square.shape[-1] - 2
            target_array = gfem_target_array_square[sid_idx, :, :, :]
            feature_array = gfem_feature_array_square[sid_idx, :, :, :]
            pred_array = rocmlm_prediction_array_square[sid_idx, :, :, :]

            if geotherm_type:
                # Get geotherms
                geotherms = self._get_geotherms_for_array_image(geotherm_type)

            for i, target in enumerate(self.rocmlm_targets):
                P = feature_array[:, :, 0 + n_feats]
                T = feature_array[:, :, 1 + n_feats]
                extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

                target_label = self.target_labels_map[target]
                title = (f"{target_label} ({self.target_units_map.get(target, '')})")

                # Get square array
                square_target, filename, r2, rmse = self._get_square_target_for_array_image(
                    sid, target, i, pred_array, target_array, plot_type)

                # Determine color mapping
                color_discrete = target in ["assemblage_index", "phase_assemblage_variance"]
                color_reverse = target not in ["phase_assemblage_variance"]
                palette = "seismic" if plot_type == "diff" else self.palette
                cmap = self._get_colormap_for_array_image(
                    palette, color_reverse, color_discrete)
                vmin, vmax = self._get_vmin_vmax_for_array_image(
                    square_target, color_discrete)

                if plot_type == "diff":
                    vmin = -np.max(np.abs(square_target))
                    vmax = np.max(np.abs(square_target))

                # Plot the array in 2d
                fig, ax = plt.subplots(figsize=(figwidth, figheight))
                im = ax.imshow(square_target, extent=extent, aspect="auto", cmap=cmap,
                               origin="lower", vmin=vmin, vmax=vmax)

                if geotherm_type:
                    # Plot geotherms based on type
                    self._plot_geotherms_on_array_image(
                        ax, geotherms, geotherm_type, linestyles)

                # Finalize plot
                ax.set_xlabel("T (K)")
                ax.set_ylabel("P (GPa)")
                if palette == "seismic":
                    cbar = plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], label="")
                else:
                    cbar = plt.colorbar(
                        im, ax=ax, label="", ticks=np.linspace(vmin, vmax, num=4))
                cbar.ax.yaxis.set_major_formatter(
                    plt.FormatStrFormatter(self.target_digits_map[target]))
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

                # Save fig
                plt.savefig(f"{self.fig_dir}/{filename}")
                plt.close()
                print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print(f"Error in _visualize_array_image():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize(self, n_samples=3):
        """
        """
        try:
            skip = len(self.gfem_sids) // n_samples

            for i, sid in enumerate(self.gfem_sids[::skip]):
                if not self._check_rocmlm_images(sid, "diff"):
                    self._visualize_array_image(sid, i, "diff")
                if not self._check_rocmlm_images(sid, "targets"):
                    self._visualize_array_image(sid, i, "targets")
                if not self._check_rocmlm_images(sid, "predictions"):
                    self._visualize_array_image(sid, i, "predictions")

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
                if self.rocmlm_cross_validate:
                    self._kfold_cv()
                    self._save_rocmlm_cv_info()
                self._fit_training_data()

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
    def inference(self, **features):
        """
        """
        try:
            if not self.model_trained:
                raise Exception("No RocMLM! Call train() first ...")

            # Ensure all necessary features are provided
            training_features = self.rocmlm_features
            missing_features = [f for f in training_features if f not in features]
            if missing_features:
                raise Exception(f"Missing required features: {missing_features}")

            # Validate that all input features are lists or numpy arrays
            for feature_name, feature_data in features.items():
                if not isinstance(feature_data, (list, np.ndarray)):
                    raise TypeError(f"Feature {feature_name} must be a list or numpy array!")

            # Convert all feature inputs to numpy arrays and check dimensions
            feature_arrays = [
                np.asarray(features[f]).reshape(-1, 1) for f in training_features]
            feature_shapes = [arr.shape[0] for arr in feature_arrays]

            if len(set(feature_shapes)) != 1:
                raise Exception("All feature arrays must have the same length!")

            # Concatenate the feature arrays
            X = np.concatenate(feature_arrays, axis=1)
            X = np.nan_to_num(X)

            # Make predictions on features
            X_scaled = self.rocmlm_scaler_X.transform(X)
            inference_start_time = time.time()
            pred_scaled = self.rocmlm.predict(X_scaled)
            inference_end_time = time.time()
            inference_time = (inference_end_time - inference_start_time) * 1e3
            inference_time_per_node = inference_time / X.shape[0]
            pred_original = self.rocmlm_scaler_y.inverse_transform(pred_scaled)

            print(f"  {X.shape[0]} nodes completed in {inference_time:.4f} "
                  f"milliseconds ({inference_time_per_node:.4f} ms per node)...")

        except Exception as e:
            print(f"Error in inference():\n  {e}")
            return None

        return pred_original

#######################################################
## .2.              Train RocMLMs                !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_rocmlms(gfem_models, ml_algos=["DT", "KN", "NN"], config_yaml=None):
    """
    """
    try:
        if not gfem_models:
            raise Exception("No GFEM models to compile!")

        models = []
        for algo in ml_algos:
            model = RocMLM(gfem_models, algo, config_yaml)
            model.train()
            models.append(model)

        rocmlms = [m for m in models if m.model_trained]
        error_count = len([m for m in models if not m.model_trained])

        if error_count > 0:
            print(f"Total RocMLMs with errors: {error_count}")
        else:
            print("All RocMLMs built successfully!")

    except Exception as e:
        print(f"Error in train_rocmlms():\n  {e}")
        return None

    return rocmlms

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_pretrained_rocmlm(rocmlm_path):
    """
    """
    try:
        if (os.path.exists(rocmlm_path)):
            print(f"Loading RocMLM object from {rocmlm_path} ...")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # Load RocMLM object
            with open(rocmlm_path, "rb") as file:
                model = joblib.load(file)

            if model.rocmlm is None:
                raise Exception("RocMLM model not loaded properly!")
        else:
            print(f"File {rocmlm_path} does not exist!")

    except Exception as e:
        print(f"Error in load_pretrained_rocmlm():\n  {e}")

    return model

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    from gfem import build_gfem_models
    try:
        gfems = []
        gfem_configs = ["assets/config_yamls/hydrated-shallow-upper-mantle-hp02m.yaml",
                        "assets/config_yamls/hydrated-shallow-upper-mantle-hp02r.yaml"]

        for yaml in gfem_configs:
            gfems.extend(build_gfem_models(config_yaml=yaml))

        rocmlm_config = "assets/config_yamls/rocmlm-default.yaml"
        rocmlms = train_rocmlms(gfems, config_yaml=rocmlm_config)

    except Exception as e:
        print(f"Error in main():\n  {e}")

if __name__ == "__main__":
    main()