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
import random
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
import torch.nn.functional as F
from scipy.integrate import simpson
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import Dataset, DataLoader, Subset
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
## .1.               PyTorch Nets                !!! ##
#######################################################
class SimpleNet(nn.Module):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        """
        super(SimpleNet, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        # Build the model layers
        layers = []
        prev_size = input_size

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LeakyReLU(inplace=True))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):
        """
        """
        return self.net(x)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params(self):
        """
        """
        params_info = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "num_layers": len(self.hidden_layer_sizes) + 1,
            "hidden_layer_sizes": self.hidden_layer_sizes
        }

        return params_info

#######################################################
class ImprovedNet(nn.Module):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, input_size, hidden_layer_sizes, output_size, dropout_rate=0.05):
        super(ImprovedNet, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        layers = []
        prev_size = input_size

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):
        """
        """
        return self.net(x)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params(self):
        """
        """
        params_info = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "dropout rate": self.dropout_rate,
            "num_layers": len(self.hidden_layer_sizes) + 1,
            "hidden_layer_sizes": self.hidden_layer_sizes
        }

        return params_info

#######################################################
class UNet(nn.Module):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, in_channels, out_channels, filters=[64, 128, 256, 512],
                 bilinear=True):
        """
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.bilinear = bilinear

        # Input convolution layer
        self.inc = self.DoubleConv(in_channels, filters[0])

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        in_filter = filters[0]
        for out_filter in filters[1:]:
            self.encoder.append(self.Down(in_filter, out_filter))
            in_filter = out_filter

        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for out_filter in reversed(filters[:-1]):
            self.decoder.append(self.Up(in_filter, out_filter, bilinear))
            in_filter = out_filter

        # Final output convolution layer
        self.outc = self.OutConv(filters[0], out_channels)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):
        """
        """
        skip_connections = []

        # Encoder (downsampling path)
        x = self.inc(x)
        skip_connections.append(x)

        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)

        # Decoder (upsampling path)
        skip_connections = skip_connections[:-1][::-1]
        for i, dec in enumerate(self.decoder):
            x = dec(x, skip_connections[i])

        # Final output layer
        x = self.outc(x)

        return x

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params(self):
        """
        """
        params_info = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "bilinear": self.bilinear,
            "num filters": len(self.filters) + 1,
            "filter sizes": self.filters
        }

        return params_info

    #######################################################
    class DoubleConv(nn.Module):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __init__(self, in_channels, out_channels, mid_channels=None):
            """
            """
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def forward(self, x):
            """
            """
            x = x.contiguous()
            return self.double_conv(x)

    #######################################################
    class Down(nn.Module):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __init__(self, in_channels, out_channels):
            """
            """
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                UNet.DoubleConv(in_channels, out_channels)
            )

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def forward(self, x):
            """
            """
            return self.maxpool_conv(x)

    #######################################################
    class Up(nn.Module):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __init__(self, in_channels, out_channels, bilinear=False):
            """
            """
            super().__init__()
            self.bilinear = bilinear

            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(
                    in_channels, in_channels // 2, kernel_size=2, stride=2)

            self.conv = UNet.DoubleConv(in_channels, out_channels)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def forward(self, x1, x2):
            """
            """
            x1 = self.up(x1)

            if self.bilinear:
                x1 = self.conv(x1)

            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            x = torch.cat([x2, x1], dim=1)

            return self.conv(x)

    #######################################################
    class OutConv(nn.Module):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def __init__(self, in_channels, out_channels):
            """
            """
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def forward(self, x):
            """
            """
            return self.conv(x)

#######################################################
## .2.             GFEMDataset Class             !!! ##
#######################################################
class GFEMDataset(Dataset):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, filepaths, features, targets, batch_size=128**2):
        """
        """
        self.filepaths = filepaths
        self.features = features
        self.targets = targets
        self.batch_size = batch_size

        self.file_idx_mapping = []
        self._create_file_idx_mapping()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _create_file_idx_mapping(self):
        """
        """
        global_idx = 0
        for file_idx, filepath in enumerate(self.filepaths):
            df_len = self._get_file_length(filepath)
            num_batches = df_len // self.batch_size
            for batch_idx in range(num_batches):
                self.file_idx_mapping.append(
                    (file_idx, batch_idx * self.batch_size, self.batch_size))

            remainder = df_len % self.batch_size
            if remainder > 0:
                self.file_idx_mapping.append(
                    (file_idx, num_batches * self.batch_size, remainder))
            global_idx += df_len

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_file_length(self, filepath):
        """
        """
        df = pd.read_csv(filepath, usecols=[0])
        return len(df)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_batch_data(self, filepath, start_idx, batch_size):
        """
        """
        data = pd.read_csv(filepath, skiprows=range(1, start_idx + 1), nrows=batch_size)
        feature_array = np.stack([data[feat].values for feat in self.features], axis=-1)
        target_array = np.stack([data[targ].values for targ in self.targets], axis=-1)
        return np.nan_to_num(feature_array), np.nan_to_num(target_array)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        """
        """
        return len(self.file_idx_mapping)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, idx):
        """
        """
        file_idx, start_idx, batch_size = self.file_idx_mapping[idx]
        filepath = self.filepaths[file_idx]
        return self._load_batch_data(filepath, start_idx, batch_size)

#######################################################
class ScaledGFEMDataset(Dataset):
    """
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, dataset, scaler_X, scaler_y):
        """
        """
        self.dataset = dataset
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        """
        """
        return len(self.dataset)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, idx):
        """
        """
        X_batch, y_batch = self.dataset[idx]
        X_batch_scaled = self.scaler_X.transform(X_batch)
        y_batch_scaled = self.scaler_y.transform(y_batch)
        return X_batch_scaled, y_batch_scaled

#######################################################
## .3.                RocMLM Class               !!! ##
#######################################################
class RocMLM:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, gfem_paths, ml_algo="DT", config_yaml=None, verbose=1):
        """
        """
        self.ml_algo = ml_algo
        self.verbose = verbose
        self.gfem_paths = gfem_paths
        self.sids = [f.split("/")[1].split("_")[0] for f in gfem_paths]
        self.config_yaml = config_yaml

        if (len(self.gfem_paths) == 0 or
                any(not os.path.exists(path) for path in self.gfem_paths)):
            raise Exception("Cannot load data from gfem_paths!")

        if self.ml_algo not in ["DT", "KN", "SimpleNet", "ImprovedNet", "UNet"]:
            raise Exception("Unrecognized ml_algo! Must be 'DT', 'KN', 'SimpleNet', "
                            "'ImprovedNet', " "or 'UNet' ...")

        # Determine torch device
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() and self.ml_algo in
            ["SimpleNet", "ImprovedNet", "UNet"] else "cpu")
        print(f"  Using device: {self.device}")

        self.model_trained = False
        self.model_cross_validated = False

        self._load_global_options()
        self._load_rocmlm_options()
        self._load_training_data()

        self.model_out_dir = "pretrained_rocmlms"
        self.model_prefix = (f"{self.ml_algo}-"
                             f"S{self.square_X_shape[0]}-"
                             f"F{self.square_X_shape[1]}-"
                             f"T{self.square_y_shape[1]}-"
                             f"R{self.square_y_shape[2]}")

        self.fig_dir = (f"figs/{self.model_out_dir}/{self.model_prefix}")
        self.rocmlm_path = f"{self.model_out_dir}/{self.model_prefix}.pkl"

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
                    "csv_batch_size": 1,
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
            self.csv_batch_size = global_options["csv_batch_size"]
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
                "phase_assemblage_variance": "",
                "SIO2":                      "wt.%",
                "AL203":                     "wt.%",
                "CAO":                       "wt.%",
                "MGO":                       "wt.%",
                "FEO":                       "wt.%",
                "NA20":                      "wt.%",
                "K2O":                       "wt.%",
                "TIO2":                      "wt.%",
                "CR2O3":                     "wt.%",
                "H2O":                       "wt.%",
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
                "phase_assemblage_variance": "%.0f",
                "SIO2":                      "%.1f",
                "AL203":                     "%.1f",
                "CAO":                       "%.1f",
                "MGO":                       "%.1f",
                "FEO":                       "%.1f",
                "NA20":                      "%.1f",
                "K2O":                       "%.1f",
                "TIO2":                      "%.1f",
                "CR2O3":                     "%.1f",
                "H2O":                       "%.1f",
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
                "phase_assemblage_variance": "Assemblage Variance",
                "SIO2":                      "SIO2",
                "AL203":                     "AL203",
                "CAO":                       "CAO",
                "MGO":                       "MGO",
                "FEO":                       "FEO",
                "NA20":                      "NA20",
                "K2O":                       "K2O",
                "TIO2":                      "TIO2",
                "CR2O3":                     "CR2O3",
                "H2O":                       "H2O",
            }

            np.random.seed(self.seed)

        except Exception as e:
            print(f"Error in _load_global_options(): {e}")

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
                        "cross_validate": True,
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
                        "cross_validate": True,
                        "label": "Decision Tree",
                        "hyperparams": {
                            "splitter": "best",
                            "min_samples_leaf": int(1),
                            "min_samples_split": int(2)
                        },
                        "grid_search": {
                            "splitter": ["best", "random"],
                            "min_samples_leaf": [int(2), int(3), int(4), int(5), int(6)],
                            "min_samples_split": [int(2), int(3), int(4), int(5), int(6)]
                        }
                    },
                    "SimpleNet": {
                        "tune": False,
                        "cross_validate": True,
                        "label": "SimpleNet",
                        "hyperparams": {
                            "hidden_layer_sizes": [int(16), int(32), int(64)],
                            "max_iter": 5000,
                            "learning_rate_init": 1e-5,
                            "gamma": 0.99,
                            "patience": 20,
                            "warmup_epochs": 200,
                            "warmup_start_lr": 1e-6,
                            "L2_regularization": False,
                            "weight_decay": 0.00001
                        },
                        "grid_search": {
                            "hidden_layer_sizes": [[int(16), int(32), int(64)]]
                        }
                    },
                    "ImprovedNet": {
                        "tune": False,
                        "cross_validate": True,
                        "label": "ImprovedNet",
                        "hyperparams": {
                            "hidden_layer_sizes": [int(64), int(128), int(256), int(512)],
                            "max_iter": 5000,
                            "learning_rate_init": 1e-5,
                            "gamma": 0.99,
                            "dropout_rate": 0.1,
                            "patience": 20,
                            "warmup_epochs": 200,
                            "warmup_start_lr": 1e-6,
                            "L2_regularization": True,
                            "weight_decay": 0.00001
                        },
                        "grid_search": {
                            "hidden_layer_sizes": [[int(64), int(128), int(256), int(512)]]
                        }
                    },
                    "UNet": {
                        "tune": False,
                        "cross_validate": True,
                        "label": "UNet",
                        "hyperparams": {
                            "filters": [int(32), int(64), int(128), int(256)],
                            "bilinear": True,
                            "max_iter": 5000,
                            "learning_rate_init": 1e-5,
                            "gamma": 0.99,
                            "patience": 20,
                            "warmup_epochs": 200,
                            "warmup_start_lr": 1e-6,
                            "L2_regularization": True,
                            "weight_decay": 0.00001
                        },
                        "grid_search": {
                            "filters": [[int(32), int(64), int(128), int(256)]]
                        }
                    }
                }

            # Assign values from rocmlm_options and rocmlm_options
            self.rocmlm_tune = rocmlm_options[self.ml_algo]["tune"]
            self.rocmlm_label = rocmlm_options[self.ml_algo]["label"]
            self.rocmlm_gridsearch = rocmlm_options[self.ml_algo]["grid_search"]
            self.default_hyperparams = rocmlm_options[self.ml_algo]["hyperparams"]
            self.rocmlm_cross_validate = rocmlm_options[self.ml_algo]["cross_validate"]

            self.lr = None
            self.gamma = None
            self.bilinear = None
            self.max_iter = None
            self.patience = None
            self.dropout_rate = None
            self.warmup_epochs = None
            self.warmup_start_lr = None
            self.L2_regularization = None
            self.weight_decay = None

            if self.ml_algo == "KN":
                self.default_rocmlm = KNeighborsRegressor(
                    weights=self.default_hyperparams["weights"],
                    n_neighbors=self.default_hyperparams["n_neighbors"]
                )
            elif self.ml_algo == "DT":
                self.default_rocmlm = DecisionTreeRegressor(
                    splitter=self.default_hyperparams["splitter"],
                    min_samples_leaf=self.default_hyperparams["min_samples_leaf"],
                    min_samples_split=self.default_hyperparams["min_samples_split"],
                    random_state=self.seed
                )
            elif self.ml_algo in ["SimpleNet", "ImprovedNet", "UNet"]:
                if self.ml_algo == "SimpleNet":
                    self.default_rocmlm = SimpleNet(
                        input_size=len(self.rocmlm_features),
                        hidden_layer_sizes=self.default_hyperparams["hidden_layer_sizes"],
                        output_size=len(self.rocmlm_targets)
                    ).to(self.device)
                elif self.ml_algo == "ImprovedNet":
                    self.default_rocmlm = ImprovedNet(
                        input_size=len(self.rocmlm_features),
                        hidden_layer_sizes=self.default_hyperparams["hidden_layer_sizes"],
                        output_size=len(self.rocmlm_targets),
                        dropout_rate=self.default_hyperparams["dropout_rate"]
                    ).to(self.device)
                    self.dropout_rate = self.default_hyperparams["dropout_rate"]
                elif self.ml_algo == "UNet":
                    self.default_rocmlm = UNet(
                        in_channels=len(self.rocmlm_features),
                        out_channels=len(self.rocmlm_targets),
                        filters=self.default_hyperparams["filters"]
                    ).to(self.device)
                    self.bilinear = self.default_hyperparams["bilinear"]
                self.max_iter = self.default_hyperparams["max_iter"]
                self.lr = self.default_hyperparams["learning_rate_init"]
                self.gamma = self.default_hyperparams["gamma"]
                self.patience = self.default_hyperparams["patience"]
                self.warmup_epochs = self.default_hyperparams["warmup_epochs"]
                self.warmup_start_lr = self.default_hyperparams["warmup_start_lr"]
                self.L2_regularization = self.default_hyperparams["L2_regularization"]
                self.weight_decay = self.default_hyperparams["weight_decay"]

        except Exception as e:
            print(f"Error in _load_rocmlm_options(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _shape_for_unet(self, input_array, array_type):
        """
        """
        try:
            if array_type == "features":
                square_shape = self.square_X_shape
            elif array_type == "targets":
                square_shape = self.square_y_shape

            C = square_shape[1]
            R = square_shape[2]

            # Reshape training data for UNet (S, F/T, R, R)
            input_array = input_array.reshape(-1, C, R, R)

        except Exception as e:
            print(f"Error in _shape_for_unet(): {e}")
            return None

        return input_array

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _unshape_from_unet(self, input_array, array_type):
        """
        """
        try:
            if array_type == "features":
                square_shape = self.square_X_shape
            elif array_type == "targets":
                square_shape = self.square_y_shape

            C = square_shape[1]

            # Unshape training data from UNet (S*R*R, F/T)
            input_array = input_array.reshape(-1, C)

        except Exception as e:
            print(f"Error in _unshape_from_unet(): {e}")
            return None

        return input_array

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _collate_fn(self, batch):
        """
        """
        return_torch = False if self.ml_algo in ["KN", "DT"] else True

        X, y = zip(*batch)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        if return_torch:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            return X_tensor, y_tensor
        else:
            return X.astype(np.float32), y.astype(np.float32)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _collate_fn_unet(self, batch):
        """
        """
        return_torch = False if self.ml_algo in ["KN", "DT"] else True

        X, y = zip(*batch)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        X = self._shape_for_unet(X, "features")
        y = self._shape_for_unet(y, "targets")

        if return_torch:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            return X_tensor, y_tensor
        else:
            return X.astype(np.float32), y.astype(np.float32)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scale_dataloader(self, dataloader):
        """
        """
        try:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            print("  Fitting scalers to the training dataset ...")
            for X_batch, y_batch in dataloader:
                scaler_X.partial_fit(X_batch)
                scaler_y.partial_fit(y_batch)

            scaled_dataset = ScaledGFEMDataset(dataloader.dataset, scaler_X, scaler_y)

            if self.ml_algo == "UNet":
                unet_shape = True

            scaled_dataloader = DataLoader(
                scaled_dataset, batch_size=self.csv_batch_size, collate_fn=self._collate_fn)

        except Exception as e:
            print(f"Error in _scale_dataloader(): {e}")
            return None, None, None

        return scaled_dataloader, scaler_X, scaler_y

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _train_test_split_dataloader(self, dataloader, test_size=0.2):
        """
        """
        try:
            if self.ml_algo == "UNet":
                collate_fn = self._collate_fn_unet
            else:
                collate_fn = self._collate_fn

            n_samples = len(dataloader.dataset)
            indices = list(range(n_samples))
            np.random.shuffle(indices)

            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=self.seed)

            train_loader = DataLoader(
                torch.utils.data.Subset(dataloader.dataset, train_idx),
                batch_size=self.csv_batch_size, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(
                torch.utils.data.Subset(dataloader.dataset, train_idx),
                batch_size=self.csv_batch_size, shuffle=False, collate_fn=collate_fn)

        except Exception as e:
            print(f"Error in _train_test_split_dataloader(): {e}")
            return None, None

        return train_loader, test_loader

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_training_data(self):
        """
        """
        try:
            # Get total dataset size and shape
            # S: number of gfem models
            # R: PT resolution (squared length of csv file)
            # F: number of training features
            # T: number of training targets
            df = pd.read_csv(self.gfem_paths[0])
            S = len(self.gfem_paths)
            R = int(np.sqrt(len(df)))
            F = len(self.rocmlm_features)
            T = len(self.rocmlm_targets)
            self.P_min, self.P_max = df["P"].min(), df["P"].max()
            self.T_min, self.T_max = df["T"].min(), df["T"].max()
            self.square_X_shape = (S, F, R, R)
            self.square_y_shape = (S, T, R, R)

            data = GFEMDataset(
                self.gfem_paths, self.rocmlm_features, self.rocmlm_targets)

            self.loader_raw = DataLoader(
                data, batch_size=self.csv_batch_size, collate_fn=self._collate_fn)

            self.loader_scaled, self.scaler_X, self.scaler_y = \
                self._scale_dataloader(self.loader_raw)

            self.train_loader, self.test_loader = \
                self._train_test_split_dataloader(self.loader_scaled)

        except Exception as e:
            print(f"Error in _load_training_data(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_pretrained_rocmlm(self, rocmlm_path):
        """
        """
        try:
            if (os.path.exists(rocmlm_path)):
                if self.verbose >= 1:
                    print(f"  Loading RocMLM object from {rocmlm_path} ...")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                # Load RocMLM object
                with open(rocmlm_path, "rb") as file:
                    loaded_rocmlm = joblib.load(file)

                # Update the current instance
                self.__dict__.update(loaded_rocmlm.__dict__)

                if self.rocmlm is None:
                    raise Exception("RocMLM not loaded properly!")
            else:
                print(f"File {rocmlm_path} does not exist!")

        except Exception as e:
            print(f"Error in _load_pretrained_rocmlm(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_existing_model(self):
        """
        """
        try:
            if os.path.exists(self.rocmlm_path):
                if self.verbose >= 1:
                    print(f"  Found pretrained model {self.model_prefix}!")
                self._load_pretrained_rocmlm(self.rocmlm_path)
            else:
                os.makedirs(self.model_out_dir, exist_ok=True)

        except Exception as e:
            print(f"Error in _check_existing_model(): {e}")

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.               RocMLMs                   !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _warmup_lr(self, epoch, warmup_epochs, base_lr, warmup_start_lr):
        """
        """
        try:
            if epoch < warmup_epochs:
                lr = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)
            else:
                lr = base_lr

        except Exception as e:
            print(f"Error in _warmup_lr(): {e}")
            return None

        return lr

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _reset_all_weights(self, model):
        """
        """
        try:
            def weight_reset(m):
                reset_parameters = getattr(m, "reset_parameters", None)
                if callable(reset_parameters):
                    m.reset_parameters()

            with torch.no_grad():
                model.apply(weight_reset)

        except Exception as e:
            print(f"Error in _reset_all_weights(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _do_inference(self, X):
        """
        """
        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            if hasattr(self.rocmlm, "eval"):
                self.rocmlm.eval()

            if self.ml_algo == "UNet":
                X = self._shape_for_unet(X, "features")

            with torch.no_grad():
                if hasattr(self.rocmlm, "predict"):
                    y_pred = self.rocmlm.predict(X)
                else:
                    X = torch.tensor(X, dtype=torch.float32).to(self.device)
                    y_pred = self.rocmlm(X)
                    y_pred = y_pred.detach().cpu().numpy()

            if self.ml_algo == "UNet":
                y_pred = self._unshape_from_unet(y_pred, "targets")

        except Exception as e:
            print(f"Error in _do_inference(): {e}")
            return None

        return y_pred

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _do_batch_inference(self, dataloader):
        """
        """
        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            y_pred_list = []

            if hasattr(self.rocmlm, "eval"):
                self.rocmlm.eval()

            with torch.no_grad():
                for X_batch, _ in dataloader:
                    X_batch = X_batch

                    if hasattr(self.rocmlm, "predict"):
                        batch_y_pred = self.rocmlm.predict(X_batch)
                        y_pred_list.append(batch_y_pred)
                    else:
                        batch_y_pred = self.rocmlm(X_batch.to(self.device))
                        y_pred_list.append(batch_y_pred.detach().cpu().numpy())

            y_pred = np.concatenate(y_pred_list, axis=0)

        except Exception as e:
            print(f"Error in _do_batch_inference(): {e}")
            return None

        return y_pred

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_all_data_from_dataloader(self, dataloader):
        """
        """
        try:
            X_list = []
            y_list = []

            for X_batch, y_batch in dataloader:
                X_list.append(X_batch)
                y_list.append(y_batch)

            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)

        except Exception as e:
            print(f"Error in _load_all_data_from_dataloader(): {e}")

        return X, y

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _tune_scikit_model(self, X, y):
        """
        """
        try:
            X, y = self._load_all_data_from_dataloader(self.loader_scaled)

            model = self.default_rocmlm
            rocmlm_gridsearch = self.rocmlm_gridsearch

            n_samples = len(self.loader_scaled.dataset)
            n_splits = min(self.kfolds, n_samples)
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
                    min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                    min_samples_split=grid_search.best_params_["min_samples_split"],
                    random_state=self.seed
                )

            self.rocmlm = model
            self.rocmlm_hyperparams = self.rocmlm.get_params()

        except Exception as e:
            print(f"Error in _tune_scikit_model(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _tune_torch_net(self):
        """
        """
        try:
            if self.ml_algo == "UNet":
                collate_fn = self._collate_fn_unet
            else:
                collate_fn = self._collate_fn

            best_params = None
            best_score = float("inf")
            rocmlm_gridsearch = self.rocmlm_gridsearch

            n_samples = len(self.loader_scaled.dataset)
            n_splits = min(self.kfolds, n_samples)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

            for i, params in enumerate(ParameterGrid(rocmlm_gridsearch)):
                fold_score, fold_auc, fold_var, fold_combined_score = [], [], [], []

                for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
                    min_train_loss, max_train_loss = float("inf"), float("-inf")
                    min_test_loss, max_test_loss = float("inf"), float("-inf")
                    epoch_, train_loss_, test_loss_ = [], [], []

                    train_loader = DataLoader(
                        torch.utils.data.Subset(self.loader_scaled.dataset, train_idx),
                        batch_size=self.csv_batch_size, shuffle=True, collate_fn=collate_fn)
                    test_loader = DataLoader(
                        torch.utils.data.Subset(self.loader_scaled.dataset, train_idx),
                        batch_size=self.csv_batch_size, shuffle=False, collate_fn=collate_fn)

                    # Configure new model
                    if self.ml_algo == "SimpleNet":
                        model = SimpleNet(
                            input_size=X.shape[1],
                            hidden_layer_sizes=params["hidden_layer_sizes"],
                            output_size=y.shape[1]
                        ).to(self.device)
                    elif self.ml_algo == "ImprovedNet":
                        model = ImprovedNet(
                            input_size=X.shape[1],
                            hidden_layer_sizes=params["hidden_layer_sizes"],
                            output_size=y.shape[1],
                            dropout_rate=self.dropout_rate
                        ).to(self.device)
                    elif self.ml_algo == "UNet":
                        model = UNet(
                            in_channels=X.shape[1],
                            filters=params["filters"],
                            out_channels=y.shape[1]
                        ).to(self.device)

                    # Reset weights before training
                    print("  Resetting model weights ...")
                    self._reset_all_weights(model)

                    # Set model to training mode
                    model.train()

                    # Optimizer and loss function
                    if self.L2_regularization:
                        optimizer = optim.Adam(model.parameters(), lr=self.lr,
                                               weight_decay=self.weight_decay)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=self.lr)

                    loss_fn = nn.MSELoss()

                    # Define the learning rate scheduler
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

                    # Early stopping parameters
                    best_test_loss = float("inf")
                    patience_counter = 0

                    # Training loop with progress bar
                    desc = f"  Training {self.rocmlm_label} ..."
                    with tqdm(total=self.max_iter, desc=desc, position=0) as pbar:
                        for epoch in range(self.max_iter):
                            # Warm-up learning rate schedule
                            if epoch < self.warmup_epochs:
                                current_lr = self._warmup_lr(
                                    epoch, self.warmup_epochs, self.lr, self.warmup_start_lr)
                            else:
                                scheduler.step()
                                current_lr = optimizer.param_groups[0]["lr"]

                            train_loss = 0
                            model.train()

                            for X_batch, y_batch in train_loader:
                                X_batch = X_batch.to(self.device)
                                y_batch = y_batch.to(self.device)

                                # Forward pass and backpropagation
                                optimizer.zero_grad()
                                y_pred = model(X_batch)
                                loss = loss_fn(y_pred, y_batch)
                                loss.backward()
                                optimizer.step()
                                train_loss += loss

                            # Evaluate on test set
                            test_loss = 0
                            model.eval()
                            with torch.no_grad():
                                for X_batch, y_batch in test_loader:
                                    X_batch = X_batch.to(self.device)
                                    y_batch = y_batch.to(self.device)
                                    y_pred = model(X_batch)
                                    loss = loss_fn(y_pred, y_batch)
                                    test_loss += loss

                            train_loss /= len(train_loader)
                            test_loss /= len(test_loader)
                            train_loss_.append(train_loss)
                            test_loss_.append(test_loss)

                            # After warm-up, step the scheduler
                            if epoch >= self.warmup_epochs:
                                scheduler.step()

                            # Update min and max loss values
                            min_train_loss = min(min_train_loss, train_loss)
                            max_train_loss = max(max_train_loss, train_loss)
                            min_test_loss = min(min_test_loss, test_loss)
                            max_test_loss = max(max_test_loss, test_loss)

                            epoch_.append(epoch + 1)
                            pbar.update(1)

                            # Early stopping
                            if test_loss < best_test_loss:
                                best_test_loss = test_loss
                                patience_counter = 0
                            else:
                                patience_counter += 1

                            # Check if patience has been exceeded
                            if (patience_counter >= self.patience and
                                    epoch >= 2 * self.warmup_epochs):
                                break

                    # Normalize losses
                    norm_train_loss_ = [
                        (t - min_train_loss) / (max_train_loss - min_train_loss + 1e-8)
                        for t in train_loss_]
                    norm_test_loss_ = [
                        (t - min_test_loss) / (max_test_loss - min_test_loss + 1e-8)
                        for t in test_loss_]
                    norm_train_loss_ = [
                        loss.detach().cpu().numpy() for loss in norm_train_loss_]
                    norm_test_loss_ = [
                        loss.detach().cpu().numpy() for loss in norm_test_loss_]

                    # Save loss curves
                    loss_curve = {"epoch": epoch_, "train_loss": norm_train_loss_,
                                  "test_loss": norm_test_loss_}
                    self._visualize_loss_curve(loss_curve)

                    # Calculate variance of test loss
                    var_test_loss = np.var(norm_test_loss_)
                    fold_var.append(var_test_loss)

                    # Calculate average loss score
                    avg_train_loss = np.mean(norm_train_loss_)
                    avg_test_loss = np.mean(norm_test_loss_)
                    avg_score = avg_test_loss
                    fold_score.append(avg_score)

                    # Calculate AUC for training and test loss curves
                    auc_train_loss = simpson(norm_train_loss_, dx=1) / self.max_iter
                    auc_test_loss = simpson(norm_test_loss_, dx=1) / self.max_iter
                    fold_auc.append(auc_test_loss)

                    combined_score = (
                        (0.33 * avg_score) + (0.33 * var_test_loss) + (0.33 * auc_test_loss))
                    fold_combined_score.append(combined_score)

                    # Average score across folds
                    print(f"--------------------")
                    print(f"  Average fold loss:    {avg_score:.4f} (normalized)")
                    print(f"  Average fold var:     {var_test_loss:.4f} (normalized)")
                    print(f"  Average fold auc:     {auc_test_loss:.4f} (normalized)")
                    print(f"  Combined fold score:  {combined_score:.4f}")

                # Calculate average fold score
                avg_combined_score = np.mean(fold_combined_score)

                if avg_combined_score < best_score:
                    print(f"********************")
                    print(f"  Previous best score: {best_score:.4f}")
                    print(f"  New best score: {avg_combined_score:.4f}")
                    print(f"  New best parameters:")
                    for name, value in params.items():
                        print(f"    {name}: {value}")
                    best_score = avg_combined_score
                    best_params = params
                else:
                    print(f"********************")
                    print(f"  No new best score ...")

            if best_params:
                if self.ml_algo == "SimpleNet":
                    best_model = SimpleNet(
                        input_size=X.shape[1],
                        hidden_layer_sizes=best_params["hidden_layer_sizes"],
                        output_size=y.shape[1]
                    ).to(self.device)
                elif self.ml_algo == "ImprovedNet":
                    best_model = ImprovedNet(
                        input_size=X.shape[1],
                        hidden_layer_sizes=best_params["hidden_layer_sizes"],
                        output_size=y.shape[1],
                        dropout_rate=self.dropout_rate
                    ).to(self.device)
                elif self.ml_algo == "UNet":
                    best_model = UNet(
                        in_channels=X.shape[1],
                        filters=best_params["filters"],
                        out_channels=y.shape[1]
                    ).to(self.device)

                self.rocmlm = best_model

            self.rocmlm_hyperparams = self.rocmlm.get_params()

        except Exception as e:
            print(f"Error in _tune_torch_net(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_rocmlm(self):
        """
        """
        try:
            if self.verbose >= 1:
                print(f"  Configuring model {self.model_prefix} ...")

            if self.rocmlm_tune:
                if self.verbose >= 1:
                    print(f"  Tuning model {self.model_prefix} ...")

                if self.ml_algo in ["DT", "KN"]:
                    self._tune_scikit_model(X_scaled, y_scaled)
                elif self.ml_algo in ["SimpleNet", "ImprovedNet", "UNet"]:
                    self._tune_torch_net(X_scaled, y_scaled)

            else:
                self.rocmlm = self.default_rocmlm
                self.rocmlm_hyperparams = self.rocmlm.get_params()

        except Exception as e:
            print(f"Error in _configure_rocmlm(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _print_rocmlm_info(self):
        """
        """
        try:
            S = self.square_X_shape[0]
            F = self.square_X_shape[1]
            R = self.square_X_shape[2]
            T = self.square_y_shape[1]
            tgwrp = textwrap.fill(", ".join(self.rocmlm_targets), width=80,
                                  subsequent_indent="                  ")
            ftwrp = textwrap.fill(", ".join(self.rocmlm_features), width=80,
                                  subsequent_indent="                  ")

            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print(f"RocMLM: {self.model_prefix}")
            print("---------------------------------------------")
            print(f"    ML model:             {self.rocmlm_label}")
            print(f"    n gfem models:        {len(self.loader_scaled.dataset)}")
            print(f"    gfem batch size:      {self.csv_batch_size}")
            print(f"    total dataset size:   {S*R*R}")
            print(f"    X array shape:        {(S*R*R, F)}")
            print(f"    y array shape:        {(S*R*R, T)}")
            print(f"    X square shape:       {(S, F, R, R)}")
            print(f"    y square shape:       {(S, T, R, R)}")
            print(f"    kfold tuning:         {self.rocmlm_tune}")
            print(f"    kfold CV:             {self.rocmlm_cross_validate}")
            print(f"    kfolds:               {self.kfolds}")
            print(f"    max iter:             {self.max_iter}")
            print(f"    warmup epochs:        {self.warmup_epochs}")
            print(f"    warmup LR:            {self.warmup_start_lr}")
            print(f"    base LR:              {self.lr}")
            print(f"    LR decay:             {self.gamma}")
            print(f"    dropout rate:         {self.dropout_rate}")
            print(f"    patience:             {self.patience}")
            print(f"    L2 regularization:    {self.L2_regularization}")
            print(f"    weight decay:         {self.weight_decay}")
            print(f"    features:             {ftwrp}")
            print(f"    targets:              {tgwrp}")
            print(f"    hyperparameters:")
            for key, value in self.rocmlm_hyperparams.items():
                print(f"        {key}: {value}")
            print("---------------------------------------------")

        except Exception as e:
            print(f"Error in _print_rocmlm_info(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _train_torch_net(self, train_loader, test_loader):
        """
        """
        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            min_train_loss, max_train_loss = float("inf"), float("-inf")
            min_test_loss, max_test_loss = float("inf"), float("-inf")
            epoch_, train_loss_, test_loss_ = [], [], []

            # Reset weights before training
            print("  Resetting model weights ...")
            self._reset_all_weights(self.rocmlm)

            # Put model on device
            self.rocmlm.to(self.device)

            # Optimizer and loss function
            if self.L2_regularization:
                optimizer = optim.Adam(self.rocmlm.parameters(), lr=self.lr,
                                       weight_decay=self.weight_decay)
            else:
                optimizer = optim.Adam(self.rocmlm.parameters(), lr=self.lr)

            loss_fn = nn.MSELoss()

            # Define the learning rate scheduler
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

            # Early stopping parameters
            best_test_loss = float("inf")
            patience_counter = 0

            # Training loop with progress bar
            desc = f"  Training {self.rocmlm_label} ..."
            with tqdm(total=self.max_iter, desc=desc, position=0) as pbar:
                for epoch in range(self.max_iter):
                    # Warm-up learning rate schedule
                    if epoch < self.warmup_epochs:
                        current_lr = self._warmup_lr(
                            epoch, self.warmup_epochs, self.lr, self.warmup_start_lr)
                    else:
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]["lr"]

                    train_loss = 0
                    self.rocmlm.train()

                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        # Forward pass and backpropagation
                        optimizer.zero_grad()
                        y_pred = self.rocmlm(X_batch)
                        loss = loss_fn(y_pred, y_batch)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss

                    # Evaluate on test set
                    test_loss = 0
                    self.rocmlm.eval()
                    with torch.no_grad():
                        for X_batch, y_batch in test_loader:
                            X_batch = X_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            y_pred = self.rocmlm(X_batch)
                            loss = loss_fn(y_pred, y_batch)
                            test_loss += loss

                    train_loss /= len(train_loader)
                    test_loss /= len(test_loader)
                    train_loss_.append(train_loss)
                    test_loss_.append(test_loss)

                    # After warm-up, step the scheduler
                    if epoch >= self.warmup_epochs:
                        scheduler.step()

                    # Update min and max loss values
                    min_train_loss = min(min_train_loss, train_loss)
                    max_train_loss = max(max_train_loss, train_loss)
                    min_test_loss = min(min_test_loss, test_loss)
                    max_test_loss = max(max_test_loss, test_loss)

                    epoch_.append(epoch + 1)
                    pbar.update(1)

                    # Early stopping
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Check if patience has been exceeded
                    if (patience_counter >= self.patience and
                            epoch >= 2 * self.warmup_epochs):
                        break

            # Normalize losses
            norm_train_loss_ = [
                (t - min_train_loss) / (max_train_loss - min_train_loss + 1e-8)
                for t in train_loss_]
            norm_test_loss_ = [
                (t - min_test_loss) / (max_test_loss - min_test_loss + 1e-8)
                for t in test_loss_]
            norm_train_loss_ = [loss.detach().cpu().numpy() for loss in norm_train_loss_]
            norm_test_loss_ = [loss.detach().cpu().numpy() for loss in norm_test_loss_]

            # Save loss curves
            loss_curve = {"epoch": epoch_, "train_loss": norm_train_loss_,
                          "test_loss": norm_test_loss_}

        except Exception as e:
            print(f"Error in _train_torch_net(): {e}")
            traceback.print_exc()
            return None

        return loss_curve

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_kfold_results(self, results):
        """
        """
        try:
            S = self.square_X_shape[0]
            R = self.square_X_shape[-1]
            sample_label = f"SMA{S}"

            training_times = []
            rmse_test_scores, r2_test_scores = [], []

            for (rmse_test, r2_test, training_time) in results:
                rmse_test_scores.append(rmse_test)
                r2_test_scores.append(r2_test)
                training_times.append(training_time)

            rmse_test_scores = np.stack(rmse_test_scores)
            r2_test_scores = np.stack(r2_test_scores)
            rmse_test_mean = np.mean(rmse_test_scores, axis=0)
            rmse_test_std = np.std(rmse_test_scores, axis=0)
            r2_test_mean = np.mean(r2_test_scores, axis=0)
            r2_test_std = np.std(r2_test_scores, axis=0)
            training_time_mean = np.mean(training_times)
            training_time_std = np.std(training_times)

            cv_info = {
                "model": [self.ml_algo],
                "sample": [sample_label],
                "size": [R],
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
            print(f"Error in _process_kfold_results(): {e}")

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
            print(f"Error in _save_rocmlm_cv_info(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _kfold_itr(self, train_loader, test_loader):
        """
        """
        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            training_start_time = time.time()

            if self.ml_algo in ["SimpleNet", "ImprovedNet", "UNet"]:
                loss_curve = self._train_torch_net(train_loader, test_loader)
                self._visualize_loss_curve(loss_curve)
            else:
                self._train_scikit_learn_model(train_loader)

            training_end_time = time.time()
            training_time = training_end_time - training_start_time

            y_pred = self._do_batch_inference(test_loader)

            y_test_list = [y_batch for _, y_batch in test_loader]
            if self.ml_algo in ["KN", "DT"]:
                y_test = np.concatenate(y_test_list, axis=0)
            else:
                y_test = torch.cat(y_test_list).numpy()

            if self.ml_algo == "UNet":
                y_test = self._unshape_from_unet(y_test, "targets")
                y_pred = self._unshape_from_unet(y_pred, "targets")

            # Calculate RMSE and R2 score
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput="raw_values"))
            r2 = r2_score(y_test, y_pred, multioutput="raw_values")

            # Normalize RMSE
            y_range = np.ptp(y_test, axis=0)
            np.seterr(divide="ignore", invalid="ignore")
            normalized_rmse = (rmse / y_range) * 100

        except Exception as e:
            print(f"Error in _kfold_itr(): {e}")
            traceback.print_exc()
            return (None, None, None, None)

        return (normalized_rmse, r2, training_time)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _kfold_cv(self):
        """
        """
        try:
            if self.ml_algo == "UNet":
                collate_fn = self._collate_fn_unet
            else:
                collate_fn = self._collate_fn

            n_samples = len(self.loader_scaled.dataset)
            n_splits = min(self.kfolds, n_samples)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

            print("  Cross-validating RocMLM ...")
            fold_results = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
                train_loader = DataLoader(
                    torch.utils.data.Subset(self.loader_scaled.dataset, train_idx),
                    batch_size=self.csv_batch_size, shuffle=True, collate_fn=collate_fn)
                test_loader = DataLoader(
                    torch.utils.data.Subset(self.loader_scaled.dataset, train_idx),
                    batch_size=self.csv_batch_size, shuffle=False, collate_fn=collate_fn)

                results = self._kfold_itr(train_loader, test_loader)
                fold_results.append(results)

            self.model_cross_validated = True
            self._process_kfold_results(fold_results)

        except Exception as e:
            print(f"Error in _kfold_cv(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _train_scikit_learn_model(self, train_loader):
        """
        """
        try:
            X, y = self._load_all_data_from_dataloader(train_loader)
            self.rocmlm.fit(X, y)

        except Exception as e:
            print(f"Error in _train_scikit_learn_model(): {e}")
            traceback.print_exc()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _fit_training_data(self):
        """
        """
        try:
            if self.rocmlm is None:
                raise Exception("No ML model! Call _configure_rocmlm() first ...")

            if not self.model_cross_validated:
                print("  Warning: ML model not cross validated. "
                      "Cannot provide performance metrics ...")

            if self.ml_algo in ["SimpleNet", "ImprovedNet", "UNet"]:
                loss_curve = self._train_torch_net(self.train_loader, self.test_loader)
                self._visualize_loss_curve(loss_curve)
            else:
                print(f"  Training model {self.model_prefix} ...")
                self._train_scikit_learn_model(self.train_loader)

            self.model_trained = True

            with open(self.rocmlm_path, "wb") as file:
                joblib.dump(self, file, compress=3)

        except Exception as e:
            print(f"Error in _fit_training_data(): {e}")
            traceback.print_exc()

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
            print(f"Error in _get_subduction_geotherm(): {e}")
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
            print(f"Error in _get_mantle_geotherm(): {e}")
            return None

        return geotherm

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_rocmlm_images(self, idx, plot_type="predictions"):
        """
        """
        try:
            if plot_type not in {"targets", "predictions", "diff"}:
                raise Exception("Unrecognized array image plot_type!")

            check = True

            for target in self.rocmlm_targets:
                path = (f"{self.fig_dir}/{self.model_prefix}-{self.sids[idx]}-"
                        f"{target.replace('_', '-')}-{plot_type}.png")
                if not os.path.exists(path):
                    check = False
                    break

        except Exception as e:
            print(f"Error in _check_rocmlm_images(): {e}")
            return None

        return check

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _visualize_loss_curve(self, loss_curve, figwidth=6.3, figheight=3.54, fontsize=14):
        """
        """
        try:
            os.makedirs(f"{self.fig_dir}", exist_ok=True)

            # Base filename
            base_filename = f"{self.fig_dir}/{self.ml_algo}-loss-curve"
            filename = f"{base_filename}-000.png"

            # Check if the file exists and append a number if necessary
            file_count = 0
            while os.path.isfile(filename):
                filename = f"{base_filename}-{str(file_count).zfill(3)}.png"
                file_count += 1

            df = pd.DataFrame.from_dict(loss_curve, orient="index").transpose()
            df.sort_values(by="epoch", inplace=True)
            df = df[df['epoch'] > df['epoch'].min()]

            plt.rcParams["font.size"] = fontsize

            fig = plt.figure(figsize=(figwidth, figheight))
            colormap = plt.get_cmap("tab10")
            plt.plot(df["epoch"], df["train_loss"], label="train loss", color=colormap(0))
            plt.plot(df["epoch"], df["test_loss"], label="test loss", color=colormap(1))
            plt.xlabel("Epoch")
            plt.ylabel(f"Loss")
            plt.title(f"{self.rocmlm_label} Loss Curve")
            plt.legend()

            plt.savefig(filename)
            plt.close()
            print(f"  Figure saved to: {filename} ...")

        except Exception as e:
            print(f"Error in _visualize_loss_curve(): {e}")
            traceback.print_exc()

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
    def _get_square_target_for_array_image(self, idx, target, target_idx, pred_array,
                                           target_array, plot_type):
        """
        """
        try:
            p = pred_array[:, :, target_idx]
            t = target_array[:, :, target_idx]
            rmse, r2 = None, None

            if plot_type == "predictions":
                square_target = p
                filename = (f"{self.model_prefix}-{self.sids[idx]}-"
                            f"{target.replace('_', '-')}-predictions.png")
            elif plot_type == "targets":
                square_target = t
                filename = (f"{self.model_prefix}-{self.sids[idx]}-"
                            f"{target.replace('_', '-')}-targets.png")
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

                filename = (f"{self.model_prefix}-{self.sids[idx]}-"
                            f"{target.replace('_', '-')}-diff.png")

        except Exception as e:
            print(f"Error in _get_square_target_for_array_image(): {e}")
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
    def _visualize_array_image(self, idx, plot_type="targets", geotherm_type=None,
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

            X, y = self.loader_raw.dataset[idx]
            X_scaled, y_scaled = self.loader_scaled.dataset[idx]

            P, T = X[:, 0], X[:, 1]
            extent = [np.nanmin(T), np.nanmax(T), np.nanmin(P), np.nanmax(P)]

            y_pred = self._do_inference(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred)

            F = self.square_X_shape[1]
            T = self.square_y_shape[1]
            R = self.square_y_shape[2]

            feature_array = X.reshape(R, R, F)
            target_array = y.reshape(R, R, T)
            pred_array = y_pred.reshape(R, R, T)

            if geotherm_type:
                geotherms = self._get_geotherms_for_array_image(geotherm_type)

            for i, target in enumerate(self.rocmlm_targets):
                target_label = self.target_labels_map[target]
                title = (f"{target_label} ({self.target_units_map.get(target, '')})")

                # Get square array
                square_target, filename, r2, rmse = self._get_square_target_for_array_image(
                    idx, target, i, pred_array, target_array, plot_type)

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
            print(f"Error in _visualize_array_image(): {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize(self, indices=[0, 1]):
        """
        """
        try:
            for i  in indices:
                if not self._check_rocmlm_images(i, "diff"):
                    self._visualize_array_image(i, "diff")
                if not self._check_rocmlm_images(i, "targets"):
                    self._visualize_array_image(i, "targets")
                if not self._check_rocmlm_images(i, "predictions"):
                    self._visualize_array_image(i, "predictions")

        except Exception as e:
            print(f"Error in visualize_rocmlm(): {e}")

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
                print(f"Error in train(): {e}")
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
            if self.rocmlm is None:
                raise Exception("No RocMLM! Load or train RocMLM first ...")

            if not self.model_trained:
                raise Exception("No RocMLM! Call train() first ...")

            training_features = self.rocmlm_features
            missing_features = [f for f in training_features if f not in features]

            if missing_features:
                raise Exception(f"Missing required features: {missing_features}")

            for key, val in features.items():
                if not isinstance(val, (list, np.ndarray)):
                    raise TypeError(f"Feature {key} must be a list or numpy array!")

            X_list = [np.asarray(features[f]).reshape(-1, 1) for f in training_features]
            X_lengths = [arr.shape[0] for arr in X_list]

            if len(set(X_lengths)) != 1:
                raise Exception("All feature arrays must have the same length!")

            X = np.concatenate(X_list, axis=1)
            X = np.nan_to_num(X)

            X_scaled = self.scaler_X.transform(X)

            inference_start_time = time.time()
            y_pred = self._do_inference(X)
            inference_end_time = time.time()

            inference_time = (inference_end_time - inference_start_time) * 1e3
            inference_time_per_node = inference_time / X.shape[0]

            y_pred = self.scaler_y.inverse_transform(y_pred)

            print(f"  {X.shape[0]} nodes completed in {inference_time:.4f} "
                  f"milliseconds ({inference_time_per_node:.4f} ms per node)...")

        except Exception as e:
            print(f"Error in inference(): {e}")
            return None

        return y_pred

#######################################################
## .4.              Train RocMLMs                !!! ##
#######################################################
def train_rocmlms(gfem_models, ml_algos=["DT", "KN", "SimpleNet", "ImprovedNet", "UNet"],
                  config_yaml=None):
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
        print(f"Error in train_rocmlms(): {e}")
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

            with open(rocmlm_path, "rb") as file:
                model = joblib.load(file)

            if model.rocmlm is None:
                raise Exception("RocMLM not loaded properly!")
        else:
            print(f"File {rocmlm_path} does not exist!")

    except Exception as e:
        print(f"Error in load_pretrained_rocmlm(): {e}")

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
        print(f"Error in main(): {e}")

if __name__ == "__main__":
    main()