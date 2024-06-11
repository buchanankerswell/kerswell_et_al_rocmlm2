#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import math
import warnings
import traceback
from scipy import stats

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#######################################################
## .1.            MixingArray Class              !!! ##
#######################################################
class MixingArray:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, res=128, verbose=1):
        # Input
        self.res = res + 1
        self.verbose = verbose

        # Mixing array sampling
        self.k = 1.5
        self.seed = 42
        self.digits = 3
        self.D_tio2 = 5e-2
        self.mc_sample = 1
        self.weighted_random = True

        # Earthchem data
        self.trace = ["CR", "NI"]
        self.volatiles = ["H2O", "CO2"]
        self.earthchem_raw = pd.DataFrame()
        self.earthchem_pca = pd.DataFrame()
        self.earthchem_imputed = pd.DataFrame()
        self.earthchem_filtered = pd.DataFrame()
        self.metadata = ["SAMPLEID", "SOURCE", "ROCKNAME"]
        self.ox_exclude = ["CR2O3", "FE2O3", "P2O5", "NIO", "MNO"]
        self.ox_data = ["SIO2", "AL2O3", "CAO", "MGO", "FEOT", "K2O", "NA2O", "TIO2",
                        "FE2O3", "CR2O3", "FE2O3T", "FEO", "NIO", "MNO", "P2O5", "LOI"]
        self.ox_gfem = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "LOI"]
        self.earthchem_filename = "earthchem-combined-deschamps-2013.txt"

        # PCA results
        self.scaler = None
        self.pca_model = None
        self.n_pca_components = 2
        self.pca_results = np.array([])

        # Mixing array results
        self.top_arrays = None
        self.mixing_arrays = None
        self.bottom_arrays = None
        self.synthetic_data_written = False
        self.mixing_array_tops = np.array([])
        self.mixing_array_bottoms = np.array([])
        self.mixing_array_endpoints = np.array([])

        # Errors
        self.error = None
        self.mixing_array_error = False

        # Paths
        self.fig_dir = "figs/mixing_array"

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read earthchem data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_earthchem_data(self):
        """
        """
        # Get self attributes
        trace = self.trace
        digits = self.digits
        ox_data = self.ox_data
        metadata = self.metadata
        volatiles = self.volatiles
        filename = self.earthchem_filename
        ox_methods = [string + "METH" for string in ox_data]
        trace_methods = [string + "METH" for string in trace]
        volatiles_methods = [string + "METH" for string in volatiles]

        # Check for earthchem data
        if not filename:
            raise Exception("No Earthchem data found!")

        # Initialize dataframes
        dataframes = {}
        df_name = []

        print("Reading Earthchem data ...")

        # Read data
        data = pd.read_csv(f"assets/data/{filename}", delimiter="\t")

        # Rename columns
        data.columns = [col.replace(" ", "") for col in data.columns]

        # Select columns
        data = data[metadata + ox_data + ox_methods + volatiles +
                    volatiles_methods + trace + trace_methods]

        # Round values
        data[ox_data] = data[ox_data].round(digits)

        # "Classic" mantle array ratios (Deschamps et al., 2013, Lithos)
        data["R_MGSI"] = round(data["MGO"] / data["SIO2"], digits)
        data["R_ALSI"] = round(data["AL2O3"] / data["SIO2"], digits)

        # Update self attribute
        self.earthchem_raw = data.copy()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # convert to cr2o3 !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_cr2o3(self, df):
        """
        """
        # Get self attributes
        digits = self.digits

        # Copy df
        data = df.copy()

        # Check for misreported units
        condition = data["CR2O3"] > data["SIO2"]
        data.loc[condition, "CR"] = data.loc[condition]["CR2O3"]
        data.loc[condition, "CR2O3"] = np.nan

        # If CR2O3 exists
        condition = data["CR2O3"].notna()
        data.loc[condition, "CR"] = np.nan

        # If CR exists but not CR2O3
        condition = data["CR2O3"].isna() & data["CR"].notna()
        data.loc[condition, "CR2O3"] = round(
            data.loc[condition]["CR"] / 1e4 * 1.4615, digits)
        data.loc[condition, "CR"] = np.nan

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # convert to nio !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_nio(self, df):
        """
        """
        # Get self attributes
        digits = self.digits

        # Copy df
        data = df.copy()

        # Check for misreported units
        condition = data["NIO"] > data["SIO2"]
        data.loc[condition, "NI"] = data.loc[condition]["NIO"]
        data.loc[condition, "NIO"] = np.nan

        # If NIO exists
        condition = data["NIO"].notna()
        data.loc[condition, "NI"] = np.nan

        # If NI exists but not NIO
        condition = data["NIO"].isna() & data["NI"].notna()
        data.loc[condition, "NIO"] = round(data.loc[condition]["NI"] / 1e4 * 1.2725, digits)
        data.loc[condition, "NI"] = np.nan

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # convert to fe2o3t !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_fe2o3t(self, df):
        """
        """
        # Get self attributes
        digits = self.digits

        # Copy df
        data = df.copy()

        # If FE2O3T exists set all Fe to nan except FE2O3T
        condition = data["FE2O3T"].notna()
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        # If FE2O3 exists but not FE2O3T, FEO, or FEOT
        condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].isna() &
                     data["FEOT"].isna())
        data.loc[condition, ["FE2O3T"]] = data.loc[condition]["FE2O3"]
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        # If FEO exists but not FE2O3, FE2O3T, or FEOT
        condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                     data["FEOT"].isna())
        data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEO"] / 0.89998, digits)
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

       # If FEOT exists but not FE2O3, FE2O3T, or FEO
        condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() & data["FEO"].isna() &
                     data["FEOT"].notna())
        data.loc[condition, ["FE2O3T"]] = round(
            data.loc[condition]["FEOT"] / 0.89998, digits)
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        # If FEO and FEOT exists but not FE2O3 or FE2O3T
        condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                     data["FEOT"].notna())
        data.loc[condition, ["FE2O3T"]] = round(
            data.loc[condition]["FEOT"] / 0.89998, digits)
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        # If FE2O3 and FEO exist but not FE2O3T or FEOT
        condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                     data["FEOT"].isna())
        data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FE2O3"] +
                                                data.loc[condition]["FEO"] / 0.89998, digits)
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        # If FE2O3 and FEOT exist but not FE2O3T or FEO
        condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].isna() &
                     data["FEOT"].notna())
        data.loc[condition, "FE2O3T"] = data.loc[condition, "FE2O3"]
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        ## If FE2O3, FEO and FEOT exist but not FE2O3T
        condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                     data["FEOT"].notna())
        data.loc[condition, ["FE2O3T"]] = round(
            data.loc[condition]["FEOT"] / 0.89998, digits)
        data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # normalize volatile free !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _normalize_volatile_free(self, df):
        """
        """
        # Get self attributes
        digits = self.digits
        ox_data = self.ox_data
        volatiles = self.volatiles

        # Copy df
        data = df.copy()

        # Sum oxides with and without volatiles
        data["total_ox"] = data[ox_data].sum(axis=1).round(digits)
        data["total_loi"] = data[ox_data + volatiles].sum(axis=1).round(digits)

        # Set total threshold
        total_threshold = [97, 103]

        # Check for samples within of threshold
        condition = (((data["total_ox"] >= total_threshold[0]) &
                      (data["total_ox"] <= total_threshold[1])) |
                     ((data["total_loi"] >= total_threshold[0]) &
                      (data["total_loi"] <= total_threshold[1])))

        # Filter data
        data = data.loc[condition]

        # Normalize to volatile free basis
        data[ox_data] = round(data[ox_data].div(data["total_ox"], axis=0) * 100, digits)

        # Re-sum oxides with and without volatiles
        data["total_ox"] = data[ox_data].sum(axis=1).round(digits)
        data["total_loi"] = data[ox_data + volatiles].sum(axis=1).round(digits)

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # convert to feot !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_feot(self, df):
        """
        """
        # Get self attributes
        digits = self.digits

        # Copy df
        data = df.copy()

        # If FEOT exists set all Fe to nan except FEOT
        condition = data["FEOT"].notna()
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        # If FEO exists but not FEOT, FE2O3, or FE2O3T
        condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].isna() &
                     data["FE2O3T"].isna())
        data.loc[condition, ["FEOT"]] = data.loc[condition]["FEO"]
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        # If FE2O3 exists but not FEO, FEOT, or FE2O3T
        condition = (data["FEO"].isna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                     data["FE2O3T"].isna())
        data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3"] * 0.89998, digits)
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        # If FE2O3T exists but not FEO, FEOT, or FE2O3
        condition = (data["FEO"].isna() & data["FEOT"].isna() & data["FE2O3"].isna() &
                     data["FE2O3T"].notna())
        data.loc[condition, ["FEOT"]] = round(
            data.loc[condition]["FE2O3T"] * 0.89998, digits)
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        # If FE2O3 and FE2O3T exists but not FEO or FEOT
        condition = (data["FEO"].isna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                     data["FE2O3T"].notna())
        data.loc[condition, ["FEOT"]] = round(
            data.loc[condition]["FE2O3T"] * 0.89998, digits)
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        # If FEO and FE2O3 exist but not FEOT or FE2O3T
        condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                     data["FE2O3T"].isna())
        data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FEO"] +
                                              data.loc[condition]["FE2O3"] * 0.89998, digits)
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        # If FEO and FE2O3T exist but not FEOT or FE2O3
        condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].isna() &
                     data["FE2O3T"].notna())
        data.loc[condition, "FEOT"] = data.loc[condition, "FEO"]
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        # If FEO, FE2O3 and FE2O3T exist but not FEOT
        condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                     data["FE2O3T"].notna())
        data.loc[condition, ["FEOT"]] = round(
            data.loc[condition]["FE2O3T"] * 0.89998, digits)
        data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_earthchem_data(self):
        """
        """
        # Get self attributes
        digits = self.digits
        ox_data = self.ox_data
        verbose = self.verbose
        volatiles = self.volatiles
        data = self.earthchem_raw.copy()

        # Rock names meta groups
        peridotite = ["peridotite", "harzburgite", "lherzolite", "dunite", "wehrlite",
                      "abyssal peridotite", "ultramafic rock", "harzburgite-lherzholite",
                      "harzburgite-dunite"]
        pyroxenite = ["pyroxenite", "websterite", "hornblendite", "clinopyroxenite",
                      "orthopyroxenite"]
        serpentinite = ["serpentinite", "serpentinized peridotite",
                        "serpentinized harzburgite", "serpentinite muds",
                        "serpentinized lherzolite", "antigorite serpentinite",
                        "serpentinized mylonite"]
        metamorphic = ["amphibolite", "blueschist", "meta-basalt", "eclogite",
                       "meta-gabbro", "metabasite", "hydrated peridotite",
                       "chlorite harzburgite"]
        xenolith = ["peridotite-xenolith", "pyroxenite-xenolith", "eclogite-xenolith"]
        other = ["limburgite", "chromitite", "unknown", "olivine-rich troctolite",
                 "hydrated cumulate"]

        # Keep only samples with required oxides
        condition = data[["SIO2", "MGO", "AL2O3", "CAO", "LOI"]].notna().all(axis=1)
        data = data.loc[condition]

        # Drop unknown rocks
        data = data[~data["ROCKNAME"].isin(xenolith + metamorphic + other + pyroxenite +
                                           ["wehrlite", "peridotite", "dunite"])]

        # Add new rock type column
        conditions = [data["ROCKNAME"].isin(peridotite),
                      data["ROCKNAME"].isin(serpentinite)]
        values = ["peridotite", "serpentinite"]
        data["ROCKTYPE"] = np.select(conditions, values, default="other")

        # Function to remove outliers based on IQR
        def remove_outliers(group, threshold):
            Q1, Q3 = group[ox_data].quantile(0.25), group[ox_data].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - threshold * IQR, Q3 + threshold * IQR
            outlier_rows = ((group[ox_data] < lower_bound) |
                            (group[ox_data] > upper_bound)).any(axis=1)
            return group[~outlier_rows]

        # Remove outliers for each rock type
        data = (data.groupby("ROCKNAME", group_keys=False)[data.columns.tolist()]
                .apply(remove_outliers, 1.5))

        # Convert all Cr to CR2O3
        data = self._convert_to_cr2o3(data)

        # Convert all Ni to NIO
        data = self._convert_to_nio(data)

        # Convert all Fe oxides to FE2O3T
        data = self._convert_to_fe2o3t(data)

        # Normalize to volatile free basis
        data = self._normalize_volatile_free(data)

        # Convert all Fe oxides to FEOT
        data = self._convert_to_feot(data)

        # Set FE2O3 to zero
        data["FE2O3"] = 0

        # Drop totals
        data = data.drop(["total_ox", "total_loi"], axis=1)

        # Drop all NA columns
        data = data.dropna(axis=1, how="all")

        # Rename FEOT
        data = data.rename(columns={"FEOT": "FEO"})

        # Consolidate unique methods
        cols = [col for col in data.columns if "METH" in col]
        unique_methods = data[cols].apply(lambda x: ", ".join(x.dropna().unique()), axis=1)
        data["METHODS"] = unique_methods.str.upper()

        # Drop individual methods
        data = data.drop(cols, axis=1)

        # Arrange columns by dtype
        cols = (data.select_dtypes(include=["int", "float"]).columns.tolist() +
                data.select_dtypes(exclude=["int", "float"]).columns.tolist())
        data = data[cols]

        # Arrange rows by SIO2
        data = data.sort_values(by=["SIO2", "MGO"], ascending=[True, False],
                                ignore_index=True)

        # Drop highly altered samples
        condition = ((data["R_MGSI"] >= 0.6) & (data["R_MGSI"] <= 1.4) &
                     (data["R_ALSI"] <= 0.3))
        data = data.loc[condition]

        # Update self attribute
        self.earthchem_filtered = data.copy()

        # Save info
        numeric_columns = data.select_dtypes(include="number").columns
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric_info = []
        for column in numeric_columns:
            numeric_info.append({
                "column": column,
                "measured": data[column].count(),
                "missing": data[column].isnull().sum(),
                "min": data[column].min().round(3),
                "max": data[column].max().round(3),
                "mean": data[column].mean().round(3),
                "median": data[column].median().round(3),
                "std": data[column].std().round(3),
                "iqr": round(data[column].quantile(0.75) -
                             data[column].quantile(0.25), digits)
            })
        info_df = pd.DataFrame(numeric_info)
        info_df.to_csv("assets/data/earthchem-counts.csv", index=False)

        if verbose >= 1:
            # Print info
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Combined and filtered samples summary:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print(data.info())
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Sample sources:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for source, count in data["SOURCE"].value_counts().items():
                print(f"[{count}] {source}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Rock names:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for name, count in data["ROCKNAME"].value_counts().items():
                print(f"[{count}] {name}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Rock types:")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            for type, count in data["ROCKTYPE"].value_counts().items():
                print(f"[{count}] {type}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # normalize sample composition !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _normalize_sample_composition(self, row):
        """
        """
        # Get self attributes
        digits = self.digits
        ox_gfem = self.ox_gfem
        ox_exclude = self.ox_exclude
        ox_sub = [oxide for oxide in ox_gfem if oxide not in ox_exclude]

        # Get sample composition
        sample_composition = row[ox_gfem].values

        # No normalizing for all components
        if not ox_exclude:
            return sample_composition

        # Check input
        if len(sample_composition) != len(ox_gfem):
            error_message = (f"The input sample list must have exactly {len(ox_gfem)} "
                             f"components!\n{ox_gfem}")

            raise ValueError(error_message)

        # Filter components
        subset_sample = [comp for comp, oxide in zip(sample_composition, ox_gfem)
                         if oxide in ox_sub]

        # Set negative compositions to zero
        subset_sample = [comp if comp >= 0 else 0 for comp in subset_sample]

        # Get total oxides
        total_subset_concentration = sum([comp for comp in subset_sample if comp != 0])

        # Normalize
        normalized_concentrations = [
            round(((comp / total_subset_concentration) * 100 if comp != 0 else 0), digits)
            for comp, oxide in zip(subset_sample, ox_sub)]

        return normalized_concentrations

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # samples to csv !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _samples_to_csv(self, sampleids, source, filename):
        """
        """
        # Check for file
        if not os.path.exists(source):
            raise Exception("Sample data source does not exist!")

        # Read data
        df = pd.read_csv(source)

        # Subset samples
        samples = df[df["SAMPLEID"].isin(sampleids)]

        # Write csv
        if os.path.exists(filename):
            df_bench = pd.read_csv(filename)
            df_bench = df_bench[~df_bench["SAMPLEID"].isin(sampleids)]
            df_bench = pd.concat([df_bench, samples])
            df_bench.to_csv(filename, index=False)
        else:
            samples.to_csv(filename, index=False)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process benchmark samples pca !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_benchmark_samples_pca(self):
        """
        """
        res = self.res
        pca = self.pca_model
        scaler = self.scaler
        D_tio2 = self.D_tio2
        oxides = self.ox_gfem

        df_bench_path = "assets/data/benchmark-samples.csv"
        df_bench_pca_path = "assets/data/benchmark-samples-pca.csv"
        df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"

        sources = ["assets/data/synthetic-samples-mixing-tops.csv",
                   "assets/data/synthetic-samples-mixing-middle.csv",
                   "assets/data/synthetic-samples-mixing-bottoms.csv"]
        sampleids = [["st000", f"st{res}"], ["sm000", f"sm{res}"], ["sb000", f"sb{res}"]]

        # Save synthetic benchmark models
        for source, sids in zip(sources, sampleids):
            self._samples_to_csv(sids, source, df_synth_bench_path)

        # Read benchmark samples
        if os.path.exists(df_bench_path) and os.path.exists(df_synth_bench_path):
            df_bench = pd.read_csv(df_bench_path)
            df_synth_bench = pd.read_csv(df_synth_bench_path)

            # Fit PCA to benchmark samples
            df_bench[["PC1", "PC2"]] = pca.transform(scaler.transform(df_bench[oxides]))
            df_bench[["PC1", "PC2"]] = df_bench[["PC1", "PC2"]].round(3)

            # Calculate F melt
            ti_init = df_synth_bench["TIO2"].iloc[-1]
            df_bench["R_TIO2"] = round(df_bench["TIO2"] / ti_init, 3)
            df_bench["F_MELT_BATCH"] = round(
                ((D_tio2 / df_bench["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            df_bench["XI_BATCH"] = round(1 - df_bench["F_MELT_BATCH"], 3)
            df_bench["F_MELT_FRAC"] = round(
                1 - df_bench["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)
            df_bench["XI_FRAC"] = round(1 - df_bench["F_MELT_FRAC"], 3)

            # Save to csv
            df_bench.to_csv(df_bench_pca_path, index=False)

        return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.1.         Create Mixing Arrays            !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # run pca !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_pca(self):
        """
        """
        # Get self attributes
        digits = self.digits
        verbose = self.verbose
        ox_data = self.ox_data
        ox_gfem = self.ox_gfem
        ox_exclude = self.ox_exclude
        data = self.earthchem_filtered.copy()
        n_pca_components = self.n_pca_components
        ox_sub = [oxide for oxide in ox_gfem if oxide not in ox_exclude]

        # Check for earthchem data
        if data.empty:
            raise Exception("No Earthchem data found! Call _read_earthchem_data() first ...")

        # Sort by composition
        data = data.sort_values(by=["SIO2", "MGO"], ascending=[True, False],
                                ignore_index=True)

        print("Imputing missing oxides ...")

        # Group by rockname and apply KNN imputer
        imputed_dataframes = []
        for rockname, subset in data.groupby("ROCKNAME"):
            subset = subset.reset_index(drop=True)

            for col in ox_gfem:
                column_to_impute = subset[[col]]
                imputer = KNNImputer(weights="distance")
                imputed_values = imputer.fit_transform(column_to_impute).round(digits)
                subset[col] = imputed_values

            imputed_dataframes.append(subset)

        # Recombine imputed subsets
        imputed_data = pd.concat(imputed_dataframes, ignore_index=True)

        # Normalize compositions
        normalized_values = data.apply(self._normalize_sample_composition, axis=1)
        data[ox_sub] = normalized_values.apply(pd.Series)
        data[ox_exclude] = 0

        # Initialize scaler
        scaler = StandardScaler()

        # Standardize data
        data_scaled = scaler.fit_transform(data[ox_gfem])

        # Update attribute
        self.scaler = scaler

        # Initialize PCA
        pca = PCA(n_components=n_pca_components)

        print(f"Running PCA to reduce to {n_pca_components} dimensions ...")

        # PCA modeling
        pca.fit(data_scaled)

        # Update self attribute
        self.pca_model = pca

        if verbose >= 1:
            # Print summary
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("PCA summary:")
            print(f"  PCA oxides: {ox_sub}")
            print(f"  number of samples: {pca.n_samples_}")
            print(f"  PCA components: {n_pca_components}")
            print( "  explained variance:")
            for i, value in enumerate(pca.explained_variance_ratio_):
                print(f"      PC{i+1}: {round(value, digits)}")
            print("  cumulative explained variance:")
            cumulative_variance = pca.explained_variance_ratio_.cumsum()
            for i, value in enumerate(cumulative_variance):
                print(f"      PC{i+1}: {round(value, digits)}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # Transform the data to obtain the principal components
        principal_components = pca.transform(data_scaled)

        # Update self attribute
        self.pca_results = principal_components

        # Update dataframe
        pca_columns = [f"PC{i+1}" for i in range(n_pca_components)]
        data[pca_columns] = principal_components

        # Round numerical data
        data[ox_gfem + pca_columns] = data[ox_gfem + pca_columns].round(3)

        # Update self attribute
        self.earthchem_pca = data.copy()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create mixing arrays !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_mixing_array(self):
        """
        """
        # Get earthchem data
        self._read_earthchem_data()

        # Process earthchem data
        self._process_earthchem_data()

        # Run PCA
        self._run_pca()

        # Get self attributes
        k = self.k
        res = self.res
        seed = self.seed
        D_tio2 = self.D_tio2
        digits = self.digits
        scaler = self.scaler
        pca = self.pca_model
        verbose = self.verbose
        ox_gfem = self.ox_gfem
        mc_sample = self.mc_sample
        data = self.earthchem_pca.copy()
        weighted_random = self.weighted_random
        principal_components = self.pca_results
        n_pca_components = self.n_pca_components

        try:
            # Define sample centroids
            centroids = data.groupby("ROCKNAME")[["PC1", "PC2"]].median()

            # Initialize endpoints
            mixing_array_endpoints = []
            mixing_array_tops = []
            mixing_array_bottoms = []

            for x, y, rockname in zip(centroids["PC1"].tolist(), centroids["PC2"].tolist(),
                                      centroids.index.tolist()):
                # Identify centroid quadrant
                if x > 0 and y > 0:
                    quadrant = "Q1"
                elif x < 0 and y > 0:
                    quadrant = "Q2"
                elif x < 0 and y < 0:
                    quadrant = "Q3"
                elif x > 0 and y < 0:
                    quadrant = "Q4"
                else:
                    raise Exception("Invalid quadrant!")

                # Subset cluster datapoints
                condition = data["ROCKNAME"] == rockname

                # Get IQR for PC1
                q1_pc1 = np.percentile(data.loc[condition, "PC1"], 25)
                q3_pc1 = np.percentile(data.loc[condition, "PC1"], 75)
                iqr_pc1 = q3_pc1 - q1_pc1

                # Get median for PC1
                median_x = np.median(data.loc[condition, "PC1"])

                # Define adjustment factor
                median_adjustment_q1x = 0
                median_adjustment_q1y = 0
                median_adjustment_q2x = -0.6
                median_adjustment_q2y = 0
                median_adjustment_q3x = 0.9
                median_adjustment_q3y = 0.1
                median_adjustment_q4x = 1.9
                median_adjustment_q4y = 1.2
                top_adjustment = 1.5
                bottom_adjustment = 1.5

                # Adjust endpoint for PC1
                if quadrant == "Q1":
                    endpoint_x = median_x + median_adjustment_q1x * iqr_pc1
                elif quadrant == "Q2":
                    endpoint_x = median_x + median_adjustment_q2x * iqr_pc1
                elif quadrant == "Q3":
                    endpoint_x = median_x + median_adjustment_q3x * iqr_pc1
                elif quadrant == "Q4":
                    endpoint_x = median_x + median_adjustment_q4x * iqr_pc1

                # Get IQR for PC2
                q1_pc2 = np.percentile(data.loc[condition, "PC2"], 25)
                q3_pc2 = np.percentile(data.loc[condition, "PC2"], 75)
                iqr_pc2 = q3_pc2 - q1_pc2

                # Get median for PC2
                median_y = np.median(data.loc[condition, "PC2"])

                # Adjust endpoint for PC2
                if quadrant == "Q1":
                    endpoint_y = median_y + median_adjustment_q1y * iqr_pc2
                elif quadrant == "Q2":
                    endpoint_y = median_y + median_adjustment_q2y * iqr_pc2
                elif quadrant == "Q3":
                    endpoint_y = median_y + median_adjustment_q3y * iqr_pc2
                elif quadrant == "Q4":
                    endpoint_y = median_y + median_adjustment_q4y * iqr_pc2

                mixing_array_endpoints.append([endpoint_x, endpoint_y])

            endpoints_sorted = sorted(mixing_array_endpoints, key=lambda x: x[0])

            mixing_array_endpoints = np.array(endpoints_sorted)

            mixing_array_tops = mixing_array_endpoints.copy()
            mixing_array_bottoms = mixing_array_endpoints.copy()

            mixing_array_tops[:,-1] += top_adjustment * iqr_pc2
            mixing_array_bottoms[:,-1] -= bottom_adjustment * iqr_pc2

            self.mixing_array_endpoints = mixing_array_endpoints
            self.mixing_array_tops = mixing_array_tops
            self.mixing_array_bottoms = mixing_array_bottoms

            # Initialize mixing lines
            mixing_lines = {}
            top_lines = {}
            bottom_lines = {}

            # Loop through PCA components
            for n in range(n_pca_components):
                for i in range(len(mixing_array_endpoints)):
                    # Calculate mixing lines between endpoints
                    if len(mixing_array_endpoints) > 1:
                        for j in range(i + 1, len(mixing_array_endpoints)):
                            nsmps = (res // (len(mixing_array_endpoints) - 1)) + 1
                            if n == 0:
                                mixing_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_endpoints[i, n],
                                                mixing_array_endpoints[j, n], nsmps))
                                top_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_tops[i, n],
                                                mixing_array_tops[j, n], nsmps))
                                bottom_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_bottoms[i, n],
                                                mixing_array_bottoms[j, n], nsmps))

                            else:
                                if (((i == 0) & (j == 1)) | ((i == 1) & (j == 2)) |
                                    ((i == 2) & (j == 3)) | ((i == 3) & (j == 4)) |
                                    ((i == 4) & (j == 5))):
                                    mixing_lines[f"{i + 1}{j + 1}"] = np.vstack((
                                        mixing_lines[f"{i + 1}{j + 1}"],
                                        np.linspace(mixing_array_endpoints[i, n],
                                                    mixing_array_endpoints[j, n], nsmps)))
                                    top_lines[f"{i + 1}{j + 1}"] = np.vstack((
                                        top_lines[f"{i + 1}{j + 1}"],
                                        np.linspace(mixing_array_tops[i, n],
                                                    mixing_array_tops[j, n], nsmps)))
                                    bottom_lines[f"{i + 1}{j + 1}"] = np.vstack((
                                        bottom_lines[f"{i + 1}{j + 1}"],
                                        np.linspace(mixing_array_bottoms[i, n],
                                                    mixing_array_bottoms[j, n], nsmps)))

            # Update self attribute
            self.mixing_arrays = mixing_lines
            self.top_arrays = top_lines
            self.bottom_arrays = bottom_lines

            # Initialize dataframes
            mixing_list = []
            top_list = []
            bottom_list = []

            # Write mixing lines to csv
            for i in range(len(mixing_array_endpoints)):
                for j in range(i + 1, len(mixing_array_endpoints)):
                    if (((i == 0) & (j == 1)) | ((i == 1) & (j == 2)) |
                        ((i == 2) & (j == 3)) | ((i == 3) & (j == 4)) |
                        ((i == 4) & (j == 5))):
                        # Create dataframe
                        mixing_synthetic = pd.DataFrame(
                            np.hstack((scaler.inverse_transform(pca.inverse_transform(
                                mixing_lines[f"{i + 1}{j + 1}"].T)),
                                mixing_lines[f"{i + 1}{j + 1}"].T)),
                            columns=ox_gfem + [f"PC{n + 1}" for n in range(n_pca_components)]
                        ).round(3)
                        tops_synthetic = pd.DataFrame(
                            np.hstack((scaler.inverse_transform(pca.inverse_transform(
                                top_lines[f"{i + 1}{j + 1}"].T)),
                                top_lines[f"{i + 1}{j + 1}"].T)),
                            columns=ox_gfem + [f"PC{n + 1}" for n in range(n_pca_components)]
                        ).round(3)
                        bottoms_synthetic = pd.DataFrame(
                            np.hstack((scaler.inverse_transform(pca.inverse_transform(
                                bottom_lines[f"{i + 1}{j + 1}"].T)),
                                bottom_lines[f"{i + 1}{j + 1}"].T)),
                            columns=ox_gfem + [f"PC{n + 1}" for n in range(n_pca_components)]
                        ).round(3)

                        # Append to list
                        mixing_list.append(mixing_synthetic)
                        top_list.append(tops_synthetic)
                        bottom_list.append(bottoms_synthetic)

            # Combine mixing arrays
            all_mixing = pd.concat(mixing_list, ignore_index=True)
            all_tops = pd.concat(top_list, ignore_index=True)
            all_bottoms = pd.concat(bottom_list, ignore_index=True)

            # Add sample id column
            all_mixing.insert(0, "SAMPLEID", [f"sm{str(n).zfill(3)}" for
                                              n in range(len(all_mixing))])
            all_tops.insert(0, "SAMPLEID", [f"st{str(n).zfill(3)}" for
                                            n in range(len(all_tops))])
            all_bottoms.insert(0, "SAMPLEID", [f"sb{str(n).zfill(3)}" for
                                               n in range(len(all_bottoms))])

            # No negative oxides
            data[ox_gfem] = data[ox_gfem].apply(lambda x: x.apply(lambda y: max(0.001, y)))
            all_mixing[ox_gfem] = all_mixing[
                ox_gfem].apply(lambda x: x.apply(lambda y: max(0.001, y)))
            all_tops[ox_gfem] = all_tops[
                ox_gfem].apply(lambda x: x.apply(lambda y: max(0.001, y)))
            all_bottoms[ox_gfem] = all_bottoms[
                ox_gfem].apply(lambda x: x.apply(lambda y: max(0.001, y)))

            # Calculate F melt
            ti_init = all_mixing["TIO2"].max()
            data["R_TIO2"] = round(data["TIO2"] / ti_init, digits)
            data["F_MELT_BATCH"] = round(
                ((D_tio2 / data["R_TIO2"]) - D_tio2) / (1 - D_tio2), digits)
            data["XI_BATCH"] = round(1 - data["F_MELT_BATCH"], digits)
            data["F_MELT_FRAC"] = round(1 - data["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), digits)
            data["XI_FRAC"] = round(1 - data["F_MELT_FRAC"], digits)

            self.earthchem_pca = data.copy()

            # Write csv file
            data.to_csv(f"assets/data/earthchem-samples-pca.csv", index=False)

            all_mixing["R_TIO2"] = round(all_mixing["TIO2"] / ti_init, digits)
            all_mixing["F_MELT_BATCH"] = round(
                ((D_tio2 / all_mixing["R_TIO2"]) - D_tio2) / (1 - D_tio2), digits)
            all_mixing["XI_BATCH"] = round(1 - all_mixing["F_MELT_BATCH"], digits)
            all_mixing["F_MELT_FRAC"] = round(
                1 - all_mixing["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), digits)
            all_mixing["XI_FRAC"] = round(1 - all_mixing["F_MELT_FRAC"], digits)

            all_tops["R_TIO2"] = round(all_tops["TIO2"] / ti_init, digits)
            all_tops["F_MELT_BATCH"] = round(
                ((D_tio2 / all_tops["R_TIO2"]) - D_tio2) / (1 - D_tio2), digits)
            all_tops["XI_BATCH"] = round(1 - all_tops["F_MELT_BATCH"], digits)
            all_tops["F_MELT_FRAC"] = round(
                1 - all_tops["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), digits)
            all_tops["XI_FRAC"] = round(1 - all_tops["F_MELT_FRAC"], digits)

            all_bottoms["R_TIO2"] = round(all_bottoms["TIO2"] / ti_init, digits)
            all_bottoms["F_MELT_BATCH"] = round(
                ((D_tio2 / all_bottoms["R_TIO2"]) - D_tio2) / (1 - D_tio2), digits)
            all_bottoms["XI_BATCH"] = round(1 - all_bottoms["F_MELT_BATCH"], digits)
            all_bottoms["F_MELT_FRAC"] = round(
                1 - all_bottoms["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), digits)
            all_bottoms["XI_FRAC"] = round(1 - all_bottoms["F_MELT_FRAC"], digits)

            # Write to csv
            all_mixing.to_csv("assets/data/synthetic-samples-mixing-middle.csv", index=False)
            all_tops.to_csv("assets/data/synthetic-samples-mixing-tops.csv", index=False)
            all_bottoms.to_csv("assets/data/synthetic-samples-mixing-bottoms.csv",
                               index=False)

            # Define bounding box around top and bottom mixing arrays
            min_x = min(mixing_array_tops[:, 0].min(), mixing_array_bottoms[:, 0].min())
            max_x = max(mixing_array_tops[:, 0].max(), mixing_array_bottoms[:, 0].max())

            # Define the sampling interval
            interval_x = (max_x - min_x) / res

            randomly_sampled_points = []
            sample_ids = []

            # Monte carlo sampling of synthetic samples
            for j in range(mc_sample):
                # Set seed
                np.random.seed(seed + j)

                # Create an array to store sampled points
                sampled_points = []
                sampled_weights = []

                # Iterate over x positions
                for x in np.linspace(min_x, max_x, res):
                    # Calculate the range of y values for the given x position
                    y_min = np.interp(x, mixing_array_tops[:, 0], mixing_array_tops[:, 1])
                    y_max = np.interp(x, mixing_array_bottoms[:, 0],
                                      mixing_array_bottoms[:, 1])
                    y_mid = (y_max + y_min) / 2

                    # Create a grid of y values for the current x position
                    y_values = np.linspace(y_min, y_max, res)

                    # Calculate exponential distance weights
                    point_weights = np.exp(-k * np.abs(y_values - y_mid))

                    # Combine x and y values to create points
                    points = np.column_stack((x * np.ones_like(y_values), y_values))

                    # Append points to the sampled_points array
                    sampled_points.extend(points)
                    sampled_weights.extend(point_weights)

                # Convert to np array
                sampled_points = np.array(sampled_points)
                sampled_weights = np.array(sampled_weights)

                # Define probability distribution for selecting random points
                if weighted_random:
                    prob_dist = sampled_weights / np.sum(sampled_weights)
                else:
                    prob_dist = None

                # Randomly select from sampled points
                sample_idx = np.random.choice(len(sampled_points), res, replace=False,
                                              p=prob_dist)
                randomly_sampled_points.append([sampled_points[i] for i in sample_idx])

                # Save random points
                sample_ids.extend([f"sr{str(n).zfill(3)}" for n in range(len(sample_idx))])

            # Combine randomly sampled points
            randomly_sampled_points = np.vstack(randomly_sampled_points)

            # Create dataframe
            random_synthetic = pd.DataFrame(
                np.hstack((scaler.inverse_transform(pca.inverse_transform(
                    randomly_sampled_points)), randomly_sampled_points)),
                columns=ox_gfem + [f"PC{n + 1}" for n in range(n_pca_components)]
            ).round(3)

            # Add sample id column
            random_synthetic.insert(0, "SAMPLEID", sample_ids)

            # No negative oxides
            random_synthetic[ox_gfem] = random_synthetic[ox_gfem].apply(
                lambda x: x.apply(lambda y: max(0.001, y)))

            # Calculate F melt
            random_synthetic["R_TIO2"] = round(random_synthetic["TIO2"] / ti_init, digits)
            random_synthetic["F_MELT_BATCH"] = round(
                ((D_tio2 / random_synthetic["R_TIO2"]) - D_tio2) / (1 - D_tio2), digits)
            random_synthetic["XI_BATCH"] = round(
                1 - random_synthetic["F_MELT_BATCH"], digits)
            random_synthetic["F_MELT_FRAC"] = round(
                1 - random_synthetic["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), digits)
            random_synthetic["XI_FRAC"] = round(1 - random_synthetic["F_MELT_FRAC"], digits)

            # Write to csv
            random_synthetic.to_csv("assets/data/synthetic-samples-mixing-random.csv",
                                    index=False)

            # Process benchmark samples
            self._process_benchmark_samples_pca()

            # Update attribute
            self.synthetic_data_written = True

            return None

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in create_mixing_array() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()

            self.mixing_array_error = True
            self.error = e

            return None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize mixing array !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_mixing_array(self, figwidth=6.3, figheight=5.2, fontsize=24):
        """
        """
        # Get self attributes
        res = self.res
        pca = self.pca_model
        oxides = self.ox_gfem
        fig_dir = self.fig_dir
        data = self.earthchem_pca

        # Fertility index column
        XI_col = "XI_FRAC"

        # Check for benchmark samples
        df_bench_path = "assets/data/benchmark-samples.csv"
        df_bench_pca_path = "assets/data/benchmark-samples-pca.csv"
        df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"
        df_synth_middle_path = "assets/data/synthetic-samples-mixing-middle.csv"
        df_synth_random_path = "assets/data/synthetic-samples-mixing-random.csv"

        # Read benchmark samples
        if (os.path.exists(df_bench_path) and os.path.exists(df_synth_bench_path) and
            os.path.exists(df_bench_pca_path)):
                df_bench = pd.read_csv(df_bench_pca_path)
                df_synth_bench = pd.read_csv(df_synth_bench_path)
                df_bench["R_MGSI"] = df_bench["MGO"] / df_bench["SIO2"]
                df_bench["R_ALSI"] = df_bench["AL2O3"] / df_bench["SIO2"]
                df_synth_bench["R_MGSI"] = df_synth_bench["MGO"] / df_synth_bench["SIO2"]
                df_synth_bench["R_ALSI"] = df_synth_bench["AL2O3"] / df_synth_bench["SIO2"]

        if (os.path.exists(df_synth_middle_path) and os.path.exists(df_synth_random_path)):
                df_synth_middle = pd.read_csv(df_synth_middle_path)
                df_synth_random = pd.read_csv(df_synth_random_path)

        # Filter Depletion < 1
        data = data[(data[XI_col] <= 1) & (data[XI_col] >= 0)]

        loadings = pd.DataFrame(
            (pca.components_.T * np.sqrt(pca.explained_variance_)).T, columns=oxides)

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

        # Colormap
        colormap = plt.colormaps["tab10"]

        # Legend order
        legend_order = ["harzburgite", "lherzolite", "serpentinite"]
        legend_lab = ["harz", "lherz", "serp"]

        fig = plt.figure(figsize=(figwidth * 2, figheight * 2))

        ax = fig.add_subplot(222)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        for i, comp in enumerate(legend_order):
            indices = data.loc[data["ROCKNAME"] == comp].index
            scatter = ax.scatter(data.loc[indices, "PC1"], data.loc[indices, "PC2"],
                                 edgecolors="none", color=colormap(i), marker=".", s=55)

        sns.kdeplot(data=data, x="PC1", y="PC2", hue="ROCKNAME", zorder=1, legend=False,
                    hue_order=legend_order, ax=ax, levels=5, warn_singular=False)

        oxs = ["SIO2", "MGO", "AL2O3", "LOI"]
        x_offset_text = [0, 0, 2.5, 0]
        y_offset_text = [0, 0, 0.8, 0]
        text_fac, arrow_fac = 2.8, 1.3
        x_offset_arrow, y_offset_arrow = 0, 0

        for oxide, x_off, y_off in zip(oxs, x_offset_text, y_offset_text):
            if oxide == "AL2O3":
                oxide_label = "Al$_2$O$_3$ TiO$_2$ FeO\nCaO Na$_2$O"
            elif oxide == "SIO2":
                oxide_label = "SiO$_2$"
            elif oxide == "MGO":
                oxide_label = "MgO"
            elif oxide == "LOI":
                oxide_label = "LOI"

            ax.arrow(x_offset_arrow, y_offset_arrow, loadings.at[0, oxide] * arrow_fac,
                     loadings.at[1, oxide] * arrow_fac, width=0.1, head_width=0.4,
                     color="black")
            ax.text(x_off + (loadings.at[0, oxide] * text_fac),
                    y_off + (loadings.at[1, oxide] * text_fac), oxide_label,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.1),
                    fontsize=fontsize * 0.833, color="black", ha="center", va="center")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

        ax2 = fig.add_subplot(221)
        legend_handles = []
        for i, comp in enumerate(legend_order):
            marker = mlines.Line2D(
                [0], [0], marker="o", color="w", label=legend_lab[i], markersize=4,
                markerfacecolor=colormap(i), markeredgewidth=0, linestyle="None")
            legend_handles.append(marker)
            indices = data.loc[data["ROCKNAME"] == comp].index
            indices_ec = data.loc[data["ROCKNAME"] == comp].index
            scatter = ax2.scatter(data.loc[indices, "R_ALSI"], data.loc[indices, "R_MGSI"],
                                 edgecolors="none", color=colormap(i), marker=".", s=55,
                                 label=legend_lab[i])

        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm000"],
                        x="R_ALSI", y="R_MGSI", facecolor="white", edgecolor="black",
                        linewidth=2, s=150, legend=False, ax=ax2, zorder=6)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm129"],
                        x="R_ALSI", y="R_MGSI", facecolor="white", edgecolor="black",
                        marker="D", linewidth=2, s=150, legend=False, ax=ax2, zorder=6)
        ax2.annotate("DSUM", xy=(
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm000", "R_ALSI"].iloc[0],
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm000", "R_MGSI"].iloc[0]),
                     xytext=(10, 10), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="black", linewidth=1.5, alpha=0.8),
                    fontsize=fontsize * 0.833, zorder=8)
        ax2.annotate("PSUM", xy=(
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm129", "R_ALSI"].iloc[0],
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm129", "R_MGSI"].iloc[0]),
                    xytext=(5, -28), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="black", linewidth=1.5, alpha=0.8),
                    fontsize=fontsize * 0.833, zorder=8)

        mrkr = ["s", "^", "P"]
        for l, name in enumerate(["PUM", "DMM", "PYR"]):
            sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="R_ALSI",
                            y="R_MGSI", marker=mrkr[l], facecolor="white", edgecolor="black",
                            linewidth=2, s=150, legend=False, ax=ax2, zorder=7)

        style = ArrowStyle("Simple", head_length=1.5, head_width=1.5, tail_width=0.15)
        arrow = FancyArrowPatch((0.13, 0.95), (0.03, 1.27), mutation_scale=6, color="black",
                                arrowstyle=style)
        ax2.add_patch(arrow)
        ax2.text(0.06, 1.0, "Melting residue", rotation=-27, fontsize=fontsize * 0.833)

        legend = ax2.legend(handles=legend_handles, loc="upper center", frameon=False,
                            bbox_to_anchor=(0.35, 0.188), ncol=3, columnspacing=0,
                            handletextpad=-0.5, markerscale=3, fontsize=fontsize * 0.833)
        # Legend order
        for i, label in enumerate(legend_lab):
            legend.get_texts()[i].set_text(label)

        ax2.set_xlabel("Al$_2$O$_3$/SiO$_2$")
        ax2.set_ylabel("MgO/SiO$_2$")

        ax3 = fig.add_subplot(223)
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        sns.scatterplot(data=data, x="PC1", y="PC2", facecolor="0.6", edgecolor="None",
                        linewidth=2, s=12, legend=False, ax=ax3, zorder=0)

        # Create colorbar
        pal = sns.color_palette("magma", as_cmap=True).reversed()
        norm = plt.Normalize(df_synth_bench[XI_col].min(), df_synth_bench[XI_col].max())
        sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
        sm.set_array([])

        sns.scatterplot(data=df_synth_middle, x="PC1", y="PC2", hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax3, zorder=0)
        sns.scatterplot(data=df_synth_random, x="PC1", y="PC2", hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax3, zorder=0)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm000"],
                        x="PC1", y="PC2", facecolor="white", edgecolor="black",
                        linewidth=2, s=150, legend=False, ax=ax3, zorder=6)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm129"],
                        x="PC1", y="PC2", facecolor="white", edgecolor="black",
                        marker="D", linewidth=2, s=150, legend=False, ax=ax3, zorder=6)

        mrkr = ["s", "^", "P"]
        for l, name in enumerate(["PUM", "DMM", "PYR"]):
            sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="PC1",
                            y="PC2", marker=mrkr[l], facecolor="white", edgecolor="black",
                            linewidth=2, s=150, legend=False, ax=ax3, zorder=7)

        plt.xlim(ax.get_xlim())
        plt.ylim(ax.get_ylim())

        # Add colorbar
        cbaxes = inset_axes(ax3, width="40%", height="3%", loc=1)
        colorbar = plt.colorbar(
            sm, ax=ax3, cax=cbaxes, label="Fertility, $\\xi$", orientation="horizontal")
        colorbar.ax.set_xticks([sm.get_clim()[0], sm.get_clim()[1]])
        colorbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))

        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

        ax4 = fig.add_subplot(224)
        sns.scatterplot(data=data, x="PC1", y="XI_FRAC", facecolor="0.6", marker=".",
                        edgecolor="None", s=55, legend=False, ax=ax4, zorder=0)
        sns.scatterplot(data=df_synth_middle, x="PC1", y=XI_col, hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax4, zorder=0)
        sns.scatterplot(data=df_synth_random, x="PC1", y=XI_col, hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax4, zorder=0)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm000"],
                        x="PC1", y="XI_FRAC", facecolor="white", edgecolor="black",
                        linewidth=2, s=150, legend=False, ax=ax4, zorder=6)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm129"],
                        x="PC1", y="XI_FRAC", facecolor="white", edgecolor="black",
                        marker="D", linewidth=2, s=150, legend=False, ax=ax4, zorder=6)

        for l, name in enumerate(["PUM", "DMM", "PYR"]):
            sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="PC1",
                            y="XI_FRAC", marker=mrkr[l], facecolor="white",
                            edgecolor="black", linewidth=2, s=150, legend=False, ax=ax4,
                            zorder=7)

        mrkr = ["D", "s", "P", "^", "o"]
        labels = ["PSUM", "PUM", "PYR", "DMM", "DSUM"]
        legend_handles = [mlines.Line2D(
            [0], [0], marker=mrkr[l], color="w", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=2, markersize=10, label=label,
            linestyle="") for l, label in enumerate(labels)]

        ax4.set_xlabel("PC1")
        ax4.set_ylabel("Fertility, $\\xi$")
        ax4.legend(handles=legend_handles, loc="best", columnspacing=0.2, handletextpad=-0.1,
                   fontsize=fontsize * 0.833)

        # Add captions
        fig.text(0.04, 0.97, "a)", fontsize=fontsize * 1.2)
        fig.text(0.04, 0.50, "c)", fontsize=fontsize * 1.2)
        fig.text(0.52, 0.97, "b)", fontsize=fontsize * 1.2)
        fig.text(0.52, 0.50, "d)", fontsize=fontsize * 1.2)

        # Save the plot to a file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plt.savefig(f"{fig_dir}/earthchem-mixing-array.png")

        # Close device
        plt.close()

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # visualize harker diagrams !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_harker_diagrams(self, figwidth=6.3, figheight=5.8, fontsize=22):
        """
        """
        # Get self attributes
        fig_dir = self.fig_dir
        data = self.earthchem_filtered
        oxides = [ox for ox in self.ox_gfem if ox not in ["SIO2", "FE2O3"]]

        # Check for benchmark samples
        df_bench_path = "assets/data/benchmark-samples.csv"
        df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"

        if os.path.exists(df_bench_path) and os.path.exists(df_synth_bench_path):
            # Read benchmark samples
            df_bench = pd.read_csv(df_bench_path)
            df_synth_bench = pd.read_csv(df_synth_bench_path)

        # Check for synthetic data
        if not self.synthetic_data_written:
            raise Exception("No synthetic data found! Call create_mixing_arrays() first ...")

        # Initialize synthetic datasets
        synthetic_samples = pd.read_csv(f"assets/data/synthetic-samples-mixing-random.csv")

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

        warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

        # Create a grid of subplots
        num_plots = len(oxides) + 1

        if num_plots == 1:
            num_cols = 1
        elif num_plots > 1 and num_plots <= 4:
            num_cols = 2
        elif num_plots > 4 and num_plots <= 9:
            num_cols = 3
        elif num_plots > 9 and num_plots <= 16:
            num_cols = 4
        else:
            num_cols = 5

        num_rows = (num_plots + 1) // num_cols

        # Total figure size
        fig_width = figwidth / 2 * num_cols
        fig_height = figheight / 2 * num_rows

        xmin, xmax = data["SIO2"].min(), data["SIO2"].max()

        # Harker diagrams
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        # Legend order
        legend_order = ["harzburgite", "lherzolite", "serpentinite"]
        legend_lab = ["harz", "lherz", "serp"]
        colormap = plt.colormaps["tab10"]

        for k, y in enumerate(oxides):
            ax = axes[k]

            sns.scatterplot(data=synthetic_samples, x="SIO2", y=y, linewidth=0, s=8,
                            color="black", legend=False, ax=ax, zorder=3)

            sns.scatterplot(data=data, x="SIO2", y=y, hue="ROCKNAME", hue_order=legend_order,
                            linewidth=0, s=8, alpha=0.5, ax=ax, zorder=1, legend=False)
            sns.kdeplot(data=data, x="SIO2", y=y, hue="ROCKNAME", hue_order=legend_order,
                        ax=ax, levels=5, zorder=1, legend=False)

            mrkr = ["s", "^", "P"]
            for l, name in enumerate(["PUM", "DMM", "PYR"]):
                sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="SIO2",
                                y=y, marker=mrkr[l], facecolor="white",
                                edgecolor="black", linewidth=2, s=150, legend=False, ax=ax,
                                zorder=7)

            sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm000"],
                            x="SIO2", y=y, facecolor="white", edgecolor="black",
                            linewidth=2, s=75, legend=False, ax=ax, zorder=6)
            sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm129"],
                            x="SIO2", y=y, facecolor="white", edgecolor="black", marker="D",
                            linewidth=2, s=75, legend=False, ax=ax, zorder=6)

            if k < (num_plots - num_cols):
                ax.set_xticks([])

            ax.set_xlim(xmin - (xmin * 0.02), xmax + (xmax * 0.02))
            ax.set_ylabel("")
            ax.set_xlabel("")

            if y in ["NA2O", "TIO", "CR2O3", "K2O", "CAO", "AL2O3"]:
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
            else:
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2g}"))

            ax.set_title(f"{y}")

        if num_plots < len(axes):
            for i in range(num_plots, len(axes)):
                fig.delaxes(axes[i])

        mrkr = ["D", "s", "P", "^", "o"]
        labels = ["PSUM", "PUM", "PYR", "DMM", "DSUM"]
        legend_handles = [mlines.Line2D([0], [0], marker=mrkr[l], color="w",
                                        markerfacecolor="white", markeredgecolor="black",
                                        markeredgewidth=2, markersize=10, label=label,
                                        linestyle="") for l, label in enumerate(labels)]

        for i, comp in enumerate(legend_order):
            marker = mlines.Line2D([0], [0], marker="o", color=colormap(i),
                                   label=legend_lab[i], markersize=10,
                                   markerfacecolor=colormap(i), markeredgewidth=2,
                                   linestyle="None")
            legend_handles.append(marker)

        marker = mlines.Line2D([0], [0], marker="o", color="black", label="synth",
                               markersize=10, markerfacecolor="black",
                               markeredgewidth=2, linestyle="None")
        legend_handles.append(marker)

        legend_ax = axes[-1]
        legend_ax.axis("off")
        legend_ax.legend(handles=legend_handles, loc="best", columnspacing=0.2, ncol=2,
                         bbox_to_anchor=(1.0, 1.0), handletextpad=-0.1,
                         fontsize=fontsize * 0.833)

        # Save the plot to a file
        plt.savefig(f"{fig_dir}/earthchem-harker-diagram.png")

        # Close device
        plt.close()

        return None


#######################################################
## .3.            Create Mixing Array            !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    if not os.path.exists("assets/data/benchmark-samples-pca.csv"):
        try:
            # Create mixing array
            mixing_array = MixingArray()
            mixing_array.create_mixing_array()
            print("Mixing array created !")

            # Visualize mixing array
            mixing_array.visualize_mixing_array()
            mixing_array.visualize_harker_diagrams()
            print("Mixing array visualized !")

        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in main() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
    else:
        print("Mixing arrays found !")

    print("=============================================")

    return None

if __name__ == "__main__":
    main()
