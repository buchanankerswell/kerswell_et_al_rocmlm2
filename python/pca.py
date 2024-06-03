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
from utils import parse_arguments, check_arguments

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

#######################################################
## .1.             Helper Functions              !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert to nio !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_nio(df, digits=3):
    """
    """
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
# convert to cr2o3 !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_cr2o3(df, digits=3):
    """
    """
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
    data.loc[condition, "CR2O3"] = round(data.loc[condition]["CR"] / 1e4 * 1.4615, digits)
    data.loc[condition, "CR"] = np.nan

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert to fe2o3t !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_fe2o3t(df, digits=3):
    """
    """
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
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    # If FEO and FEOT exists but not FE2O3 or FE2O3T
    condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() & data["FEO"].notna() &
                 data["FEOT"].notna())
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
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
    data.loc[condition, ["FE2O3T"]] = round(data.loc[condition]["FEOT"] / 0.89998, digits)
    data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# convert to feot !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_to_feot(df, digits=3):
    """
    """
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
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    # If FE2O3 and FE2O3T exists but not FEO or FEOT
    condition = (data["FEO"].isna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                 data["FE2O3T"].notna())
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
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

    ## If FEO, FE2O3 and FE2O3T exist but not FEOT
    condition = (data["FEO"].notna() & data["FEOT"].isna() & data["FE2O3"].notna() &
                 data["FE2O3T"].notna())
    data.loc[condition, ["FEOT"]] = round(data.loc[condition]["FE2O3T"] * 0.89998, digits)
    data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# normalize volatile free !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def normalize_volatile_free(df, oxides, volatiles, digits=3):
    """
    """
    # Copy df
    data = df.copy()

    # Sum oxides with and without volatiles
    data["total_ox"] = data[oxides].sum(axis=1).round(digits)
    data["total_loi"] = data[oxides + volatiles].sum(axis=1).round(digits)

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
    data[oxides] = round(data[oxides].div(data["total_ox"], axis=0) * 100, digits)

    # Re-sum oxides with and without volatiles
    data["total_ox"] = data[oxides].sum(axis=1).round(digits)
    data["total_loi"] = data[oxides + volatiles].sum(axis=1).round(digits)

    return data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# samples to csv !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def samples_to_csv(sampleids, source, filename):
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

#######################################################
## .2.            MixingArray Class              !!! ##
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
        data = convert_to_cr2o3(data, digits)

        # Convert all Ni to NIO
        data = convert_to_nio(data, digits)

        # Convert all Fe oxides to FE2O3T
        data = convert_to_fe2o3t(data, digits)

        # Normalize to volatile free basis
        data = normalize_volatile_free(data, ox_data, volatiles, digits)

        # Convert all Fe oxides to FEOT
        data = convert_to_feot(data)

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
            print(f"    PCA oxides: {ox_sub}")
            print(f"    number of samples: {pca.n_samples_}")
            print(f"    PCA components: {n_pca_components}")
            print("    explained variance:")
            for i, value in enumerate(pca.explained_variance_ratio_):
                print(f"        PC{i+1}: {round(value, digits)}")
            print("    cumulative explained variance:")
            cumulative_variance = pca.explained_variance_ratio_.cumsum()
            for i, value in enumerate(cumulative_variance):
                print(f"        PC{i+1}: {round(value, digits)}")
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
            samples_to_csv(sids, source, df_synth_bench_path)

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
                median_adjustment_q2x = -0.5
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
            random_synthetic["XI_BATCH"] = round(1 - random_synthetic["F_MELT_BATCH"], digits)
            random_synthetic["F_MELT_FRAC"] = round(
                1 - random_synthetic["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), digits)
            random_synthetic["XI_FRAC"] = round(1 - random_synthetic["F_MELT_FRAC"], digits)

            # Write to csv
            random_synthetic.to_csv("assets/data/synthetic-samples-mixing-random.csv",
                                    index=False)

            # Update attribute
            self.synthetic_data_written = True

            # Process benchmark samples
            self._process_benchmark_samples_pca()

            return None

        except Exception as e:
            print("Error occurred when computing mixing arrays!")
            traceback.print_exc()

            self.mixing_array_error = True
            self.error = e

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
    from visualize import visualize_mixing_array, visualize_harker_diagrams

    # Parse arguments and check
    args = parse_arguments()
    valid_args = check_arguments(args, "pca.py")

    # Load valid arguments
    locals().update(valid_args)

    # Create mixing array
    mixing_array = MixingArray()
    mixing_array.create_mixing_array()
    print("Mixing array created!")

    # Visualize mixing array
    visualize_harker_diagrams(mixing_array)
    visualize_mixing_array(mixing_array)

    print("Mixing array visualized!")

    return None

if __name__ == "__main__":
    main()
