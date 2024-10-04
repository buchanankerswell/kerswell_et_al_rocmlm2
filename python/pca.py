#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import warnings
import traceback
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    def __init__(self, res=14, res_h2o=8, verbose=1):
        """
        """
        self.res = res + 1
        self.res_h2o = res_h2o
        self.verbose = verbose

        self._load_global_options()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.0.           Helper Functions              !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _load_global_options(self):
        """
        """
        # Mixing array sampling
        self.k = 1.5
        self.seed = 42
        self.digits = 3
        self.max_h2o = 14
        self.D_tio2 = 5e-2
        self.mc_sample = 1
        self.n_pca_components = 2
        self.weighted_random = True

        # Earthchem data
        self.trace = ["CR", "NI"]
        self.volatiles = ["H2O", "CO2", "LOI"]
        self.metadata = ["SAMPLEID", "SOURCE", "ROCKNAME"]
        self.ox_exclude = ["FE2O3", "P2O5", "NIO", "MNO", "H2O", "CO2"]
        self.ox_pca = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "CR2O3"]
        self.ox_data = ["SIO2", "AL2O3", "CAO", "MGO", "FEOT", "K2O", "NA2O", "TIO2",
                        "FE2O3", "CR2O3", "FE2O3T", "FEO", "NIO", "MNO", "P2O5"]

        # Paths
        self.fig_dir = "figs/mixing_array"

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_earthchem_data(self):
        """
        Reads and processes Earthchem data from a specified file.

        This function reads an Earthchem dataset file, processes it by selecting
        columns for metadata, oxide data, volatile components, and trace elements,
        then calculates classic mantle array ratios (Mg/Si and Al/Si). The
        processed data is stored in `self.earthchem_raw`.

        Updates:
            self.earthchem_raw: DataFrame containing the processed Earthchem data.

        Raises:
            Exception: If the Earthchem data file is not found.
        """
        try:
            filename = "earthchem-combined-deschamps-2013.txt"
            if not filename:
                raise Exception("No Earthchem data found!")

            # Get METHOD column for each oxide component
            ox_methods = [string + "METH" for string in self.ox_data]
            trace_methods = [string + "METH" for string in self.trace]
            volatiles_methods = [string + "METH" for string in self.volatiles]

            print("Reading Earthchem data ...")
            data = pd.read_csv(f"assets/{filename}", delimiter="\t")
            data.columns = data.columns.str.replace(" ", "")

            # Select relevant columns and round oxide data
            cols_to_keep = (self.metadata + self.ox_data + ox_methods + self.volatiles +
                            volatiles_methods + self.trace + trace_methods)
            data = data[cols_to_keep]
            data[self.ox_data] = data[self.ox_data].round(self.digits)

            # Calculate mantle array ratios (e.g., Deschamps et al., 2013, Lithos)
            data["R_MGSI"] = round(data["MGO"] / data["SIO2"], self.digits)
            data["R_ALSI"] = round(data["AL2O3"] / data["SIO2"], self.digits)

            # Update self attribute
            self.earthchem_raw = data.copy()

        except Exception as e:
            print(f"Error in _read_earthchem_data():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_cr2o3(self, df):
        """
        Converts Cr to Cr2O3 in the given DataFrame and handles potential
        unit misreporting.

        This function identifies rows where Cr2O3 is misreported (i.e., values of
        Cr2O3 greater than SiO2) and transfers the misreported Cr2O3 values to
        Cr, replacing Cr2O3 with NaN. If Cr2O3 is missing but Cr exists, it
        converts Cr to Cr2O3 using a specific factor (1.4615) and rounds
        the result.

        Inputs:
            df (DataFrame): The input DataFrame with Cr and Cr2O3 columns.

        Outputs:
            DataFrame: A modified DataFrame with corrected Cr2O3 and Cr values.

        Updates:
            Rounds converted Cr2O3 values to `self.digits`.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            data = df.copy()

            # Handle misreported Cr2O3 > SiO2
            condition = data["CR2O3"] > data["SIO2"]
            data.loc[condition, "CR"] = data.loc[condition]["CR2O3"]
            data.loc[condition, "CR2O3"] = np.nan

            # If CR2O3 exists, set CR to NaN
            condition = data["CR2O3"].notna()
            data.loc[condition, "CR"] = np.nan

            # Convert CR to CR2O3 if CR2O3 is NaN but CR exists
            condition = data["CR2O3"].isna() & data["CR"].notna()
            data.loc[condition, "CR2O3"] = round(
                data.loc[condition]["CR"] / 1e4 * 1.4615, self.digits)
            data.loc[condition, "CR"] = np.nan

        except Exception as e:
            print(f"Error in _convert_to_cr2o3():\n  {e}")
            return None

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_nio(self, df):
        """
        Converts Ni to NiO in the given DataFrame and handles potential unit
        misreporting.

        This function checks for misreported NiO values (e.g., NiO > SiO2) and
        transfers them to Ni, setting NiO to NaN. If NiO is missing but Ni exists,
        it converts Ni to NiO using a specific factor (1.2725) and rounds the
        result to the defined number of digits.

        Inputs:
            df (DataFrame): The input DataFrame with Ni and NiO columns.

        Outputs:
            DataFrame: A modified DataFrame with corrected NiO and Ni values.

        Updates:
            Rounds converted NiO values to `self.digits`.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            data = df.copy()

            # Handle misreported NiO > SiO2
            condition = data["NIO"] > data["SIO2"]
            data.loc[condition, "NI"] = data.loc[condition]["NIO"]
            data.loc[condition, "NIO"] = np.nan

            # If NiO exists, set Ni to NaN
            condition = data["NIO"].notna()
            data.loc[condition, "NI"] = np.nan

            # Convert Ni to NiO if NiO is NaN but Ni exists
            condition = data["NIO"].isna() & data["NI"].notna()
            data.loc[condition, "NIO"] = round(
                data.loc[condition]["NI"] / 1e4 * 1.2725, self.digits)
            data.loc[condition, "NI"] = np.nan

        except Exception as e:
            print(f"Error in _convert_to_nio():\n  {e}")
            return None

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_fe2o3t(self, df):
        """
        Converts iron oxide measurements (FeO, Fe2O3, FEOT) into a single
        Fe2O3T column while handling missing values and potential unit
        misreporting.

        The method processes the DataFrame as follows:
        1. If FE2O3T exists, sets FeO, Fe2O3, and FEOT to NaN.
        2. If only FE2O3 exists, converts it to FE2O3T.
        3. If only FeO exists, converts it to FE2O3T using a factor (0.89998).
        4. If FEOT exists, it is used to compute FE2O3T.
        5. Handles cases where multiple columns exist, prioritizing FE2O3T.

        Inputs:
            df (DataFrame): A DataFrame with FeO, Fe2O3, FEOT, and FE2O3T columns.

        Outputs:
            DataFrame: A modified DataFrame with FE2O3T populated, and FeO, Fe2O3,
            FEOT set to NaN after conversion.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            data = df.copy()

            # If FE2O3T exists set all Fe to nan except FE2O3T
            condition = data["FE2O3T"].notna()
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

            # If FE2O3 exists but not FE2O3T, FEO, or FEOT
            condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() &
                         data["FEO"].isna() & data["FEOT"].isna())
            data.loc[condition, ["FE2O3T"]] = data.loc[condition]["FE2O3"]
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

            # If FEO exists but not FE2O3, FE2O3T, or FEOT
            condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() &
                         data["FEO"].notna() & data["FEOT"].isna())
            data.loc[condition, ["FE2O3T"]] = round(
                data.loc[condition]["FEO"] / 0.89998, self.digits)
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

            # If FEOT exists but not FE2O3, FE2O3T, or FEO
            condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() &
                         data["FEO"].isna() & data["FEOT"].notna())
            data.loc[condition, ["FE2O3T"]] = round(
                data.loc[condition]["FEOT"] / 0.89998, self.digits)
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

            # If FEO and FEOT exists but not FE2O3 or FE2O3T
            condition = (data["FE2O3"].isna() & data["FE2O3T"].isna() &
                         data["FEO"].notna() & data["FEOT"].notna())
            data.loc[condition, ["FE2O3T"]] = round(
                data.loc[condition]["FEOT"] / 0.89998, self.digits)
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

            # If FE2O3 and FEO exist but not FE2O3T or FEOT
            condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() &
                         data["FEO"].notna() & data["FEOT"].isna())
            data.loc[condition, ["FE2O3T"]] = round(
                data.loc[condition]["FE2O3"] + data.loc[condition]["FEO"] / 0.89998,
                self.digits)
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

            # If FE2O3 and FEOT exist but not FE2O3T or FEO
            condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() &
                         data["FEO"].isna() & data["FEOT"].notna())
            data.loc[condition, "FE2O3T"] = data.loc[condition, "FE2O3"]
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

            ## If FE2O3, FEO and FEOT exist but not FE2O3T
            condition = (data["FE2O3"].notna() & data["FE2O3T"].isna() &
                         data["FEO"].notna() & data["FEOT"].notna())
            data.loc[condition, ["FE2O3T"]] = round(
                data.loc[condition]["FEOT"] / 0.89998, self.digits)
            data.loc[condition, ["FE2O3", "FEO", "FEOT"]] = np.nan

        except Exception as e:
            print(f"Error in _convert_to_fe2o3t():\n  {e}")
            return None

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _normalize_volatile_free(self, df):
        """
        Normalizes oxide data to a volatile-free basis, ensuring the sum of oxides
        falls within a defined threshold range.

        This method performs the following steps:
        1. Calculates the total oxide content (with and without volatiles).
        2. Filters out samples based on the total oxide threshold (default: 97–103%).
        3. Normalizes oxide data to a 100% volatile-free basis.
        4. Re-calculates the total oxide content after normalization.

        Inputs:
            df (DataFrame): A DataFrame containing oxide and volatile data.

        Outputs:
            DataFrame: A DataFrame with normalized oxide data on a volatile-free basis.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            data = df.copy()

            # Sum oxides with and without volatiles
            data["total_ox"] = data[self.ox_data].sum(axis=1).round(self.digits)
            data["total_loi"] = data[
                self.ox_data + self.volatiles].sum(axis=1).round(self.digits)

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
            data[self.ox_data] = round(
                data[self.ox_data].div(data["total_ox"], axis=0) * 100, self.digits)

            # Re-sum oxides with and without volatiles
            data["total_ox"] = data[self.ox_data].sum(axis=1).round(self.digits)
            data["total_loi"] = data[
                self.ox_data + self.volatiles].sum(axis=1).round(self.digits)

        except Exception as e:
            print(f"Error in _normalize_volatile_free():\n  {e}")
            return None

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _convert_to_feot(self, df):
        """
        Converts FeO and Fe2O3 values into FeOT (total iron as FeO) by applying conversion
        formulas based on the available iron oxide data.

        This method processes the following scenarios:
        1. If FEOT exists, it sets all other Fe forms (FeO, Fe2O3, FE2O3T) to NaN.
        2. If FEO exists and FEOT is missing, it calculates FEOT from FEO.
        3. If FE2O3 exists and FEOT is missing, it converts FE2O3 to FEOT.
        4. If FE2O3T exists and FEOT is missing, it converts FE2O3T to FEOT.
        5. Various combinations of FeO, Fe2O3, and FE2O3T are handled similarly.

        Inputs:
            df (DataFrame): A DataFrame containing FeO, Fe2O3, FE2O3T, and FEOT values.

        Outputs:
            DataFrame: A DataFrame where Fe values are normalized to FEOT, with other Fe
            forms set to NaN.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            data = df.copy()

            # If FEOT exists set all Fe to nan except FEOT
            condition = data["FEOT"].notna()
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

            # If FEO exists but not FEOT, FE2O3, or FE2O3T
            condition = (data["FEO"].notna() & data["FEOT"].isna() &
                         data["FE2O3"].isna() & data["FE2O3T"].isna())
            data.loc[condition, ["FEOT"]] = data.loc[condition]["FEO"]
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

            # If FE2O3 exists but not FEO, FEOT, or FE2O3T
            condition = (data["FEO"].isna() & data["FEOT"].isna() &
                         data["FE2O3"].notna() & data["FE2O3T"].isna())
            data.loc[condition, ["FEOT"]] = round(
                data.loc[condition]["FE2O3"] * 0.89998, self.digits)
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

            # If FE2O3T exists but not FEO, FEOT, or FE2O3
            condition = (data["FEO"].isna() & data["FEOT"].isna() &
                         data["FE2O3"].isna() & data["FE2O3T"].notna())
            data.loc[condition, ["FEOT"]] = round(
                data.loc[condition]["FE2O3T"] * 0.89998, self.digits)
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

            # If FE2O3 and FE2O3T exists but not FEO or FEOT
            condition = (data["FEO"].isna() & data["FEOT"].isna() &
                         data["FE2O3"].notna() & data["FE2O3T"].notna())
            data.loc[condition, ["FEOT"]] = round(
                data.loc[condition]["FE2O3T"] * 0.89998, self.digits)
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

            # If FEO and FE2O3 exist but not FEOT or FE2O3T
            condition = (data["FEO"].notna() & data["FEOT"].isna() &
                         data["FE2O3"].notna() & data["FE2O3T"].isna())
            data.loc[condition, ["FEOT"]] = round(
                data.loc[condition]["FEO"] + data.loc[condition]["FE2O3"] * 0.89998,
                self.digits)
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

            # If FEO and FE2O3T exist but not FEOT or FE2O3
            condition = (data["FEO"].notna() & data["FEOT"].isna() &
                         data["FE2O3"].isna() & data["FE2O3T"].notna())
            data.loc[condition, "FEOT"] = data.loc[condition, "FEO"]
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

            # If FEO, FE2O3 and FE2O3T exist but not FEOT
            condition = (data["FEO"].notna() & data["FEOT"].isna() &
                         data["FE2O3"].notna() & data["FE2O3T"].notna())
            data.loc[condition, ["FEOT"]] = round(
                data.loc[condition]["FE2O3T"] * 0.89998, self.digits)
            data.loc[condition, ["FEO", "FE2O3", "FE2O3T"]] = np.nan

        except Exception as e:
            print(f"Error in _convert_to_feot():\n  {e}")
            return None

        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_earthchem_data(self):
        """
        Processes raw EarthChem data by filtering, converting, normalizing, and
        summarizing rock samples.

        This method performs the following steps:
        - Filters out samples missing required oxide values (SIO2, MGO, AL2O3, CAO).
        - Excludes specific rock types from analysis and assigns samples to general
          rock type categories.
        - Removes outliers based on the interquartile range (IQR) method.
        - Converts elemental compositions (Cr to CR2O3, Ni to NIO, Fe oxides to FE2O3T,
          and FEOT).
        - Normalizes data to a volatile-free basis and cleans up redundant columns.
        - Consolidates methods into a single column and arranges data by specific rules.
        - Filters samples based on alteration ratios (R_MGSI, R_ALSI).
        - Summarizes and saves numeric data (counts, min, max, mean, etc.) to a CSV file.
        - Optionally, prints a summary of the combined and filtered samples if verbose
          mode is enabled.

        Attributes updated:
            self.earthchem_filtered : pandas.DataFrame
                The processed EarthChem data after filtering, conversion, and summarization.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
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
            condition = data[["SIO2", "MGO", "AL2O3", "CAO"]].notna().all(axis=1)
            data = data.loc[condition]

            # Drop certain rocks
            data = data[~data["ROCKNAME"].isin(
                xenolith + metamorphic + other + pyroxenite + serpentinite +
                ["wehrlite", "wehlerite", "peridotite", "dunite"])]

            # Add new rock type column
            conditions = [data["ROCKNAME"].isin(peridotite),
                          data["ROCKNAME"].isin(pyroxenite),
                          data["ROCKNAME"].isin(serpentinite),
                          data["ROCKNAME"].isin(metamorphic),
                          data["ROCKNAME"].isin(xenolith),
                          data["ROCKNAME"].isin(other)]
            values = ["peridotite", "pyroxenite", "serpentinite", "metamorphic", "xenolith",
                      "other"]
            data["ROCKTYPE"] = np.select(conditions, values, default="other")

            # Function to remove outliers based on IQR
            def remove_outliers(group, threshold):
                Q1 = group[self.ox_data].quantile(0.25)
                Q3 = group[self.ox_data].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - threshold * IQR, Q3 + threshold * IQR
                outlier_rows = ((group[self.ox_data] < lower_bound) |
                                (group[self.ox_data] > upper_bound)).any(axis=1)
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

            # Drop totals
            data = data.drop(["total_ox", "total_loi"], axis=1)

            # Drop all NA columns
            data = data.dropna(axis=1, how="all")

            # Rename FEOT
            data = data.rename(columns={"FEOT": "FEO"})

            # Consolidate unique methods
            cols = [col for col in data.columns if "METH" in col]
            unique_methods = data[cols].apply(
                lambda x: ", ".join(x.dropna().unique()), axis=1)
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
            data[numeric_columns] = data[numeric_columns].apply(
                pd.to_numeric, errors="coerce")
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
                                 data[column].quantile(0.25), self.digits)
                })
            info_df = pd.DataFrame(numeric_info)
            info_df.to_csv("assets/earthchem-counts.csv", index=False)

            if self.verbose >= 1:
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

        except Exception as e:
            print(f"Error in _process_earthchem_data():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _normalize_sample_composition(self, row):
        """
        Normalizes a sample composition by excluding specified oxides and rescaling
        the remainder.

        This method performs the following steps:
        - Filters out excluded oxides based on the `ox_exclude` list.
        - Sets negative component values to zero.
        - Calculates the total of the remaining (non-excluded) components.
        - Normalizes the components to ensure their total sums to 100.
        - Handles cases where no oxides are excluded by returning the
          unnormalized composition.

        Parameters:
            row : pandas.Series
                A row containing oxide values for a single sample, with column names
                corresponding to oxide names.

        Returns:
            list:
                A list of normalized oxide compositions, where excluded oxides have been
                removed and the remaining components are rescaled to sum to 100.

        Raises:
            Exception: If any error occurs during the normalization process.
        """
        try:
            ox_sub = [oxide for oxide in self.ox_pca if oxide not in self.ox_exclude]

            # Get sample composition
            sample_composition = row[self.ox_pca].values

            # No normalizing for all components
            if not self.ox_exclude:
                return sample_composition

            # Filter components
            subset_sample = [comp for comp, oxide in zip(sample_composition, self.ox_pca)
                             if oxide in ox_sub]

            # Set negative compositions to zero
            subset_sample = [comp if comp >= 0 else 0 for comp in subset_sample]

            # Get total oxides
            total_subset_comps = sum([comp for comp in subset_sample if comp != 0])

            # Normalize
            normalized_comps = [
                round(((comp / total_subset_comps) * 100 if comp != 0 else 0),
                      self.digits) for comp, oxide in zip(subset_sample, ox_sub)]

        except Exception as e:
            print(f"Error in _normalize_sample_composition():\n  {e}")
            return None

        return normalized_comps

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_benchmark_samples_pca(self):
        """
        Processes benchmark samples for PCA analysis by normalizing compositions,
        calculating ratios, and fitting PCA models.

        This method performs the following steps:
        - Reads benchmark sample data from a CSV file.
        - Calculates classic mantle array ratios (R_MGSI and R_ALSI) based on oxide
          compositions.
        - Normalizes sample compositions while excluding specified oxides.
        - Standardizes the data using a pre-fitted scaler.
        - Fits the PCA model to the standardized benchmark samples.
        - Updates the benchmark DataFrame with principal component scores.
        - Calculates additional melt fractions and ratios (F_MELT_BATCH, XI_BATCH,
          F_MELT_FRAC, XI_FRAC).
        - Selects relevant columns for the final DataFrame.
        - Saves the processed benchmark data to a new CSV file.

        Attributes updated:
            None: This method does not modify any attributes directly, but it generates
            a processed DataFrame that is saved to a CSV file.

        Raises:
            Exception: If any error occurs during the processing of benchmark samples.
        """
        try:
            metadata = ["SAMPLEID"]
            ox_sub = [oxide for oxide in self.ox_pca if oxide not in self.ox_exclude]
            other_cols = ["R_MGSI", "R_ALSI", "R_TIO2", "F_MELT_BATCH", "XI_BATCH",
                          "F_MELT_FRAC", "XI_FRAC"]

            df_bench_path = "assets/benchmark-samples.csv"
            df_bench_pca_path = "assets/bench-pca.csv"

            # Get dry synthetic endmember compositions
            df_mids = pd.read_csv("assets/synth-mids.csv")
            df_dry = df_mids[df_mids["H2O"] == 0]
            sids = [df_dry["SAMPLEID"].head(1).values[0],
                    df_dry["SAMPLEID"].tail(1).values[0]]
            df_synth_bench = df_mids[df_mids["SAMPLEID"].isin(sids) & (df_mids["H2O"] == 0)]

            # Read benchmark samples
            if os.path.exists(df_bench_path):
                # Get max TiO2 from mixing array endpoints
                ti_init = df_synth_bench["TIO2"].max()

                # Read benchmark samples
                df_bench = pd.read_csv(df_bench_path)

                # "Classic" mantle array ratios (Deschamps et al., 2013, Lithos)
                df_bench["R_MGSI"] = round(df_bench["MGO"] / df_bench["SIO2"], self.digits)
                df_bench["R_ALSI"] = round(df_bench["AL2O3"] / df_bench["SIO2"], self.digits)

                # Normalize compositions
                normalized_values = df_bench.apply(
                    self._normalize_sample_composition, axis=1)
                df_bench[ox_sub] = normalized_values.apply(pd.Series)
                df_bench[self.ox_exclude] = float(0)

                # Standardize data
                df_bench_scaled = self.scaler.transform(df_bench[self.ox_pca])

                # Fit PCA to benchmark samples
                principal_components = self.pca_model.transform(df_bench_scaled)

                # Update dataframe
                pca_columns = [f"PC{i+1}" for i in range(self.n_pca_components)]
                df_bench[pca_columns] = principal_components

                # Round numerical data
                df_bench[self.ox_pca + pca_columns] = df_bench[
                    self.ox_pca + pca_columns].round(self.digits)

                # Calculate F melt
                df_bench["R_TIO2"] = round(df_bench["TIO2"] / ti_init, self.digits)
                df_bench["F_MELT_BATCH"] = round(
                    ((self.D_tio2 / df_bench["R_TIO2"]) - self.D_tio2) /
                    (1 - self.D_tio2), self.digits)
                df_bench["XI_BATCH"] = round(1 - df_bench["F_MELT_BATCH"], self.digits)
                df_bench["F_MELT_FRAC"] = round(
                    1 - df_bench["R_TIO2"]**(1 / ((1 / self.D_tio2) - 1)), self.digits)
                df_bench["XI_FRAC"] = round(1 - df_bench["F_MELT_FRAC"], self.digits)

                # Select columns
                df_bench = df_bench[
                    metadata + self.ox_pca + ["H2O"] + pca_columns + other_cols]

                # Save to csv
                df_bench.to_csv(df_bench_pca_path, index=False)

        except Exception as e:
            print(f"Error in _process_benchmark_samples_pca():\n  {e}")

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.1.         Create Mixing Arrays            !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _run_pca(self):
        """
        Executes PCA analysis on filtered EarthChem data to reduce dimensionality and
        impute missing values.

        This method performs the following steps:
        - Checks for the existence of filtered EarthChem data; raises an exception if not
          found.
        - Sorts the data by SiO2 and MgO compositions.
        - Imputes missing oxide values using KNN imputer grouped by rock name.
        - Normalizes the compositions of the imputed data, excluding specified oxides.
        - Initializes and applies a standard scaler to the normalized data.
        - Initializes PCA with the specified number of components and fits the model to the
          standardized data.
        - Outputs a summary of the PCA results, including explained variance and cumulative
          explained variance if verbose mode is enabled.
        - Transforms the standardized data to obtain principal components and updates the
          original DataFrame.
        - Rounds the numerical data in the resulting DataFrame for better readability.
        - Saves the PCA results in an attribute for later use.

        Attributes updated:
            self.scaler : StandardScaler
                The fitted scaler used for standardizing data.
            self.pca_model : PCA
                The fitted PCA model after running the analysis.
            self.pca_results : numpy.ndarray
                The principal component scores for the samples.
            self.earthchem_pca : pandas.DataFrame
                The updated DataFrame containing original data and PCA results.

        Raises:
            Exception: If any error occurs during PCA execution, including missing data or
              imputation failures.
        """
        try:
            data = self.earthchem_filtered.copy()
            ox_sub = [oxide for oxide in self.ox_pca if oxide not in self.ox_exclude]

            # Check for earthchem data
            if data.empty:
                raise Exception(
                    "No Earthchem data found! Call _read_earthchem_data() first ...")

            # Sort by composition
            data = data.sort_values(by=["SIO2", "MGO"], ascending=[True, False],
                                    ignore_index=True)

            print("Imputing missing oxides ...")

            # Group by rockname and apply KNN imputer
            imputed_dataframes = []
            for rockname, subset in data.groupby("ROCKNAME"):
                subset = subset.reset_index(drop=True)

                for col in self.ox_pca:
                    column_to_impute = subset[[col]]
                    imputer = KNNImputer(weights="distance")
                    imputed_values = imputer.fit_transform(
                        column_to_impute).round(self.digits)
                    subset[col] = imputed_values

                imputed_dataframes.append(subset)

            # Recombine imputed subsets
            imputed_data = pd.concat(imputed_dataframes, ignore_index=True)

            # Normalize compositions
            normalized_values = data.apply(self._normalize_sample_composition, axis=1)
            data[ox_sub] = normalized_values.apply(pd.Series)
            data[self.ox_exclude] = float(0)

            # Initialize scaler
            scaler = StandardScaler()

            # Standardize data
            data_scaled = scaler.fit_transform(data[self.ox_pca])

            # Update attribute
            self.scaler = scaler

            # Initialize PCA
            pca = PCA(n_components=self.n_pca_components)

            print(f"Running PCA to reduce to {self.n_pca_components} dimensions ...")

            # PCA modeling
            pca.fit(data_scaled)

            # Update self attribute
            self.pca_model = pca

            if self.verbose >= 1:
                # Print summary
                print("+++++++++++++++++++++++++++++++++++++++++++++")
                print("PCA summary:")
                print(f"  PCA oxides: {ox_sub}")
                print(f"  number of samples: {pca.n_samples_}")
                print(f"  PCA components: {self.n_pca_components}")
                print( "  explained variance:")
                for i, value in enumerate(pca.explained_variance_ratio_):
                    print(f"      PC{i+1}: {round(value, self.digits)}")
                print("  cumulative explained variance:")
                cumulative_variance = pca.explained_variance_ratio_.cumsum()
                for i, value in enumerate(cumulative_variance):
                    print(f"      PC{i+1}: {round(value, self.digits)}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # Transform the data to obtain the principal components
            principal_components = pca.transform(data_scaled)

            # Update self attribute
            self.pca_results = principal_components

            # Update dataframe
            pca_columns = [f"PC{i+1}" for i in range(self.n_pca_components)]
            data[pca_columns] = principal_components

            # Round numerical data
            data[self.ox_pca + pca_columns] = data[self.ox_pca + pca_columns].round(3)

            # Update self attribute
            self.earthchem_pca = data.copy()

        except Exception as e:
            print(f"Error in _run_pca():\n  {e}")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _add_h2o(self, df):
        """
        Adds a range of H2O values to the provided DataFrame, duplicating samples
        for each H2O value and updating the SAMPLEID accordingly.

        This method performs the following steps:
        - Constructs a linear array of H2O values from 0 to the specified maximum
          H2O content, with a resolution defined by `res_h2o`.
        - Duplicates each sample ID in the DataFrame to accommodate the new H2O values.
        - Assigns the generated H2O values to the newly duplicated rows in the DataFrame.
        - Updates the SAMPLEID to include a suffix indicating the H2O iteration count,
          formatted with leading zeros.

        Attributes updated:
            df : pandas.DataFrame
                The modified DataFrame containing the original samples and additional
                H2O values, with updated SAMPLEIDs.

        Returns:
            pandas.DataFrame: The updated DataFrame with H2O values added.

        Raises:
            Exception: If any error occurs during the addition of H2O values, including
            issues with indexing or DataFrame operations.
        """
        try:
            ox_pca = self.ox_pca + ["H2O"]
            ox_sub = [oxide for oxide in ox_pca if oxide not in self.ox_exclude]

            # Create linear array of ad hoc h2o
            h2o = np.linspace(0.0, self.max_h2o, self.res_h2o).round(self.digits)
            sid_append = np.arange(0, len(h2o))

            # Duplicate each sampleid
            df = df.reindex(df.index.repeat(self.res_h2o))

            # Add h2o to each sampleid
            df["H2O"] = np.tile(h2o, len(df) // len(h2o))
            df["SAMPLEID"] = (
                df["SAMPLEID"] +
                df.groupby(level=0).cumcount().map(lambda x: f"-h2o{str(x).zfill(3)}"))

        except Exception as e:
            print(f"Error in _add_h2o():\n  {e}")
            return None

        return df

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def create_mixing_array(self):
        """
        Create a mixing array from EarthChem data using PCA (Principal Component Analysis).

        This method reads EarthChem data, processes it, and performs PCA to extract
        principal components. It calculates endpoints, tops, and bottoms for a mixing
        array based on the PCA results. Additionally, it generates synthetic data for
        mixing lines and exports the results to CSV files.

        Steps involved:
            1. Read and process EarthChem data.
            2. Run PCA on the processed data.
            3. Define sample centroids based on PCA components.
            4. Calculate the endpoints, tops, and bottoms of mixing arrays for each rock
               type.
            5. Generate mixing lines between endpoints and store them.
            6. Create synthetic data for the mixing arrays and save to CSV files.
            7. Ensure non-negative oxide values in the results.
            8. Calculate and add additional properties like R_MGSI, R_ALSI, and H2O.

        Raises:
            Exception: If an invalid quadrant is detected during endpoint calculation.

        Returns:
            None: The method modifies instance attributes to store the resulting mixing
            arrays and writes the data to CSV files without returning any values.

        Notes:
            - This method relies on several instance variables, such as
              `self.n_pca_components`, `self.earthchem_pca`, `self.scaler`, and
              `self.pca_model`, among others.
            - The results are saved to files in the 'assets' directory, specifically:
              - `earthchem-pca.csv`
              - `synth-mids.csv`
              - `synth-tops.csv`
              - `synth-bots.csv`
        """
        try:
            self._read_earthchem_data()
            self._process_earthchem_data()
            self._run_pca()

            data = self.earthchem_pca.copy()
            pca_columns = [f"PC{i+1}" for i in range(self.n_pca_components)]
            other_cols = ["R_MGSI", "R_ALSI", "R_TIO2", "F_MELT_BATCH", "XI_BATCH",
                          "F_MELT_FRAC", "XI_FRAC"]

            # Define sample centroids
            centroids = data.groupby("ROCKNAME")[["PC1", "PC2"]].median()

            # Initialize endpoints
            mixing_array_endpoints = []
            mixing_array_tops = []
            mixing_array_bots = []

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
                median_adjustment_q2x = 0
                median_adjustment_q2y = 0
                median_adjustment_q3x = -1.4
                median_adjustment_q3y = 0.6
                median_adjustment_q4x = 1.0
                median_adjustment_q4y = 0.05
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
            mixing_array_bots = mixing_array_endpoints.copy()

            mixing_array_tops[:,-1] += top_adjustment * iqr_pc2
            mixing_array_bots[:,-1] -= bottom_adjustment * iqr_pc2

            self.mixing_array_endpoints = mixing_array_endpoints
            self.mixing_array_tops = mixing_array_tops
            self.mixing_array_bots = mixing_array_bots

            # Initialize mixing lines
            mixing_lines = {}
            top_lines = {}
            bottom_lines = {}

            # Loop through PCA components
            for n in range(self.n_pca_components):
                for i in range(len(mixing_array_endpoints)):
                    # Calculate mixing lines between endpoints
                    if len(mixing_array_endpoints) > 1:
                        for j in range(i + 1, len(mixing_array_endpoints)):
                            nsmps = (self.res // (len(mixing_array_endpoints) - 1)) + 1
                            if n == 0:
                                mixing_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_endpoints[i, n],
                                                mixing_array_endpoints[j, n], nsmps))
                                top_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_tops[i, n],
                                                mixing_array_tops[j, n], nsmps))
                                bottom_lines[f"{i + 1}{j + 1}"] = (
                                    np.linspace(mixing_array_bots[i, n],
                                                mixing_array_bots[j, n], nsmps))

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
                                        np.linspace(mixing_array_bots[i, n],
                                                    mixing_array_bots[j, n], nsmps)))

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
                            np.hstack((self.scaler.inverse_transform(
                                self.pca_model.inverse_transform(
                                    mixing_lines[f"{i + 1}{j + 1}"].T)),
                                    mixing_lines[f"{i + 1}{j + 1}"].T)),
                            columns=self.ox_pca + [f"PC{n + 1}" for n
                                                   in range(self.n_pca_components)]).round(3)
                        tops_synthetic = pd.DataFrame(
                            np.hstack((self.scaler.inverse_transform(
                                self.pca_model.inverse_transform(
                                    top_lines[f"{i + 1}{j + 1}"].T)),
                                    top_lines[f"{i + 1}{j + 1}"].T)),
                            columns=self.ox_pca + [f"PC{n + 1}" for n in
                                                   range(self.n_pca_components)]).round(3)
                        bots_synthetic = pd.DataFrame(
                            np.hstack((self.scaler.inverse_transform(
                                self.pca_model.inverse_transform(
                                    bottom_lines[f"{i + 1}{j + 1}"].T)),
                                    bottom_lines[f"{i + 1}{j + 1}"].T)),
                            columns=self.ox_pca + [f"PC{n + 1}" for n in
                                                   range(self.n_pca_components)]).round(3)

                        # Append to list
                        mixing_list.append(mixing_synthetic)
                        top_list.append(tops_synthetic)
                        bottom_list.append(bots_synthetic)

            # Combine mixing arrays
            all_mixs = pd.concat(mixing_list, ignore_index=True)
            all_tops = pd.concat(top_list, ignore_index=True)
            all_bots = pd.concat(bottom_list, ignore_index=True)

            # Add sample id column
            all_mixs.insert(
                0, "SAMPLEID", [f"sm{str(n).zfill(3)}" for n in range(len(all_mixs))])
            all_tops.insert(
                0, "SAMPLEID", [f"st{str(n).zfill(3)}" for n in range(len(all_tops))])
            all_bots.insert(
                0, "SAMPLEID", [f"sb{str(n).zfill(3)}" for n in range(len(all_bots))])

            # No negative oxides
            data[self.ox_pca] = data[
                self.ox_pca].apply(lambda x: x.apply(lambda y: max(0.001, y)))
            all_mixs[self.ox_pca] = all_mixs[
                self.ox_pca].apply(lambda x: x.apply(lambda y: max(0.001, y)))
            all_tops[self.ox_pca] = all_tops[
                self.ox_pca].apply(lambda x: x.apply(lambda y: max(0.001, y)))
            all_bots[self.ox_pca] = all_bots[
                self.ox_pca].apply(lambda x: x.apply(lambda y: max(0.001, y)))

            # Increase TIO2 by 10% for mixing arrays so that F melt is consistent with PUM
            all_mixs["TIO2"] = all_mixs["TIO2"] + (all_mixs["TIO2"] * 0.1)
            all_tops["TIO2"] = all_mixs["TIO2"] + (all_mixs["TIO2"] * 0.1)
            all_bots["TIO2"] = all_mixs["TIO2"] + (all_mixs["TIO2"] * 0.1)

            # Calculate F melt
            ti_init = all_mixs["TIO2"].max()
            data["R_TIO2"] = round(data["TIO2"] / ti_init, self.digits)
            data["F_MELT_BATCH"] = round(
                ((self.D_tio2 / data["R_TIO2"]) - self.D_tio2) / (1 - self.D_tio2),
                self.digits)
            data["XI_BATCH"] = round(1 - data["F_MELT_BATCH"], self.digits)
            data["F_MELT_FRAC"] = round(1 - data["R_TIO2"]**(1 / ((1 / self.D_tio2) - 1)),
                                        self.digits)
            data["XI_FRAC"] = round(1 - data["F_MELT_FRAC"], self.digits)

            # Select columns
            data = data[self.metadata + self.ox_pca + ["LOI"] + pca_columns + other_cols]
            data = data.rename(columns={"LOI": "H2O"})

            self.earthchem_pca = data.copy()

            # Write csv file
            data.to_csv(f"assets/earthchem-pca.csv", index=False)

            # Calculate F melt
            all_mixs["R_TIO2"] = round(all_mixs["TIO2"] / ti_init, self.digits)
            all_mixs["F_MELT_BATCH"] = round(
                ((self.D_tio2 / all_mixs["R_TIO2"]) - self.D_tio2) / (1 - self.D_tio2),
                self.digits)
            all_mixs["XI_BATCH"] = round(1 - all_mixs["F_MELT_BATCH"], self.digits)
            all_mixs["F_MELT_FRAC"] = round(
                1 - all_mixs["R_TIO2"]**(1 / ((1 / self.D_tio2) - 1)), self.digits)
            all_mixs["XI_FRAC"] = round(1 - all_mixs["F_MELT_FRAC"], self.digits)

            all_tops["R_TIO2"] = round(all_tops["TIO2"] / ti_init, self.digits)
            all_tops["F_MELT_BATCH"] = round(
                ((self.D_tio2 / all_tops["R_TIO2"]) - self.D_tio2) / (1 - self.D_tio2),
                self.digits)
            all_tops["XI_BATCH"] = round(1 - all_tops["F_MELT_BATCH"], self.digits)
            all_tops["F_MELT_FRAC"] = round(
                1 - all_tops["R_TIO2"]**(1 / ((1 / self.D_tio2) - 1)), self.digits)
            all_tops["XI_FRAC"] = round(1 - all_tops["F_MELT_FRAC"], self.digits)

            all_bots["R_TIO2"] = round(all_bots["TIO2"] / ti_init, self.digits)
            all_bots["F_MELT_BATCH"] = round(
                ((self.D_tio2 / all_bots["R_TIO2"]) - self.D_tio2) / (1 - self.D_tio2),
                self.digits)
            all_bots["XI_BATCH"] = round(1 - all_bots["F_MELT_BATCH"], self.digits)
            all_bots["F_MELT_FRAC"] = round(
                1 - all_bots["R_TIO2"]**(1 / ((1 / self.D_tio2) - 1)), self.digits)
            all_bots["XI_FRAC"] = round(1 - all_bots["F_MELT_FRAC"], self.digits)

            # "Classic" mantle array ratios (Deschamps et al., 2013, Lithos)
            all_mixs["R_MGSI"] = round(all_mixs["MGO"] / all_mixs["SIO2"], self.digits)
            all_mixs["R_ALSI"] = round(all_mixs["AL2O3"] / all_mixs["SIO2"], self.digits)
            all_tops["R_MGSI"] = round(all_tops["MGO"] / all_tops["SIO2"], self.digits)
            all_tops["R_ALSI"] = round(all_tops["AL2O3"] / all_tops["SIO2"], self.digits)
            all_bots["R_MGSI"] = round(all_bots["MGO"] / all_bots["SIO2"], self.digits)
            all_bots["R_ALSI"] = round(all_bots["AL2O3"] / all_bots["SIO2"], self.digits)

            # Add H2O ad hoc
            all_tops = self._add_h2o(all_tops)
            all_mixs = self._add_h2o(all_mixs)
            all_bots = self._add_h2o(all_bots)

            # Select columns
            all_mixs = all_mixs[["SAMPLEID"] + self.ox_pca + ["H2O"] + pca_columns +
                                other_cols]
            all_tops = all_tops[["SAMPLEID"] + self.ox_pca + ["H2O"] + pca_columns +
                                other_cols]
            all_bots = all_bots[["SAMPLEID"] + self.ox_pca + ["H2O"] + pca_columns +
                                other_cols]

            # Write to csv
            all_mixs.to_csv("assets/synth-mids.csv", index=False)
            all_tops.to_csv("assets/synth-tops.csv", index=False)
            all_bots.to_csv("assets/synth-bots.csv", index=False)

            # Define bounding box around top and bottom mixing arrays
            min_x = min(mixing_array_tops[:, 0].min(), mixing_array_bots[:, 0].min())
            max_x = max(mixing_array_tops[:, 0].max(), mixing_array_bots[:, 0].max())

            # Define the sampling interval
            interval_x = (max_x - min_x) / self.res

            randomly_sampled_points = []
            sample_ids = []

            # Monte carlo sampling of synthetic samples
            for j in range(self.mc_sample):
                # Set seed
                np.random.seed(self.seed + j)

                # Create an array to store sampled points
                sampled_points = []
                sampled_weights = []

                # Iterate over x positions
                for x in np.linspace(min_x, max_x, self.res):
                    # Calculate the range of y values for the given x position
                    y_min = np.interp(x, mixing_array_tops[:, 0], mixing_array_tops[:, 1])
                    y_max = np.interp(x, mixing_array_bots[:, 0], mixing_array_bots[:, 1])
                    y_mid = (y_max + y_min) / 2

                    # Create a grid of y values for the current x position
                    y_values = np.linspace(y_min, y_max, self.res)

                    # Calculate exponential distance weights
                    point_weights = np.exp(-self.k * np.abs(y_values - y_mid))

                    # Combine x and y values to create points
                    points = np.column_stack((x * np.ones_like(y_values), y_values))

                    # Append points to the sampled_points array
                    sampled_points.extend(points)
                    sampled_weights.extend(point_weights)

                # Convert to np array
                sampled_points = np.array(sampled_points)
                sampled_weights = np.array(sampled_weights)

                # Define probability distribution for selecting random points
                if self.weighted_random:
                    prob_dist = sampled_weights / np.sum(sampled_weights)
                else:
                    prob_dist = None

                # Randomly select from sampled points
                sample_idx = np.random.choice(
                    len(sampled_points), self.res, replace=False, p=prob_dist)
                randomly_sampled_points.append([sampled_points[i] for i in sample_idx])

                # Save random points
                sample_ids.extend([f"sr{str(n).zfill(3)}" for n in range(len(sample_idx))])

            # Combine randomly sampled points
            randomly_sampled_points = np.vstack(randomly_sampled_points)

            # Create dataframe
            all_rnds = pd.DataFrame(
                np.hstack((self.scaler.inverse_transform(self.pca_model.inverse_transform(
                    randomly_sampled_points)), randomly_sampled_points)),
                columns=self.ox_pca + [f"PC{n + 1}" for n in range(self.n_pca_components)]
            ).round(3)

            # Add sample id column
            all_rnds.insert(0, "SAMPLEID", sample_ids)

            # No negative oxides
            all_rnds[self.ox_pca] = all_rnds[self.ox_pca].apply(
                lambda x: x.apply(lambda y: max(0.001, y)))

            # Increase TIO2 by 10% for mixing arrays so that F melt is consistent with PUM
            all_rnds["TIO2"] = (all_rnds["TIO2"] + (all_rnds["TIO2"] * 0.1))

            # Calculate F melt
            all_rnds["R_TIO2"] = round(all_rnds["TIO2"] / ti_init, self.digits)
            all_rnds["F_MELT_BATCH"] = round(
                ((self.D_tio2 / all_rnds["R_TIO2"]) - self.D_tio2) / (1 - self.D_tio2),
                self.digits)
            all_rnds["XI_BATCH"] = round(
                1 - all_rnds["F_MELT_BATCH"], self.digits)
            all_rnds["F_MELT_FRAC"] = round(
                1 - all_rnds["R_TIO2"]**(1 / ((1 / self.D_tio2) - 1)), self.digits)
            all_rnds["XI_FRAC"] = round(1 - all_rnds["F_MELT_FRAC"], self.digits)

            # "Classic" mantle array ratios (Deschamps et al., 2013, Lithos)
            all_rnds["R_MGSI"] = round(all_rnds["MGO"] / all_rnds["SIO2"], self.digits)
            all_rnds["R_ALSI"] = round(all_rnds["AL2O3"] / all_rnds["SIO2"], self.digits)

            # Add H2O ad hoc
            all_rnds = self._add_h2o(all_rnds)

            # Select columns
            all_rnds = all_rnds[["SAMPLEID"] + self.ox_pca + ["H2O"] + pca_columns +
                                other_cols]

            # Write to csv
            all_rnds.to_csv("assets/synth-rnds.csv", index=False)

            # Process benchmark samples
            self._process_benchmark_samples_pca()

        except Exception as e:
            print(f"Error in create_mixing_array():\n  {e}")
            traceback.print_exc()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+ .1.2.              Visualize                  !!! ++
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def visualize_mixing_array(self, figwidth=6.3, figheight=5.2, fontsize=24):
        """
        """
        # Get self attributes
        res = self.res
        pca = self.pca_model
        oxides = self.ox_pca
        fig_dir = self.fig_dir
        data = self.earthchem_pca

        # Fertility index column
        XI_col = "XI_FRAC"

        # Check for benchmark samples
        df_bench_pca_path = "assets/bench-pca.csv"
        df_synth_mids_path = "assets/synth-mids.csv"
        df_synth_random_path = "assets/synth-rnds.csv"

        # Get dry synthetic endmember compositions
        df_mids = pd.read_csv("assets/synth-mids.csv")
        df_dry = df_mids[df_mids["H2O"] == 0]
        sids = [df_dry["SAMPLEID"].head(1).values[0], df_dry["SAMPLEID"].tail(1).values[0]]
        df_synth_bench = df_mids[df_mids["SAMPLEID"].isin(sids) & (df_mids["H2O"] == 0)]

        # Mixing array endmembers
        bend = df_synth_bench["SAMPLEID"].iloc[0]
        tend = df_synth_bench["SAMPLEID"].iloc[-1]

        # Read benchmark samples
        if os.path.exists(df_bench_pca_path):
                df_bench = pd.read_csv(df_bench_pca_path)

        if (os.path.exists(df_synth_mids_path) and os.path.exists(df_synth_random_path)):
                df_synth_mids = pd.read_csv(df_synth_mids_path)
                df_synth_random = pd.read_csv(df_synth_random_path)

        # Filter Depletion < 1
        data = data[(data[XI_col] <= 1) & (data[XI_col] >= 0)]

        loadings = pd.DataFrame(
            (pca.components_.T * np.sqrt(pca.explained_variance_)).T, columns=oxides)

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Set plot style and settings
        plt.rcParams["font.size"] = fontsize

        # Colormap
        colormap = plt.colormaps["tab10"]

        # Legend order
        legend_order = ["lherzolite", "harzburgite"]
        legend_lab = ["lherzolite", "harzburgite"]

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


        oxs = ["SIO2", "MGO", "FEO", "AL2O3", "TIO2"]
        x_offset_text = [-1.7, -1.2, -1.2, -2.5, -2.3]
        y_offset_text = [5.8, 6.0, 6.0, 5.7, 6.7]
        text_fac, arrow_fac = 3.5, 1.8
        x_offset_arrow, y_offset_arrow= -1.2, 6.0

        for oxide, x_off, y_off in zip(oxs, x_offset_text, y_offset_text):
            if oxide == "AL2O3":
                oxide_label = "Al$_2$O$_3$ CaO"
            elif oxide == "TIO2":
                oxide_label = "TiO$_2$ Na$_2$O"
            elif oxide == "SIO2":
                oxide_label = "SiO$_2$"
            elif oxide == "MGO":
                oxide_label = "MgO"
            elif oxide == "FEO":
                oxide_label = "FeOT"

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

        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == tend],
                        x="R_ALSI", y="R_MGSI", facecolor="white", edgecolor="black",
                        linewidth=2, s=150, legend=False, ax=ax2, zorder=6)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == bend],
                        x="R_ALSI", y="R_MGSI", facecolor="white", edgecolor="black",
                        marker="D", linewidth=2, s=150, legend=False, ax=ax2, zorder=6)
        ax2.annotate("DSUM", xy=(
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == tend, "R_ALSI"].iloc[0],
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == tend, "R_MGSI"].iloc[0]),
                     xytext=(10, 10), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="black", linewidth=1.5, alpha=0.8),
                    fontsize=fontsize * 0.833, zorder=8)
        ax2.annotate("PSUM", xy=(
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == bend, "R_ALSI"].iloc[0],
            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == bend, "R_MGSI"].iloc[0]),
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
        ax2.text(0.06, 1.0, "Melting residue", rotation=-25, fontsize=fontsize * 0.833)

        legend = ax.legend(handles=legend_handles, loc="lower center", frameon=False,
                           ncol=2, columnspacing=0, handletextpad=-0.5, markerscale=3,
                           fontsize=fontsize * 0.833)

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

        sns.scatterplot(data=df_synth_mids, x="PC1", y="PC2", hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax3, zorder=0)
        sns.scatterplot(data=df_synth_random, x="PC1", y="PC2", hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax3, zorder=0)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == tend],
                        x="PC1", y="PC2", facecolor="white", edgecolor="black",
                        linewidth=2, s=150, legend=False, ax=ax3, zorder=6)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == bend],
                        x="PC1", y="PC2", facecolor="white", edgecolor="black",
                        marker="D", linewidth=2, s=150, legend=False, ax=ax3, zorder=6)

        mrkr = ["s", "^", "P"]
        for l, name in enumerate(["PUM", "DMM", "PYR"]):
            sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="PC1",
                            y="PC2", marker=mrkr[l], facecolor="white", edgecolor="black",
                            linewidth=2, s=150, legend=False, ax=ax3, zorder=7)

        # Add colorbar
        cbaxes = inset_axes(ax3, width="40%", height="3%", loc=2)
        colorbar = plt.colorbar(
            sm, ax=ax3, cax=cbaxes, label="Fertility, $\\xi$", orientation="horizontal")
        colorbar.ax.set_xticks([sm.get_clim()[0], sm.get_clim()[1]])
        colorbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))

        for label in colorbar.ax.get_xticklabels():
            label.set_ha("left")

        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

        ax4 = fig.add_subplot(224)
        sns.scatterplot(data=data, x="PC1", y="XI_FRAC", facecolor="0.6", marker=".",
                        edgecolor="None", s=55, legend=False, ax=ax4, zorder=0)
        sns.scatterplot(data=df_synth_mids, x="PC1", y=XI_col, hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax4, zorder=0)
        sns.scatterplot(data=df_synth_random, x="PC1", y=XI_col, hue=XI_col, palette=pal,
                        edgecolor="None", linewidth=2, s=55, legend=False, ax=ax4, zorder=0)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == tend],
                        x="PC1", y="XI_FRAC", facecolor="white", edgecolor="black",
                        linewidth=2, s=150, legend=False, ax=ax4, zorder=6)
        sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == bend],
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
        fig.text(0.02, 0.97, "a)", fontsize=fontsize * 1.2)
        fig.text(0.02, 0.50, "c)", fontsize=fontsize * 1.2)
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
    def visualize_harker_diagrams(self, figwidth=6.3, figheight=5.5, fontsize=22):
        """
        """
        # Get self attributes
        res = self.res
        fig_dir = self.fig_dir
        data = self.earthchem_filtered
        oxides = [ox for ox in self.ox_pca if ox not in ["SIO2", "FE2O3", "K2O"]]

        df_bench_path = "assets/benchmark-samples.csv"

        # Check for benchmark samples
        if os.path.exists(df_bench_path):
            df_bench = pd.read_csv(df_bench_path)

        # Get dry synthetic endmember compositions
        df_mids = pd.read_csv("assets/synth-mids.csv")
        df_dry = df_mids[df_mids["H2O"] == 0]
        sids = [df_dry["SAMPLEID"].head(1).values[0], df_dry["SAMPLEID"].tail(1).values[0]]
        df_synth_bench = df_mids[df_mids["SAMPLEID"].isin(sids) & (df_mids["H2O"] == 0)]

        # Mixing array endmembers
        bend = df_synth_bench["SAMPLEID"].iloc[0]
        tend = df_synth_bench["SAMPLEID"].iloc[-1]

        # Initialize synthetic datasets
        synthetic_samples = pd.read_csv(f"assets/synth-rnds.csv")

        # Check for figs directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir, exist_ok=True)

        # Set plot style and settings
        plt.rcParams["font.size"] = fontsize

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
        legend_order = ["lherzolite", "harzburgite"]
        legend_lab = ["lherz", "harz"]
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

            sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == tend],
                            x="SIO2", y=y, facecolor="white", edgecolor="black",
                            linewidth=2, s=75, legend=False, ax=ax, zorder=6)
            sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == bend],
                            x="SIO2", y=y, facecolor="white", edgecolor="black", marker="D",
                            linewidth=2, s=75, legend=False, ax=ax, zorder=6)

            if k < (num_plots - num_cols - 1):
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
def main():
    """
    Main function to manage the creation and visualization of mixing arrays.

    This function performs the following steps:
    - Checks if the benchmark PCA CSV file exists. If it does not, it proceeds
      to create a mixing array.
    - Initializes the MixingArray class and calls methods to create and visualize
      the mixing array and Harker diagrams.
    - Prints status messages indicating the success of the creation and visualization
      processes.
    - If the benchmark PCA CSV file already exists, it notifies that mixing arrays
      are found.

    Attributes checked:
        assets/bench-pca.csv : str
            The path to the benchmark PCA CSV file that determines if mixing
            arrays need to be created.

    Returns:
        None

    Raises:
        Exception: If any error occurs during the creation or visualization of
        mixing arrays, including issues with file handling or method execution.
    """
    if not os.path.exists("assets/bench-pca.csv"):
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
            print(f"Error in main():\n  {e}")
            traceback.print_exc()
    else:
        print("Mixing arrays found!")

    return None

if __name__ == "__main__":
    main()
