#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import glob
import shutil
import warnings
import subprocess
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error
warnings.simplefilter("ignore", category=RuntimeWarning)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GFEM models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from gfem import GFEMModel

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import seaborn as sns
from scipy import ndimage
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, Normalize, SymLogNorm

#######################################################
## .1.              Visualizations               !!! ##
#######################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .1.1            Helper Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
# create dataset movies !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_dataset_movies(gfem_models):
    """
    """
    # Parse and sort models
    perplex_models = gfem_models
    perplex_models = sorted(perplex_models, key=lambda m: (m.sid if m else ""))

    # Iterate through all models
    for perplex_model in perplex_models:
        perplex = True if perplex_model is not None else False

        if not perplex:
            continue

        # Get model data
        sid = perplex_model.sid
        res = perplex_model.res
        targets = perplex_model.targets
        fig_dir = perplex_model.fig_dir
        model_prefix = perplex_model.model_prefix
        verbose = perplex_model.verbose

        # Rename targets
        targets_rename = [target.replace("_", "-") for target in targets]

        # Check for existing movies
        if sid not in ["PUM", "DMM", "PYR"]:
            if "sb" in sid:
                pattern = "sb???"
                prefix = "sb"
            if "sm" in sid:
                pattern = "sm???"
                prefix = "sm"
            if "st" in sid:
                pattern = "st???"
                prefix = "st"
            if "sr" in sid:
                pattern = "sr???"
                prefix = "sr"

            existing_movs = []
            for target in targets_rename:
                mov_1 = f"figs/movies/image2-{prefix}-{target}.mp4"
                mov_2 = f"figs/movies/image3-{prefix}-{target}.mp4"
                mov_3 = f"figs/movies/image9-{prefix}.mp4"

                check = ((os.path.exists(mov_1) and os.path.exists(mov_3)) |
                         (os.path.exists(mov_2) and os.path.exists(mov_3)))

                if check:
                    existing_movs.append(check)

            if existing_movs:
                return None

            else:
                print(f"Creating movie for {prefix} samples ...")

                if not os.path.exists("figs/movies"):
                    os.makedirs("figs/movies", exist_ok=True)

                for target in targets_rename:
                    ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                              f"'figs/gfem/{pattern}_{res}/image2-"
                              f"{pattern}-{target}.png' -vf 'scale=3915:1432' "
                              f"-c:v h264 -pix_fmt yuv420p 'figs/movies/image2-{prefix}-"
                              f"{target}.mp4'")

                    try:
                        subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, shell=True)

                    except subprocess.CalledProcessError as e:
                        print(f"Error running FFmpeg command: {e}")

                if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                    for target in ["rho", "Vp", "Vs"]:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{pattern}_{res}/image3-"
                                  f"{pattern}-{target}.png' -vf 'scale=5832:"
                                  f"1432' -c:v h264 -pix_fmt yuv420p 'figs/movies/"
                                  f"image3-{prefix}-{target}.mp4'")
                        ffmpeg2 = (f"ffmpeg -i 'figs/movies/image3-"
                                   f"{prefix}-{target}.mp4' -filter_complex "
                                   f"'[0:v]reverse,fifo[r];[0:v][r] concat=n=2:v=1 [v]' "
                                   f"-map '[v]' 'figs/movies/image3-{prefix}-{target}-"
                                   f"bounce.mp4'")

                        try:
                            subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, shell=True)
                            subprocess.run(ffmpeg2, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, shell=True)

                        except subprocess.CalledProcessError as e:
                            print(f"Error running FFmpeg command: {e}")

                    ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                              f"'figs/gfem/{pattern}_{res}/image9-"
                              f"{pattern}.png' -vf 'scale=5842:4296' -c:v "
                              f"h264 -pix_fmt yuv420p 'figs/movies/image9-{prefix}.mp4'")
                    ffmpeg2 = (f"ffmpeg -i 'figs/movies/image9-{prefix}.mp4' -"
                               f"filter_complex '[0:v]reverse,fifo[r];[0:v][r] "
                               f"concat=n=2:v=1 [v]' -map '[v]' 'figs/movies/image9-"
                               f"{prefix}-bounce.mp4'")

                    try:
                        subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, shell=True)
                        subprocess.run(ffmpeg2, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, shell=True)

                    except subprocess.CalledProcessError as e:
                        print(f"Error running FFmpeg command: {e}")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose rocmlm plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocmlm_plots(rocmlm, skip=1):
    """
    """
    # Get ml model attributes
    model_prefix = rocmlm.model_prefix
    ml_model_label = rocmlm.ml_model_label
    sids = rocmlm.sids
    res = rocmlm.res
    targets = rocmlm.targets
    fig_dir = rocmlm.fig_dir
    fig_dir_perf = "figs/rocmlm"
    verbose = rocmlm.verbose

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    # Don't skip benchmark samples
    if any(sample in sids for sample in ["PUM", "DMM", "PYR"]):
        skip = 1
    else:
        # Need to skip double for synthetic samples bc X_res training starts at 2 ...
        skip = skip * 2

    # Check for existing plots
    existing_figs = []
    for target in targets_rename:
        for sid in rocmlm.sids[::skip]:
            fig_1 = f"{fig_dir}/prem-{sid}-{ml_model_label}-{target}.png"
            fig_2 = f"{fig_dir}/surf-{sid}-{ml_model_label}-{target}.png"
            fig_3 = f"{fig_dir}/surf9-{sid}-{ml_model_label}.png"
            fig_4 = f"{fig_dir}/image-{sid}-{ml_model_label}-{target}.png"
            fig_5 = f"{fig_dir}/image9-{sid}-{ml_model_label}-profile.png"
            fig_6 = f"{fig_dir}/image9-{sid}-{ml_model_label}-diff.png"
            fig_7 = f"{fig_dir}/image12-{sid}-{ml_model_label}.png"

            check = ((os.path.exists(fig_1) and os.path.exists(fig_2) and
                      os.path.exists(fig_3) and os.path.exists(fig_4) and
                      os.path.exists(fig_5) and os.path.exists(fig_6) and
                      os.path.exists(fig_7)) |
                     (os.path.exists(fig_2) and os.path.exists(fig_3) and
                      os.path.exists(fig_4) and os.path.exists(fig_5) and
                      os.path.exists(fig_6) and os.path.exists(fig_7)))

            if check:
                existing_figs.append(check)

    if existing_figs:
        return None

    for target in targets_rename:
        for sid in rocmlm.sids[::skip]:
            if verbose >= 1:
                print(f"Composing {model_prefix}-{sid}-{target} ...")

            if target in ["rho", "Vp", "Vs"]:
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
                    f"{fig_dir}/prem-{sid}-{ml_model_label}-{target}.png",
                    caption1="",
                    caption2="c)"
                )

            combine_plots_horizontally(
                f"{fig_dir}/{model_prefix}-{sid}-{target}-targets-surf.png",
                f"{fig_dir}/{model_prefix}-{sid}-{target}-surf.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{model_prefix}-{sid}-{target}-diff-surf.png",
                f"{fig_dir}/surf-{sid}-{ml_model_label}-{target}.png",
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
                f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png",
                f"{fig_dir}/image-{sid}-{ml_model_label}-{target}.png",
                caption1="",
                caption2="c)"
            )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "Vp", "Vs"]

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
                    f"{fig_dir}/temp-Vp.png",
                    f"{fig_dir}/temp1.png",
                    caption1="",
                    caption2=""
                )

                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp-Vs.png",
                    f"{fig_dir}/image9-{sid}-{ml_model_label}-profile.png",
                    caption1="",
                    caption2=""
                )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "Vp", "Vs"]

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
                    f"{fig_dir}/temp-Vp.png",
                    f"{fig_dir}/temp1.png",
                    caption1="",
                    caption2=""
                )

                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp-Vs.png",
                    f"{fig_dir}/image9-{sid}-{ml_model_label}-diff.png",
                    caption1="",
                    caption2=""
                )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "d)", "g)", "j)"), ("b)", "e)", "h)", "k)"),
                            ("c)", "f)", "i)", "l)")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_vertically(
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-targets.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-predictions.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_vertically(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-diff.png",
                        f"{fig_dir}/temp2.png",
                        caption1="",
                        caption2=captions[i][2]
                    )

                    combine_plots_vertically(
                        f"{fig_dir}/temp2.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-prem.png",
                        f"{fig_dir}/temp-{target}.png",
                        caption1="",
                        caption2=captions[i][3]
                    )

                combine_plots_horizontally(
                    f"{fig_dir}/temp-rho.png",
                    f"{fig_dir}/temp-Vp.png",
                    f"{fig_dir}/temp1.png",
                    caption1="",
                    caption2=""
                )

                combine_plots_horizontally(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp-Vs.png",
                    f"{fig_dir}/image12-{sid}-{ml_model_label}.png",
                    caption1="",
                    caption2=""
                )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-targets-surf.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-surf.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sid}-{target}-diff-surf.png",
                        f"{fig_dir}/temp-{target}.png",
                        caption1="",
                        caption2=captions[i][2]
                    )

                combine_plots_vertically(
                    f"{fig_dir}/temp-rho.png",
                    f"{fig_dir}/temp-Vp.png",
                    f"{fig_dir}/temp1.png",
                    caption1="",
                    caption2=""
                )

                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp-Vs.png",
                    f"{fig_dir}/surf9-{sid}-{ml_model_label}.png",
                    caption1="",
                    caption2=""
                )

    # Clean up directory
    rocmlm_files = glob.glob(f"{fig_dir}/rocmlm*.png")
    tmp_files = glob.glob(f"{fig_dir}/temp*.png")

    for file in rocmlm_files + tmp_files:
        os.remove(file)

    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .1.2          Plotting Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    data = data[data["sample"].isin(["SYNTH258", "SYNTH129", "SYNTH65", "SYNTH33"])]
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
    cmap = cm.get_cmap("mako_r")

    # Create a ScalarMappable to map rmse to colors
    norm = Normalize(data["rmse_val_mean_rho"].min(), data["rmse_val_mean_rho"].max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize target surf !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_target_surf(P, T, target_array, target, title, palette, color_discrete,
                          color_reverse, vmin, vmax, fig_dir, filename, figwidth=6.3,
                          figheight=4.725, fontsize=22):
    """
    """
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

    if color_discrete:
        # Discrete color palette
        num_colors = vmax - vmin + 1

        if palette == "viridis":
            if color_reverse:
                pal = plt.cm.get_cmap("viridis_r", num_colors)
            else:
                pal = plt.cm.get_cmap("viridis", num_colors)
        elif palette == "bone":
            if color_reverse:
                pal = plt.cm.get_cmap("bone_r", num_colors)
            else:
                pal = plt.cm.get_cmap("bone", num_colors)
        elif palette == "pink":
            if color_reverse:
                pal = plt.cm.get_cmap("pink_r", num_colors)
            else:
                pal = plt.cm.get_cmap("pink", num_colors)
        elif palette == "seismic":
            if color_reverse:
                pal = plt.cm.get_cmap("seismic_r", num_colors)
            else:
                pal = plt.cm.get_cmap("seismic", num_colors)
        elif palette == "grey":
            if color_reverse:
                pal = plt.cm.get_cmap("Greys_r", num_colors)
            else:
                pal = plt.cm.get_cmap("Greys", num_colors)
        elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
            if color_reverse:
                pal = plt.cm.get_cmap("Blues_r", num_colors)
            else:
                pal = plt.cm.get_cmap("Blues", num_colors)

        # Descritize
        color_palette = pal(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(color_palette)

        # Set nan color
        cmap.set_bad(color="white")

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(T, P, target_array, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(title, y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")
        cbar = fig.colorbar(surf, ax=ax, ticks=np.arange(vmin, vmax, num_colors // 4),
                            label="", shrink=0.6)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        cbar.ax.set_ylim(vmax, vmin)

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
            vmin=-np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))
            vmax=np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))

        else:
            vmin, vmax = vmin, vmax

            # Adjust vmin close to zero
            if vmin <= 1e-4:
                vmin = 0

            # Set melt fraction to 0–100 %
            if target == "melt":
                vmin, vmax = 0, 100

        # Set nan color
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color="white")

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(T, P, target_array, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.set_zlim(vmin - (vmin * 0.05), vmax + (vmax * 0.05))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(title, y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")

        # Diverging colorbar
        if palette == "seismic":
            cbar = fig.colorbar(surf, ax=ax, ticks=[vmin, 0, vmax], label="", shrink=0.6)

        # Continous colorbar
        else:
            cbar = fig.colorbar(surf, ax=ax, ticks=np.linspace(vmin, vmax, num=4),
                                label="", shrink=0.6)

        # Set z and colorbar limits and number formatting
        if target == "rho":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vp":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vs":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "melt":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "variance":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

        cbar.ax.set_ylim(vmin, vmax)

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/{filename}")

    # Close fig
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocmlm !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocmlm(rocmlm, skip=1, figwidth=6.3, figheight=4.725, fontsize=22):
    """
    """
    # Get ml model attributes
    model_label_full = rocmlm.ml_model_label_full
    model_label = rocmlm.ml_model_label
    model_prefix = rocmlm.model_prefix
    sids = rocmlm.sids
    feature_arrays = rocmlm.feature_square
    target_arrays = rocmlm.target_square
    pred_arrays = rocmlm.prediction_square
    cv_info = rocmlm.cv_info
    targets = rocmlm.targets
    fig_dir = rocmlm.fig_dir
    palette = rocmlm.palette
    n_feats = feature_arrays.shape[-1] - 2
    n_models = feature_arrays.shape[0]
    w = feature_arrays.shape[1]
    verbose = rocmlm.verbose

    # Don't skip benchmark samples
    if any(sample in sids for sample in ["PUM", "DMM", "PYR"]):
        skip = 1
    else:
        # Need to skip double for synthetic samples bc X_res training starts at 2 ...
        skip = skip * 2

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

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    # Check for existing plots
    existing_figs = []
    for target in targets_rename:
        for s in sids[::skip]:
            fig_1 = f"{fig_dir}/prem-{s}-{model_label}-{target}.png"
            fig_2 = f"{fig_dir}/surf-{s}-{model_label}-{target}.png"
            fig_3 = f"{fig_dir}/image-{s}-{model_label}-{target}.png"

            check = (os.path.exists(fig_1) and os.path.exists(fig_2) and
                     os.path.exists(fig_3))

            if check:
                existing_figs.append(check)

    if existing_figs:
        return None

    for s, sid in enumerate(sids[::skip]):
        if verbose >= 1:
            print(f"Visualizing {model_prefix}-{sid} ...")

        # Slice arrays
        feature_array = feature_arrays[s, :, :, :]
        target_array = target_arrays[s, :, :, :]
        pred_array = pred_arrays[s, :, :, :]

        # Plotting variables
        units = []
        units_labels = []
        vmin = []
        vmax = []

        # Get units and colorbar limits
        for i, target in enumerate(targets):
            # Units
            if target == "rho":
                units.append("g/cm$^3$")

            elif target in ["Vp", "Vs"]:
                units.append("km/s")

            elif target == "melt":
                units.append("%")

            else:
                units.append("")

            # Get min max of target array
            vmin.append(np.nanmin(target_array[:, :, i]))
            vmax.append(np.nanmax(target_array[:, :, i]))

        units_labels = [f"({unit})" if unit is not None else "" for unit in units]

        # Colormap
        cmap = plt.cm.get_cmap("bone_r")
        cmap.set_bad(color="white")

        for i, target in enumerate(targets):
            # Rename target
            target_rename = target.replace("_", "-")

            # Create nan mask for targets
            mask = np.isnan(target_array[:, :, i])

            # Match nans between predictions and original targets
            pred_array[:, :, i][mask] = np.nan

            # Compute normalized diff
            diff = target_array[: ,: , i] - pred_array[: , :, i]

            # Get relevant metrics for target array plot
            rmse = cv_info[f"rmse_val_mean_{target}"][0]
            r2 = cv_info[f"r2_val_mean_{target}"][0]

            # Make nans consistent
            diff[mask] = np.nan

            # Plot training data distribution and ML model predictions
            colormap = plt.cm.get_cmap("tab10")

            # Reverse color scale
            if palette in ["grey"]:
                color_reverse = False

            else:
                color_reverse = True

            # Filename
            filename = f"{model_prefix}-{sid}-{target_rename}"

            # Plot target array 2d
            P = feature_array[:, :, 0 + n_feats]
            T = feature_array[:, :, 1 + n_feats]
            t = target_array[:, :, i]
            p = pred_array[:, :, i]

            visualize_target_array(P.flatten(), T.flatten(), t, target,
                                   palette, False, color_reverse, vmin[i], vmax[i], None,
                                   None, fig_dir, f"{filename}-targets.png")
            # Plot target array 3d
            visualize_target_surf(P, T, t, target, palette, False,
                                  color_reverse, vmin[i], vmax[i], fig_dir,
                                  f"{filename}-targets-surf.png")

            # Plot ML model predictions array 2d
            visualize_target_array(P.flatten(), T.flatten(), p, target, model_label_full,
                                   palette, False, color_reverse, vmin[i], vmax[i], None,
                                   None, fig_dir, f"{filename}-predictions.png", False)

            # Plot ML model predictions array 3d
            visualize_target_surf(P, T, p, target, model_label_full, palette, False,
                                  color_reverse, vmin[i], vmax[i], fig_dir,
                                  f"{filename}-surf.png")

            # Plot PT normalized diff targets vs. ML model predictions 2d
            visualize_target_array(P.flatten(), T.flatten(), diff, target, "Residuals",
                                   "seismic", False, False, vmin[i], vmax[i], rmse, r2,
                                   fig_dir, f"{filename}-diff.png", False)

            # Plot PT normalized diff targets vs. ML model predictions 3d
            visualize_target_surf(P, T, diff, target, "Residuals", "seismic", False, False,
                                  vmin[i], vmax[i], fig_dir, f"{filename}-diff-surf.png")

            # Reshape results and transform units for Perple_X
            results_gfem = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                           target: t.flatten().tolist()}

            # Reshape results and transform units for ML model
            results_rocmlm = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                              target: p.flatten().tolist()}

            # Set geotherm threshold for extracting depth profiles
            res = w - 1

            # Plot PREM comparisons
            if target == "rho":
                visualize_prem(sid, "train", res, target, "g/cm$^3$",
                               ["low", "mid", "high"], results_gfem,
                               results_rocmlm, model_label, title="Depth Profile",
                               fig_dir=fig_dir, filename=f"{filename}-prem.png")

            if target in ["Vp", "Vs"]:
                visualize_prem(sid, "train", res, target, "km/s",
                               ["low", "mid", "high"], results_gfem,
                               results_rocmlm, model_label, title="Depth Profile",
                               fig_dir=fig_dir, filename=f"{filename}-prem.png")
    return None