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
from gfem import GFEMModel, get_geotherm, get_1d_reference_models

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
# compose dataset plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_dataset_plots(gfem_models):
    """
    """
    # Parse and sort models
    magemin_models = [m if m.program == "magemin" and m.dataset == "valid" else
                      None for m in gfem_models]
    magemin_models = sorted(magemin_models, key=lambda m: (m.sample_id if m else ""))

    perplex_models = [m if m.program == "perplex" and m.dataset == "valid" else
                      None for m in gfem_models]
    perplex_models = sorted(perplex_models, key=lambda m: (m.sample_id if m else ""))

    # Iterate through all models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        magemin = True if magemin_model is not None else False
        perplex = True if perplex_model is not None else False

        if not magemin and not perplex:
            continue

        if magemin and perplex:
            # Get model data
            if magemin_model.sample_id == perplex_model.sample_id:
                sample_id = magemin_model.sample_id
            else:
                raise ValueError("Model samples are not the same!")
            if magemin_model.res == perplex_model.res:
                res = magemin_model.res
            else:
                raise ValueError("Model resolutions are not the same!")
            if magemin_model.dataset == perplex_model.dataset:
                dataset = magemin_model.dataset
            else:
                raise ValueError("Model datasets are not the same!")
            if magemin_model.targets == perplex_model.targets:
                targets = magemin_model.targets
            else:
                raise ValueError("Model targets are not the same!")
            if magemin_model.model_prefix == perplex_model.model_prefix:
                model_prefix = magemin_model.model_prefix
            else:
                raise ValueError("Model prefix are not the same!")
            if magemin_model.verbose == perplex_model.verbose:
                verbose = magemin_model.verbose
            else:
                verbose = 1

            program = "magemin + perplex"
            fig_dir_mage = magemin_model.fig_dir
            fig_dir_perp = perplex_model.fig_dir
            fig_dir_diff = f"figs/gfem/diff_{sample_id}_{res}"

            fig_dir = f"figs/gfem/{sample_id}_{res}"
            os.makedirs(fig_dir, exist_ok=True)

        elif magemin and not perplex:
            # Get model data
            program = "magemin"
            sample_id = magemin_model.sample_id
            res = magemin_model.res
            dataset = magemin_model.dataset
            targets = magemin_model.targets
            fig_dir = magemin_model.fig_dir
            model_prefix = magemin_model.model_prefix
            verbose = magemin_model.verbose

        elif perplex and not magemin:
            # Get model data
            program = "perplex"
            sample_id = perplex_model.sample_id
            res = perplex_model.res
            dataset = perplex_model.dataset
            targets = perplex_model.targets
            fig_dir = perplex_model.fig_dir
            model_prefix = perplex_model.model_prefix
            verbose = perplex_model.verbose

        # Rename targets
        targets_rename = [target.replace("_", "-") for target in targets]

        if verbose >= 1:
            print(f"Composing {model_prefix} [{program}] ...")

        # Check for existing plots
        existing_figs = []
        for target in targets_rename:
            fig_1 = f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png"
            fig_2 = f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png"
            fig_3 = f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png"
            fig_4 = f"{fig_dir}/image9-{sample_id}-{dataset}.png"

            check = ((os.path.exists(fig_3) and os.path.exists(fig_4)) |
                     (os.path.exists(fig_1) and os.path.exists(fig_2)) |
                     (os.path.exists(fig_1) and os.path.exists(fig_2) and
                      os.path.exists(fig_4)))

            if check:
                existing_figs.append(check)

        if existing_figs:
            return None

        if magemin and perplex:
            for target in targets_rename:
                if target not in ["assemblage", "variance"]:
                    combine_plots_horizontally(
                        f"{fig_dir_mage}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir_perp}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir_diff}/diff-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                        caption1="",
                        caption2="c)"
                    )

                if target in ["rho", "Vp", "Vs"]:
                    combine_plots_horizontally(
                        f"{fig_dir_mage}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir_perp}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    combine_plots_horizontally(
                        f"{fig_dir_diff}/diff-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir_diff}/prem-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp2.png",
                        caption1="c)",
                        caption2="d)"
                    )

                    combine_plots_vertically(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/temp2.png",
                        f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png",
                        caption1="",
                        caption2=""
                    )

                for dir in [fig_dir_mage, fig_dir_perp, fig_dir_diff]:
                    shutil.rmtree(dir)

        elif magemin and not perplex:
            for target in targets_rename:
                if target not in ["assemblage", "variance"]:
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/grad-magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    if target in ["rho", "Vp", "Vs"]:
                        combine_plots_horizontally(
                            f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/grad-magemin-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/temp1.png",
                            caption1="a)",
                            caption2="b)"
                        )

                        combine_plots_horizontally(
                            f"{fig_dir}/temp1.png",
                            f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                            caption1="",
                            caption2="c)"
                        )

        elif perplex and not magemin:
            for target in targets_rename:
                if target not in ["assemblage", "variance"]:
                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/grad-perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    if target in ["rho", "Vp", "Vs"]:
                        combine_plots_horizontally(
                            f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/grad-perplex-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/temp1.png",
                            caption1="a)",
                            caption2="b)"
                        )

                        combine_plots_horizontally(
                            f"{fig_dir}/temp1.png",
                            f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                            f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                            caption1="",
                            caption2="c)"
                        )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/grad-perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
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
                    f"{fig_dir}/image9-{sample_id}-{dataset}.png",
                    caption1="",
                    caption2=""
                )

        # Clean up directory
        tmp_files = glob.glob(f"{fig_dir}/temp*.png")
        prem_files = glob.glob(f"{fig_dir}/prem*.png")
        grad_files = glob.glob(f"{fig_dir}/grad*.png")
        diff_files = glob.glob(f"{fig_dir}/diff*.png")
        mgm_files = glob.glob(f"{fig_dir}/mage*.png")
        ppx_files = glob.glob(f"{fig_dir}/perp*.png")

        for file in tmp_files + prem_files + grad_files + mgm_files + ppx_files:
            os.remove(file)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create dataset movies !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_dataset_movies(gfem_models):
    """
    """
    # Parse and sort models
    magemin_models = [m if m.program == "magemin" and m.dataset == "valid" else
                      None for m in gfem_models]
    magemin_models = sorted(magemin_models, key=lambda m: (m.sample_id if m else ""))

    perplex_models = [m if m.program == "perplex" and m.dataset == "valid" else
                      None for m in gfem_models]
    perplex_models = sorted(perplex_models, key=lambda m: (m.sample_id if m else ""))

    # Iterate through all models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        magemin = True if magemin_model is not None else False
        perplex = True if perplex_model is not None else False

        if not magemin and not perplex:
            continue

        if magemin and perplex:
            # Get model data
            if magemin_model.sample_id == perplex_model.sample_id:
                sample_id = magemin_model.sample_id
            else:
                raise ValueError("Model samples are not the same!")
            if magemin_model.res == perplex_model.res:
                res = magemin_model.res
            else:
                raise ValueError("Model resolutions are not the same!")
            if magemin_model.dataset == perplex_model.dataset:
                dataset = magemin_model.dataset
            else:
                raise ValueError("Model datasets are not the same!")
            if magemin_model.targets == perplex_model.targets:
                targets = magemin_model.targets
            else:
                raise ValueError("Model targets are not the same!")
            if magemin_model.model_prefix == perplex_model.model_prefix:
                model_prefix = magemin_model.model_prefix
            else:
                raise ValueError("Model prefix are not the same!")
            if magemin_model.verbose == perplex_model.verbose:
                verbose = magemin_model.verbose
            else:
                verbose = 1

            program = "magemin + perplex"
            fig_dir_mage = magemin_model.fig_dir
            fig_dir_perp = perplex_model.fig_dir
            fig_dir_diff = f"figs/gfem/diff_{sample_id}_{res}"

            fig_dir = f"figs/gfem/{sample_id}_{res}"
            os.makedirs(fig_dir, exist_ok=True)

        elif magemin and not perplex:
            # Get model data
            program = "magemin"
            sample_id = magemin_model.sample_id
            res = magemin_model.res
            dataset = magemin_model.dataset
            targets = magemin_model.targets
            fig_dir = magemin_model.fig_dir
            model_prefix = magemin_model.model_prefix
            verbose = magemin_model.verbose

        elif perplex and not magemin:
            # Get model data
            program = "perplex"
            sample_id = perplex_model.sample_id
            res = perplex_model.res
            dataset = perplex_model.dataset
            targets = perplex_model.targets
            fig_dir = perplex_model.fig_dir
            model_prefix = perplex_model.model_prefix
            verbose = perplex_model.verbose

        # Rename targets
        targets_rename = [target.replace("_", "-") for target in targets]

        # Check for existing movies
        if sample_id not in ["PUM", "DMM", "PYR"]:
            if "sb" in sample_id:
                pattern = "sb???"
                prefix = "sb"
            if "sm" in sample_id:
                pattern = "sm???"
                prefix = "sm"
            if "st" in sample_id:
                pattern = "st???"
                prefix = "st"
            if "sr" in sample_id:
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
                print(f"Creating movie for {prefix} samples [{program}] ...")

                if not os.path.exists("figs/movies"):
                    os.makedirs("figs/movies", exist_ok=True)

                for target in targets_rename:
                    if perplex and magemin:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{pattern}_{res}/image2-{pattern}-{dataset}-"
                                  f"{target}.png' -vf 'scale=3915:1432' -c:v h264 -pix_fmt "
                                  f"yuv420p 'figs/movies/image2-{prefix}-{target}.mp4'")
                    else:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{program[:4]}_{pattern}_{res}/image2-"
                                  f"{pattern}-{dataset}-{target}.png' -vf 'scale=3915:1432' "
                                  f"-c:v h264 -pix_fmt yuv420p 'figs/movies/image2-{prefix}-"
                                  f"{target}.mp4'")

                    try:
                        subprocess.run(ffmpeg, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, shell=True)

                    except subprocess.CalledProcessError as e:
                        print(f"Error running FFmpeg command: {e}")

                if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                    for target in ["rho", "Vp", "Vs"]:
                        if perplex and magemin:
                            ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                      f"'figs/gfem/{pattern}_{res}/image3-{pattern}-"
                                      f"{dataset}-{target}.png' -vf 'scale=5832:1432' -c:v "
                                      f"h264 -pix_fmt yuv420p 'figs/movies/image3-{prefix}-"
                                      f"{target}.mp4'")
                        else:
                            ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                      f"'figs/gfem/{program[:4]}_{pattern}_{res}/image3-"
                                      f"{pattern}-{dataset}-{target}.png' -vf 'scale=5832:"
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

                    if perplex and magemin:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{pattern}_{res}/image9-{pattern}-{dataset}."
                                  f"png' -vf 'scale=5842:4296' -c:v h264 -pix_fmt yuv420p "
                                  f"'figs/movies/image9-{prefix}.mp4'")
                    else:
                        ffmpeg = (f"ffmpeg -framerate 15 -pattern_type glob -i "
                                  f"'figs/gfem/{program[:4]}_{pattern}_{res}/image9-"
                                  f"{pattern}-{dataset}.png' -vf 'scale=5842:4296' -c:v "
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
    program = rocmlm.program
    model_prefix = rocmlm.model_prefix
    ml_model_label = rocmlm.ml_model_label
    sample_ids = rocmlm.sample_ids
    res = rocmlm.res
    targets = rocmlm.targets
    fig_dir = rocmlm.fig_dir
    fig_dir_perf = "figs/rocmlm"
    verbose = rocmlm.verbose

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    # Don't skip benchmark samples
    if any(sample in sample_ids for sample in ["PUM", "DMM", "PYR"]):
        skip = 1
    else:
        # Need to skip double for synthetic samples bc X_res training starts at 2 ...
        skip = skip * 2

    # Check for existing plots
    existing_figs = []
    for target in targets_rename:
        for sample_id in rocmlm.sample_ids[::skip]:
            fig_1 = f"{fig_dir}/prem-{sample_id}-{ml_model_label}-{target}.png"
            fig_2 = f"{fig_dir}/surf-{sample_id}-{ml_model_label}-{target}.png"
            fig_3 = f"{fig_dir}/surf9-{sample_id}-{ml_model_label}.png"
            fig_4 = f"{fig_dir}/image-{sample_id}-{ml_model_label}-{target}.png"
            fig_5 = f"{fig_dir}/image9-{sample_id}-{ml_model_label}-profile.png"
            fig_6 = f"{fig_dir}/image9-{sample_id}-{ml_model_label}-diff.png"
            fig_7 = f"{fig_dir}/image12-{sample_id}-{ml_model_label}.png"

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
        for sample_id in rocmlm.sample_ids[::skip]:
            if verbose >= 1:
                print(f"Composing {model_prefix}-{sample_id}-{target} [{program}] ...")

            if target in ["rho", "Vp", "Vs"]:
                combine_plots_horizontally(
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                    f"{fig_dir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                combine_plots_horizontally(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/{model_prefix}-{sample_id}-{target}-prem.png",
                    f"{fig_dir}/prem-{sample_id}-{ml_model_label}-{target}.png",
                    caption1="",
                    caption2="c)"
                )

            combine_plots_horizontally(
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets-surf.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-surf.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-diff-surf.png",
                f"{fig_dir}/surf-{sample_id}-{ml_model_label}-{target}.png",
                caption1="",
                caption2="c)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            combine_plots_horizontally(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/{model_prefix}-{sample_id}-{target}-diff.png",
                f"{fig_dir}/image-{sample_id}-{ml_model_label}-{target}.png",
                caption1="",
                caption2="c)"
            )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-prem.png",
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
                    f"{fig_dir}/image9-{sample_id}-{ml_model_label}-profile.png",
                    caption1="",
                    caption2=""
                )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-diff.png",
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
                    f"{fig_dir}/image9-{sample_id}-{ml_model_label}-diff.png",
                    caption1="",
                    caption2=""
                )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "d)", "g)", "j)"), ("b)", "e)", "h)", "k)"),
                            ("c)", "f)", "i)", "l)")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_vertically(
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-predictions.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_vertically(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-diff.png",
                        f"{fig_dir}/temp2.png",
                        caption1="",
                        caption2=captions[i][2]
                    )

                    combine_plots_vertically(
                        f"{fig_dir}/temp2.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-prem.png",
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
                    f"{fig_dir}/image12-{sample_id}-{ml_model_label}.png",
                    caption1="",
                    caption2=""
                )

            if all(item in targets_rename for item in ["rho", "Vp", "Vs"]):
                captions = [("a)", "b)", "c)"), ("d)", "e)", "f)"), ("g)", "h)", "i)")]
                targets = ["rho", "Vp", "Vs"]

                for i, target in enumerate(targets):
                    combine_plots_horizontally(
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-targets-surf.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-surf.png",
                        f"{fig_dir}/temp1.png",
                        caption1=captions[i][0],
                        caption2=captions[i][1]
                    )

                    combine_plots_horizontally(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/{model_prefix}-{sample_id}-{target}-diff-surf.png",
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
                    f"{fig_dir}/surf9-{sample_id}-{ml_model_label}.png",
                    caption1="",
                    caption2=""
                )

    # Clean up directory
    rocmlm_files = glob.glob(f"{fig_dir}/rocmlm*.png")
    tmp_files = glob.glob(f"{fig_dir}/temp*.png")
    program_files = glob.glob(f"{fig_dir}/{program[:4]}*.png")

    for file in rocmlm_files + tmp_files + program_files:
        os.remove(file)

    return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .1.2          Plotting Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem pt range !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem_pt_range(gfem_model, fig_dir="figs/other", T_mantle1=673, T_mantle2=1773,
                            grad_mantle1=0.5, grad_mantle2=0.5, fontsize=12, figwidth=6.3,
                            figheight=3.3):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Get gfem model data
    res = gfem_model.res
    targets = gfem_model.targets
    results = gfem_model.results
    P_gt, T_gt, rho_gt = results["P"], results["T"], results["rho"]

    # Get min/max PT
    P_min, P_max, T_min, T_max = np.min(P_gt), np.max(P_gt), np.min(T_gt), np.max(T_gt)

    # T range
    T = np.arange(0, T_max + 728)

    # Olivine --> Ringwoodite Clapeyron slopes
    references_410 = {"ol$\\rightarrow$wad (Akaogi89)": [0.001, 0.002],
                      "ol$\\rightarrow$wad (Katsura89)": [0.0025],
                      "ol$\\rightarrow$wad (Morishima94)": [0.0034, 0.0038]}

    # Ringwoodite --> Bridgmanite + Ferropericlase Clapeyron slopes
    references_670 = {"ring$\\rightarrow$brg (Ito82)": [-0.002],
                      "ring$\\rightarrow$brg (Ito89 & Hirose02)": [-0.0028],
                      "ring$\\rightarrow$brg (Ito90)": [-0.002, -0.006],
                      "ring$\\rightarrow$brg (Katsura03)": [-0.0004, -0.002],
                      "ring$\\rightarrow$brg (Akaogi07)": [-0.0024, -0.0028]}

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Legend colors
    colormap = plt.cm.get_cmap("tab10")
    colors = [colormap(i) for i in range(9)]

    # Calculate phase boundaries:
    # Olivine --> Ringwoodite
    lines_410 = []
    labels_410 = set()

    for i, (ref, c_values) in enumerate(references_410.items()):
        ref_lines = []

        for j, c in enumerate(c_values):
            P = (T - 1758) * c + 13.4

            ref_lines.append(P)

            label = f"{ref}"
            labels_410.add(label)

        lines_410.append(ref_lines)

    # Ringwoodite --> Bridgmanite + Ferropericlase
    lines_670 = []
    labels_670 = set()

    for i, (ref, c_values) in enumerate(references_670.items()):
        ref_lines = []

        for j, c in enumerate(c_values):
            P = (T - 1883) * c + 23.0

            ref_lines.append(P)

            label = f"{ref}"
            labels_670.add(label)

        lines_670.append(ref_lines)

    # Plotting
    plt.figure()

    # Map labels to colors
    label_color_mapping = {}

    # Olivine --> Ringwoodite
    for i, (ref, ref_lines) in enumerate(zip(references_410.keys(), lines_410)):
        color = colors[i % len(colors)]

        for j, line in enumerate(ref_lines):
            label = f"{ref}" if j == 0 else None

            plt.plot(T[(T >= 1200) & (T <= 2000)], line[(T >= 1200) & (T <= 2000)],
                     color=color, label=label)

            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Ringwoodite --> Bridgmanite + Ferropericlase
    for j, (ref, ref_lines) in enumerate(zip(references_670.keys(), lines_670)):
        color = colors[j + i + 1 % len(colors)]

        for j, line in enumerate(ref_lines):
            label = f"{ref}" if j == 0 else None

            plt.plot(T[(T >= 1250) & (T <= 2100)], line[(T >= 1250) & (T <= 2100)],
                     color=color, label=label)

            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Plot shaded rectangle for PT range of training dataset
    fill = plt.fill_between(T, P_min, P_max, where=(T >= T_min) & (T <= T_max),
                            hatch="++", facecolor="none", alpha=0.1)

    # Calculate mantle geotherms
    geotherm1 = (T - T_mantle1) / (grad_mantle1 * 35)
    geotherm2 = (T - T_mantle2) / (grad_mantle2 * 35)

    # Find boundaries
    T1_Pmax = (P_max * grad_mantle1 * 35) + T_mantle1
    P1_Tmin = (T_min - T_mantle1) / (grad_mantle1 * 35)
    T2_Pmin = (P_min * grad_mantle2 * 35) + T_mantle2
    T2_Pmax = (P_max * grad_mantle2 * 35) + T_mantle2

    # Crop geotherms
    geotherm1_cropped = geotherm1[geotherm1 >= P1_Tmin]
    geotherm1_cropped = geotherm1_cropped[geotherm1_cropped <= P_max]
    geotherm2_cropped = geotherm2[geotherm2 >= P_min]
    geotherm2_cropped = geotherm2_cropped[geotherm2_cropped <= P_max]

    # Crop T vectors
    T_cropped_geotherm1= T[T >= T_min]
    T_cropped_geotherm1 = T_cropped_geotherm1[T_cropped_geotherm1 <= T1_Pmax]
    T_cropped_geotherm2= T[T >= T2_Pmin]
    T_cropped_geotherm2 = T_cropped_geotherm2[T_cropped_geotherm2 <= T2_Pmax]

    if res <= 8:
        geotherm_threshold = 40
    elif res <= 16:
        geotherm_threshold = 20
    elif res <= 32:
        geotherm_threshold = 10
    elif res <= 64:
        geotherm_threshold = 5
    elif res <= 128:
        geotherm_threshold = 2.5
    else:
        geotherm_threshold = 1.25

    # Get geotherm (non-adiabatic)
    results = pd.DataFrame({"P": P_gt, "T": T_gt, "rho": rho_gt})
    P_geotherm, T_geotherm, _ = get_geotherm(results, "rho", geotherm_threshold,
                                             Qs=250e-3, A1=2.2e-8, k1=3.0,
                                             litho_thickness=1, mantle_potential=1573)

    # Plot mantle geotherms
    plt.plot(T_cropped_geotherm1, geotherm1_cropped, ":", color="black")
    plt.plot(T_cropped_geotherm2, geotherm2_cropped, ":", color="black")
    plt.plot(T_geotherm, P_geotherm, linestyle="--", color="black")

    # Interpolate the geotherms to have the same length as temperature vectors
    geotherm1_interp = np.interp(T_cropped_geotherm1, T, geotherm1)
    geotherm2_interp = np.interp(T_cropped_geotherm2, T, geotherm2)

    # Define the vertices for the polygon
    vertices = np.vstack(
        (
            np.vstack((T_cropped_geotherm1, geotherm1_interp)).T,
            (T_cropped_geotherm2[-1], geotherm2_interp[-1]),
            np.vstack((T_cropped_geotherm2[::-1], geotherm2_interp[::-1])).T,
            np.array([T_min, P_min]),
            (T_cropped_geotherm1[0], geotherm1_interp[0])
        )
    )

    # Fill the area within the polygon
    plt.fill(vertices[:, 0], vertices[:, 1], facecolor="white", edgecolor=None, alpha=1)
    plt.fill(vertices[:, 0], vertices[:, 1], facecolor="none", edgecolor="whitesmoke",
             alpha=1, hatch="++")
    plt.fill(vertices[:, 0], vertices[:, 1], facecolor="white", edgecolor=None, alpha=0.2)
    plt.fill_between(T, P_min, P_max, where=(T >= T_min) & (T <= T_max),
                     facecolor="none", edgecolor="black", linewidth=1.5, alpha=1)

    # Geotherm legend handles
    geotherm_handle = mlines.Line2D([], [], linestyle="--", color="black",
                                    label="Avg. Mid-Ocean Ridge")

    # Phase boundaries legend handles
    ref_line_handles = [
        mlines.Line2D([], [], color=color, label=label)
        for label, color in label_color_mapping.items() if label
    ]

    # Add geotherms to legend handles
    ref_line_handles.extend([geotherm_handle])

    db_data_handle = mpatches.Patch(facecolor="white", edgecolor="black", alpha=0.8,
                                    hatch="++", label="RocMLM Training Data")

    labels_670.add("RocMLM Training Data")
    label_color_mapping["RocMLM Training Data"] = "black"

    training_data_handle = mpatches.Patch(facecolor="white", edgecolor="black", alpha=1,
                                          linestyle=":", label="Hypothetical Mantle PTs")

    labels_670.add("Hypothetical Mantle PTs")
    label_color_mapping["Hypothetical Mantle PTs"] = "gray"

    # Define the desired order of the legend items
    desired_order = ["RocMLM Training Data",
                     "Hypothetical Mantle PTs",
                     "Avg. Mid-Ocean Ridge",
                     "ol$\\rightarrow$wad (Akaogi89)",
                     "ol$\\rightarrow$wad (Katsura89)",
                     "ol$\\rightarrow$wad (Morishima94)",
                     "ring$\\rightarrow$brg (Ito82)",
                     "ring$\\rightarrow$brg (Ito89 & Hirose02)",
                     "ring$\\rightarrow$brg (Ito90)",
                     "ring$\\rightarrow$brg (Katsura03)",
                     "ring$\\rightarrow$brg (Akaogi07)"]

    # Sort the legend handles based on the desired order
    legend_handles = sorted(ref_line_handles + [db_data_handle, training_data_handle],
                            key=lambda x: desired_order.index(x.get_label()))

    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (GPa)")
    plt.xlim(T_min - 100, T_max + 100)
    plt.ylim(P_min - 1, P_max + 1)

    # Move the legend outside the plot to the right
    plt.legend(title="", handles=legend_handles, loc="center left", handleheight=1.2,
               bbox_to_anchor=(1.02, 0.5))

    # Adjust the figure size
    fig = plt.gcf()
    fig.set_size_inches(figwidth, figheight)

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/training-dataset-design.png")

    # Close device
    plt.close()

    return None

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
    data_rocmlm["dataset"] = "train"

    # Calculate efficiency in milliseconds/Megabyte
    data_rocmlm["model_efficiency"] = (data_rocmlm["time"] * 1e3 *
                                       data_rocmlm["model_size_mb"])

    # Select columns
    data_rocmlm = data_rocmlm[["sample", "program", "dataset", "size", "time",
                               "model_size_mb", "model_efficiency", "rmse_val_mean_rho",
                               "rmse_val_mean_Vp", "rmse_val_mean_Vs"]]

    # Combine data
    data = pd.concat([data_lut, data_rocmlm], axis=0, ignore_index=True)

    # Relabel programs
    def label_programs(row):
        if row["program"] == "magemin":
            return "MAGEMin"
        elif row["program"] == "perplex":
            return "Perple_X"
        elif row["program"] == "lut":
            return "Lookup Table"
        else:
            return row["program"]

    data["program"] = data.apply(label_programs, axis=1)

    # Filter out validation dataset
    data = data[data["dataset"] == "train"]

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
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

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
# visualize prem !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_prem(program, sample_id, dataset, res, target, target_unit,
                   geotherms=["low", "mid", "high"], results_mgm=None, results_ppx=None,
                   results_ml=None, model=None, title=None, fig_dir="figs", filename=None,
                   figwidth=6.3, figheight=4.725, fontsize=22):
    """
    """
    # Data asset dir
    data_dir = "assets/data"

    # Check for data dir
    if not os.path.exists(data_dir):
        raise Exception(f"Data not found at {data_dir}!")

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Check for average geotherm
    if "mid" not in geotherms:
        geotherms = geotherms + ["mid"]

    # Get 1D reference models
    ref_models = get_1d_reference_models()

    # Get 1D refernce model profiles
    P_prem, target_prem = ref_models["prem"]["P"], ref_models["prem"][target]
    P_stw105, target_stw105 = ref_models["stw105"]["P"], ref_models["stw105"][target]

    # Initialize geotherms
    P_mgm, P_ppx, P_ml, P_pyr = None, None, None, None
    P_mgm2, P_ppx2, P_ml2, P_pyr2 = None, None, None, None
    P_mgm3, P_ppx3, P_ml3, P_pyr3 = None, None, None, None
    target_mgm, target_ppx, target_ml, target_pyr = None, None, None, None
    target_mgm2, target_ppx2, target_ml2, target_pyr2 = None, None, None, None
    target_mgm3, target_ppx3, target_ml3, target_pyr3 = None, None, None, None

    # Get benchmark models
    pyr_path = f"gfems/{program[:4]}_PYR_{dataset[0]}{res}/results.csv"
    source = "assets/data/benchmark-samples-pca.csv"
    targets = ["rho", "Vp", "Vs"]

    if os.path.exists(pyr_path) and sample_id != "PYR":
        pyr_model = GFEMModel(program, dataset, "PYR", source, res, 1, 28, 773, 2273, "all",
                              targets, False, 0, False)
        results_pyr = pyr_model.results
    else:
        results_pyr = None

    # Set geotherm threshold for extracting depth profiles
    if res <= 8:
        geotherm_threshold = 40
    elif res <= 16:
        geotherm_threshold = 20
    elif res <= 32:
        geotherm_threshold = 10
    elif res <= 64:
        geotherm_threshold = 5
    elif res <= 128:
        geotherm_threshold = 2.5
    else:
        geotherm_threshold = 1.25

    # Extract target values along a geotherm
    if results_mgm:
        if "low" in geotherms:
            P_mgm, _, target_mgm = get_geotherm(results_mgm, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1173)
        if "mid" in geotherms:
            P_mgm2, _, target_mgm2 = get_geotherm(results_mgm, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1573)
        if "high" in geotherms:
            P_mgm3, _, target_mgm3 = get_geotherm(results_mgm, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1773)
    if results_ppx:
        if "low" in geotherms:
            P_ppx, _, target_ppx = get_geotherm(results_ppx, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1173)
        if "mid" in geotherms:
            P_ppx2, _, target_ppx2 = get_geotherm(results_ppx, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1573)
        if "high" in geotherms:
            P_ppx3, _, target_ppx3 = get_geotherm(results_ppx, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1773)
    if results_ml:
        if "low" in geotherms:
            P_ml, _, target_ml = get_geotherm(results_ml, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1173)
        if "mid" in geotherms:
            P_ml2, _, target_ml2 = get_geotherm(results_ml, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1573)
        if "high" in geotherms:
            P_ml3, _, target_ml3 = get_geotherm(results_ml, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1773)
    if results_pyr:
        if "low" in geotherms:
            P_pyr, _, target_pyr = get_geotherm(results_pyr, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1173)
        if "mid" in geotherms:
            P_pyr2, _, target_pyr2 = get_geotherm(results_pyr, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1573)
        if "high" in geotherms:
            P_pyr3, _, target_pyr3 = get_geotherm(results_pyr, target, geotherm_threshold,
                                                  Qs=250e-3, A1=2.2e-8, k1=3.0,
                                                  litho_thickness=1, mantle_potential=1773)

    # Get min and max P
    P_min = min(np.nanmin(P) for P in [P_mgm, P_mgm2, P_mgm3, P_ppx, P_ppx2, P_ppx3, P_ml,
                                       P_ml2, P_ml3, P_pyr, P_pyr2, P_pyr3] if P is not None)
    P_max = max(np.nanmax(P) for P in [P_mgm, P_mgm2, P_mgm3, P_ppx, P_ppx2, P_ppx3, P_ml,
                                       P_ml2, P_ml3, P_pyr, P_pyr2, P_pyr3] if P is not None)

    # Create cropping mask
    mask_prem = (P_prem >= P_min) & (P_prem <= P_max)
    mask_stw105 = (P_stw105 >= P_min) & (P_stw105 <= P_max)

    # Crop profiles
    P_prem, target_prem = P_prem[mask_prem], target_prem[mask_prem]
    P_stw105, target_stw105 = P_stw105[mask_stw105], target_stw105[mask_stw105]

    # Crop results
    if results_mgm:
        if "low" in geotherms:
            mask_mgm = (P_mgm >= P_min) & (P_mgm <= P_max)
            P_mgm, target_mgm = P_mgm[mask_mgm], target_mgm[mask_mgm]
        if "mid" in geotherms:
            mask_mgm2 = (P_mgm2 >= P_min) & (P_mgm2 <= P_max)
            P_mgm2, target_mgm2 = P_mgm2[mask_mgm2], target_mgm2[mask_mgm2]
        if "high" in geotherms:
            mask_mgm3 = (P_mgm3 >= P_min) & (P_mgm3 <= P_max)
            P_mgm3, target_mgm3 = P_mgm3[mask_mgm3], target_mgm3[mask_mgm3]
    if results_ppx:
        if "low" in geotherms:
            mask_ppx = (P_ppx >= P_min) & (P_ppx <= P_max)
            P_ppx, target_ppx = P_ppx[mask_ppx], target_ppx[mask_ppx]
        if "mid" in geotherms:
            mask_ppx2 = (P_ppx2 >= P_min) & (P_ppx2 <= P_max)
            P_ppx2, target_ppx2 = P_ppx2[mask_ppx2], target_ppx2[mask_ppx2]
        if "high" in geotherms:
            mask_ppx3 = (P_ppx3 >= P_min) & (P_ppx3 <= P_max)
            P_ppx3, target_ppx3 = P_ppx3[mask_ppx3], target_ppx3[mask_ppx3]
    if results_ml:
        if "low" in geotherms:
            mask_ml = (P_ml >= P_min) & (P_ml <= P_max)
            P_ml, target_ml = P_ml[mask_ml], target_ml[mask_ml]
        if "mid" in geotherms:
            mask_ml2 = (P_ml2 >= P_min) & (P_ml2 <= P_max)
            P_ml2, target_ml2 = P_ml2[mask_ml2], target_ml2[mask_ml2]
        if "high" in geotherms:
            mask_ml3 = (P_ml3 >= P_min) & (P_ml3 <= P_max)
            P_ml3, target_ml3 = P_ml3[mask_ml3], target_ml3[mask_ml3]
    if results_pyr:
        if "low" in geotherms:
            mask_pyr = (P_pyr >= P_min) & (P_pyr <= P_max)
            P_pyr, target_pyr = P_pyr[mask_pyr], target_pyr[mask_pyr]
        if "mid" in geotherms:
            mask_pyr2 = (P_pyr2 >= P_min) & (P_pyr2 <= P_max)
            P_pyr2, target_pyr2 = P_pyr2[mask_pyr2], target_pyr2[mask_pyr2]
        if "high" in geotherms:
            mask_pyr3 = (P_pyr3 >= P_min) & (P_pyr3 <= P_max)
            P_pyr3, target_pyr3 = P_pyr3[mask_pyr3], target_pyr3[mask_pyr3]

    # Initialize interpolators
    interp_prem = interp1d(P_prem, target_prem, fill_value="extrapolate")
    interp_stw105 = interp1d(P_stw105, target_stw105, fill_value="extrapolate")

    # New x values for interpolation
    if results_ppx:
        x_new = np.linspace(P_min, P_max, len(P_ppx2))
    if results_ml:
        x_new = np.linspace(P_min, P_max, len(P_ml2))

    # Interpolate profiles
    P_prem, target_prem = x_new, interp_prem(x_new)
    P_stw105, target_stw105 = x_new, interp_stw105(x_new)

    # Create nan and inf masks for interpolated profiles
    nan_mask_prem = np.isnan(target_prem)
    inf_mask_prem = np.isinf(target_prem)
    nan_mask_stw105 = np.isnan(target_stw105)
    inf_mask_stw105 = np.isinf(target_stw105)

    # Change endmember sampleids
    if sample_id == "sm000":
        sample_id = "DSUM"
    elif sample_id == "sm129":
        sample_id = "PSUM"

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Colormap
    colormap = plt.cm.get_cmap("tab10")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

    # Plot GFEM and RocMLM profiles
    if results_ppx and not results_ml:
        if "low" in geotherms:
            ax1.plot(target_ppx, P_ppx, "-", linewidth=3, color=colormap(0),
                     label=f"{sample_id}-1173")
            ax1.fill_betweenx(P_ppx, target_ppx * (1 - 0.05), target_ppx * (1 + 0.05),
                              color=colormap(0), alpha=0.2)
        if "mid" in geotherms:
            ax1.plot(target_ppx2, P_ppx2, "-", linewidth=3, color=colormap(2),
                     label=f"{sample_id}-1573")
            ax1.fill_betweenx(P_ppx2, target_ppx2 * (1 - 0.05), target_ppx2 * (1 + 0.05),
                              color=colormap(2), alpha=0.2)
        if "high" in geotherms:
            ax1.plot(target_ppx3, P_ppx3, "-", linewidth=3, color=colormap(1),
                     label=f"{sample_id}-1773")
            ax1.fill_betweenx(P_ppx3, target_ppx3 * (1 - 0.05), target_ppx3 * (1 + 0.05),
                              color=colormap(1), alpha=0.3)
    if results_ml:
        if "low" in geotherms:
            ax1.plot(target_ml, P_ml, "-", linewidth=3, color=colormap(0),
                     label=f"{model}-1173")
            ax1.fill_betweenx(P_ml, target_ml * (1 - 0.05), target_ml * (1 + 0.05),
                              color=colormap(0), alpha=0.2)
        if "mid" in geotherms:
            ax1.plot(target_ml2, P_ml2, "-", linewidth=3, color=colormap(2),
                     label=f"{model}-1573")
            ax1.fill_betweenx(P_ml2, target_ml2 * (1 - 0.05), target_ml2 * (1 + 0.05),
                              color=colormap(2), alpha=0.2)
        if "high" in geotherms:
            ax1.plot(target_ml3, P_ml3, "-", linewidth=3, color=colormap(1),
                     label=f"{model}-1773")
            ax1.fill_betweenx(P_ml3, target_ml3 * (1 - 0.05), target_ml3 * (1 + 0.05),
                              color=colormap(1), alpha=0.3)

    # Plot reference models
    ax1.plot(target_prem, P_prem, "-", linewidth=2, color="black")
    ax1.plot(target_stw105, P_stw105, ":", linewidth=2, color="black")

    if target == "rho":
        target_label = "Density"
    else:
        target_label = target

    ax1.set_xlabel(f"{target_label} ({target_unit})")
    ax1.set_ylabel("P (GPa)")

    if target in ["Vp", "Vs", "rho"]:
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    # Vertical text spacing
    text_margin_x = 0.04
    text_margin_y = 0.15
    text_spacing_y = 0.1

    # Compute metrics
    if results_mgm and not results_ml:
        nan_mask_mgm2 = np.isnan(target_mgm2)
        nan_mask = nan_mask_mgm2 | nan_mask_prem | inf_mask_prem
        P_mgm2, target_mgm2 = P_mgm2[~nan_mask], target_mgm2[~nan_mask]
        P_prem, target_prem = P_prem[~nan_mask], target_prem[~nan_mask]
        rmse = np.sqrt(mean_squared_error(target_prem, target_mgm2))
        r2 = r2_score(target_prem, target_mgm2)
    elif results_ppx and not results_ml:
        nan_mask_ppx2 = np.isnan(target_ppx2)
        nan_mask = nan_mask_ppx2 | nan_mask_prem | inf_mask_prem
        P_ppx2, target_ppx2 = P_ppx2[~nan_mask], target_ppx2[~nan_mask]
        P_prem, target_prem = P_prem[~nan_mask], target_prem[~nan_mask]
        rmse = np.sqrt(mean_squared_error(target_prem, target_ppx2))
        r2 = r2_score(target_prem, target_ppx2)
    elif results_ml:
        nan_mask_ml2 = np.isnan(target_ml2)
        nan_mask = nan_mask_ml2 | nan_mask_prem | inf_mask_prem
        P_ml2, target_ml2 = P_ml2[~nan_mask], target_ml2[~nan_mask]
        P_prem, target_prem = P_prem[~nan_mask], target_prem[~nan_mask]
        rmse = np.sqrt(mean_squared_error(target_prem, target_ml2))
        r2 = r2_score(target_prem, target_ml2)
    else:
        rmse, r2 = None, None

    # Add R-squared and RMSE values as text annotations in the plot
    plt.text(text_margin_x, 1 - (text_margin_y - (text_spacing_y * 0)), f"R$^2$: {r2:.3f}",
             transform=plt.gca().transAxes, fontsize=fontsize * 0.833,
             horizontalalignment="left", verticalalignment="top")
    plt.text(text_margin_x, 1 - (text_margin_y - (text_spacing_y * 1)), f"RMSE: {rmse:.3f}",
             transform=plt.gca().transAxes, fontsize=fontsize * 0.833,
             horizontalalignment="left", verticalalignment="top")

    # Convert the primary y-axis data (pressure) to depth
    depth_conversion = lambda P: P * 30
    depth_values = depth_conversion(P_prem)

    # Create the secondary y-axis and plot depth on it
    ax2 = ax1.secondary_yaxis("right", functions=(depth_conversion, depth_conversion))
    ax2.set_yticks([410, 670])
    ax2.set_ylabel("Depth (km)")

    plt.legend(loc="lower right", columnspacing=0, handletextpad=0.2,
               fontsize=fontsize * 0.833)

    if title:
        plt.title(title)

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/{filename}")

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize prem comps !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_prem_comps(gfem_models, fig_dir="figs/other", filename="prem-comps.png",
                         figwidth=6.3, figheight=5.8, fontsize=28):
    """
    """
    # Data asset dir
    data_dir = "assets/data"

    # Check for data dir
    if not os.path.exists(data_dir):
        raise Exception(f"Data not found at {data_dir}!")

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Get 1D reference models
    ref_models = get_1d_reference_models()

    # Get correct Depletion column
    D_col = "D_FRAC"

    # Check for benchmark samples
    df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"

    # Read benchmark samples
    if os.path.exists(df_synth_bench_path):
        df_synth_bench = pd.read_csv(df_synth_bench_path)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Colormap
    colormap = plt.cm.get_cmap("tab10")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(figwidth * 3, figheight))

    for j, model in enumerate([model for model in gfem_models if model.dataset == "train"]):
        # Get gfem model data
        res = model.res
        xi = model.fertility_index
        targets = model.targets
        results = model.results
        sample_id = model.sample_id

        # Set geotherm threshold for extracting depth profiles
        if res <= 8:
            geotherm_threshold = 40
        elif res <= 16:
            geotherm_threshold = 20
        elif res <= 32:
            geotherm_threshold = 10
        elif res <= 64:
            geotherm_threshold = 5
        elif res <= 128:
            geotherm_threshold = 2.5
        else:
            geotherm_threshold = 1.25

        # Change endmember sampleids
        if sample_id == "sm000":
            sample_id = "DSUM"
        elif sample_id == "sm129":
            sample_id = "PSUM"

        for i, target in enumerate(targets):
            if target == "rho":
                target_unit = "g/cm$^3$"
            else:
                target_unit = "km/s"

            # Get 1D refernce model profiles
            P_prem, target_prem = ref_models["prem"]["P"], ref_models["prem"][target]
            P_stw105, target_stw105 = ref_models["stw105"]["P"], ref_models["stw105"][target]

            # Extract target values along a geotherm
            P2, _, target2 = get_geotherm(
                results, target, geotherm_threshold, Qs=250e-3, A1=2.2e-8,
                k1=3.0, litho_thickness=1, mantle_potential=1573)

            # Get min and max P
            P_min = np.nanmin(P2)
            P_max = np.nanmax(P2)

            # Create cropping mask
            mask_prem = (P_prem >= P_min) & (P_prem <= P_max)
            mask_stw105 = (P_stw105 >= P_min) & (P_stw105 <= P_max)

            # Crop profiles
            P_prem, target_prem = P_prem[mask_prem], target_prem[mask_prem]
            P_stw105, target_stw105 = P_stw105[mask_stw105], target_stw105[mask_stw105]

            # Crop results
            mask2 = (P2 >= P_min) & (P2 <= P_max)
            P2, target2 = P2[mask2], target2[mask2]

            # Initialize interpolators
            interp_prem = interp1d(P_prem, target_prem, fill_value="extrapolate")
            interp_stw105 = interp1d(P_stw105, target_stw105, fill_value="extrapolate")

            # New x values for interpolation
            x_new = np.linspace(P_min, P_max, len(P2))

            # Interpolate profiles
            P_prem, target_prem = x_new, interp_prem(x_new)
            P_stw105, target_stw105 = x_new, interp_stw105(x_new)

            # Create colorbar
            pal = sns.color_palette("magma", as_cmap=True).reversed()
            norm = plt.Normalize(df_synth_bench[D_col].min(),
                                 df_synth_bench[D_col].max())
            sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
            sm.set_array([])

            ax = axes[i]

            if j == 0:
                # Plot reference models
                ax.plot(target_prem, P_prem, "-", linewidth=4.5, color="forestgreen",
                        label="PREM", zorder=7)
                ax.plot(target_stw105, P_stw105, ":", linewidth=4.5, color="forestgreen",
                        label="STW105", zorder=7)

            if sample_id == "PSUM":
                ax.plot(target2, P2, "-", linewidth=6.5, color=sm.to_rgba(xi),
                        label="PSUM", zorder=6)
            if sample_id == "DSUM":
                ax.plot(target2, P2, "-", linewidth=6.5, color=sm.to_rgba(xi),
                        label="DSUM", zorder=6)

            # Plot GFEM and RocMLM profiles
            ax.plot(target2, P2, "-", linewidth=1, color=sm.to_rgba(xi), alpha=0.1)

            if target == "rho":
                target_label = "Density"
            else:
                target_label = target

            if i != 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("P (GPa)")

            ax.set_xlabel(f"{target_label} ({target_unit})")

            if target in ["Vp", "Vs", "rho"]:
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

            # Convert the primary y-axis data (pressure) to depth
            depth_conversion = lambda P: P * 30
            depth_values = depth_conversion(P_prem)

            if i == 2:
                # Create the secondary y-axis and plot depth on it
                ax2 = ax.secondary_yaxis(
                    "right", functions=(depth_conversion, depth_conversion))
                ax2.set_yticks([410, 670])
                ax2.set_ylabel("Depth (km)")
                cbaxes = inset_axes(ax, width="40%", height="3%", loc=2)
                colorbar = plt.colorbar(sm, ax=ax, cax=cbaxes, label="$\\xi$",
                                        orientation="horizontal")

                ax.legend(loc="lower right", columnspacing=0, handletextpad=0.2,
                           fontsize=fontsize * 0.833)

    fig.text(0.00, 0.98, "a)", fontsize=fontsize * 1.4)
    fig.text(0.33, 0.98, "b)", fontsize=fontsize * 1.4)
    fig.text(0.65, 0.98, "c)", fontsize=fontsize * 1.4)

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/{filename}")

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize target array  !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_target_array(P, T, target_array, target, title, palette, color_discrete,
                           color_reverse, vmin, vmax, rmse, r2, fig_dir, filename,
                           plot_geotherms=True, figwidth=6.3, figheight=4.725, fontsize=22):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Set geotherm threshold for extracting depth profiles
    res = target_array.shape[0]
    if res <= 8:
        geotherm_threshold = 40
    elif res <= 16:
        geotherm_threshold = 20
    elif res <= 32:
        geotherm_threshold = 10
    elif res <= 64:
        geotherm_threshold = 5
    elif res <= 128:
        geotherm_threshold = 2.5
    else:
        geotherm_threshold = 1.25

    # Get geotherm
    results = pd.DataFrame({"P": P, "T": T, target: target_array.flatten()})
    P_geotherm, T_geotherm, _ = get_geotherm(results, target, geotherm_threshold,
                                             Qs=250e-3, A1=2.2e-8, k1=3.0,
                                             litho_thickness=1, mantle_potential=1173)
    P_geotherm2, T_geotherm2, _ = get_geotherm(results, target, geotherm_threshold,
                                               Qs=250e-3, A1=2.2e-8, k1=3.0,
                                               litho_thickness=1, mantle_potential=1573)
    P_geotherm3, T_geotherm3, _ = get_geotherm(results, target, geotherm_threshold,
                                               Qs=250e-3, A1=2.2e-8, k1=3.0,
                                               litho_thickness=1, mantle_potential=1773)

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

        # Plot as a raster using imshow
        fig, ax = plt.subplots(figsize=(figwidth, figheight))

        im = ax.imshow(target_array, extent=[np.nanmin(T), np.nanmax(T), np.nanmin(P),
                                             np.nanmax(P)],
                       aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        if plot_geotherms:
            ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white", linewidth=3)
            ax.plot(T_geotherm2, P_geotherm2, linestyle="--", color="white", linewidth=3)
            ax.plot(T_geotherm3, P_geotherm3, linestyle="-.", color="white", linewidth=3)
            plt.text(1173 + (14 * 0.5 * 35), 14, "1173", fontsize=fontsize * 0.833,
                     horizontalalignment="left", verticalalignment="bottom")
            plt.text(1163 + (6 * 0.5 * 35), 6, "1173", fontsize=fontsize * 0.833,
                     horizontalalignment="center", verticalalignment="bottom", rotation=67,
                     color="white")
            plt.text(1563 + (6 * 0.5 * 35), 6, "1573", fontsize=fontsize * 0.833,
                     horizontalalignment="center", verticalalignment="bottom", rotation=67,
                     color="white")
            plt.text(1763 + (6 * 0.5 * 35), 6, "1773", fontsize=fontsize * 0.833,
                     horizontalalignment="center", verticalalignment="bottom", rotation=67,
                     color="white")
        ax.set_xlabel("T (K)")
        ax.set_ylabel("P (GPa)")
        plt.colorbar(im, ax=ax, ticks=np.arange(vmin, vmax, num_colors // 4), label="")

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

        # Plot as a raster using imshow
        fig, ax = plt.subplots()

        im = ax.imshow(target_array, extent=[np.nanmin(T), np.nanmax(T), np.nanmin(P),
                                             np.nanmax(P)],
                       aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        if plot_geotherms:
            ax.plot(T_geotherm, P_geotherm, linestyle="-", color="white", linewidth=3)
            ax.plot(T_geotherm2, P_geotherm2, linestyle="--", color="white", linewidth=3)
            ax.plot(T_geotherm3, P_geotherm3, linestyle="-.", color="white", linewidth=3)
            plt.text(1163 + (6 * 0.5 * 35), 6, "1173", fontsize=fontsize * 0.833,
                     horizontalalignment="center", verticalalignment="bottom", rotation=67,
                     color="white")
            plt.text(1563 + (6 * 0.5 * 35), 6, "1573", fontsize=fontsize * 0.833,
                     horizontalalignment="center", verticalalignment="bottom", rotation=67,
                     color="white")
            plt.text(1763 + (6 * 0.5 * 35), 6, "1773", fontsize=fontsize * 0.833,
                     horizontalalignment="center", verticalalignment="bottom", rotation=67,
                     color="white")
        ax.set_xlabel("T (K)")
        ax.set_ylabel("P (GPa)")

        # Diverging colorbar
        if palette == "seismic":
            cbar = plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], label="")

        # Continuous colorbar
        else:
            cbar = plt.colorbar(im, ax=ax, ticks=np.linspace(vmin, vmax, num=4), label="")

        # Set colorbar limits and number formatting
        if target == "rho":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vp":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vs":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "melt":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "variance":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

    # Add title
    if title:
        plt.title(title)

    # Vertical text spacing
    text_margin_x = 0.04
    text_margin_y = 0.15
    text_spacing_y = 0.1

    # Add rmse and r2
    if rmse is not None and r2 is not None:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=1.5,
                          alpha=0.3)
        plt.text(text_margin_x, text_margin_y - (text_spacing_y * 0), f"R$^2$: {r2:.3f}",
                 transform=plt.gca().transAxes, fontsize=fontsize * 0.833,
                 horizontalalignment="left", verticalalignment="bottom", bbox=bbox_props)
        plt.text(text_margin_x, text_margin_y - (text_spacing_y * 1),
                 f"RMSE: {rmse:.3f}", transform=plt.gca().transAxes,
                 fontsize=fontsize * 0.833, horizontalalignment="left",
                 verticalalignment="bottom", bbox=bbox_props)

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
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

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
# visualize gfem !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem(gfem_models, edges=True, palette="bone", verbose=1):
    """
    """
    for model in [m if m.dataset == "valid" else None for m in gfem_models]:
        # Check for model
        if model is None:
            continue

        # Get model data
        program = model.program
        sample_id = model.sample_id
        model_prefix = model.model_prefix
        res = model.res
        dataset = model.dataset
        targets = model.targets
        mask_geotherm = model.mask_geotherm
        results = model.results
        P, T = results["P"], results["T"]
        target_array = model.target_array
        fig_dir = model.fig_dir
        verbose = model.verbose

        if program == "magemin":
            program_title = "MAGEMin"
        elif program == "perplex":
            program_title = "Perple_X"

        if verbose >= 1:
            print(f"Visualizing {model_prefix} [{program}] ...")

        # Check for existing plots
        existing_figs = []
        for i, target in enumerate(targets):
            fig_1 = f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png"
            fig_2 = f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png"
            fig_3 = f"{fig_dir}/image9-{sample_id}-{dataset}.png"

            check = ((os.path.exists(fig_1) and os.path.exists(fig_3)) |
                     (os.path.exists(fig_1) and os.path.exists(fig_3)))

            if check:
                existing_figs.append(check)

        if existing_figs:
            return None

        for i, target in enumerate(targets):
            # Reshape targets into square array
            square_target = target_array[:, i].reshape(res + 1, res + 1)

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
                vmin=np.min(square_target[np.logical_not(np.isnan(square_target))])
                vmax=np.max(square_target[np.logical_not(np.isnan(square_target))])

            else:
                vmin = int(np.nanmin(np.unique(square_target)))
                vmax = int(np.nanmax(np.unique(square_target)))

            # Rename target
            target_rename = target.replace("_", "-")

            # Print filepath
            filename = f"{program}-{sample_id}-{dataset}-{target_rename}.png"
            if verbose >= 2:
                print(f"Saving figure: {filename}")

            # Plot targets
            visualize_target_array(P, T, square_target, target, program_title, palette,
                                   color_discrete, color_reverse, vmin, vmax, None, None,
                                   fig_dir, filename)
            if edges:
                original_image = square_target.copy()

                # Apply Sobel edge detection
                edges_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
                edges_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate the magnitude of the gradient
                magnitude = np.sqrt(edges_x**2 + edges_y**2) / np.nanmax(original_image)

                if not color_discrete:
                    vmin_mag = np.min(magnitude[np.logical_not(np.isnan(magnitude))])
                    vmax_mag = np.max(magnitude[np.logical_not(np.isnan(magnitude))])

                else:
                    vmin_mag = int(np.nanmin(np.unique(magnitude)))
                    vmax_mag = int(np.nanmax(np.unique(magnitude)))

                visualize_target_array(P, T, magnitude, target, "Gradient", palette,
                                       color_discrete, color_reverse, vmin_mag, vmax_mag,
                                       None, None, fig_dir, f"grad-{filename}", False)

            filename = f"prem-{sample_id}-{dataset}-{target_rename}.png"
            filename2 = f"prem-{sample_id}-{dataset}-{target_rename}-1557.png"

            # Plot PREM comparisons
            if target == "rho":
                # Print filepath
                if verbose >= 2:
                    print(f"Saving figure: {filename}")

                if program == "magemin":
                    results_mgm = results
                    results_ppx = None
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   ["low", "mid", "high"], results_mgm, results_ppx,
                                   title="Depth Profile", fig_dir=fig_dir, filename=filename)
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   ["mid"], results_mgm, results_ppx, title="Depth Profile",
                                   fig_dir=fig_dir, filename=filename2)

                elif program == "perplex":
                    results_mgm = None
                    results_ppx = results
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   ["low", "mid", "high"], results_mgm, results_ppx,
                                   title="Depth Profile", fig_dir=fig_dir, filename=filename)
                    visualize_prem(program, sample_id, dataset, res, target, "g/cm$^3$",
                                   ["mid"], results_mgm, results_ppx, title="Depth Profile",
                                   fig_dir=fig_dir, filename=filename2)

            if target in ["Vp", "Vs"]:
                # Print filepath
                if verbose >= 2:
                    print(f"Saving figure: {filename}")

                if program == "magemin":
                    results_mgm = results
                    results_ppx = None
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   ["low", "mid", "high"], results_mgm, results_ppx,
                                   title="Depth Profile", fig_dir=fig_dir, filename=filename)
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   ["mid"], results_mgm, results_ppx, title="Depth Profile",
                                   fig_dir=fig_dir, filename=filename2)

                elif program == "perplex":
                    results_mgm = None
                    results_ppx = results
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   ["low", "mid", "high"], results_mgm, results_ppx,
                                   title="Depth Profile", fig_dir=fig_dir, filename=filename)
                    visualize_prem(program, sample_id, dataset, res, target, "km/s",
                                   ["mid"], results_mgm, results_ppx, title="Depth Profile",
                                   fig_dir=fig_dir, filename=filename2)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize gfem diff !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_gfem_diff(gfem_models, palette="bone", verbose=1):
    """
    """
    # Parse models
    magemin_models = [m if m.program == "magemin" and m.dataset == "valid" else
                      None for m in gfem_models]
    magemin_models = sorted(magemin_models, key=lambda m: (m.sample_id if m else ""))

    perplex_models = [m if m.program == "perplex" and m.dataset == "valid" else
                      None for m in gfem_models]
    perplex_models = sorted(perplex_models, key=lambda m: (m.sample_id if m else ""))

    # Iterate through models
    for magemin_model, perplex_model in zip(magemin_models, perplex_models):
        # Check for model
        if magemin_model is None or perplex_model is None:
            continue

        # Get model data
        if magemin_model.sample_id == perplex_model.sample_id:
            sample_id = magemin_model.sample_id
        else:
            raise ValueError("Model samples are not the same!")
        if magemin_model.res == perplex_model.res:
            res = magemin_model.res
        else:
            raise ValueError("Model resolutions are not the same!")
        if magemin_model.dataset == perplex_model.dataset:
            dataset = magemin_model.dataset
        else:
            raise ValueError("Model datasets are not the same!")
        if magemin_model.targets == perplex_model.targets:
            targets = magemin_model.targets
        else:
            raise ValueError("Model datasets are not the same!")
        if magemin_model.mask_geotherm == perplex_model.mask_geotherm:
            mask_geotherm = magemin_model.mask_geotherm
        else:
            raise ValueError("Model geotherm masks are not the same!")
        if magemin_model.verbose == perplex_model.verbose:
            verbose = magemin_model.verbose
        else:
            verbose = 1

        fig_dir = f"figs/gfem/diff_{sample_id}_{res}"

        results_mgm, results_ppx = magemin_model.results, perplex_model.results
        P_mgm, T_mgm = results_mgm["P"], results_mgm["T"]
        P_ppx, T_ppx = results_ppx["P"], results_ppx["T"]
        target_array_mgm = magemin_model.target_array
        target_array_ppx = perplex_model.target_array

        for i, target in enumerate(targets):
            # Check for existing figures
            fig_1 = f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png"
            fig_2 = f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png"
            fig_3 = f"{fig_dir}/image9-{sample_id}-{dataset}.png"

            if os.path.exists(fig_1) and os.path.exists(fig_2) and os.path.exists(fig_3):
                print(f"Found composed plots at {fig_1}!")
                print(f"Found composed plots at {fig_2}!")
                print(f"Found composed plots at {fig_3}!")

            else:
                # Reshape targets into square array
                square_array_mgm = target_array_mgm[:, i].reshape(res + 1, res + 1)
                square_array_ppx = target_array_ppx[:, i].reshape(res + 1, res + 1)

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
                    vmin_mgm=np.min(
                        square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
                    vmax_mgm=np.max(
                        square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
                    vmin_ppx=np.min(
                        square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])
                    vmax_ppx=np.max(
                        square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])

                    vmin = min(vmin_mgm, vmin_ppx)
                    vmax = max(vmax_mgm, vmax_ppx)
                else:
                    num_colors_mgm = len(np.unique(square_array_mgm))
                    num_colors_ppx = len(np.unique(square_array_ppx))

                    vmin = 1
                    vmax = max(num_colors_mgm, num_colors_ppx) + 1

                if not color_discrete:
                    # Define a filter to ignore the specific warning
                    warnings.filterwarnings("ignore",
                                            message="invalid value encountered in divide")

                    # Create nan mask
                    mask = ~np.isnan(square_array_mgm) & ~np.isnan(square_array_ppx)

                    # Compute normalized diff
                    diff = square_array_mgm - square_array_ppx

                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(square_array_mgm, square_array_ppx))

                    # Calculate R2
                    r2 = r2_score(square_array_mgm, square_array_ppx)

                    # Add nans to match original target arrays
                    diff[~mask] = np.nan

                    # Rename target
                    target_rename = target.replace("_", "-")

                    # Print filepath
                    filename = f"diff-{sample_id}-{dataset}-{target_rename}.png"
                    if verbose >= 2:
                        print(f"Saving figure: {filename}")

                    # Plot target array normalized diff mgm-ppx
                    visualize_target_array(P_ppx, T_ppx, diff, target, "Residuals",
                                           "seismic", color_discrete, False, vmin, vmax,
                                           rmse, r2, fig_dir, filename)

                    filename = f"prem-{sample_id}-{dataset}-{target_rename}.png"

                    # Plot PREM comparisons
                    if target == "rho":
                        # Print filepath
                        if verbose >= 2:
                            print(f"Saving figure: {filename}")

                        visualize_prem("perplex", sample_id, dataset, res, target,
                                       ["low", "mid", "high"], "g/cm$^3$", results_mgm,
                                       results_ppx, title="Depth Profile", fig_dir=fig_dir,
                                       filename=filename)

                    if target in ["Vp", "Vs"]:
                        # Print filepath
                        if verbose >= 2:
                            print(f"Saving figure: {filename}")

                        visualize_prem("perplex", sample_id, dataset, res, target, "km/s",
                                       ["low", "mid", "high"], results_mgm, results_ppx,
                                       title="Depth Profile", fig_dir=fig_dir,
                                       filename=filename)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocmlm !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocmlm(rocmlm, skip=1, figwidth=6.3, figheight=4.725, fontsize=22):
    """
    """
    # Get ml model attributes
    program = rocmlm.program
    if program == "perplex":
        program_label = "Perple_X"
    elif program == "magemin":
        program_label = "MAGEMin"
    model_label_full = rocmlm.ml_model_label_full
    model_label = rocmlm.ml_model_label
    model_prefix = rocmlm.model_prefix
    sample_ids = rocmlm.sample_ids
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
    if any(sample in sample_ids for sample in ["PUM", "DMM", "PYR"]):
        skip = 1
    else:
        # Need to skip double for synthetic samples bc X_res training starts at 2 ...
        skip = skip * 2

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.constrained_layout.use"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    # Check for existing plots
    existing_figs = []
    for target in targets_rename:
        for s in sample_ids[::skip]:
            fig_1 = f"{fig_dir}/prem-{s}-{model_label}-{target}.png"
            fig_2 = f"{fig_dir}/surf-{s}-{model_label}-{target}.png"
            fig_3 = f"{fig_dir}/image-{s}-{model_label}-{target}.png"

            check = (os.path.exists(fig_1) and os.path.exists(fig_2) and
                     os.path.exists(fig_3))

            if check:
                existing_figs.append(check)

    if existing_figs:
        return None

    for s, sample_id in enumerate(sample_ids[::skip]):
        if verbose >= 1:
            print(f"Visualizing {model_prefix}-{sample_id} [{program}] ...")

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
            filename = f"{model_prefix}-{sample_id}-{target_rename}"

            # Plot target array 2d
            P = feature_array[:, :, 0 + n_feats]
            T = feature_array[:, :, 1 + n_feats]
            t = target_array[:, :, i]
            p = pred_array[:, :, i]

            visualize_target_array(P.flatten(), T.flatten(), t, target, program_label,
                                   palette, False, color_reverse, vmin[i], vmax[i], None,
                                   None, fig_dir, f"{filename}-targets.png")
            # Plot target array 3d
            visualize_target_surf(P, T, t, target, program_label, palette, False,
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

            # Reshape results and transform units for MAGEMin
            if program == "magemin":
                results_mgm = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                               target: t.flatten().tolist()}

                results_ppx = None

            # Reshape results and transform units for Perple_X
            elif program == "perplex":
                results_ppx = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                               target: t.flatten().tolist()}

                results_mgm = None

            # Reshape results and transform units for ML model
            results_rocmlm = {"P": P.flatten().tolist(), "T": T.flatten().tolist(),
                              target: p.flatten().tolist()}

            # Set geotherm threshold for extracting depth profiles
            res = w - 1

            # Plot PREM comparisons
            if target == "rho":
                visualize_prem(program, sample_id, "train", res, target, "g/cm$^3$",
                               ["low", "mid", "high"], results_mgm, results_ppx,
                               results_rocmlm, model_label, title="Depth Profile",
                               fig_dir=fig_dir, filename=f"{filename}-prem.png")

            if target in ["Vp", "Vs"]:
                visualize_prem(program, sample_id, "train", res, target, "km/s",
                               ["low", "mid", "high"], results_mgm, results_ppx,
                               results_rocmlm, model_label, title="Depth Profile",
                               fig_dir=fig_dir, filename=f"{filename}-prem.png")
    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize mixing array !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_mixing_array(mixing_array, fig_dir="figs/mixing_array", filename="earthchem",
                           batch=False, figwidth=6.3, figheight=4.7, fontsize=24):
    """
    """
    # Get mixing array attributes
    res = mixing_array.res
    pca = mixing_array.pca_model
    oxides = mixing_array.oxides_system
    n_pca_components = mixing_array.n_pca_components
    pca_model = mixing_array.pca_model
    pca_scaler = mixing_array.scaler
    data = mixing_array.earthchem_pca
    D_tio2 = mixing_array.D_tio2

    # Get correct Depletion column
    if batch:
        D_col = "D_BATCH"
    else:
        D_col = "D_FRAC"

    # Check for benchmark samples
    df_bench_path = "assets/data/benchmark-samples.csv"
    df_bench_pca_path = "assets/data/benchmark-samples-pca.csv"
    df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"
    df_synth_tops_path = "assets/data/synthetic-samples-mixing-tops.csv"
    df_synth_middle_path = "assets/data/synthetic-samples-mixing-middle.csv"
    df_synth_bottoms_path = "assets/data/synthetic-samples-mixing-bottoms.csv"
    df_synth_random_path = "assets/data/synthetic-samples-mixing-random.csv"

    # Read benchmark samples
    if os.path.exists(df_bench_path) and os.path.exists(df_synth_bench_path):
        if os.path.exists(df_bench_pca_path):
            df_bench = pd.read_csv(df_bench_pca_path)
            df_synth_bench = pd.read_csv(df_synth_bench_path)

        else:
            df_bench = pd.read_csv(df_bench_path)
            df_synth_bench = pd.read_csv(df_synth_bench_path)

            # Fit PCA to benchmark samples
            df_bench[["PC1", "PC2"]] = pca_model.transform(
                pca_scaler.transform(df_bench[oxides]))
            df_bench[["PC1", "PC2"]] = df_bench[["PC1", "PC2"]].round(3)

            # Calculate F melt
            ti_init = df_synth_bench.loc[
                df_synth_bench["SAMPLEID"] == "sm129", "TIO2"].iloc[0]
            df_bench["R_TIO2"] = round(df_bench["TIO2"] / ti_init, 3)
            df_bench["F_MELT_BATCH"] = round(
                ((D_tio2 / df_bench["R_TIO2"]) - D_tio2) / (1 - D_tio2), 3)
            df_bench["D_BATCH"] = round(1 - df_bench["F_MELT_BATCH"], 3)
            df_bench["F_MELT_FRAC"] = round(
                1 - df_bench["R_TIO2"]**(1 / ((1 / D_tio2) - 1)), 3)
            df_bench["D_FRAC"] = round(1 - df_bench["F_MELT_FRAC"], 3)

            # Save to csv
            df_bench.to_csv("assets/data/benchmark-samples-pca.csv", index=False)

    if (os.path.exists(df_synth_tops_path) and os.path.exists(df_synth_middle_path) and
        os.path.exists(df_synth_bottoms_path)):
            df_synth_tops = pd.read_csv(df_synth_tops_path)
            df_synth_middle = pd.read_csv(df_synth_middle_path)
            df_synth_bottoms = pd.read_csv(df_synth_bottoms_path)
            df_synth_random = pd.read_csv(df_synth_random_path)

    # Filter Depletion < 1
    data = data[(data[D_col] <= 1) & (data[D_col] >= 0)]

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    loadings = pd.DataFrame((pca.components_.T * np.sqrt(pca.explained_variance_)).T,
                            columns=oxides)

    # Colormap
    colormap = plt.cm.get_cmap("tab10")

    # Legend order
    legend_order = ["peridotite", "serpentinite", "pyroxenite", "metamorphic"]
    legend_lab = ["perid", "serp", "pyrx", "meta"]

    fname = f"{filename}-mixing-array"

    fig = plt.figure(figsize=(figwidth * 2, figheight * 1.2))

    ax = fig.add_subplot(121)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    oxs = ["SIO2", "MGO", "FEO", "AL2O3", "TIO2", "LOI", "CAO", "NA2O"]
    x_offset_text = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    y_offset_text = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
    text_fac, arrow_fac = 2.8, 1.3
    x_offset_arrow, y_offset_arrow = 1.0, 4.5

    for oxide, x_off, y_off in zip(oxs, x_offset_text, y_offset_text):
        if oxide == "AL2O3":
            oxide_label = "Al$_2$O$_3$"
        elif oxide == "TIO2":
            oxide_label = "TiO$_2$"
        elif oxide == "SIO2":
            oxide_label = "SiO$_2$"
        elif oxide == "MGO":
            oxide_label = "MgO"
        elif oxide == "FEO":
            oxide_label = "FeOT"
        elif oxide == "LOI":
            oxide_label = "LOI"
        elif oxide == "CAO":
            oxide_label = "CaO"
        elif oxide == "NA2O":
            oxide_label = "Na$_2$O"

        ax.arrow(x_offset_arrow, y_offset_arrow, loadings.at[0, oxide] *
                 arrow_fac, loadings.at[1, oxide] * arrow_fac, width=0.1,
                 head_width=0.4, color="black")
        ax.text(x_off + (loadings.at[0, oxide] * text_fac),
                y_off + (loadings.at[1, oxide] * text_fac), oxide_label,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                          pad=0.1), fontsize=fontsize * 0.833, color="black",
                ha="center", va="center")

    legend_handles = []
    for i, comp in enumerate(legend_order):
        marker = mlines.Line2D([0], [0], marker="o", color="w", label=legend_lab[i],
                               markersize=4, markerfacecolor=colormap(i),
                               markeredgewidth=0, linestyle="None", alpha=1)
        legend_handles.append(marker)

        indices = data.loc[data["ROCKTYPE"] == comp].index

        scatter = ax.scatter(data.loc[indices, "PC1"],
                             data.loc[indices, "PC2"], edgecolors="none",
                             color=colormap(i), marker=".", s=55, label=legend_lab[i],
                             alpha=1)

    sns.kdeplot(data=data, x="PC1", y="PC2", hue="ROCKTYPE", zorder=1,
                hue_order=legend_order, ax=ax, levels=5, warn_singular=False)

    legend = ax.legend(handles=legend_handles, loc="upper center", frameon=False,
                       bbox_to_anchor=(0.5, 0.13), ncol=4, columnspacing=0,
                       handletextpad=-0.5, markerscale=3,
                       fontsize=fontsize * 0.833)
    # Legend order
    for i, label in enumerate(legend_lab):
        legend.get_texts()[i].set_text(label)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    ax2 = fig.add_subplot(122)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    sns.scatterplot(data=data, x="PC1", y="PC2", facecolor="0.6",
                    edgecolor="None", linewidth=2, s=12, legend=False, ax=ax2,
                    zorder=0)

    # Create colorbar
    pal = sns.color_palette("magma", as_cmap=True).reversed()
    norm = plt.Normalize(df_synth_bench[D_col].min(),
                         df_synth_bench[D_col].max())
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
    sm.set_array([])

    sns.scatterplot(data=df_synth_middle, x="PC1", y="PC2", hue=D_col,
                    palette=pal, edgecolor="None", linewidth=2, s=82,
                    legend=False, ax=ax2, zorder=0)
    sns.scatterplot(data=df_synth_random, x="PC1", y="PC2", hue=D_col,
                    palette=pal, edgecolor="None", linewidth=2, s=52,
                    legend=False, ax=ax2, zorder=0)
    sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm000"],
                    x="PC1", y="PC2", facecolor="white", edgecolor="black",
                    linewidth=2, s=150, legend=False, ax=ax2, zorder=6)
    sns.scatterplot(data=df_synth_bench[df_synth_bench["SAMPLEID"] == "sm129"],
                    x="PC1", y="PC2", facecolor="white", edgecolor="black",
                    marker="D", linewidth=2, s=150, legend=False, ax=ax2,
                    zorder=6)
    ax2.annotate("DSUM", xy=(
        df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm000", "PC1"].iloc[0],
        df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm000",
                           "PC2"].iloc[0]), xytext=(-65, -25),
                 textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="black", linewidth=1.5, alpha=0.8),
                fontsize=fontsize * 0.833, zorder=8)
    ax2.annotate("PSUM", xy=(
        df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm129", "PC1"].iloc[0],
        df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm129",
                           "PC2"].iloc[0]), xytext=(10, -25),
                 textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="black", linewidth=1.5, alpha=0.8),
                fontsize=fontsize * 0.833, zorder=8)

    mrkr = ["s", "^", "P"]
    for l, name in enumerate(["PUM", "DMM", "PYR"]):
        if name == "PUM":
            offst = (0, -29)
        elif name == "DMM":
            offst = (13, 15)
        else:
            offst = (-43, -25)

        sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="PC1",
                        y="PC2", marker=mrkr[l], facecolor="white",
                        edgecolor="black", linewidth=2, s=150, legend=False,
                        ax=ax2, zorder=7)
        ax2.annotate(
            name, xy=(df_bench.loc[df_bench["SAMPLEID"] == name, "PC1"].iloc[0],
                      df_bench.loc[df_bench["SAMPLEID"] == name, "PC2"].iloc[0]),
            xytext=offst, textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                      edgecolor="black", linewidth=1.5, alpha=0.8),
            fontsize=fontsize * 0.833, zorder=8)

    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())

    # Add colorbar
    cbaxes = inset_axes(ax2, width="40%", height="3%", loc=1)
    colorbar = plt.colorbar(sm, ax=ax2, cax=cbaxes, label="Fertility, $\\xi$",
                            orientation="horizontal")
    colorbar.ax.set_xticks([sm.get_clim()[0], sm.get_clim()[1]])
    colorbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("")
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax2.set_yticks([])

    # Create inset
    left, bottom, width, height = [0.520, 0.705, 0.185, 0.32]
    ax2 = fig.add_axes([left, bottom, width, height])
    sns.scatterplot(data=data, x="PC1", y="D_FRAC", facecolor="0.6",
                    edgecolor="None", linewidth=2, s=8, legend=False, ax=ax2,
                    zorder=0)
    sns.scatterplot(data=df_synth_middle, x="PC1", y=D_col, hue=D_col,
                    palette=pal, edgecolor="None", linewidth=2, s=31,
                    legend=False, ax=ax2, zorder=0)
    sns.scatterplot(data=df_synth_random, x="PC1", y=D_col, hue=D_col,
                    palette=pal, edgecolor="None", linewidth=2, s=21,
                    legend=False, ax=ax2, zorder=0)
    mrkr = ["s", "^", "P"]
    for l, name in enumerate(["PUM", "DMM", "PYR"]):
        sns.scatterplot(data=df_bench[df_bench["SAMPLEID"] == name], x="PC1",
                        y="D_FRAC", marker=mrkr[l], facecolor="white",
                        edgecolor="black", linewidth=2, s=100, legend=False,
                        ax=ax2, zorder=7)

    ax2.set_ylabel("$\\xi$")
    ax2.yaxis.set_label_coords(0.20, 0.85)
    ax2.xaxis.set_label_coords(0.50, 0.20)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_facecolor("0.8")
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    # Add captions
    fig.text(0.03, 0.97, "a)", fontsize=fontsize * 1.2)
    fig.text(0.72, 0.97, "b)", fontsize=fontsize * 1.2)
    fig.text(0.67, 0.74, "c)", fontsize=fontsize * 1.2)

    # Save the plot to a file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plt.savefig(f"{fig_dir}/{fname}.png")

    # Close device
    plt.close()

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize harker diagrams !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_harker_diagrams(mixing_array, fig_dir="figs/mixing_array",
                              filename="earthchem", figwidth=6.3, figheight=5.8,
                              fontsize=22):
    """
    """
    # Get mixing array attributes
    oxides = [ox for ox in mixing_array.oxides_system if ox not in ["SIO2", "FE2O3"]]
    n_pca_components = mixing_array.n_pca_components
    data = mixing_array.earthchem_filtered

    # Check for benchmark samples
    df_bench_path = "assets/data/benchmark-samples.csv"
    df_synth_bench_path = "assets/data/synthetic-samples-benchmarks.csv"

    if os.path.exists(df_bench_path) and os.path.exists(df_synth_bench_path):
        # Read benchmark samples
        df_bench = pd.read_csv(df_bench_path)
        df_synth_bench = pd.read_csv(df_synth_bench_path)

    # Check for synthetic data
    if not mixing_array.synthetic_data_written:
        raise Exception("No synthetic data found! Call create_mixing_arrays() first ...")

    # Initialize synthetic datasets
    synthetic_samples = pd.read_csv(f"assets/data/synthetic-samples-mixing-random.csv")

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # Create a grid of subplots
    num_plots = len(oxides)

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
    legend_order = ["peridotite", "serpentinite", "pyroxenite", "metamorphic"]

    for k, y in enumerate(oxides):
        ax = axes[k]

        sns.scatterplot(data=synthetic_samples, x="SIO2", y=y, linewidth=0, s=8,
                        color="black", alpha=1, legend=False, ax=ax, zorder=3)

        sns.scatterplot(data=data, x="SIO2", y=y, hue="ROCKTYPE", hue_order=legend_order,
                        linewidth=0, s=8, alpha=0.5, ax=ax, zorder=1, legend=False)
        sns.kdeplot(data=data, x="SIO2", y=y, hue="ROCKTYPE", hue_order=legend_order,
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

        if k == 4:
            ax.annotate(
                "DSUM", xy=(df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm000",
                                               "SIO2"].iloc[0],
                            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm000",
                                               y].iloc[0]),
                xytext=(-18, -18), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="black",
                          linewidth=1.5, alpha=1),
                fontsize=fontsize * 0.579, zorder=8)
            ax.annotate(
                "PSUM", xy=(df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm129",
                                               "SIO2"].iloc[0],
                            df_synth_bench.loc[df_synth_bench["SAMPLEID"] == "sm129",
                                               y].iloc[0]),
                xytext=(10, 0), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="black",
                          linewidth=1.5, alpha=1),
                fontsize=fontsize * 0.579, zorder=8)

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

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/{filename}-harker-diagram.png")

    # Close device
    plt.close()

    return None