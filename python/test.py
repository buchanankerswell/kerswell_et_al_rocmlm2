from gfem import GFEMModel, get_sampleids, build_gfem_models
from rocmlm import train_rocmlms, visualize_rocmlm_performance, compose_rocmlm_plots

def test():
    """
    """
    return None

def main():
    """
    """
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
        rocmlms[name] = train_rocmlms(models, ["DT"])

#    # Visualize RocMLMs
#    visualize_rocmlm_performance()
#
    for name, models in rocmlms.items():
        for model in models:
            compose_rocmlm_plots(model)

    return None

if __name__ == "__main__":
    main()