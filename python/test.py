from gfem import GFEMModel, get_sampleids, build_gfem_models
from rocmlm import RocMLM, train_rocmlms, visualize_rocmlm_performance, compose_rocmlm_plots

def main():
    """
    """
    gfems = {}
    sources = {"m": "assets/data/synth-mids.csv", "r": "assets/data/synth-rnds.csv"}
    for name, source in sources.items():
        sids = get_sampleids(source, "all")
        gfems[name] = build_gfem_models(source, sids)
    training_data = gfems["m"] + gfems["r"]
    rocmlms = train_rocmlms(training_data, PT_steps=[1], X_steps=[1])
    visualize_rocmlm_performance()
    for model in rocmlms: compose_rocmlm_plots(model)
    return None

if __name__ == "__main__":
    main()