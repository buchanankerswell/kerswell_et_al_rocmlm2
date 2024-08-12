from hymatz import HyMaTZ
from rocmlm import RocMLM, train_rocmlms, visualize_rocmlm_performance, compose_rocmlm_plots
from gfem import (GFEMModel, get_sampleids, build_gfem_models, compose_gfem_plots,
                  visualize_prem_comps)

def main():
    """
    """
#    model = HyMaTZ(1573, "Pyrolite", 50)

    model = GFEMModel("hp633", f"sm000-loi005", "assets/data/synth-mids.csv")
    model.build_model()

#    models = []
#    for i in range(0, 8):
#        models.append(GFEMModel("hp633", f"sm000-loi00{i}", "assets/data/synth-mids.csv"))
#        models.append(GFEMModel("koma06", f"sm000-loi00{i}", "assets/data/synth-mids.csv"))
#        models.append(GFEMModel("stx21", f"sm000-loi00{i}", "assets/data/synth-mids.csv"))
#    compose_gfem_plots(models)

#    gfems = {}
#    sources = {"m": "assets/data/synth-mids.csv", "r": "assets/data/synth-rnds.csv"}
#    for name, source in sources.items():
#        sids = get_sampleids(source)
#        gfems[name] = build_gfem_models(source, sids)
#    visualize_prem_comps(gfems["m"] + gfems["r"])

#    gfems = {}
#    sources = {"m": "assets/data/synth-mids.csv", "r": "assets/data/synth-rnds.csv"}
#    for name, source in sources.items():
#        sids = get_sampleids(source, "all")
#        gfems[name] = build_gfem_models(source, sids)
#    training_data = gfems["m"] + gfems["r"]
#    rocmlms = train_rocmlms(training_data, PT_steps=[1], X_steps=[1])
#    visualize_rocmlm_performance()
#    for model in rocmlms: compose_rocmlm_plots(model)

    return None

if __name__ == "__main__":
    main()