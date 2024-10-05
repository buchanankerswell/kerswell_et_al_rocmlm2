import numpy as np
from gfem import GFEMModel, build_gfem_models
from rocmlm import RocMLM, load_pretrained_rocmlm

def main():
    """
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build training database and train RocMLM
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gfems = []
#    gfem_configs = ["assets/config_yamls/hydrated-shallow-upper-mantle-hp02m.yaml",
#                    "assets/config_yamls/hydrated-shallow-upper-mantle-hp02r.yaml"]
    gfem_configs = ["assets/config_yamls/hydrated-shallow-upper-mantle-hp02m.yaml"]
    rocmlm_config = "assets/config_yamls/rocmlm-default.yaml"

    for yaml in gfem_configs:
        gfems.extend(build_gfem_models(config_yaml=yaml))

#    mod = RocMLM(gfems, "SimpleNet", config_yaml=rocmlm_config)
#    mod.train()
#    mod.visualize()
#
#    mod = RocMLM(gfems, "ImprovedNet", config_yaml=rocmlm_config)
#    mod.train()
#    mod.visualize()
#
#    mod = RocMLM(gfems, "UNet", config_yaml=rocmlm_config)
#    mod.train()
#    mod.visualize()
#
    mod = RocMLM(gfems, "KN", config_yaml=rocmlm_config)
    mod.train()
    mod.visualize()

    mod = RocMLM(gfems, "DT", config_yaml=rocmlm_config)
    mod.train()
    mod.visualize()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load RocMLM from pkl file and Test inference speed
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#    # Simulate a 921x301 grid of dummy features
#    xi = np.full(921*301, 1)
#    h2o = np.full(921*301, 5)
#    P = np.full(921*301, 15)
#    T = np.full(921*301, 1773)
#
#    pretrained_path = "rocmlms/DT-S248-R65-F4-T5-hp02-pretrained.pkl"
#    mod = load_pretrained_rocmlm(pretrained_path)
#    mod.visualize()
#
#    for _ in range(10):
#        pred = mod.inference(P=P, T=T, XI_FRAC=xi, H2O=h2o)

if __name__ == "__main__":
    main()