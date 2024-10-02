import numpy as np
from rocmlm import RocMLM
from gfem import GFEMModel, build_gfem_models

def main():
    """
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train rocmlm and test inference speed
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gfems = []
    gfem_configs = ["assets/config_yamls/hydrated-shallow-upper-mantle-hp02m.yaml",
                    "assets/config_yamls/hydrated-shallow-upper-mantle-hp02r.yaml"]

    for yaml in gfem_configs:
        gfems.extend(build_gfem_models(config_yaml=yaml))

    rocmlm_config = "assets/config_yamls/rocmlm-default.yaml"
    mod_default = RocMLM(gfems, "DT", config_yaml=rocmlm_config)
    mod_default.train()
    mod_default.visualize()

    rocmlm_config = "assets/config_yamls/rocmlm-test.yaml"
    mod_test = RocMLM(gfems, "DT", config_yaml=rocmlm_config)
    mod_test.train()
    mod_test.visualize()

    # Simulate a 921x301 grid of dummy features
    xi = np.full(921*301, 1)
    h2o = np.full(921*301, 5)
    r_mgsi = np.full(921*301, 0.8)
    r_alsi = np.full(921*301, 0.03)
    P = np.full(921*301, 15)
    T = np.full(921*301, 1773)

    for _ in range(10):
        pred = mod_default.inference(P=P, T=T, XI_FRAC=xi, H2O=h2o)
        pred = mod_test.inference(P=P, T=T, XI_FRAC=xi, H2O=h2o, R_MGSI=r_mgsi, R_ALSI=r_alsi)

    return None

if __name__ == "__main__":
    main()