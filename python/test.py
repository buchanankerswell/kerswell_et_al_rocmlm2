import glob
import numpy as np
from gfem import GFEMModel, build_gfem_models
from rocmlm import RocMLM, load_pretrained_rocmlm

def main():
    """
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build training database and train RocMLM
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gfem_config = "assets/config_yamls/gfem-cerpa2022-demo.yaml"
    gfems = build_gfem_models(gfem_config)

    paths = glob.glob("gfems/cerpa*/results.csv")
    rocmlm_config = "assets/config_yamls/rocmlm-cerpa2022-demo.yaml"

#    gfem_config = "assets/config_yamls/gfem-stx-demo.yaml"
#    gfems = build_gfem_models(config_yaml=gfem_config)

#    paths = glob.glob("gfems/sm*/results.csv")
#    rocmlm_config = "assets/config_yamls/rocmlm-stx-demo.yaml"

    mod = RocMLM(paths, "SimpleNet", rocmlm_config)
    mod.train()
    mod.visualize()

#    mod = RocMLM(paths, "ImprovedNet", rocmlm_config)
#    mod.train()
#    mod.visualize()

#    mod = RocMLM(paths, "KN", rocmlm_config)
#    mod.train()
#    mod.visualize()

#    mod = RocMLM(paths, "DT", rocmlm_config)
#    mod.train()
#    mod.visualize()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test inference speed
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    # Simulate a 921x301 grid of dummy features
#    xi = np.full(921*301, 1)
#    h2o = np.full(921*301, 5)
#    P = np.full(921*301, 15)
#    T = np.full(921*301, 1773)
#
#    model = "SimpleNet"
#    pretrained_path = glob.glob(f"pretrained_rocmlms/{model}*.pkl")
#
#    mod = load_pretrained_rocmlm(pretrained_path[0])
#
#    for _ in range(10):
#        pred = mod.inference(P=P, T=T, XI_FRAC_FEAT=xi, H2O_FEAT=h2o)

if __name__ == "__main__":
    main()