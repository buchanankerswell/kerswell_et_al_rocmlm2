import glob
import numpy as np
from rocmlm import RocMLM
from gfem import GFEMModel, build_gfem_models

def main():
    """
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build training database and train RocMLM
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gfem_config_yaml = "assets/config_yamls/gfem-stx21-demo.yaml"
    gfems = build_gfem_models(gfem_config_yaml)

    paths = glob.glob("gfems/sm*/results.csv")
    rocmlm_config_yaml = "assets/config_yamls/rocmlm-stx21-demo.yaml"

    mod = RocMLM(paths, "SimpleNet", rocmlm_config_yaml)
    mod.train()
#    mod.visualize()

#    mod = RocMLM(paths, "ImprovedNet", rocmlm_config_yaml)
#    mod.train()
#    mod.visualize()

#    mod = RocMLM(paths, "KN", rocmlm_config_yaml)
#    mod.train()
#    mod.visualize()

#    mod = RocMLM(paths, "DT", rocmlm_config_yaml)
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