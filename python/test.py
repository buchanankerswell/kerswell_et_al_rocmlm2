import glob
from rocmlm import RocMLM
from gfem import build_gfem_models

def main():
    """
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build training database and train RocMLM
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gfem_config = "assets/config_yamls/demo-dataset.yaml"
    rocmlm_config = "assets/config_yamls/rocmlm-default.yaml"

    gfems = build_gfem_models(config_yaml=gfem_config)
    paths = glob.glob("gfems/*/results.csv")

#    mod = RocMLM(paths, "SimpleNet", config_yaml=rocmlm_config)
#    mod.train()

#    mod = RocMLM(paths, "ImprovedNet", config_yaml=rocmlm_config)
#    mod.train()

    mod = RocMLM(paths, "UNet", config_yaml=rocmlm_config)
    mod.train()

#    mod = RocMLM(paths, "KN", config_yaml=rocmlm_config)
#    mod.train()

#    mod = RocMLM(paths, "DT", config_yaml=rocmlm_config)
#    mod.train()

if __name__ == "__main__":
    main()