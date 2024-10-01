from rocmlm import RocMLM
from gfem import GFEMModel, build_gfem_models

def main():
    """
    """
    ####################################################################################
    # Building training dataset
    ####################################################################################
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GFEM models
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # makes default PYR stx21 model from GFEMModel class
    model = GFEMModel()
    model.build()
    model.visualize()

    # makes default PYR stx21 model from yaml file
    model = GFEMModel(config_yaml="assets/config_yamls/PYR-stx21.yaml")
    model.build()
    model.visualize()

    res, src, samp = 32, "assets/synth-mids.csv", "sm005-h2o005"

    # makes hp02 model from GFEMModel class
    P_min, P_max, T_min, T_max = 0.1, 8.1, 273, 1973
    model_shallow = GFEMModel("hp02", samp, src, res, P_min, P_max, T_min, T_max)
    model_shallow.build()
    model_shallow.visualize()

    # makes stx21 model from GFEMModel class
    P_min, P_max, T_min, T_max = 8.1, 136.1, 773, 4273
    model_deep = GFEMModel("stx21", samp, src, res, P_min, P_max, T_min, T_max)
    model_deep.build()
    model_deep.visualize()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Builds multiple GFEM models in parallel
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    # makes multiple hp02 models from yaml file
#    yaml = "assets/config_yamls/hydrated-shallow-upper-mantle-hp02m.yaml"
#    gfems = build_gfem_models(config_yaml=yaml)
#
#    # makes multiple stx21 models from yaml file
#    yaml = "assets/config_yamls/dry-deep-mantle-stx21m.yaml"
#    gfems = build_gfem_models(config_yaml=yaml)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train RocMLMs
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    # makes rocmlm from RocMLM class
#    model_nn = RocMLM(gfems, "NN", tune=False)
#    model_nn.train()
#    model_nn.visualize()
#
#    # makes rocmlm from yaml file
#    yaml = "assets/config_yamls/rocmlm-test.yaml"
#    model = RocMLM(gfems, "NN", tune=True, config_yaml=yaml)
#    model.train()
#    model.visualize()
#
#    model = RocMLM(gfems, "KN", tune=True, config_yaml=yaml)
#    model.train()
#    model.visualize()
#
#    model = RocMLM(gfems, "DT", tune=True, config_yaml=yaml)
#    model.train()
#    model.visualize()

#    ####################################################################################
#    Test rocmlm inference speed
#    ####################################################################################
#    xi = [1]
#    h2o_in = [1]
#    P = [15]
#    T = [1773]
#
#    print("STX speed")
#    for _ in range(10):
#        pred = mod_st.predict_old(xi, P, T)
#    print("HP speed")
#    for _ in range(10):
#        pred = mod_hp.predict(xi, h2o_in, P, T)
#
#    ####################################################################################
#    End test rocmlm inference speed
#    ####################################################################################

    return None

if __name__ == "__main__":
    main()