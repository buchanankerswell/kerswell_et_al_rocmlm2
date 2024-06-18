from gfem import GFEMModel

def test():
    """
    """
    return None

def main():
    """
    """
    res = 32
    source = "assets/data/synth-mids.csv"

    for db in ["stx21", "hp633"]:
        for i in [0, 2, 4, 6, 8]:
            sid = f"sm000-loi00{i}"
            model = GFEMModel(db, sid, source, res)
            model.build_model()

    return None

if __name__ == "__main__":
    main()