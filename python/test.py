from gfem import GFEMModel

def test():
    """
    """
    return None

def main():
    """
    """
    res = 64
    db = "koma06"
    source = "assets/data/synth-mids.csv"

    for i in range(2):
#    for i in range(16):
        sid = f"sm000-loi{str(i).zfill(3)}"
        model = GFEMModel(db, sid, source, res)
        model.build_model()

    return None

if __name__ == "__main__":
    main()