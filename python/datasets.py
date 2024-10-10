#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
import os
import glob
import traceback
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

#######################################################
## .1.              Torch Datasets               !!! ##
#######################################################
class GFEMDataset(Dataset):
    def __init__(self, gfem_paths, features, targets):
        """
        """
        self.gfem_paths = gfem_paths
        self.features = features
        self.targets = targets

        if not gfem_paths or any([not os.path.exists(p) for p in self.gfem_paths]):
            raise Exception(f"Cannot load gfem data!")

        output_file = "assets/temp-dataset.parquet"
        self._combine_and_save_csv(self.gfem_paths, output_file)
        self.data = pd.read_parquet(output_file)
        self.total_rows = len(self.data)

    def _combine_and_save_csv(self, filepaths, output_file):
        """
        """
        combined_df = pd.concat(
            map(pd.read_csv, filepaths)
        )[self.features + self.targets].dropna().astype("float32")
        combined_df.to_parquet(output_file, index=False)

    def __len__(self):
        """
        """
        return self.total_rows

    def __getitem__(self, idx):
        """
        """
        row = self.data.iloc[idx].to_numpy(dtype=np.float32, copy=False)
        X, y = row[:len(self.features)], row[len(self.features):]
        return np.nan_to_num(X).reshape(1, -1), np.nan_to_num(y).reshape(1, -1)

#######################################################
class ScaledGFEMDataset(Dataset):
    def __init__(self, scaled_X, scaled_y):
        """
        """
        self.scaled_X = scaled_X
        self.scaled_y = scaled_y

    def __len__(self):
        """
        """
        return len(self.scaled_X)

    def __getitem__(self, idx):
        """
        """
        X = self.scaled_X[idx].reshape(1, -1)
        y = self.scaled_y[idx].reshape(1, -1)

        return X, y

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    try:
        paths = glob.glob("gfems/*/results.csv")

        features = ["P", "T"]
        targets = ["density", "molar_entropy"]

        if paths is not None:
            data = GFEMDataset(paths, features, targets)

            print("Data loading ...")
            for i, (X, y) in enumerate(data):
                if i % 1e3 == 0: print(f"Row {i}: {X} {y}")

    except Exception as e:
        print(f"Error in main(): {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
