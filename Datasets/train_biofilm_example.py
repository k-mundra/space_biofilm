"""
train_biofilm_example.py

Minimal end-to-end example using biofilm_ground_to_space.py.

YOU MUST:
  1. Replace the dummy TIFF paths with real paths from OSD-627/554.
  2. Replace the dummy RNA features with PCA features from your RNA data.
  3. Fill MATERIAL_PROPS in biofilm_ground_to_space.py with real numbers.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from biofilm_ground_to_space import (
    TrajectorySpec,
    BiofilmGroundSpaceDataset,
    BiofilmConvLSTMModel,
    MATERIAL_PROPS,
)


def main():
    # ----- 1. RNA feature dict (REPLACE with your PCA vectors) -----
    # keys: RNA group ids (e.g. 'G4', 'F4')
    rna_feat_dict = {
        "G4": np.random.randn(8).astype("float32"),
        "F4": np.random.randn(8).astype("float32"),
    }

    # ----- 2. Trajectories list (REPLACE TIFF paths + ids) -----
    # Each TrajectorySpec is one material+gravity sequence over time.
    # tiff_paths should be ordered [Day1, Day2, Day3] for that sample.
    traj_specs = [
        TrajectorySpec(
            tiff_paths=[
                "path/to/day1_ground_SS316_stack.tif",
                "path/to/day2_ground_SS316_stack.tif",
                "path/to/day3_ground_SS316_stack.tif",
            ],
            material="SS316",
            gravity="Ground",
            rna_id="G4",
            # predict Day3 space morphology from ground sequence, for example
            target_path="path/to/day3_space_SS316_stack.tif",
        )
    ]

    dataset = BiofilmGroundSpaceDataset(traj_specs, rna_feat_dict)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ----- 3. Build model -----
    rna_dim = dataset.rna_dim if dataset.rna_dim > 0 else 8
    mat_dim = MATERIAL_PROPS.shape[1]

    model = BiofilmConvLSTMModel(rna_dim=rna_dim, mat_dim=mat_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # ----- 4. Tiny training loop -----
    for epoch in range(3):
        model.train()
        for batch in loader:
            X   = batch["morphology"].to(device)
            rna = batch["rna"].to(device)
            mat = batch["material"].to(device)
            env = batch["env"].to(device)
            y   = batch["target"].to(device)

            optimizer.zero_grad()
            y_pred = model(X, rna, mat, env)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
