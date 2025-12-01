"""
biofilm_ground_to_space.py
==========================

Core scientific-ML components for modeling ground→space biofilm dynamics.

What this file does
-------------------
1. Defines a MATERIAL_PROPS table and helper:
     - YOU fill real numbers from your material characterization.
2. Loads full 3D TIFF stacks (keeps all z-layers).
3. Defines a TrajectorySpec dataclass describing one trajectory.
4. Defines BiofilmGroundSpaceDataset:
     - returns morphology sequence + RNA + material vector + gravity one-hot.
5. Defines BiofilmConvLSTMModel:
     - ConvLSTM3D backbone conditioned on RNA + material + gravity.

Your separate preprocessing scripts
-----------------------------------
Keep your existing TIFF merging/splitting utilities in separate files,
e.g. `merge_tiff_layers.py`, `split_png_tiles.py`. This module just
consumes final TIFF stack paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# 1. Material physics table (YOU fill in real values here)
# ----------------------------------------------------------------------

# NOTE: these numbers are placeholders. Replace them with your real
# cos(theta), roughness, adhesion energy, etc.
MATERIAL_PROPS = pd.DataFrame(
    {
        "material": [
            "SS316",
            "pSS316",
            "LIS",
            "Silicone",
            "DLIP_Silicone",
            "Cellulose",
        ],
        "cos_theta":      [0.42, 0.53, -0.42, -0.17, -0.42, 0.71],
        "roughness_log":  [-2.1, -2.3, -2.2, -1.8, -1.5, -2.6],
        "adhesion_energy":[102.0, 110.0, 42.0, 60.0, 42.0, 123.0],
        "lis_flag":       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
).set_index("material")


def get_material_feats(material_name: str) -> np.ndarray:
    """
    Return a physics-informed feature vector for a material.

    Parameters
    ----------
    material_name : str
        Must match an index in MATERIAL_PROPS.

    Returns
    -------
    np.ndarray, shape (d_mat,), dtype float32
    """
    try:
        vec = MATERIAL_PROPS.loc[material_name].values.astype(np.float32)
    except KeyError as exc:
        raise KeyError(
            f"Material {material_name!r} not found in MATERIAL_PROPS. "
            "Check spelling or update MATERIAL_PROPS."
        ) from exc
    return vec


# ----------------------------------------------------------------------
# 2. TIFF loader that keeps ALL z-layers
# ----------------------------------------------------------------------

def load_tiff_stack(path: str, normalize: bool = True) -> np.ndarray:
    """
    Load a multi-layer TIFF stack as [1, Z, H, W].

    This mirrors the COMSTAT2 idea of using the full 3D volume.

    Parameters
    ----------
    path : str
        Path to .tif/.tiff.
    normalize : bool
        If True, divide by max intensity so values are in [0,1].

    Returns
    -------
    np.ndarray, shape (1, Z, H, W)
    """
    arr = tifffile.imread(path)  # [Z,H,W] or [H,W]
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[None, :, :]      # -> [1,H,W]
    elif arr.ndim == 3:
        pass                       # already [Z,H,W]
    else:
        raise ValueError(f"Unexpected TIFF ndim={arr.ndim} for {path!r}")

    if normalize and arr.max() > 0:
        arr /= float(arr.max())

    # Add channel dimension: [C=1,Z,H,W]
    arr = arr[None, ...]
    return arr


# ----------------------------------------------------------------------
# 3. TrajectorySpec + Dataset
# ----------------------------------------------------------------------

@dataclass
class TrajectorySpec:
    """
    Describe one (material, gravity) trajectory.

    Attributes
    ----------
    tiff_paths : list[str]
        Time-ordered TIFF stack paths for this trajectory
        (e.g. [day1, day2, day3]).
    material : str
        Material name (must exist in MATERIAL_PROPS index).
    gravity : str
        'Ground'/'Earth'/'G' or 'Space'/'Flight'/'F'.
    rna_id : str, optional
        Key into RNA feature dict (e.g. 'G4', 'F4').
    target_path : str, optional
        Target TIFF stack (e.g. microgravity Day3).
    target_scalar : float, optional
        Scalar target instead of image (e.g. growth rate).
    """
    tiff_paths: Sequence[str]
    material: str
    gravity: str
    rna_id: Optional[str] = None
    target_path: Optional[str] = None
    target_scalar: Optional[float] = None


class BiofilmGroundSpaceDataset(Dataset):
    """
    PyTorch Dataset giving:
        morphology : [T,1,Z,H,W]
        rna        : [d_rna]
        material   : [d_mat]
        env        : [2]  (Earth, Space)
        target     : [1,Z,H,W] or scalar
    """

    def __init__(
        self,
        traj_specs: Sequence[TrajectorySpec],
        rna_feat_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.traj_specs: List[TrajectorySpec] = list(traj_specs)
        self.rna_feat_dict = rna_feat_dict or {}

        # Infer RNA dim if possible
        self._rna_dim: int = 0
        for spec in self.traj_specs:
            if spec.rna_id is not None and spec.rna_id in self.rna_feat_dict:
                self._rna_dim = int(self.rna_feat_dict[spec.rna_id].shape[-1])
                break

    def __len__(self) -> int:
        return len(self.traj_specs)

    @property
    def rna_dim(self) -> int:
        return self._rna_dim

    @staticmethod
    def _gravity_to_onehot(label: str) -> np.ndarray:
        s = label.strip().lower()
        if s in {"ground", "earth", "g", "control"}:
            return np.array([1.0, 0.0], dtype=np.float32)
        if s in {"space", "flight", "f", "spaceflight"}:
            return np.array([0.0, 1.0], dtype=np.float32)
        raise ValueError(f"Unrecognized gravity label: {label!r}")

    def __getitem__(self, idx: int):
        spec = self.traj_specs[idx]

        # --- morphology sequence ---
        stacks = [load_tiff_stack(p) for p in spec.tiff_paths]  # each [1,Z,H,W]
        X = np.stack(stacks, axis=0)                            # [T,1,Z,H,W]
        X_t = torch.from_numpy(X).float()

        # --- RNA ---
        if spec.rna_id is not None and spec.rna_id in self.rna_feat_dict:
            rna_np = self.rna_feat_dict[spec.rna_id].astype(np.float32)
        else:
            rna_np = np.zeros((self._rna_dim,), dtype=np.float32)
        rna_t = torch.from_numpy(rna_np).float()

        # --- material ---
        mat_np = get_material_feats(spec.material)
        mat_t = torch.from_numpy(mat_np).float()

        # --- gravity env one-hot ---
        env_np = self._gravity_to_onehot(spec.gravity)
        env_t = torch.from_numpy(env_np).float()

        # --- target ---
        if spec.target_path is not None:
            tgt_np = load_tiff_stack(spec.target_path)
            y_t = torch.from_numpy(tgt_np).float()
        else:
            if spec.target_scalar is None:
                raise ValueError(
                    "TrajectorySpec must have target_path or target_scalar."
                )
            y_t = torch.tensor(spec.target_scalar, dtype=torch.float32)

        return {
            "morphology": X_t,   # [T,1,Z,H,W]
            "rna":        rna_t, # [d_rna]
            "material":   mat_t, # [d_mat]
            "env":        env_t, # [2]
            "target":     y_t,   # [1,Z,H,W] or scalar
        }


# ----------------------------------------------------------------------
# 4. ConvLSTM3D cell + conditioned ConvLSTM model
# ----------------------------------------------------------------------

class ConvLSTM3DCell(nn.Module):
    """
    Standard ConvLSTM cell for 3D volumes (Z,H,W).
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv3d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        c_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if h_prev is None or c_prev is None:
            B, _, Z, H, W = x_t.shape
            device = x_t.device
            h_prev = torch.zeros(
                B, self.hidden_channels, Z, H, W,
                device=device, dtype=x_t.dtype
            )
            c_prev = torch.zeros_like(h_prev)

        combined = torch.cat([x_t, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class BiofilmConvLSTMModel(nn.Module):
    """
    ConvLSTM backbone conditioned on RNA + material + gravity.

    forward(X, rna_vec, mat_vec, env_vec) -> predicted 3D stack.
    """

    def __init__(
        self,
        rna_dim: int,
        mat_dim: int,
        env_dim: int = 2,
        hidden_channels: int = 32,
        aux_dim: int = 32,
    ) -> None:
        super().__init__()

        self.cell = ConvLSTM3DCell(in_channels=1, hidden_channels=hidden_channels)

        self.aux_mlp = nn.Sequential(
            nn.Linear(rna_dim + mat_dim + env_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, aux_dim),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv3d(
            in_channels=hidden_channels + aux_dim,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        X: torch.Tensor,
        rna_vec: torch.Tensor,
        mat_vec: torch.Tensor,
        env_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        X      : [B,T,1,Z,H,W]
        rna_vec: [B,d_rna]
        mat_vec: [B,d_mat]
        env_vec: [B,2]
        """
        B, T, C, Z, H, W = X.shape
        h, c = None, None

        for t in range(T):
            x_t = X[:, t]  # [B,1,Z,H,W]
            h, c = self.cell(x_t, h, c)

        aux_in = torch.cat([rna_vec, mat_vec, env_vec], dim=-1)
        aux_emb = self.aux_mlp(aux_in)   # [B,aux_dim]

        aux_expanded = aux_emb.view(B, aux_emb.shape[1], 1, 1, 1)
        aux_expanded = aux_expanded.expand(-1, -1, Z, H, W)

        h_cond = torch.cat([h, aux_expanded], dim=1)
        y_pred = self.out_conv(h_cond)   # [B,1,Z,H,W]
        return y_pred
