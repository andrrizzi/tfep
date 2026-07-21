#!/usr/bin/env python

"""Shell-conditioned trajectory dataset utilities."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

from tfep.io.dataset.traj import TrajectoryDataset
from tfep.solvation.mapping import minimum_image_displacements, residue_sort_key


class SolvationShellPermutingTrajectoryDataset(TrajectoryDataset):
    """Dataset that permutes whole water residues to keep nearest-shell slots fixed.

    The first ``k1 + k2`` water residues (in topology order) are treated as shell
    slots. At each frame, the nearest waters to the specified center are moved to
    those slots by permutation of entire residue coordinates.
    """

    def __init__(
        self,
        universe,
        *,
        center_atoms: Union[str, Sequence[int]],
        water_selection: str,
        oxygen_names: Sequence[str],
        k1: int,
        k2: int,
    ):
        super().__init__(
            universe=universe,
            return_dataset_sample_index=True,
            return_trajectory_sample_index=True,
        )

        self.k1 = int(k1)
        self.k2 = int(k2)
        self.k_total = int(k1) + int(k2)
        if self.k_total <= 0:
            raise ValueError("SolvationShellPermutingTrajectoryDataset requires k1+k2 > 0")

        self._water_selection = str(water_selection)
        self._oxygen_names = list(oxygen_names)

        if isinstance(center_atoms, str):
            center_group = self.universe.select_atoms(center_atoms)
            if len(center_group) == 0:
                raise ValueError(f"Shell center selection returned 0 atoms: {center_atoms!r}")
            self._center_indices = center_group.indices.astype(int)
        else:
            center_idx = np.asarray(list(center_atoms), dtype=int)
            if center_idx.size == 0:
                raise ValueError("Shell center atom indices are empty")
            self._center_indices = center_idx

        water_atoms = self.universe.select_atoms(self._water_selection)
        if len(water_atoms) == 0:
            raise ValueError(f"Water selection returned 0 atoms: {self._water_selection!r}")

        residues = list(water_atoms.residues)
        residues.sort(key=residue_sort_key)
        self._water_residues = residues

        if self.k_total > len(self._water_residues):
            raise ValueError(
                f"k1+k2={self.k_total} > number of water residues={len(self._water_residues)} "
                f"selected by {self._water_selection!r}"
            )

        self._res_atom_indices = []
        self._res_oxygen_index = []

        for residue in self._water_residues:
            atom_idx = residue.atoms.indices.astype(int)
            self._res_atom_indices.append(atom_idx)

            oxygen = None
            for name in self._oxygen_names:
                ag = residue.atoms.select_atoms(f"name {name}")
                if len(ag) > 0:
                    oxygen = int(ag.indices[0])
                    break
            if oxygen is None:
                ag = residue.atoms.select_atoms("element O")
                if len(ag) > 0:
                    oxygen = int(ag.indices[0])
            if oxygen is None:
                raise RuntimeError(
                    f"Could not find oxygen atom in water residue {residue} using names {self._oxygen_names}"
                )
            self._res_oxygen_index.append(oxygen)

        self._res_oxygen_index = np.asarray(self._res_oxygen_index, dtype=int)

    def __getitem__(self, idx):
        timestep = self.get_timestep(idx)
        positions = np.array(timestep.positions, copy=True)

        center = positions[self._center_indices].mean(axis=0)
        oxygen_pos = positions[self._res_oxygen_index]
        displacements = minimum_image_displacements(oxygen_pos - center, timestep.dimensions)
        distances2 = np.sum(displacements**2, axis=1)

        n_waters = len(self._water_residues)
        k_total = self.k_total

        if k_total >= n_waters:
            perm = np.argsort(distances2)
        else:
            nearest = np.argpartition(distances2, k_total - 1)[:k_total]
            nearest = nearest[np.argsort(distances2[nearest])]
            mask = np.ones(n_waters, dtype=bool)
            mask[nearest] = False
            rest = np.nonzero(mask)[0]
            perm = np.concatenate([nearest, rest], axis=0)

        permuted = positions.copy()
        for target_idx, source_idx in enumerate(perm):
            target_atoms = self._res_atom_indices[target_idx]
            source_atoms = self._res_atom_indices[source_idx]
            permuted[target_atoms] = positions[source_atoms]

        sample = {
            "positions": torch.tensor(np.ravel(permuted), dtype=torch.get_default_dtype())
        }

        if timestep.dimensions is not None:
            sample["dimensions"] = torch.tensor(timestep.dimensions, dtype=torch.get_default_dtype())

        for aux_name, aux_info in self.universe.trajectory.ts.aux.items():
            sample[aux_name] = torch.tensor(aux_info)

        if self.return_dataset_sample_index:
            sample["dataset_sample_index"] = int(idx)

        if self.return_trajectory_sample_index:
            if self.trajectory_sample_indices is None:
                trajectory_sample_index = int(idx)
            else:
                trajectory_sample_index = int(self.trajectory_sample_indices[int(idx)])
            sample["trajectory_sample_index"] = trajectory_sample_index
        else:
            trajectory_sample_index = int(idx) if self.trajectory_sample_indices is None else int(self.trajectory_sample_indices[int(idx)])

        if self._log_weights is not None:
            sample["log_weights"] = torch.tensor(
                self._log_weights[int(trajectory_sample_index)],
                dtype=torch.get_default_dtype(),
            )

        return sample
