"""Ozone bidirectional TMBAR demo map.

This is a small example map that trains a single invertible flow bidirectionally
between two OpenMM potentials, using :class:`tfep.app.base.TMBARMapBase`.

The goal of this module is to keep the ozone demo *thin*: reusable components
such as FlowFactory and BAR-like regularizers live in the core library.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

import tfep
from tfep.app.base import TMBARMapBase
from tfep.nn.flows import CenteredCentroidFlow, OrientedFlow
from tfep.nn.flows.factory import FlowFactory, FlowSpec
from tfep.potentials.base import MultiStatePotential
from tfep.potentials.openmm import OpenMMPotential
from tfep.regularizers.bar import BARLikeRegularizer


class OzoneBidirectionalTMBARMap(TMBARMapBase):
    """Bidirectional two-state TMBAR map for ozone."""

    def __init__(
        self,
        potential_0: OpenMMPotential,
        potential_1: OpenMMPotential,
        topology_file_path: str,
        coordinates_file_path: Union[str, Sequence[str]],
        coordinates_file_path_2: Union[str, Sequence[str]],
        temperature,
        *,
        batch_size: int = 1,
        mapped_atoms: Optional[Union[Sequence[int], str]] = None,
        conditioning_atoms: Optional[Union[Sequence[int], str]] = None,
        origin_atom: Optional[Union[int, str]] = None,
        axes_atoms: Optional[Union[Sequence[int], str]] = None,
        tfep_logger_dir_path: str = "tfep_logs",
        dataloader_kwargs: Optional[dict] = None,
        flow_spec: Optional[FlowSpec] = None,
        bar_reg_weight: float = 0.0,
        train_indices: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        # Optional minibatch regularizer.
        bar_reg = BARLikeRegularizer(lam=float(bar_reg_weight)) if float(bar_reg_weight) > 0 else None

        super().__init__(
            potential_energy_func=MultiStatePotential(potential_0, potential_1),
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            coordinates_file_path_2=coordinates_file_path_2,
            temperature=temperature,
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,
            dataloader_kwargs=dataloader_kwargs,
            n_states=2,
            state_names=["state0_ref", "state1_tgt"],
            bar_regularizer=bar_reg,
            **kwargs,
        )

        self._dataset_full0 = None
        self._dataset_full1 = None
        self._pending_train_indices = np.asarray(list(train_indices), dtype=int) if train_indices is not None else None
        self._train_indices = None
        self._flow_spec = flow_spec

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        # Keep references to the full datasets.
        if self._dataset_full0 is None and getattr(self, "dataset", None) is not None:
            self._dataset_full0 = self.dataset
        if self._dataset_full1 is None and getattr(self, "dataset_2", None) is not None:
            self._dataset_full1 = self.dataset_2

        # Apply a common subset to both datasets (e.g., for k-fold CV).
        if (
            self._pending_train_indices is not None
            and self._dataset_full0 is not None
            and self._dataset_full1 is not None
        ):
            idx = np.asarray(self._pending_train_indices, dtype=int)
            self._train_indices = idx
            self.dataset = torch.utils.data.Subset(self._dataset_full0, idx.tolist())
            self.dataset_2 = torch.utils.data.Subset(self._dataset_full1, idx.tolist())
            self._pending_train_indices = None

    def configure_flow(self):
        """Create the flow via the factory.

        The factory supports both Cartesian MAFs (with optional centering/orienting
        wrappers) and the triatomic Z-matrix flow used in the ozone toy demo.
        """
        spec = self._flow_spec
        if spec is None:
            # Reasonable default for the ozone demo.
            spec = FlowSpec(name="triatomic_zmat", kwargs={}, wrappers=[])

        if spec.name == "cartesian_maf":
            conditioning_indices = self.get_conditioning_indices(
                idx_type="dof",
                remove_fixed=True,
                remove_reference=True,
            )
            n_cond = int(conditioning_indices.numel()) if conditioning_indices is not None else 0
            if n_cond >= int(self.n_nonfixed_dofs):
                raise RuntimeError(
                    f"Conditioning DOFs cover all non-fixed DOFs (n_nonfixed_dofs={self.n_nonfixed_dofs}, n_conditioning={n_cond})."
                )

            reference_atom_indices = self.get_reference_atoms_indices(remove_fixed=True)
            ctx = {
                "n_nonfixed_dofs": int(self.n_nonfixed_dofs),
                "conditioning_indices": conditioning_indices,
                "reference_atom_indices": reference_atom_indices,
            }
            flow = FlowFactory.build(spec, ctx)

            # Apply reference wrappers automatically if reference atoms exist and wrappers were not explicitly specified.
            if (reference_atom_indices is not None) and (len(spec.wrappers) == 0):
                has_origin = len(reference_atom_indices) in {1, 3}
                if has_origin:
                    flow = CenteredCentroidFlow(
                        flow,
                        subset_point_indices=reference_atom_indices[:1],
                        fixed_point_idx=0,
                        space_dimension=3,
                        translate_back=True,
                    )
                has_axes = len(reference_atom_indices) > 1
                if has_axes:
                    axis_point_idx = int(reference_atom_indices[-2])
                    plane_point_idx = int(reference_atom_indices[-1])
                    flow = OrientedFlow(
                        flow,
                        axis_point_idx=axis_point_idx,
                        plane_point_idx=plane_point_idx,
                        axis="z",
                        plane="xz",
                    )

            return flow

        if spec.name == "triatomic_zmat":
            reference_atom_indices = self.get_reference_atoms_indices(remove_fixed=True)
            # Fallback if reference indices missing.
            if reference_atom_indices is None or len(reference_atom_indices) < 3:
                origin_idx, axis_idx, plane_idx = 1, 0, 2
            else:
                origin_idx = int(reference_atom_indices[0])
                axis_idx = int(reference_atom_indices[-2])
                plane_idx = int(reference_atom_indices[-1])

            ctx = {
                "origin_idx": origin_idx,
                "axis_idx": axis_idx,
                "plane_idx": plane_idx,
            }
            return FlowFactory.build(spec, ctx)

        # Delegate to factory for other custom flows.
        ctx = {
            "n_nonfixed_dofs": int(self.n_nonfixed_dofs),
            "conditioning_indices": self.get_conditioning_indices(idx_type="dof", remove_fixed=True),
            "reference_atom_indices": self.get_reference_atoms_indices(remove_fixed=True),
        }
        return FlowFactory.build(spec, ctx)
