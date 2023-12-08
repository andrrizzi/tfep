#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Base ``LightningModule`` class to implement TFEP maps."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
from typing import Optional, Union

import pint
import torch

from tfep.app.base import TFEPMapBase
import tfep.nn.flows
from tfep.utils.misc import atom_to_flattened_indices


# =============================================================================
# TFEP MAP BASE CLASS
# =============================================================================

class CartesianMAFMap(TFEPMapBase):
    """A TFEP map using a masked autoregressive flow in Cartesian coordinates.

    The class divides the atoms of the entire system in mapped, conditioning,
    and fixed. Mapped atoms are defined as those that the flow maps. Conditioning
    atoms are not mapped but are given as input to the flow to condition the
    mapping. Fixed atoms are instead ignored.

    The flow uses three atoms to fix the frame of reference and remove the global
    rototranslational degrees of freedom from the mapped DOFs.

    .. warning::

        Currently, this class is not multi-process or thread safe. Running with
        multiple processes may result in the corrupted logging of the potentials
        and Jacobians.

    Examples
    --------

    >>> from tfep.potentials.psi4 import PotentialPsi4
    >>> units = pint.UnitRegistry()
    >>>
    >>> tfep_map = CartesianMAFMap(
    ...     potential_energy_func=PotentialPsi4(name='mp2'),
    ...     topology_file_path='path/to/topology.psf',
    ...     coordinates_file_path='path/to/trajectory.dcd',
    ...     temperature=300*units.kelvin,
    ...     reference_atoms=[13, 15, 18],
    ...     mapped_atoms='resname MOL',  # MDAnalysis selection syntax.
    ...     conditioning_atoms=range(10),
    ...     batch_size=64,
    ... )
    >>>
    >>> # Train the flow and save the potential energies.
    >>> trainer = lightning.Trainer()
    >>> trainer.fit(tfep_map)  # doctest: +SKIP

    """

    def __init__(
            self,
            potential_energy_func: torch.nn.Module,
            topology_file_path: str,
            coordinates_file_path: Union[str, Sequence[str]],
            temperature: pint.Quantity,
            reference_atoms: Sequence[int],
            batch_size: int = 1,
            mapped_atoms: Optional[Union[Sequence[int], str]] = None,
            conditioning_atoms: Optional[Union[Sequence[int], str]] = None,
            tfep_logger_dir_path: str = 'tfep_logs',
            n_maf_layers: int = 6,
            **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        potential_energy_func : torch.nn.Module
            A PyTorch module encapsulating the target potential energy function
            (e.g. :class:`tfep.potentials.psi4.PotentialASE`).
        topology_file_path : str
            The path to the topology file. The file can be in `any format supported
            by MDAnalysis <https://docs.mdanalysis.org/stable/documentation_pages/topology/init.html#supported-topology-formats>`__
            which is automatically detected from the file extension.
        coordinates_file_path : str or Sequence[str]
            The path(s) to the trajectory file(s). If a sequence of files is given,
            the trajectories are concatenated into a single large dataset. The
            file(s) can be in `any format supported by MDAnalysis <https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html#id2>`__
            which is automatically detected from the file extension.
        temperature : pint.Quantity
            The temperature of the ensemble.
        reference_atoms : Sequence[int]
            A triplet of atom indices ``(center, axis, plane)``. The frame of
            reference of the mapping will be fixed so that the ``center``-th atom
            will be at the origin, the  ``axis``-th atom will lay on the x axis
            , and the ``plane``-th atom will lay on the plane spanned by x and y
            axes. The z-axis will be set as the cross product of x and y.
        batch_size : int, optional
            The batch size.
        mapped_atoms : Sequence[int] or str, optional
            The indices (0-based) of the atoms to map or a selection string in
            MDAnalysis syntax. If not passed, all atoms that are not conditioning
            are mapped (i.e., all atoms are mapped if also ``conditioning_atoms``
            is not given.
        conditioning_atoms : Sequence[int] or str, optional
            The indices (0-based) of the atoms conditioning the mapping or a
            selection string in MDAnalysis syntax. If not passed, no atom will
            condition the map.
        tfep_logger_dir_path : str, optional
            The path where to save TFEP-related information (potential energies,
            sample indices, etc.).
        n_maf_layers : int, optional
            The number of MAF layers.
        **kwargs
            Other keyword arguments to pass to the constructor of :class:`tfep.nn.flows.MAF`.


        See Also
        --------
        `MDAnalysis Universe object <https://docs.mdanalysis.org/2.6.1/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe>`_

        """
        super().__init__(
            potential_energy_func=potential_energy_func,
            topology_file_path=topology_file_path,
            coordinates_file_path=coordinates_file_path,
            temperature=temperature,
            tfep_logger_dir_path=tfep_logger_dir_path,
            batch_size=batch_size,
        )
        self.reference_atoms = reference_atoms
        self.save_hyperparameters('n_maf_layers')
        self.kwargs = kwargs

    def configure_flow(self):
        """Initialize the normalizing flow.

        Returns
        -------
        flow : torch.nn.Module
            The normalizing flow.

        """
        if len(self.mapped_atom_indices) < 3:
            raise ValueError("There must be at least 3 mapped atoms to define the frame of reference")

        # Check that center, axis and plane atoms are different.
        try:  # Tensor
            unique = self.reference_atoms.unique()
        except AttributeError:
            unique = set(self.reference_atoms)
        if len(unique) != 3:
            raise ValueError("center, axis, and plane atoms must be different")

        # Make tensor version of reference atoms to simplify code.
        center_atom_idx, axis_atom_idx, plane_atom_idx = self.reference_atoms
        center_atom_idx = torch.tensor([center_atom_idx])
        axis_atom_idx = torch.tensor([axis_atom_idx])
        plane_atom_idx = torch.tensor([plane_atom_idx])

        # center atom refers to the atom index after the fixed atoms have been
        # removed, while the axis/plane atom indices must account for the removed
        # fixed and center atoms. First remove the fixed atoms from all, which
        # is needed to update the conditioning atoms. We will fix the axis/plane
        # atoms later.
        if self.fixed_atom_indices is not None:
            center_atom_idx = center_atom_idx - torch.searchsorted(self.fixed_atom_indices, center_atom_idx)
            axis_atom_idx = axis_atom_idx - torch.searchsorted(self.fixed_atom_indices, axis_atom_idx)
            plane_atom_idx = plane_atom_idx - torch.searchsorted(self.fixed_atom_indices, plane_atom_idx)

        # Now we can test whether this are mapped atoms. mapped_atom_indices
        # refers to the indices after the fixed atoms have been removed.
        if center_atom_idx[0] not in self.mapped_atom_indices:
            raise ValueError("center atom is not a mapped atom")
        if axis_atom_idx[0] not in self.mapped_atom_indices:
            raise ValueError("axis atom is not a mapped atom")
        if plane_atom_idx[0] not in self.mapped_atom_indices:
            raise ValueError("plane atom is not a mapped atom")

        # Determine conditioning indices.
        if self.conditioning_atom_indices is None:
            conditioning_indices = None
        else:
            # Convert from atom indices to DOF indices.
            conditioning_indices = atom_to_flattened_indices(self.conditioning_atom_indices)

            # Conditioning DOFs indices already account for the fixed atoms, but we need
            # to account for the removed rototranslational DOFs.
            # The center atom accounts for 3 lost DOFs.
            atom_indices_diff = 3 * torch.searchsorted(center_atom_idx, self.conditioning_atom_indices)
            # The axis atom accounts for 2 lost DOFs.
            atom_indices_diff = atom_indices_diff + 2 * torch.searchsorted(axis_atom_idx, self.conditioning_atom_indices)
            # The plane atom accounts for 1 lost DOFs.
            atom_indices_diff = atom_indices_diff + torch.searchsorted(plane_atom_idx, self.conditioning_atom_indices)

            # Now convert from atom_indices to DOF indices.
            dof_indices_diff = atom_indices_diff.expand(3, -1).T.flatten()
            conditioning_indices = conditioning_indices - dof_indices_diff

        # Update the index of axis/plane atom.
        if center_atom_idx < axis_atom_idx:
            axis_atom_idx = axis_atom_idx - 1
        if center_atom_idx < plane_atom_idx:
            plane_atom_idx = plane_atom_idx - 1

        # Remove the 6 rototranslational degrees of freedom from the number of
        # DOFs mapped by the MAF layers.
        dimension_in = 3*(self.dataset.n_atoms-self.n_fixed_atoms) - 6

        # Build MAF layers.
        maf_layers = []
        for layer_idx in range(self.hparams.n_maf_layers):
            maf_layers.append(tfep.nn.flows.MAF(
                dimension_in=dimension_in,
                conditioning_indices=conditioning_indices,
                degrees_in='input' if (layer_idx%2 == 0) else 'reversed',
                **self.kwargs,
            ))
        flow = tfep.nn.flows.SequentialFlow(*maf_layers)

        # Add rotational invariance.
        flow = tfep.nn.flows.OrientedFlow(
            flow,
            axis_point_idx=axis_atom_idx[0],
            plane_point_idx=plane_atom_idx[0],
        )

        # Add translational invariance.
        flow = tfep.nn.flows.CenteredCentroidFlow(
            flow,
            space_dimension=3,
            subset_point_indices=center_atom_idx,
        )

        return flow
