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

    Optionally, the flow can map the atoms in a relative frame of reference
    which based on the position of an ``origin_atom`` and two ``axes_atoms``
    that determine the origin and the orientation of the axes, respectively.

    The class further supports logging the potential energies computed during
    training (required for the multimap TFEP analysis) and mid-epoch resuming.

    .. warning::

        Currently, this class is not multi-process or thread safe. Running with
        multiple processes may result in the corrupted logging of the potentials
        and Jacobians.

    See Also
    --------
    :class:`tfep.app.base.TFEPMapBase`
        The base class for TFEP maps with more detailed explanations of how the
        relative reference frame and the division in mapped/conditioning/fixed
        atoms work.

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
    ...     batch_size=64,
    ...     mapped_atoms='resname MOL',  # MDAnalysis selection syntax.
    ...     conditioning_atoms=range(10, 20),
    ...     origin_atom=12,  # Fix the origin of the relative reference frame on atom 123.
    ...     axes_atoms=[13, 16],  # Determine the orientation of the reference frame.
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
            batch_size: int = 1,
            mapped_atoms: Optional[Union[Sequence[int], str]] = None,
            conditioning_atoms: Optional[Union[Sequence[int], str]] = None,
            origin_atom: Optional[Union[int, str]] = None,
            axes_atoms: Optional[Union[Sequence[int], str]] = None,
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
        origin_atom : int or str or None, optional
            The index (0-based) or a selection string in MDAnalysis syntax of an
            atom on which to center the origin of the relative frame of reference.
            While this atom affects the mapping of the mapped atoms, its position
            will be constrained during the mapping, and thus it must be a conditioning
            atom by definition.
        axes_atoms : Sequence[int] or str or None, optional
            A pair of indices (0-based) or a selection string in MDAnalysis syntax
            for the two atoms determining the relative frame of reference. The
            ``axes_atoms[0]``-th atom will lay on the ``x`` axis , and the
            ``axes_atoms[1]``-th atom will lay on the plane spanned by the ``x``
            and ``y`` axes. The ``z`` axis will be set as the cross product of
            ``x`` and ``y``.

            These atoms can be either conditioning or mapped. ``axes_atoms[0]``
            has only 1 degree of freedom (DOF) while ``axes_atoms[1]`` has 2.
            Whether these DOFs are mapped or not depends on whether their atoms
            are indicated as mapped or conditioning, respectively.
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
            batch_size=batch_size,
            mapped_atoms=mapped_atoms,
            conditioning_atoms=conditioning_atoms,
            origin_atom=origin_atom,
            axes_atoms=axes_atoms,
            tfep_logger_dir_path=tfep_logger_dir_path,

        )
        self.save_hyperparameters('n_maf_layers')
        self.kwargs = kwargs

    def configure_flow(self) -> torch.nn.Module:
        """Initialize the normalizing flow.

        Returns
        -------
        flow : torch.nn.Module
            The normalizing flow.

        """
        dimension_in = self.n_mapped_dofs
        conditioning_indices = self.get_conditioning_indices(idx_type='dof', remove_constrained=True)
        if conditioning_indices is not None:
            dimension_in += len(conditioning_indices)

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

        return flow
