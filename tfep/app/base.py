#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Base ``LightningModule`` class to implement TFEP maps."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Optional

import lightning
import pint
import MDAnalysis
import torch

import tfep.loss
import tfep.io.sampler


# =============================================================================
# TFEP MAP BASE CLASS
# =============================================================================

class TFEPMapBase(ABC, lightning.LightningModule):
    """A ``LightningModule`` to run TFEP calculations.

    This abstract class implements several data-related utilities that are shared
    by all TFEP maps. In particular to:

    - Support correct mid-epoch resuming.
    - Log vectorial quantities such as the calculated potential energies and the
      absolute Jacobian terms that can later be used to estimate the free energy.

    To implement a concrete class, you must implement the following methods

    - :func:`~TFEPMapBase.configure_flow`

    .. warning::

        Currently, this class is not multi-process or thread safe. Running with
        multiple processes may result in the corrupted logging of the potentials
        and Jacobians.

    Examples
    --------

    Here is an example of how to implement a working map using ``TFEPMapBase``.

    >>> class TFEPMap(TFEPMapBase):
    ...
    ...     def configure_flow(self):
    ...         # A simple 1-layer affine autoregressive flow.
    ...         n_dofs = self.dataset.n_atoms * 3
    ...         return tfep.nn.flows.MAF(dimension_in=n_dofs)
    ...

    """

    def __init__(
            self,
            potential_energy_func : torch.nn.Module,
            topology_file_path : str,
            coordinates_file_path : str | Sequence[str],
            temperature : pint.Quantity,
            tfep_logger_dir_path : str = 'tfep_logs',
            batch_size : int = 1,
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
        tfep_logger_dir_path : str, optional
            The path where to save TFEP-related information (potential energies,
            sample indices, etc.).
        batch_size : int, optional
            The batch size.

        See Also
        --------
        `MDAnalysis Universe object <https://docs.mdanalysis.org/2.6.1/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe>`_

        """
        super().__init__()

        # Make sure coordinates_file_paths is a sequence.
        if isinstance(coordinates_file_path, str):
            coordinates_file_path = [coordinates_file_path]

        # batch_size is saved as a hyperparmeter of the module.
        self.save_hyperparameters('batch_size')

        # Potential energy.
        self._potential_energy_func = potential_energy_func

        # Paths to files.
        self._topology_file_path = topology_file_path
        self._coordinates_file_path = coordinates_file_path
        self._tfep_logger_dir_path = tfep_logger_dir_path

        # Internally, rather than the temperature, save the (unitless) value of
        # kT, in the same units of energy returned by potential_energy_func.
        units = temperature._REGISTRY
        self._kT = (temperature * units.molar_gas_constant).to(potential_energy_func.energy_unit).magnitude

        # KL divergence loss function.
        self._loss_func = tfep.loss.BoltzmannKLDivLoss()

        # The following variables can be data-dependent and are thus initialized
        # dynamically in setup(), which is called by Lightning by all processes.
        # This class is not currently parallel-safe as the TFEPLogger is not, but
        # I organized the code as suggested by Lightning's docs anyway as I plan
        # to add support for this at some point.
        self.dataset : Optional[tfep.io.dataset.TrajectoryDataset] = None  #: The dataset.
        self._flow = None  # The normalizing flow model.
        self._stateful_batch_sampler = None  # Batch sampler for mid-epoch resuming.
        self._tfep_logger = None  # The logger where to save the potentials.

    def setup(self, stage: str):
        """Lightning method.

        This is executed on all processes by Lightning in DDP mode (contrary to
        ``__init__``) and can be used to initialize objects like the ``Dataset``
        and all data-dependent objects.

        """
        # Create TrajectoryDataset.
        universe = MDAnalysis.Universe(self._topology_file_path, *self._coordinates_file_path)
        self.dataset = tfep.io.TrajectoryDataset(universe=universe)

        # Create model.
        self._flow = self.configure_flow()

    @abstractmethod
    def configure_flow(self):
        """Initialize the normalizing flow.

        When this method is called, the :class:`~tfep.io.dataset.traj.TrajectoryDataset`
        is already initialized and available through the :attr:`~.TFEPMapBase.dataset`
        attribute.

        Returns
        -------
        flow : torch.nn.Module
            The normalizing flow.

        """
        pass

    def configure_optimizers(self):
        """Lightning method.

        Returns
        -------
        optimizer : torch.optim.optimizer.Optimizer
            The optimizer to use for the training.

        """
        return torch.optim.AdamW(self._flow.parameters())

    def forward(self, x):
        """Execute the normalizing flow in the forward direction.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates for the flow.


        Returns
        -------
        y : torch.Tensor
            Mapped coordinates for the flow.
        log_det_J : torch.Tensor
            Logarithm of the absolute value of the Jacobian determinant.

        """
        return self._flow(x)

    def inverse(self, y):
        """Execute the normalizing flow in the inverse direction.

        Parameters
        ----------
        y : torch.Tensor
            Mapped coordinates for the flow.


        Returns
        -------
        x : torch.Tensor
            Input coordinates for the flow.
        log_det_J : torch.Tensor
            Logarithm of the absolute value of the Jacobian determinant.

        """
        return self._flow.inverse(y)

    def training_step(self, batch, batch_idx):
        """Lightning method.

        Execute a training step.

        """
        x = batch['positions']

        # Forward.
        result = self._flow(x)

        # Continuous flows also return a regularization term.
        try:
            y, log_det_J = result
            reg = None
        except ValueError:
            y, log_det_J, reg = result

        # Compute potentials and loss.
        try:
            potential_y = self._potential_energy_func(y, batch['dimensions'])
        except KeyError:
            # There are no box vectors.
            potential_y = self._potential_energy_func(y)

        # Convert potentials to units of kT.
        potential_y = potential_y / self._kT

        # Convert bias to units of kT.
        try:
            log_weights = batch['log_weights']
        except KeyError:  # Unbiased simulation.
            try:
                log_weights = batch['bias'] / self._kT
            except KeyError:
                log_weights = None

        # Compute loss.
        loss = self._loss_func(target_potentials=potential_y, log_det_J=log_det_J, log_weights=log_weights)

        # Add regularization for continuous flows.
        if reg is not None:
            loss = loss + reg.mean()

        # Log potentials.
        self._tfep_logger.save_train_tensors(
            tensors={
                'dataset_sample_index': batch['dataset_sample_index'],
                'trajectory_sample_index': batch['trajectory_sample_index'],
                'potential': potential_y,
                'log_det_J': log_det_J,
            },
            epoch_idx=self.trainer.current_epoch,
            batch_idx=batch_idx,
        )

        # Log loss.
        self._tfep_logger.save_train_metrics(
            tensors={'loss': loss},
            epoch_idx=self.trainer.current_epoch,
            batch_idx=batch_idx,
        )

        return loss

    def train_dataloader(self):
        """Lightning method.

        Returns
        -------
        data_loader : torch.utils.data.DataLoader
            The training data loader.

        """
        # If this was loaded from a checkpoint, we need to restore the state of the batch sampler.
        if isinstance(self._stateful_batch_sampler, dict):
            batch_sampler_state = self._stateful_batch_sampler
        else:
            batch_sampler_state = None

        # Initialize the batch sampler for a correct mid-epoch resuming.
        self._stateful_batch_sampler = tfep.io.StatefulBatchSampler(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            trainer=self.trainer,
        )
        if batch_sampler_state is not None:
            self._stateful_batch_sampler.load_state_dict(batch_sampler_state)

        # Create the training dataloader.
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self._stateful_batch_sampler,
        )

        # Initialize the TFEPLogger.
        self._tfep_logger = tfep.io.TFEPLogger(
            save_dir_path=self._tfep_logger_dir_path,
            data_loader=data_loader,
        )

        return data_loader

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook.

        Used to restore the state of the batch sampler for mid-epoch resuming.

        """
        # Normally, if this is a resumed training after a crash, StatefulBatchSampler
        # won't be initialized at this point, so we just save the state.
        self._stateful_batch_sampler = checkpoint['stateful_batch_sampler']

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Lightning hook.

        Used to store the state of the batch sampler for mid-epoch resuming.

        """
        checkpoint['stateful_batch_sampler'] = self._stateful_batch_sampler.state_dict()
