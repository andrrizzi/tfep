#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.io.sampler``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import tempfile

import lightning
import pint
import pytest
import torch

import tfep.nn.dynamics
import tfep.nn.flows
from tfep.potentials.base import PotentialBase
from tfep.app.base import TFEPMapBase


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

SCRIPT_DIR_PATH = os.path.dirname(__file__)
CHLOROMETHANE_PDB_FILE_PATH = os.path.join(SCRIPT_DIR_PATH, '..', 'data', 'chloro-fluoromethane.pdb')

UNITS = pint.UnitRegistry()


# =============================================================================
# TEST UTILITIES
# =============================================================================

class MockPotential(PotentialBase):
    """Mock potential to test TFEPMapBase."""

    DEFAULT_ENERGY_UNIT = 'kcal'
    DEFAULT_POSITION_UNIT = 'angstrom'

    def forward(self, x):
        return x.sum(dim=1)


class TFEPMap(TFEPMapBase):
    """Light-weight TFEPMap for unit testing.

    fail_after is the number of global_steps after which an exception is raised
    to simulate a crashed training.

    flow can be 'maf' or 'continuous'.

    """

    def __init__(self, energy_unit, tfep_logger_dir_path, flow='maf', fail_after=None):
        super().__init__(
            potential_energy_func=MockPotential(energy_unit=energy_unit),
            topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            temperature=300*UNITS.kelvin,
            tfep_logger_dir_path=tfep_logger_dir_path,
            batch_size=2,
        )
        self._flow_type = flow
        self.fail_after = fail_after
        self.epoch_visited_samples = None
        self.n_processed_steps = 0

    def configure_flow(self):
        n_dofs = self.dataset.n_atoms * 3

        # MAF.
        if self._flow_type == 'maf':
            return tfep.nn.flows.MAF(dimension_in=n_dofs)

        # Continuous.
        egnn_dynamics = tfep.nn.dynamics.EGNNDynamics(
            particle_types=torch.tensor([0, 1, 2, 2, 2, 3]),
            r_cutoff=6.0,
            time_feat_dim=2,
            node_feat_dim=4,
            distance_feat_dim=4,
            n_layers=2,
        )
        return tfep.nn.flows.ContinuousFlow(
            dynamics=egnn_dynamics,
            solver='euler',
            solver_options={'step_size': 1.0},
        )

    def training_step(self, batch, batch_idx):
        # Call the actual training step.
        loss = super().training_step(batch, batch_idx)

        # Simulate a crashed training.
        if self.trainer.global_step == self.fail_after:
            if self.epoch_visited_samples is None:
                # This is a resumed run. Verify this the trainer epoch is correct.
                assert self.trainer.current_epoch == (self.fail_after-1) // len(self._stateful_batch_sampler)
            else:
                # Interrupt.
                raise RuntimeError()

        # Keep track of the samples that have been seen so far.
        sample_indices = batch['dataset_sample_index'].detach().tolist()
        if (self.epoch_visited_samples is None or
                        self.trainer.global_step % len(self._stateful_batch_sampler) == 0):
            # New epoch.
            self.epoch_visited_samples = sample_indices
        else:
            self.epoch_visited_samples.extend(sample_indices)

        # Update total number of steps.
        self.n_processed_steps += 1

        return loss


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('energy_unit', [None, UNITS.joule, UNITS.kcal/UNITS.mole])
@pytest.mark.parametrize('flow', ['maf', 'continuous'])
def test_resuming_mid_epoch(energy_unit, flow):
    """Test that resuming mid-epoch works correctly.

    In particular, that training does not restart from the next epoch, and that
    the training does not process the same samples twice in the restarted epoch.

    """
    # Check if the optional dependency torchdiffeq required for continuous flows is present.
    if flow == 'continuous':
        try:
            import torchdiffeq
        except ImportError:
            pytest.skip('The torchdiffeq package is required for continuous flows.')

    # The trajectory should have a total of 5 frames so 3 step per epoch.
    fail_after = 4
    max_epochs = 2

    def init_map_trainer(tmp_dir_path):
        tfep_map = TFEPMap(
            energy_unit=energy_unit,
            tfep_logger_dir_path=tmp_dir_path,
            flow=flow,
            fail_after=fail_after,  # steps
        )

        # This first training should fail.
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=tmp_dir_path,
            save_last=True,
            every_n_train_steps=1,
        )
        trainer = lightning.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=max_epochs,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_dir_path
        )
        return tfep_map, trainer

    # Store all logging files in a temporary directory.
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        tfep_map, trainer = init_map_trainer(tmp_dir_path)
        try:
            trainer.fit(tfep_map)
        except RuntimeError:
            assert trainer.global_step == tfep_map.fail_after

        # Check the visited samples.
        visited_samples1 = set(tfep_map.epoch_visited_samples)
        n_batches_per_epoch = len(tfep_map._stateful_batch_sampler)
        n_processed_batches = tfep_map.fail_after % n_batches_per_epoch
        assert len(visited_samples1) == n_processed_batches * tfep_map.hparams.batch_size

        # Now simulate resuming.
        tfep_map, trainer = init_map_trainer(tmp_dir_path)
        trainer.fit(tfep_map, ckpt_path='last')

        # Check that the visited samples on resuming do not overlap.
        visited_samples2 = set(tfep_map.epoch_visited_samples)

        # Total number of steps amount to max_epochs.
        assert tfep_map.n_processed_steps + fail_after == max_epochs * n_batches_per_epoch

        # During the interrupted epoch, the training visited all samples and only once.
        assert visited_samples1.union(visited_samples2) == set(range(len(tfep_map.dataset)))
        assert len(visited_samples1.intersection(visited_samples2)) == 0
