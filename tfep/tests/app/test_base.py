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

import tfep.nn.flows
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

class MockPotential(torch.nn.Module):
    """Mock potential to test TFEPMapBase."""

    energy_unit = UNITS.joule/UNITS.mole

    def forward(self, x):
        return x.sum(dim=1)


class TFEPMap(TFEPMapBase):
    """Light-weight TFEPMap for unit testing.

    fail_after is the number of global_steps after which an exception is raised
    to simulate a crashed training.

    """

    def __init__(self, tfep_logger_dir_path, fail_after=None):
        super().__init__(
            potential_energy_func=MockPotential(),
            topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            temperature=300*UNITS.kelvin,
            tfep_logger_dir_path=tfep_logger_dir_path,
            batch_size=2,
        )
        self.fail_after = fail_after
        self.epoch_visited_samples = None
        self.n_processed_steps = 0

    def configure_flow(self):
        n_dofs = self.dataset.n_atoms * 3
        return tfep.nn.flows.MAF(dimension_in=n_dofs)

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

def test_resuming_mid_epoch():
    """Test that resuming mid-epoch works correctly.

    In particular, that training does not restart from the next epoch, and that
    the training does not process the same samples twice in the restarted epoch.

    """
    # The trajectory should have a total of 5 frames so 3 step per epoch.
    fail_after = 4
    max_epochs = 2

    def init_map_trainer(tmp_dir_path):
        tfep_map = TFEPMap(
            tfep_logger_dir_path=tmp_dir_path,
            fail_after=fail_after  # steps
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
