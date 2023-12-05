#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.app.base``.
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
from tfep.utils.misc import atom_to_flattened_indices, flattened_to_atom

from tfep.app.base import TFEPMapBase


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

SCRIPT_DIR_PATH = os.path.dirname(__file__)
CHLOROMETHANE_PDB_FILE_PATH = os.path.join(SCRIPT_DIR_PATH, '..', 'data', 'chloro-fluoromethane.pdb')

UNITS = pint.UnitRegistry()


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

_old_default_dtype = None

def setup_module(module):
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)


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

    def __init__(self, energy_unit=None, flow='maf', fail_after=None, **kwargs):
        super().__init__(
            potential_energy_func=MockPotential(energy_unit=energy_unit),
            topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
            temperature=300*UNITS.kelvin,
            batch_size=2,
            **kwargs,
        )
        self._flow_type = flow
        self.fail_after = fail_after
        self.epoch_visited_samples = None
        self.n_processed_steps = 0

    def configure_flow(self):
        # MAF.
        if self._flow_type == 'maf':
            if self.conditioning_atom_indices is None:
                conditioning_indices = None
            else:
                conditioning_indices=atom_to_flattened_indices(self.conditioning_atom_indices)

            return tfep.nn.flows.MAF(
                dimension_in=3*(self.dataset.n_atoms-self.n_fixed_atoms),
                conditioning_indices=conditioning_indices,
                initialize_identity=False,
            )

        # Continuous.
        if (self.conditioning_atom_indices is not None) or (self.fixed_atom_indices is not None):
            raise NotImplementedError()
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

@pytest.mark.parametrize('mapped_atoms,conditioning_atoms,expected_mapped,expected_conditioning,expected_fixed,expected_mapped_remove_fixed,expected_conditioning_remove_fixed', [
    # If neither mapped nor conditioning are given, all atoms are mapped.
    (None, None, list(range(6)), None, None, list(range(6)), None),
    # If only mapped is given, the non-mapped are fixed.
    ('index 0:1', None, [0, 1], None, [2, 3, 4, 5], [0, 1], None),
    ([2, 5], None, [2, 5], None, [0, 1, 3, 4], [0, 1], None),
    ('index 1:5', None, [1, 2, 3, 4, 5], None, [0], [0, 1, 2, 3, 4], None),
    ([0, 2, 3, 4, 5], None, [0, 2, 3, 4, 5], None, [1], [0, 1, 2, 3, 4], None),
    # If only conditioning is given, the non-conditioning are mapped.
    (None, 'index 3:4', [0, 1, 2, 5], [3, 4], None, [0, 1, 2, 5], [3, 4]),
    (None, torch.tensor([0, 4, 5]), [1, 2, 3], [0, 4, 5], None, [1, 2, 3], [0, 4, 5]),
    # If both are given, everything else is fixed.
    ('index 3:4', [1], [3, 4], [1], [0, 2, 5], [1, 2], [0]),
    ([1, 4], [2, 5], [1, 4], [2, 5], [0, 3], [0, 2], [1, 3]),
    ([0, 4], [2, 3, 5], [0, 4], [2, 3, 5], [1], [0, 3], [1, 2, 4]),
])
def test_atom_selection(
        mapped_atoms,
        conditioning_atoms,
        expected_mapped,
        expected_conditioning,
        expected_fixed,
        expected_mapped_remove_fixed,
        expected_conditioning_remove_fixed,
):
    """Mapped, conditioning, and fixed atoms are selected correctly."""
    tfep_map = TFEPMap(
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
    )
    tfep_map.setup()

    # Convert to tensor for comparison.
    for expected_indices, tfep_indices in zip([expected_mapped_remove_fixed, expected_conditioning_remove_fixed, expected_fixed],
                                              [tfep_map.mapped_atom_indices, tfep_map.conditioning_atom_indices, tfep_map.fixed_atom_indices]):
        if expected_indices is None:
            assert tfep_indices is None
        else:
            assert torch.all(tfep_indices == torch.tensor(expected_indices))

    # Generate random positions.
    batch_size, n_features = 2, 18
    x = torch.randn(batch_size, n_features)

    # Test forward and inverse.
    y, log_det_J = tfep_map(x)
    x_inv, log_det_J_inv = tfep_map.inverse(y)
    assert torch.allclose(x, x_inv)

    # The flow must take care of only mapped and conditioning.
    # The fixed atoms are handled automatically.
    x = flattened_to_atom(x)
    y = flattened_to_atom(y)
    assert not torch.allclose(x[:, expected_mapped], y[:, expected_mapped])
    if expected_conditioning is not None:
        assert torch.allclose(x[:, expected_conditioning], y[:, expected_conditioning])
    if expected_fixed is not None:
        assert torch.allclose(x[:, expected_fixed], y[:, expected_fixed])


def test_no_mapped_atoms():
    """If only conditioning is given and there are no atoms left to map, an error is raised."""
    tfep_map = TFEPMap(conditioning_atoms=list(range(6)))
    with pytest.raises(ValueError, match='no atoms to map'):
        tfep_map.setup()


def test_overlapping_atom_selections():
    """If mapped and conditioning atoms overlap, an error is raised."""
    tfep_map = TFEPMap(mapped_atoms=[0, 1, 2], conditioning_atoms=[2, 3])
    with pytest.raises(ValueError, match='overlapping atoms'):
        tfep_map.setup()


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
