#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test shared functionalities of the TFEP maps in in the ``tfep.app`` package.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import tempfile

import lightning
import numpy as np
import pint
import pytest
import torch

import tfep.nn.dynamics
import tfep.nn.flows
from tfep.utils.math import batchwise_dot
from tfep.utils.misc import flattened_to_atom

from tfep.app import TFEPMapBase, CartesianMAFMap

from . import MockPotential


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

SCRIPT_DIR_PATH = os.path.dirname(__file__)
CHLOROMETHANE_PDB_FILE_PATH = os.path.join(SCRIPT_DIR_PATH, '..', 'data', 'chloro-fluoromethane.pdb')

UNITS = pint.UnitRegistry()

# Shared kwargs for TFEPMap.__init__().
MAP_INIT_KWARGS = dict(
    potential_energy_func=MockPotential(),
    topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
    coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
    temperature=298*UNITS.kelvin,
    batch_size=2,
    initialize_identity=False,
)


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

class ContinuousTFEPMap(TFEPMapBase):
    """TFEPMap for testing inheritance from TFEPMapBase and compatibility with continuous flows."""

    def __init__(self, initialize_identity=False, **kwargs):
        super().__init__(**kwargs)
        self.initialize_identity = initialize_identity

    def configure_flow(self):
        try:
            import torchdiffeq
        except ImportError:
            pytest.skip('The torchdiffeq package is required for continuous flows.')

        # Continuous.
        if self.n_conditioning_atoms > 0:
            pytest.skip("ContinuousTFEPMap doesn't support conditioning atoms.")

        egnn_dynamics = tfep.nn.dynamics.EGNNDynamics(
            particle_types=torch.tensor([0] * self.n_mapped_atoms),
            r_cutoff=6.0,
            time_feat_dim=2,
            node_feat_dim=4,
            distance_feat_dim=4,
            n_layers=2,
            initialize_identity=self.initialize_identity,
        )
        return tfep.nn.flows.ContinuousFlow(
            dynamics=egnn_dynamics,
            solver='euler',
            solver_options={'step_size': .01},
        )


TESTED_TFEP_MAPS = [ContinuousTFEPMap, CartesianMAFMap]


# =============================================================================
# TESTS
# =============================================================================

# ContinuousTFEPMap doesn't support conditioning atoms.
@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS[1:])
@pytest.mark.parametrize('fix_origin', [False, True])
@pytest.mark.parametrize('fix_orientation', [False, True])
@pytest.mark.parametrize('mapped_atoms,conditioning_atoms,expected_mapped,expected_conditioning,expected_fixed,expected_mapped_fixed_removed,expected_conditioning_fixed_removed', [
    # If neither mapped nor conditioning are given, all atoms are mapped.
    (None, None, list(range(6)), None, None, list(range(6)), None),
    # If only mapped is given, the non-mapped are fixed.
    ('index 0:2', None, [0, 1, 2], None, [3, 4, 5], [0, 1, 2], None),
    ([2, 3, 5], None, [2, 3, 5], None, [0, 1, 4], [0, 1, 2], None),
    ('index 1:5', None, [1, 2, 3, 4, 5], None, [0], [0, 1, 2, 3, 4], None),
    (np.array([0, 2, 3, 4, 5]), None, [0, 2, 3, 4, 5], None, [1], [0, 1, 2, 3, 4], None),
    # If only conditioning is given, the non-conditioning are mapped.
    (None, 'index 3:4', [0, 1, 2, 5], [3, 4], None, [0, 1, 2, 5], [3, 4]),
    (None, torch.tensor([0, 4, 5]), [1, 2, 3], [0, 4, 5], None, [1, 2, 3], [0, 4, 5]),
    # If both are given, everything else is fixed.
    ('index 2:4', [1], [2, 3, 4], [1], [0, 5], [1, 2, 3], [0]),
    (torch.tensor([1, 4]), [2, 5], [1, 4], [2, 5], [0, 3], [0, 2], [1, 3]),
    ([0, 2, 4], np.array([3, 5]), [0, 2, 4], [3, 5], [1], [0, 1, 3], [2, 4]),
])
def test_atom_selection(
        tfep_map_cls,
        fix_origin,
        fix_orientation,
        mapped_atoms,
        conditioning_atoms,
        expected_mapped,
        expected_conditioning,
        expected_fixed,
        expected_mapped_fixed_removed,
        expected_conditioning_fixed_removed,
):
    """Mapped, conditioning, fixed, and reference frame atoms are selected and handled correctly."""
    # Select a random fixed atom to fix the rotational degrees of freedom.
    if fix_origin:
        if expected_conditioning is None:
            pytest.skip('fixing the translational DOFs require the presence of a conditioning atom')
        origin_atom = np.random.choice(expected_conditioning)
    else:
        origin_atom = None

    # Select axis and plane atoms among the remaining atoms.
    if fix_orientation:
        remaining = []
        [remaining.extend(l) for l in (expected_mapped, expected_conditioning) if l is not None]
        remaining = sorted([i for i in remaining if i != origin_atom])
        if len(remaining) < 2:
            pytest.skip('fixing the orientation of the reference frame requires at least 2 mapped or conditioning atoms.')
        axes_atoms = np.random.choice(remaining, size=2, replace=False).tolist()
    else:
        axes_atoms = None

    # Initialize the map.
    tfep_map = tfep_map_cls(
        mapped_atoms=mapped_atoms,
        conditioning_atoms=conditioning_atoms,
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
        **MAP_INIT_KWARGS,
    )
    tfep_map.setup()

    # Compare expected indices.
    for expected_indices, tfep_indices in zip(
            [
                expected_mapped_fixed_removed,
                expected_conditioning_fixed_removed,
                expected_fixed
            ], [
                tfep_map.get_mapped_indices(idx_type='atom', remove_constrained=True),
                tfep_map.get_conditioning_indices(idx_type='atom', remove_constrained=True),
                tfep_map._fixed_atom_indices
            ]):
        if expected_indices is None:
            assert tfep_indices is None
        else:
            assert torch.all(tfep_indices == torch.tensor(expected_indices))

    # Generate random positions.
    n_features = 3 * tfep_map.dataset.n_atoms
    x = torch.randn(tfep_map.hparams.batch_size, n_features, requires_grad=True)

    # Test forward and inverse.
    y, log_det_J = tfep_map(x)
    x_inv, log_det_J_inv = tfep_map.inverse(y)
    assert torch.allclose(x, x_inv)

    # Compute gradients w.r.t. the input.
    loss = y.sum()
    loss.backward()
    x_grad = flattened_to_atom(x.grad)

    # The flow must take care of mapped and conditioning, while the fixed atoms
    # are handled automatically.
    x = flattened_to_atom(x)
    y = flattened_to_atom(y)

    # Check that the map is not the identity or this test doesn't make sense.
    assert not torch.allclose(x[:, expected_mapped], y[:, expected_mapped])

    # The flow doesn't alter but still depends on the conditioning DOFs.
    if expected_conditioning is not None:
        assert torch.allclose(x[:, expected_conditioning], y[:, expected_conditioning])
        assert torch.all(~torch.isclose(x_grad[:, expected_conditioning], torch.ones(*x[:, expected_conditioning].shape)))

    # The flow doesn't alter and doesn't depend on the fixed DOFs.
    if expected_fixed is not None:
        assert torch.allclose(x[:, expected_fixed], y[:, expected_fixed])
        # The output does not depend on the fixed DOFs.
        assert torch.allclose(x_grad[:, expected_fixed], torch.ones(*x[:, expected_fixed].shape))

    # The center atom should be left untouched.
    if fix_origin:
        assert torch.allclose(x[:, origin_atom], y[:, origin_atom])

    # Check rotational frame of reference.
    if fix_orientation:
        if fix_origin:
            origin = x[:, origin_atom]
        else:
            origin = torch.zeros(tfep_map.hparams.batch_size, 3)

        # The direction center-axis should be the same (up to a flip).
        dir_01_x = torch.nn.functional.normalize(x[:, axes_atoms[0]] - origin)
        dir_01_y = torch.nn.functional.normalize(y[:, axes_atoms[0]] - origin)
        sign = torch.sign(batchwise_dot(dir_01_x, dir_01_y)).unsqueeze(1)
        assert torch.allclose(sign*dir_01_x, dir_01_y)

        # The mapped plane atom should be orthogonal to the plane-center-axis plane normal.
        dir_02_x = torch.nn.functional.normalize(x[:, axes_atoms[1]] - origin)
        plane_x = torch.cross(dir_01_x, dir_02_x)
        dir_02_y = torch.nn.functional.normalize(y[:, axes_atoms[1]] - origin)
        assert torch.allclose(batchwise_dot(plane_x, dir_02_y), torch.zeros(len(dir_02_y)))


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_no_mapped_atoms(tfep_map_cls):
    """If only conditioning is given and there are no atoms left to map, an error is raised."""
    tfep_map = tfep_map_cls(conditioning_atoms=list(range(6)), **MAP_INIT_KWARGS)
    with pytest.raises(ValueError, match='no atoms to map'):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_overlapping_atom_selections(tfep_map_cls):
    """If mapped and conditioning atoms overlap, an error is raised."""
    tfep_map = tfep_map_cls(mapped_atoms=[0, 1, 2], conditioning_atoms=[2, 3], **MAP_INIT_KWARGS)
    with pytest.raises(ValueError, match='overlapping atoms'):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_duplicate_atoms(tfep_map_cls):
    """If mapped and conditioning atoms overlap, an error is raised."""
    tfep_map = tfep_map_cls(mapped_atoms=[0, 0, 2], **MAP_INIT_KWARGS)
    with pytest.raises(ValueError, match='duplicate mapped atom'):
        tfep_map.setup()

    tfep_map = tfep_map_cls(conditioning_atoms=[3, 4, 4], **MAP_INIT_KWARGS)
    with pytest.raises(ValueError, match='duplicate conditioning atom'):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
@pytest.mark.parametrize('origin_atom,axes_atoms', [
    [0, (0, 2)],
    [0, (2, 2)],
    [0, (2, 0)],
    [None, (1, 1)],
])
def test_error_reference_frame_atoms_overlap(tfep_map_cls, origin_atom, axes_atoms):
    """An error is raised if the origin, axis, and/or plane atoms overlap."""
    tfep_map = tfep_map_cls(
        conditioning_atoms=[0],
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
        **MAP_INIT_KWARGS,
    )
    with pytest.raises(ValueError, match="must be different"):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_origin_atom_not_fixed(tfep_map_cls):
    """An error is raised if the origin atom is not a conditioning atom."""
    tfep_map = tfep_map_cls(
        mapped_atoms=range(6),
        origin_atom=[1],
        **MAP_INIT_KWARGS,
    )
    with pytest.raises(ValueError, match="is not a conditioning atom"):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_axes_atom_not_fixed(tfep_map_cls):
    """An error is raised if the origin atom is not a conditioning atom."""
    tfep_map = tfep_map_cls(
        mapped_atoms=range(1, 6),
        axes_atoms=[0, 2],
        **MAP_INIT_KWARGS,
    )
    with pytest.raises(ValueError, match="must be mapped or conditioning atoms"):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_handling_energy_unit(tfep_map_cls):
    """The TFEPMap correctly recognizes the units returned by the potential energy function."""
    # We'll test that all kT have different values so let's make sure the default
    # (i.e. energy_unit=None) is different than the ones we use.
    assert MockPotential.DEFAULT_ENERGY_UNIT == 'kcal'
    tested_energy_units = [None, UNITS.joule, UNITS.kcal/UNITS.mole]

    # Collect different values of kTs.
    kTs = []
    kwargs = MAP_INIT_KWARGS.copy()
    for energy_unit in tested_energy_units:
        kwargs['potential_energy_func'] = MockPotential(energy_unit=energy_unit)
        tfep_map = tfep_map_cls(**kwargs)
        kTs.append(tfep_map._kT)

    # These are the expected values at 298 K.
    assert kwargs['temperature'] == 298*UNITS.kelvin
    kT_kcal = 9.83349431166348e-25
    kT_joule = 4.11433402e-21
    RT_kcalmol = 0.592186869074968
    assert np.allclose(kTs, [kT_kcal, kT_joule, RT_kcalmol])


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_resuming_mid_epoch(tfep_map_cls):
    """Test that resuming mid-epoch works correctly.

    In particular, that training does not restart from the next epoch, and that
    the training does not process the same samples twice in the restarted epoch.

    """
    # Create at runtime a subclass of tfep_map_cls that can be interrupted after
    # a fixed number of steps.
    class InterruptableTFEPMap(tfep_map_cls):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fail_after = None
            self.epoch_visited_samples = None
            self.n_processed_steps = 0

        def training_step(self, batch, batch_idx):
            # On the very first training step, configure fail_after so that we
            # 1.5 epochs of training before interrupting.
            if self.fail_after is None:
                self.fail_after = int(len(self._stateful_batch_sampler) * 1.5)

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

    # We perform exactly 2 epochs of training in total.
    max_epochs = 2

    def init_map_and_trainer(tmp_dir_path):
        tfep_map = InterruptableTFEPMap(tfep_logger_dir_path=tmp_dir_path, **MAP_INIT_KWARGS)

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
        tfep_map, trainer = init_map_and_trainer(tmp_dir_path)
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
        tfep_map, trainer = init_map_and_trainer(tmp_dir_path)
        trainer.fit(tfep_map, ckpt_path='last')

        # Check that the visited samples on resuming do not overlap.
        visited_samples2 = set(tfep_map.epoch_visited_samples)

        # Total number of steps amount to max_epochs.
        assert tfep_map.n_processed_steps + tfep_map.fail_after == max_epochs * n_batches_per_epoch

        # During the interrupted epoch, the training visited all samples and only once.
        assert visited_samples1.union(visited_samples2) == set(range(len(tfep_map.dataset)))
        assert len(visited_samples1.intersection(visited_samples2)) == 0
