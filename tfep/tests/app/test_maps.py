#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test shared functionalities of the TFEP maps in the ``tfep.app`` package.
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

from tfep.app import TFEPMapBase, CartesianMAFMap, MixedMAFMap

from .. import MockPotential, DATA_DIR_PATH


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

CHLOROMETHANE_PDB_FILE_PATH = os.path.join(DATA_DIR_PATH, 'chloro-fluoromethane.pdb')

UNITS = pint.UnitRegistry()


# =============================================================================
# TEST MODULE CONFIGURATION
# =============================================================================

def set_worker_default_dtype(worker):
    """Passed as init function to DataLoader workers."""
    torch.set_default_dtype(torch.double)


_old_default_dtype = None


def setup_module(module):
    global _old_default_dtype
    _old_default_dtype = torch.get_default_dtype()
    set_worker_default_dtype(None)


def teardown_module(module):
    torch.set_default_dtype(_old_default_dtype)


# =============================================================================
# TEST UTILITIES
# =============================================================================

class ContinuousTFEPMap(TFEPMapBase):
    """TFEPMap for testing inheritance from TFEPMapBase and compatibility with continuous flows."""

    def configure_flow(self):
        try:
            import torchdiffeq
        except ImportError:
            pytest.skip('The torchdiffeq package is required for continuous flows.')

        # Continuous.
        if self.n_conditioning_atoms > 0:
            pytest.skip("ContinuousTFEPMap doesn't support conditioning atoms.")

        egnn_dynamics = tfep.nn.dynamics.EGNNDynamics(
            node_types=torch.tensor([0] * self.n_mapped_atoms),
            r_cutoff=6.0,
            time_feat_dim=2,
            node_feat_dim=4,
            distance_feat_dim=4,
            n_layers=2,
        )
        return tfep.nn.flows.ContinuousFlow(
            dynamics=egnn_dynamics,
            solver='euler',
            solver_options={'step_size': .01},
        )


# Maps to be tested.
TESTED_TFEP_MAPS = [ContinuousTFEPMap, CartesianMAFMap, MixedMAFMap]


# Shared kwargs for TFEPMap.__init__().
MAP_INIT_KWARGS = dict(
    potential_energy_func=MockPotential(),
    topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
    coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
    temperature=298*UNITS.kelvin,
    batch_size=2,
)
MAP_INIT_KWARGS = {cls.__name__: MAP_INIT_KWARGS.copy() for cls in TESTED_TFEP_MAPS}


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_no_mapped_atoms(tfep_map_cls):
    """If only conditioning is given and there are no atoms left to map, an error is raised."""
    tfep_map = tfep_map_cls(conditioning_atoms=list(range(6)), **MAP_INIT_KWARGS[tfep_map_cls.__name__])
    with pytest.raises(ValueError, match='no atoms to map'):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_overlapping_atom_selections(tfep_map_cls):
    """If mapped and conditioning atoms overlap, an error is raised."""
    tfep_map = tfep_map_cls(mapped_atoms=[0, 1, 2], conditioning_atoms=[2, 3], **MAP_INIT_KWARGS[tfep_map_cls.__name__])
    with pytest.raises(ValueError, match='overlapping atoms'):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
@pytest.mark.parametrize('origin_atom,axes_atoms,match', [
    [0, (0, 2), "must be different"],
    [0, (2, 2), "2 axes atoms must be given"],
    [0, (2, 0), "must be different"],
])
def test_error_reference_frame_atoms_overlap(tfep_map_cls, origin_atom, axes_atoms, match):
    """An error is raised if the origin, axis, and/or plane atoms overlap."""
    tfep_map = tfep_map_cls(
        conditioning_atoms=[0],
        origin_atom=origin_atom,
        axes_atoms=axes_atoms,
        **MAP_INIT_KWARGS[tfep_map_cls.__name__],
    )
    with pytest.raises(ValueError, match=match):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_multiple_origin_atom(tfep_map_cls):
    """An error is raised if multiple atoms are selected."""
    tfep_map = tfep_map_cls(
        mapped_atoms=range(6),
        origin_atom='element H',
        **MAP_INIT_KWARGS[tfep_map_cls.__name__],
    )
    with pytest.raises(ValueError, match="multiple atoms as the origin atom"):
        tfep_map.setup()


@pytest.mark.parametrize('tfep_map_cls', TESTED_TFEP_MAPS)
def test_error_axes_atom_fixed(tfep_map_cls):
    """An error is raised if the origin atom is not a conditioning atom."""
    tfep_map = tfep_map_cls(
        mapped_atoms=range(1, 6),
        conditioning_atoms=[7],
        origin_atom=7,
        axes_atoms=[0, 2],
        **MAP_INIT_KWARGS[tfep_map_cls.__name__],
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
    kwargs = MAP_INIT_KWARGS[tfep_map_cls.__name__].copy()
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
@pytest.mark.parametrize('num_workers', [0, 2])
def test_resuming_mid_epoch(tfep_map_cls, num_workers):
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
        tfep_map = InterruptableTFEPMap(
            tfep_logger_dir_path=tmp_dir_path,
            dataloader_kwargs=dict(num_workers=num_workers, worker_init_fn=set_worker_default_dtype),
            **MAP_INIT_KWARGS[tfep_map_cls.__name__],
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
