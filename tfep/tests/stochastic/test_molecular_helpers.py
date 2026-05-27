import numpy as np

from tfep.stochastic.molecular import MolecularStochasticConfig, atom_indices_to_flat_indices, metadata_from_config


def test_atom_indices_to_flat_indices():
    got = atom_indices_to_flat_indices([0, 2, 5])
    expected = np.asarray([0, 1, 2, 6, 7, 8, 15, 16, 17])
    assert np.array_equal(got, expected)


def test_molecular_config_defaults_are_deterministic_and_metadata_records_terms():
    cfg = MolecularStochasticConfig()
    assert not cfg.enabled
    metadata = metadata_from_config(cfg, selected_flat_indices=[0, 1, 2])
    assert metadata["work_convention"] == "u_target-u_source-logJ+logq_forward-logq_reverse"
    assert metadata["selected_flat_coordinate_count"] == 3


def test_molecular_config_step_count_is_at_least_one():
    cfg = MolecularStochasticConfig(estimator="stochastic-path-tfep", num_blocks=0, steps_per_block=0)
    assert cfg.enabled
    assert cfg.n_stochastic_steps == 1
