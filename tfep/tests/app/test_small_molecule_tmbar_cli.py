import numpy as np
import pytest

from tfep.app import small_molecule_tmbar


def _required_args():
    return [
        "--state0-dir", "state0",
        "--state1-dir", "state1",
        "--traj0", "traj0.xtc",
        "--traj1", "traj1.xtc",
        "--temperature", "298.15",
    ]


def test_gradient_clipping_cli_defaults_preserve_behavior(tmp_path):
    args = small_molecule_tmbar.build_argparser().parse_args(_required_args())

    assert args.gradient_clip_val == 0.0
    assert args.gradient_clip_algorithm == "norm"
    assert args.bar_df_solver == "newton"

    trainer = small_molecule_tmbar.build_trainer(args, tmp_path)
    assert float(trainer.gradient_clip_val or 0.0) == 0.0


def test_gradient_clipping_cli_configures_lightning(tmp_path):
    args = small_molecule_tmbar.build_argparser().parse_args(
        _required_args() + [
            "--gradient-clip-val", "1.5",
            "--gradient-clip-algorithm", "value",
        ]
    )

    trainer = small_molecule_tmbar.build_trainer(args, tmp_path)

    assert trainer.gradient_clip_val == 1.5
    assert trainer.gradient_clip_algorithm == "value"


def test_negative_gradient_clip_value_rejected(tmp_path):
    args = small_molecule_tmbar.build_argparser().parse_args(
        _required_args() + ["--gradient-clip-val", "-1.0"]
    )

    with pytest.raises(ValueError, match="gradient-clip-val"):
        small_molecule_tmbar.build_trainer(args, tmp_path)


def test_bar_df_solver_cli_accepts_robust():
    args = small_molecule_tmbar.build_argparser().parse_args(
        _required_args() + ["--bar-df-solver", "robust"]
    )

    assert args.bar_df_solver == "robust"


def test_reweighted_bar_uses_global_normalization_by_default():
    args = small_molecule_tmbar.build_argparser().parse_args(_required_args())

    assert args.reweighted_bar_weight_normalization == "global"


def test_global_training_weight_stats_use_only_selected_frames():
    log_weights = np.array([-10.0, 0.0, 2.0, 1.0, -4.0])
    indices = np.array([1, 2, 3])

    stats = small_molecule_tmbar._global_training_weight_stats(
        log_weights,
        indices,
        label="test",
    )
    selected = log_weights[indices]

    assert stats["population_size"] == 3
    assert np.isclose(stats["log_normalizer"], np.log(np.exp(selected).sum()))
    assert np.isclose(
        stats["ess"],
        small_molecule_tmbar.rwlib.effective_sample_size(selected),
    )
    assert stats["log_weight_min"] == 0.0
    assert stats["log_weight_max"] == 2.0
