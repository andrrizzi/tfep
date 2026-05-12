#!/usr/bin/env python

import os
import types

import pint
import pytest
import torch

from tfep.app.base import TMBARMapBase
from tfep.potentials.base import MultiStatePotential
from tfep.regularizers import bar as barlib

from .. import DATA_DIR_PATH, MockPotential


CHLOROMETHANE_PDB_FILE_PATH = os.path.join(DATA_DIR_PATH, "chloro-fluoromethane.pdb")


class _IdentityFlow(torch.nn.Module):
    def forward(self, x):
        return x, torch.zeros(len(x), dtype=x.dtype, device=x.device)

    def inverse(self, y):
        return y, torch.zeros(len(y), dtype=y.dtype, device=y.device)


class _DummyTMBARMap(TMBARMapBase):
    def configure_flow(self):
        return _IdentityFlow()


def _make_map(*, objective: str, lambda_bar: float = 1.0, logj_penalty_weight: float = 0.0):
    units = pint.UnitRegistry()
    return _DummyTMBARMap(
        potential_energy_func=MultiStatePotential(MockPotential(), MockPotential()),
        topology_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path=CHLOROMETHANE_PDB_FILE_PATH,
        coordinates_file_path_2=CHLOROMETHANE_PDB_FILE_PATH,
        temperature=298 * units.kelvin,
        batch_size=2,
        objective=objective,
        lambda_bar=lambda_bar,
        bar_warm_start=False,
        logJ_penalty_weight=logj_penalty_weight,
    )


@pytest.mark.parametrize(
    "objective,lambda_bar,logj_weight",
    [
        ("kl", 1.0, 0.0),
        ("bar", 1.0, 0.0),
        ("hybrid", 2.5, 0.0),
        ("hybrid", 2.5, 0.3),
    ],
)
def test_tmbar_objective_composition(objective, lambda_bar, logj_weight):
    torch.set_default_dtype(torch.double)
    module = _make_map(objective=objective, lambda_bar=lambda_bar, logj_penalty_weight=logj_weight)

    loss_01 = torch.tensor(2.0, dtype=torch.double)
    loss_10 = torch.tensor(4.0, dtype=torch.double)
    u1_y0 = torch.tensor([1.2, 1.4], dtype=torch.double)
    logj01 = torch.tensor([0.2, -0.1], dtype=torch.double)
    u0_x0 = torch.tensor([0.5, 0.6], dtype=torch.double)

    u0_y1 = torch.tensor([0.8, 0.9], dtype=torch.double)
    logj10 = torch.tensor([0.3, -0.2], dtype=torch.double)
    u1_x1 = torch.tensor([0.4, 0.3], dtype=torch.double)

    def _fake_step(self, batch_data, direction_func, state_mapping, batch_idx):
        if tuple(state_mapping) == (0, 1):
            return loss_01, u1_y0, logj01, u0_x0
        return loss_10, u0_y1, logj10, u1_x1

    module._compute_direction_step = types.MethodType(_fake_step, module)
    module.log = lambda *args, **kwargs: None

    batch = {"batch_1": {}, "batch_2": {}}
    got = module.training_step(batch, 0)

    mean_kl = 0.5 * (loss_01 + loss_10)
    w01, w10 = module._compute_bidirectional_works(
        u1_y0=u1_y0,
        logJ01=logj01,
        u0_x0=u0_x0,
        u0_y1=u0_y1,
        logJ10=logj10,
        u1_x1=u1_x1,
    )

    n0 = int(w01.numel())
    n1 = int(w10.numel())
    log_ratio = torch.log(
        torch.as_tensor(float(n1), device=w01.device, dtype=w01.dtype)
        / torch.as_tensor(float(n0), device=w01.device, dtype=w01.dtype)
    )
    df_bar = barlib._bar_newton_solve_detached(
        w01.detach(),
        w10.detach(),
        log_ratio=log_ratio.detach(),
        df_init=None,
        max_iter=module._bar_max_iter,
        tol=module._bar_tol,
    ).detach()
    bar_obj = barlib.bar_objective(w01, w10, df_bar.detach(), log_ratio)

    expected = mean_kl
    if objective == "bar":
        expected = bar_obj
    elif objective == "hybrid":
        expected = mean_kl + torch.as_tensor(lambda_bar, dtype=bar_obj.dtype) * bar_obj

    if logj_weight > 0.0:
        expected = expected + logj_weight * (0.5 * (logj01.pow(2).mean() + logj10.pow(2).mean()))

    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_tmbar_checkpoint_sampler_compat_roundtrip():
    module = _make_map(objective="kl")

    module.on_load_checkpoint({"stateful_batch_samplers": {"state0": {"a": 1}, "state1": {"b": 2}}})
    assert module._stateful_batch_sampler_0 == {"a": 1}
    assert module._stateful_batch_sampler_1 == {"b": 2}

    module.on_load_checkpoint({"stateful_batch_sampler": {"legacy": 42}})
    assert module._stateful_batch_sampler_0 == {"legacy": 42}
    assert module._stateful_batch_sampler_1 is None

    class _Sampler:
        def __init__(self, payload):
            self._payload = payload

        def state_dict(self):
            return dict(self._payload)

    module._stateful_batch_sampler_0 = _Sampler({"k0": 10})
    module._stateful_batch_sampler_1 = _Sampler({"k1": 20})

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["stateful_batch_samplers"]["state0"] == {"k0": 10}
    assert checkpoint["stateful_batch_samplers"]["state1"] == {"k1": 20}


def test_tmbar_invalid_objective_rejected():
    with pytest.raises(ValueError, match="Unsupported objective"):
        _make_map(objective="nope")
