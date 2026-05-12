#!/usr/bin/env python

import torch

from tfep.nn.flows.shell_water import (
    JointSoluteMAFAndShellWaterInternalFlow,
    ShellEquivariantWaterInternalFlow,
)


def _toy_coordinates(batch_size: int = 4) -> torch.Tensor:
    # Layout: [solute, W0(O,H1,H2), W1(O,H1,H2)] => 7 atoms total.
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.95, 0.0, 0.0],
            [1.76, 0.93, 0.0],
            [4.0, 0.2, 0.0],
            [4.95, 0.2, 0.0],
            [3.76, 1.13, 0.0],
        ],
        dtype=torch.double,
    )
    return coords.reshape(1, -1).repeat(batch_size, 1)


def test_shell_equivariant_internal_flow_inverse_consistency():
    torch.set_default_dtype(torch.double)
    flow = ShellEquivariantWaterInternalFlow(
        n_mapped_atoms=7,
        solute_local_indices=[0],
        water_oxygen_local_indices=[1, 4],
        water_h1_local_indices=[2, 5],
        water_h2_local_indices=[3, 6],
        cutoff_angstrom=4.5,
        tau_angstrom=0.3,
        hidden_dim=32,
        top_k=2,
    )

    x = _toy_coordinates(batch_size=3)
    y, log_det = flow(x)
    x_rec, log_det_inv = flow.inverse(y)

    assert y.shape == x.shape
    assert log_det.shape == torch.Size([x.shape[0]])
    assert torch.allclose(x, x_rec, atol=5e-6, rtol=5e-6)
    assert torch.allclose(log_det + log_det_inv, torch.zeros_like(log_det), atol=5e-6, rtol=5e-6)


def test_joint_solute_and_water_flow_inverse_consistency():
    torch.set_default_dtype(torch.double)
    water_flow = ShellEquivariantWaterInternalFlow(
        n_mapped_atoms=7,
        solute_local_indices=[0],
        water_oxygen_local_indices=[1, 4],
        water_h1_local_indices=[2, 5],
        water_h2_local_indices=[3, 6],
        hidden_dim=16,
        top_k=2,
    )
    flow = JointSoluteMAFAndShellWaterInternalFlow(
        n_mapped_atoms=7,
        solute_local_indices=[0],
        water_flow=water_flow,
        solute_maf_layers=2,
        solute_maf_hidden_dim=32,
    )

    x = _toy_coordinates(batch_size=2)
    y, log_det = flow(x)
    x_rec, log_det_inv = flow.inverse(y)

    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_det + log_det_inv, torch.zeros_like(log_det), atol=1e-5, rtol=1e-5)
