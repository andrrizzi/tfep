#!/usr/bin/env python

"""Shell-equivariant internal-water flow utilities for solvated systems."""

from __future__ import annotations

import torch


class ShellEquivariantWaterInternalFlow(torch.nn.Module):
    """Solute-atom-resolved equivariant internal-water coupling flow.

    This leaves water oxygen positions unchanged and transforms only hydrogen
    coordinates through bounded rotations/scalings conditioned on local
    solute-water geometry. The log-determinant is analytic.
    """

    def __init__(
        self,
        n_mapped_atoms: int,
        solute_local_indices,
        water_oxygen_local_indices,
        water_h1_local_indices,
        water_h2_local_indices,
        water_atom_local_flat_indices=None,
        water_atom_owner_indices=None,
        cutoff_angstrom: float = 4.5,
        tau_angstrom: float = 0.3,
        hidden_dim: int = 64,
        max_displacement_angstrom: float = 0.05,
        initial_log_scale: float = -1.0,
        top_k: int = 12,
        max_rotation_radians: float = 0.35,
        max_internal_log_scale: float = 0.06,
    ):
        super().__init__()

        self.n_mapped_atoms = int(n_mapped_atoms)
        self.cutoff = float(cutoff_angstrom)
        self.tau = float(tau_angstrom)
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
        self.max_rotation = float(max_rotation_radians)
        self.max_internal_log_scale = float(max_internal_log_scale)
        self._max_displacement_angstrom = float(max_displacement_angstrom)

        self.register_buffer(
            "solute_local_indices",
            torch.as_tensor(solute_local_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "water_oxygen_local_indices",
            torch.as_tensor(water_oxygen_local_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "water_h1_local_indices",
            torch.as_tensor(water_h1_local_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "water_h2_local_indices",
            torch.as_tensor(water_h2_local_indices, dtype=torch.long),
            persistent=False,
        )

        n_solute_atoms = int(torch.as_tensor(solute_local_indices).numel())
        if n_solute_atoms <= 0:
            raise ValueError("ShellEquivariantWaterInternalFlow requires at least one solute atom")
        self.n_solute_atoms = n_solute_atoms

        n_waters = int(torch.as_tensor(water_oxygen_local_indices).numel())
        if n_waters <= 0:
            raise ValueError("ShellEquivariantWaterInternalFlow requires at least one water molecule")
        if int(torch.as_tensor(water_h1_local_indices).numel()) != n_waters:
            raise ValueError("water_h1_local_indices length must equal water_oxygen_local_indices length")
        if int(torch.as_tensor(water_h2_local_indices).numel()) != n_waters:
            raise ValueError("water_h2_local_indices length must equal water_oxygen_local_indices length")
        self.n_waters = n_waters

        self.solute_atom_embedding = torch.nn.Embedding(self.n_solute_atoms, self.hidden_dim)

        pair_in_dim = 4 + self.hidden_dim
        self.solute_pair_mlp = torch.nn.Sequential(
            torch.nn.Linear(pair_in_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, 3),
        )

        self.internal_param_mlp = torch.nn.Sequential(
            torch.nn.Linear(5, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, 4),
        )

        self.log_scale = torch.nn.Parameter(torch.tensor(float(initial_log_scale)))

    def _shell_mask(self, r_c: torch.Tensor) -> torch.Tensor:
        """Return either hard per-frame top-k mask or smooth sigmoid shell gate."""
        if self.top_k > 0:
            k = min(int(self.top_k), int(r_c.shape[1]))
            idx = torch.topk(r_c, k=k, dim=1, largest=False).indices
            mask = torch.zeros_like(r_c)
            mask.scatter_(1, idx, 1.0)
            return mask
        return torch.sigmoid((self.cutoff - r_c) / self.tau)

    @staticmethod
    def _rotate(v: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-8)
        ct = torch.cos(theta)[..., None]
        st = torch.sin(theta)[..., None]
        cross = torch.cross(axis, v, dim=-1)
        dot = (axis * v).sum(dim=-1, keepdim=True)
        return v * ct + cross * st + axis * dot * (1.0 - ct)

    def _compute_internal_parameters(self, x3: torch.Tensor):
        solute = x3[:, self.solute_local_indices, :]
        water_o = x3[:, self.water_oxygen_local_indices, :]
        center = solute.mean(dim=1, keepdim=True)

        vec_c = water_o - center
        r_c = torch.linalg.norm(vec_c, dim=-1).clamp_min(1.0e-8)
        unit_c = vec_c / r_c[..., None]
        shell = self._shell_mask(r_c)

        vec_ws = water_o[:, :, None, :] - solute[:, None, :, :]
        d_ws = torch.linalg.norm(vec_ws, dim=-1).clamp_min(1.0e-8)
        unit_ws = vec_ws / d_ws[..., None]

        scalar_pair_feat = torch.stack(
            [
                d_ws / self.cutoff,
                1.0 / (1.0 + d_ws),
                torch.exp(-d_ws / self.cutoff),
                shell[:, :, None].expand_as(d_ws),
            ],
            dim=-1,
        )
        solute_ids = torch.arange(self.n_solute_atoms, device=x3.device, dtype=torch.long)
        solute_emb = self.solute_atom_embedding(solute_ids)
        solute_emb = solute_emb.view(1, 1, self.n_solute_atoms, self.hidden_dim).expand(
            x3.shape[0], water_o.shape[1], self.n_solute_atoms, self.hidden_dim
        )

        pair_feat = torch.cat([scalar_pair_feat, solute_emb], dim=-1)
        pair_coeff = torch.tanh(self.solute_pair_mlp(pair_feat))

        atom_vecs = (pair_coeff[..., None] * unit_ws[..., None, :]).sum(dim=2) / (self.n_solute_atoms**0.5)
        axis1 = atom_vecs[:, :, 0, :] + 1.0e-3 * unit_c
        axis2 = atom_vecs[:, :, 1, :] + 1.0e-3 * unit_c
        field = atom_vecs[:, :, 2, :] + unit_c
        field_norm = torch.linalg.norm(field, dim=-1).clamp_min(1.0e-8)

        feat = torch.stack(
            [
                r_c / self.cutoff,
                shell,
                1.0 / (1.0 + r_c),
                torch.tanh(field_norm),
                torch.exp(-r_c / self.cutoff),
            ],
            dim=-1,
        )
        raw = self.internal_param_mlp(feat)
        global_scale = torch.exp(self.log_scale).clamp(max=1.0)

        theta1 = self.max_rotation * global_scale * shell * torch.tanh(raw[..., 0])
        theta2 = self.max_rotation * global_scale * shell * torch.tanh(raw[..., 1])
        s1 = self.max_internal_log_scale * global_scale * shell * torch.tanh(raw[..., 2])
        s2 = self.max_internal_log_scale * global_scale * shell * torch.tanh(raw[..., 3])

        return axis1, axis2, theta1, theta2, s1, s2

    def forward(self, x, inverse: bool = False, **kwargs):
        if x.ndim != 2:
            raise ValueError(
                "ShellEquivariantWaterInternalFlow expects flattened coordinates "
                f"[batch, 3*n_mapped_atoms], got shape {tuple(x.shape)}"
            )

        batch = x.shape[0]
        expected = 3 * self.n_mapped_atoms
        if x.shape[1] != expected:
            raise ValueError(
                f"ShellEquivariantWaterInternalFlow expected {expected} coordinates "
                f"for {self.n_mapped_atoms} mapped atoms, got {x.shape[1]}"
            )

        x3 = x.reshape(batch, self.n_mapped_atoms, 3)
        y3 = x3.clone()

        o = x3[:, self.water_oxygen_local_indices, :]
        h1 = x3[:, self.water_h1_local_indices, :]
        h2 = x3[:, self.water_h2_local_indices, :]
        r1 = h1 - o
        r2 = h2 - o

        axis1, axis2, theta1, theta2, s1, s2 = self._compute_internal_parameters(x3)

        if inverse:
            r1_new = self._rotate(torch.exp(-s1[..., None]) * r1, axis1, -theta1)
            r2_new = self._rotate(torch.exp(-s2[..., None]) * r2, axis2, -theta2)
            log_det = -(3.0 * s1 + 3.0 * s2).sum(dim=1)
        else:
            r1_new = torch.exp(s1[..., None]) * self._rotate(r1, axis1, theta1)
            r2_new = torch.exp(s2[..., None]) * self._rotate(r2, axis2, theta2)
            log_det = (3.0 * s1 + 3.0 * s2).sum(dim=1)

        y3[:, self.water_h1_local_indices, :] = o + r1_new
        y3[:, self.water_h2_local_indices, :] = o + r2_new

        return y3.reshape(batch, expected), log_det.to(dtype=x.dtype, device=x.device)

    def inverse(self, x, **kwargs):
        return self.forward(x, inverse=True, **kwargs)


class _MaskedAutoregressiveAffineLayer(torch.nn.Module):
    """Small exact masked autoregressive affine layer."""

    def __init__(
        self,
        n_dims: int,
        hidden_dim: int = 128,
        max_log_scale: float = 0.20,
        max_shift: float = 0.20,
        reverse_order: bool = False,
    ):
        super().__init__()
        self.n_dims = int(n_dims)
        self.max_log_scale = float(max_log_scale)
        self.max_shift = float(max_shift)

        order = list(range(self.n_dims))
        if bool(reverse_order):
            order = list(reversed(order))
        self.register_buffer("order", torch.as_tensor(order, dtype=torch.long), persistent=False)

        masks = []
        previous = []
        for dim in order:
            mask = torch.zeros(self.n_dims, dtype=torch.get_default_dtype())
            if previous:
                mask[torch.as_tensor(previous, dtype=torch.long)] = 1.0
            masks.append(mask)
            previous.append(int(dim))
        self.register_buffer("context_masks", torch.stack(masks, dim=0), persistent=False)

        self.mlps = torch.nn.ModuleList()
        for _ in range(self.n_dims):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dims, int(hidden_dim)),
                torch.nn.SiLU(),
                torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
                torch.nn.SiLU(),
                torch.nn.Linear(int(hidden_dim), 2),
            )
            torch.nn.init.zeros_(mlp[-1].weight)
            torch.nn.init.zeros_(mlp[-1].bias)
            self.mlps.append(mlp)

    def _params(self, context: torch.Tensor, step: int):
        dim = int(self.order[int(step)].item())
        mask = self.context_masks[int(step)].to(dtype=context.dtype, device=context.device)
        raw = self.mlps[dim](context * mask)
        shift = self.max_shift * torch.tanh(raw[:, 0])
        log_scale = self.max_log_scale * torch.tanh(raw[:, 1])
        return dim, shift, log_scale

    def forward(self, x: torch.Tensor, inverse: bool = False):
        if x.ndim != 2 or x.shape[1] != self.n_dims:
            raise ValueError(
                f"_MaskedAutoregressiveAffineLayer expected [B,{self.n_dims}], got {tuple(x.shape)}"
            )

        if not inverse:
            y = x.clone()
            log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for step in range(self.n_dims):
                dim, shift, log_scale = self._params(x, step)
                y[:, dim] = torch.exp(log_scale) * x[:, dim] + shift
                log_det = log_det + log_scale
            return y, log_det

        y = x
        x_rec = torch.zeros_like(y)
        log_det = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
        for step in range(self.n_dims):
            dim, shift, log_scale = self._params(x_rec, step)
            x_rec[:, dim] = (y[:, dim] - shift) * torch.exp(-log_scale)
            log_det = log_det - log_scale
        return x_rec, log_det

    def inverse(self, x: torch.Tensor, **kwargs):
        return self.forward(x, inverse=True)


class SoluteAutoregressiveMAFFlow(torch.nn.Module):
    """Stack of lightweight exact MAF layers for a solute coordinate block."""

    def __init__(
        self,
        n_dims: int,
        n_layers: int = 2,
        hidden_dim: int = 128,
        max_log_scale: float = 0.20,
        max_shift: float = 0.20,
    ):
        super().__init__()
        self.n_dims = int(n_dims)
        self.layers = torch.nn.ModuleList(
            [
                _MaskedAutoregressiveAffineLayer(
                    n_dims=self.n_dims,
                    hidden_dim=int(hidden_dim),
                    max_log_scale=float(max_log_scale),
                    max_shift=float(max_shift),
                    reverse_order=bool(i % 2),
                )
                for i in range(max(1, int(n_layers)))
            ]
        )

    def forward(self, x: torch.Tensor, inverse: bool = False):
        log_det_total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        y = x
        layers = reversed(self.layers) if inverse else self.layers
        for layer in layers:
            y, log_det = layer(y, inverse=bool(inverse))
            log_det_total = log_det_total + log_det
        return y, log_det_total

    def inverse(self, x: torch.Tensor, **kwargs):
        return self.forward(x, inverse=True)


class JointSoluteMAFAndShellWaterInternalFlow(torch.nn.Module):
    """Joint flow: solute exact MAF plus shell-water internal equivariant flow."""

    def __init__(
        self,
        n_mapped_atoms: int,
        solute_local_indices,
        water_flow: ShellEquivariantWaterInternalFlow,
        solute_maf_layers: int = 2,
        solute_maf_hidden_dim: int = 128,
        solute_maf_max_log_scale: float = 0.20,
        solute_maf_max_shift_angstrom: float = 0.20,
    ):
        super().__init__()
        self.n_mapped_atoms = int(n_mapped_atoms)
        self.water_flow = water_flow

        solute_local_indices = torch.as_tensor(solute_local_indices, dtype=torch.long)
        self.register_buffer("solute_local_indices", solute_local_indices, persistent=False)

        flat_idx = []
        for idx in solute_local_indices.tolist():
            base = 3 * int(idx)
            flat_idx.extend([base, base + 1, base + 2])
        self.register_buffer("solute_flat_indices", torch.as_tensor(flat_idx, dtype=torch.long), persistent=False)

        self.solute_flow = SoluteAutoregressiveMAFFlow(
            n_dims=int(len(flat_idx)),
            n_layers=int(solute_maf_layers),
            hidden_dim=int(solute_maf_hidden_dim),
            max_log_scale=float(solute_maf_max_log_scale),
            max_shift=float(solute_maf_max_shift_angstrom),
        )

    def _apply_solute_flow(self, x: torch.Tensor, inverse: bool = False):
        y = x.clone()
        x_sol = x.index_select(1, self.solute_flat_indices)
        y_sol, log_det = self.solute_flow(x_sol, inverse=bool(inverse))
        y[:, self.solute_flat_indices] = y_sol
        return y, log_det

    def forward(self, x: torch.Tensor, inverse: bool = False, **kwargs):
        if not inverse:
            y, log_s = self._apply_solute_flow(x, inverse=False)
            y, log_w = self.water_flow(y, inverse=False)
            return y, log_s + log_w

        y, log_w = self.water_flow(x, inverse=True)
        y, log_s = self._apply_solute_flow(y, inverse=True)
        return y, log_w + log_s

    def inverse(self, x: torch.Tensor, **kwargs):
        return self.forward(x, inverse=True)


# Backward-compatible alias used by experimental wrappers.
ShellEquivariantWaterFlatFlow = ShellEquivariantWaterInternalFlow
