"""
Test objects and function in the package ``tfep.app``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from collections.abc import Sequence
from typing import List, Optional, Type, Union

import tfep.app.base


# =============================================================================
# SHARED APP TEST UTILITIES
# =============================================================================

def check_atom_groups(
        tfep_map_cls: Type[tfep.app.base.TFEPMapBase],
        fix_origin: bool,
        fix_orientation: bool,
        mapped_atoms: Optional[Union[Sequence[int], str]],
        conditioning_atoms: Optional[Union[Sequence[int], str]],
        expected_mapped: Optional[List[int]],
        expected_conditioning: Optional[List[int]],
        expected_fixed: Optional[List[int]],
        expected_mapped_fixed_removed: Optional[List[int]],
        expected_conditioning_fixed_removed: Optional[List[int]],
        **kwargs,
):
    """Test selection of mapped, conditioning, fixed, and reference frame atoms."""
    import numpy as np
    import pytest
    import torch
    from tfep.utils.misc import flattened_to_atom
    from tfep.utils.geometry import batchwise_dot

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
        **kwargs,
    )
    tfep_map.setup()

    # Compare expected indices.
    for expected_indices, tfep_indices in zip(
            [
                expected_mapped_fixed_removed,
                expected_conditioning_fixed_removed,
                expected_fixed
            ], [
                tfep_map.get_mapped_indices(idx_type='atom', remove_fixed=True),
                tfep_map.get_conditioning_indices(idx_type='atom', remove_fixed=True),
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
