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
        round_trip: bool = True,
        **kwargs,
):
    """Test selection of mapped, conditioning, fixed, and reference frame atoms.

    This also tests:
    - That a forward-inverse round trip yields the original input.
    - That the conditioning atoms are not changed but affect the output.
    - That the fixed atoms are not changed and do not affect the output.

    Parameters
    ----------
    round_trip : bool, optional
        If ``False``, this will not check that a forward-inverse round trip
        yields the original input.

    """
    import lightning
    import numpy as np
    import pytest
    import tempfile
    import torch
    from tfep.utils.misc import atom_to_flattened, flattened_to_atom, temporary_cd
    from tfep.utils.geometry import batchwise_dot

    # Since we select randomly the reference atoms, there's a chance to pick
    # collinear atoms. We repeat the selection until the error is not thrown.
    max_n_attempts = 10
    for attempt_idx in range(max_n_attempts):
        try:
            # Select a random fixed atom to fix the rotational degrees of freedom.
            if fix_origin:
                if expected_conditioning is None:
                    pytest.skip('fixing the translational DOFs require the presence of a conditioning atom')
                origin_atom = np.random.choice(expected_conditioning)
                kwargs['origin_atom'] = origin_atom

            # Select axis and plane atoms among the remaining atoms.
            if fix_orientation:
                remaining = []
                [remaining.extend(l) for l in (expected_mapped, expected_conditioning) if l is not None]
                remaining = sorted([i for i in remaining if i != kwargs.get('origin_atom', None)])
                if len(remaining) < 2:
                    pytest.skip('fixing the orientation of the reference frame requires at least 2 mapped or conditioning atoms.')
                axes_atoms = np.random.choice(remaining, size=2, replace=False).tolist()
                kwargs['axes_atoms'] = axes_atoms

            with tempfile.TemporaryDirectory() as tmp_dir_path:
                with temporary_cd(tmp_dir_path):
                    # Initialize the map.
                    tfep_map = tfep_map_cls(
                        mapped_atoms=mapped_atoms,
                        conditioning_atoms=conditioning_atoms,
                        **kwargs,
                    )

                    # Train for one step to make sure that the map is not the identity.
                    trainer = lightning.Trainer(
                        max_steps=1,
                        logger=False,
                        enable_checkpointing=False,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                    )
                    trainer.fit(tfep_map)
            break
        except RuntimeError as err:
            if 'collinear' in str(err):
                continue
            raise
    else:
        raise err

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

    # Create position input.
    x = torch.tensor([tfep_map.dataset.universe.trajectory[i].positions
                      for i in range(tfep_map.hparams.batch_size)],
                     dtype=torch.get_default_dtype())
    x = atom_to_flattened(x)
    x.requires_grad = True

    # Test forward and inverse.
    result = tfep_map({'positions': x})
    y, log_det_J = result['positions'], result['log_det_J']
    result = tfep_map.inverse({'positions': y})
    if round_trip:
        x_inv, log_det_J_inv = result['positions'], result['log_det_J']
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
        assert torch.all(~torch.isclose(x_grad[:, expected_conditioning], torch.ones(*x[:, expected_conditioning].shape), rtol=0.0))

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
