#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Base class to implement potential energy functions.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional

import pint
import torch


# =============================================================================
# TORCH MODULE API
# =============================================================================

class PotentialBase(torch.nn.Module):
    """Base class for potential energy functions.

    This ``Module`` implements units related utilities to easily define default
    units and handle ``pint`` unit registries.

    To inherit from this class one needs to define the following class variables.

    - :attr:`~PotentialBase.DEFAULT_ENERGY_UNIT`
    - :attr:`~PotentialBase.DEFAULT_POSITION_UNIT`

    """

    #: The default energy unit.
    DEFAULT_ENERGY_UNIT : str = ''

    #: The default position unit.
    DEFAULT_POSITION_UNIT : str = ''

    def __init__(
            self,
            position_unit: Optional[pint.Quantity] = None,
            energy_unit: Optional[pint.Quantity] = None,
    ):
        r"""Constructor.

        Parameters
        ----------
        positions_unit : pint.Unit, optional
            The unit of the positions passed to the class methods. Since input
            ``Tensor``\ s do not have units attached, this is used to appropriately
            convert ``batch_positions`` to ASE units. If ``None``, no conversion
            is performed, which assumes that the input positions are in the units
            specified by the class attribute :attr:`~PotentialBase.DEFAULT_POSITION_UNIT`.
        energy_unit : pint.Unit, optional
            The unit used for the returned energies (and as a consequence forces).
            Since ``Tensor``\ s do not have units attached, this is used to
            appropriately convert ASE energies into the desired units. If ``None``,
            no conversion is performed, which means that energies will be returned
            in the units specified by the class attribute :attr:`~PotentialBase.DEFAULT_ENERGY_UNIT`.

        """
        super().__init__()
        self._position_unit = position_unit
        self._energy_unit = energy_unit

    @property
    def position_unit(self) -> pint.Quantity:
        """The position units requested for the input."""
        if self._position_unit is None:
            ureg = self._get_unit_registry()
            return getattr(ureg, self.DEFAULT_POSITION_UNIT)
        return self._position_unit

    @property
    def energy_unit(self) -> pint.Quantity:
        """The energy units of the returned potential."""
        if self._energy_unit is None:
            ureg = self._get_unit_registry()
            return getattr(ureg, self.DEFAULT_ENERGY_UNIT)
        return self._energy_unit

    @classmethod
    def default_position_unit(cls, unit_registry) -> pint.Quantity:
        """Return the default position units."""
        return getattr(unit_registry, cls.DEFAULT_POSITION_UNIT)

    @classmethod
    def default_energy_unit(cls, unit_registry) -> pint.Quantity:
        """Return the default energy units."""
        return getattr(unit_registry, cls.DEFAULT_ENERGY_UNIT)

    def _get_unit_registry(self):
        """Return a unit registry.

        The class tries to obtain a ``pint.UnitRegistry`` from the units passed
        on initialization. If none was found, it creates a new one.

        """
        if self._position_unit is not None:
            return self._position_unit._REGISTRY
        if self._energy_unit is not None:
            return self._energy_unit._REGISTRY
        return pint.UnitRegistry()
