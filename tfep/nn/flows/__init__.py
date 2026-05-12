"""
Normalizing flow models for PyTorch.

All the layers defined in this module are invertible and implement an
``inverse()`` method (not to be comfused with the ``Tensor``'s ``backward()``
method which backpropagate the gradients).

The forward propagation of the modules here return both the transformation
of the input plus the log determinant of the Jacobian.
"""

from tfep.nn.flows.centroid import CenteredCentroidFlow
from tfep.nn.flows.maf import MAF
from tfep.nn.flows.continuous import ContinuousFlow
from tfep.nn.flows.oriented import OrientedFlow
from tfep.nn.flows.partial import PartialFlow
from tfep.nn.flows.pca import PCAWhitenedFlow
from tfep.nn.flows.sequential import SequentialFlow

# Optional developer-facing utilities.
from tfep.nn.flows.factory import FlowFactory, FlowSpec
from tfep.nn.flows.triatomic_zmatrix import TriatomicZMatrixFlow, VectorCouplingFlow
from tfep.nn.flows.shell_water import (
    JointSoluteMAFAndShellWaterInternalFlow,
    ShellEquivariantWaterFlatFlow,
    ShellEquivariantWaterInternalFlow,
    SoluteAutoregressiveMAFFlow,
)

__all__ = [
    "CenteredCentroidFlow",
    "MAF",
    "ContinuousFlow",
    "OrientedFlow",
    "PartialFlow",
    "PCAWhitenedFlow",
    "SequentialFlow",
    "FlowFactory",
    "FlowSpec",
    "TriatomicZMatrixFlow",
    "VectorCouplingFlow",
    "JointSoluteMAFAndShellWaterInternalFlow",
    "ShellEquivariantWaterFlatFlow",
    "ShellEquivariantWaterInternalFlow",
    "SoluteAutoregressiveMAFFlow",
]
