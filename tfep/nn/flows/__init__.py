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
from tfep.nn.flows.oriented import OrientedFlow
from tfep.nn.flows.partial import PartialFlow
from tfep.nn.flows.pca import PCAWhitenedFlow
from tfep.nn.flows.sequential import SequentialFlow
