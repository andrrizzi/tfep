"""
Transformers for autoregressive normalizing flows in PyTorch.

All the layers defined in this module are invertible and implement an
``inverse()`` method (not to be comfused with the ``Tensor``'s ``backward()``
method which backpropagate the gradients).

The forward propagation of the modules here return both the transformation
of the input plus the log determinant of the Jacobian.
"""

from tfep.nn.transformers.affine import (
    AffineTransformer, affine_transformer, affine_transformer_inverse,
    VolumePreservingShiftTransformer, volume_preserving_shift_transformer, volume_preserving_shift_transformer_inverse,
)
from tfep.nn.transformers.moebius import (
    MoebiusTransformer, moebius_transformer)
from tfep.nn.transformers.sos import (
    SOSPolynomialTransformer, SOSPolynomialTransformerFunc, sos_polynomial_transformer)
from tfep.nn.transformers.spline import (
    NeuralSplineTransformer, neural_spline_transformer)
