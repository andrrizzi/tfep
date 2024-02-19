TFEP
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/andrrizzi/tfep/workflows/CI/badge.svg)](https://github.com/andrrizzi/tfep/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/andrrizzi/tfep/branch/master/graph/badge.svg)](https://codecov.io/gh/andrrizzi/tfep/branch/master)
[![Documentation Status](https://readthedocs.org/projects/tfep/badge/?version=latest)](https://tfep.readthedocs.io/en/latest/?badge=latest)

A Python utility library to perform (multimap) targeted free energy perturbation.

### Features

The library includes implementations for the following flows:

- Masked Autoregressive Flows [1] with a ready-to-use MADE [2] conditioner and the following transformers:
  - Affine
  - (Circular) neural spline [3, 4]
  - Sum-of-squares polynomial [5]
  - Moebius (based on that proposed in [4])
- Continuous normalizing flows [6] with a dynamics layer based on Equivariant graph neural networks [7].
- ``Centroid`` and ``Oriented`` flows, which can be used to fix the reference system before the map.
- A ``PCA`` flow to perform the transformation in uncorrelated coordinates.

The library also includes several utilities:

- PyTorch wrappers for [psi4](https://psicode.org/) and the [Atomistic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)
  to evaluate potential energies and forces with multiple molecular simulation engines.
- A PyTorch ``Dataset`` that can wrap an MDAnalysis ``Universe`` to train on molecular simulation trajectories.
- A PyTorch-accelerated (T)FEP estimator.
- A PyTorch-accelerated bootstrap analysis utility.
- A simple storage utility class to save potential energies and log training information.


### Installation

The library has the following required dependencies
```
python >= 3.9
pytorch >= 1.11
mdanalysis >= 2.0
pint
numpy
lightning >= 2.0
```
and the following optional dependencies
```
openmm       # To evaluate the target potentials using the OpenMM Python library.
psi4         # To evaluate the target potentials using the psi4 Python library.
ase          # To evaluate the target potentials using the Atomistic Simulation Environment (ASE) Python library.
torchdiffeq  # To use continuous normalizing flows.
bgflow       # To use the mixed internal-Cartesian coordinates Lightning module
einops       # Required by bgflow
```

The suggested way of installing ``tfep`` is by first installing all the dependencies through ``conda``/``pip``/``setuptools``,
and then installing ``tfep`` from the source (I plan to add a ``tfep`` conda package in the near future). Here is an
example that creates a separate conda environment with all the dependencies and installs ``tfep``.

```bash
# Required dependencies.
conda create --name tfepenv python">=3.9" pytorch">=1.11" mdanalysis">=2.0" pint numpy lightning">=2.0" -c conda-forge
conda activate tfepenv

# Optional dependency using conda.
conda install einops -c conda-forge

# Optional dependency from source code.
git clone https://github.com/noegroup/bgflow.git
cd bgflow
pip install .
cd ..

# Optional dependency using pip.
pip install ase

# Install the package.
git clone https://github.com/andrrizzi/tfep.git
cd tfep
pip install .

# Or if you want to modify the source code, install it in editable mode.
# pip install -e .
```


### Citation

If you find this code useful, please cite the following paper:

Andrea Rizzi, Paolo Carloni, Michele Parrinello. *Multimap targeted free energy estimation.* [arXiv preprint arXiv:2302.07683](http://arxiv.org/abs/2302.07683).


### References

1. Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for density estimation. In [Advances in Neural
   Information Processing Systems](https://doi.org/10.48550/arXiv.1705.07057) (2017).
2. Germain M, Gregor K, Murray I, Larochelle H. Made: Masked autoencoder for distribution estimation. In [International
   Conference on Machine Learning](https://doi.org/10.48550/arXiv.1502.03509) (2015).
3. Durkan C, Bekasov A, Murray I, Papamakarios G. Neural spline flows. Advances in [Neural Information Processing
   Systems](https://doi.org/10.48550/arXiv.1906.04032) (2019).
4. Rezende DJ, Papamakarios G, Racaniere S, Albergo M, Kanwar G, Shanahan P, Cranmer K. Normalizing flows on tori and
   spheres. In [International Conference on Machine Learning](https://doi.org/10.48550/arXiv.2002.02428) (2020).
5. Jaini P, Selby KA, Yu Y. Sum-of-squares polynomial flow. In [International Conference on Machine Learning](https://doi.org/10.48550/arXiv.1905.02325) (2019).
6. Chen RT, Rubanova Y, Bettencourt J, Duvenaud DK. Neural ordinary differential equations. Advances in [Neural
   Information Processing Systems](https://doi.org/10.48550/arXiv.1806.07366) (2018).
7. Garcia Satorras V, Hoogeboom E, Fuchs F, Posner I, Welling M. E(n) Equivariant Normalizing Flows. Advances in
   [Neural Information Processing Systems](https://doi.org/10.48550/arXiv.2105.09016). (2021).


### Acknowledgements

Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.


