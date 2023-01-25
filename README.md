tfep
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/andrrizzi/tfep/workflows/CI/badge.svg)](https://github.com/andrrizzi/tfep/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/andrrizzi/tfep/branch/master/graph/badge.svg)](https://codecov.io/gh/andrrizzi/tfep/branch/master)

A Python library to perform targeted free energy perturbation.


### Installation

The library has the following required dependencies
```
pytorch >= 1.11
mdanalysis >= 2.0
pint
numpy
```
and the following optional dependencies
```
psi4         # To evaluate the target potentials using the psi4 Python library.
ase          # To evaluate the target potentials using the Atomistic Simulation Environment (ASE) Python library.
torchdiffeq  # To use continuous normalizing flows.
```

The suggested way of installing ``tfep`` is by first installing all the dependencies through ``conda``/``pip``, and then
installing ``tfep`` from the source (I plan to add a ``tfep`` conda package in the near future). Here is an example that
creates a separate conda environment with all the dependencies and installs ``tfep`` in editable mode.

```bash
# Required dependencies.
conda create --name tfepenv pytorch">=1.11" mdanalysis">=2.0" pint numpy -c conda-forge
conda activate tfepenv

# Optional dependency using pip.
pip install ase

# Install package.
git clone https://github.com/andrrizzi/tfep.git
cd tfep
pip install -e .
```


### Copyright

Copyright (c) 2021, Andrea Rizzi


#### Acknowledgements

Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
