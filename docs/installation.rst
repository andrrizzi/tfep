Installation
============

The library has the following required dependencies

.. code-block::

    pytorch >= 1.11
    mdanalysis >= 2.0
    pint
    numpy
    lightning >= 2.0

and the following optional dependencies

.. code-block::

    psi4         # To evaluate the target potentials using the psi4 Python library.
    ase          # To evaluate the target potentials using the Atomistic Simulation Environment (ASE) Python library.
    torchdiffeq  # To use continuous normalizing flows.

The suggested way of installing ``tfep`` is by first installing all the dependencies through ``conda``/``pip``/``setuptools``,
and then installing ``tfep`` from the source (I plan to add a ``tfep`` pip/conda package in the near future). Here is an
example that creates a separate conda environment with all the dependencies and installs ``tfep``.

.. code-block:: bash

    # Required dependencies.
    conda create --name tfepenv pytorch">=1.11" mdanalysis">=2.0" pint numpy lightning">=2.0" -c conda-forge
    conda activate tfepenv

    # Optional dependency using pip.
    pip install ase

    # Install the package in editable mode.
    git clone https://github.com/andrrizzi/tfep.git
    cd tfep
    pip install .

    # Or if you want to modify the source code, install it in editable mode.
    # pip install -e .
