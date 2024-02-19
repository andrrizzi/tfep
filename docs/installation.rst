Installation
============

The library has the following required dependencies

.. code-block::

    python >= 3.9
    pytorch >= 1.11
    mdanalysis >= 2.0
    pint
    numpy
    lightning >= 2.0

and the following optional dependencies

.. code-block::

    openmm       # To evaluate the target potentials using the OpenMM Python library.
    psi4         # To evaluate the target potentials using the psi4 Python library.
    ase          # To evaluate the target potentials using the Atomistic Simulation Environment (ASE) Python library.
    torchdiffeq  # To use continuous normalizing flows.
    bgflow       # To use the mixed internal-Cartesian coordinates Lightning module
    einops       # Required by bgflow

The suggested way of installing ``tfep`` is by first installing all the dependencies through ``conda``/``pip``/``setuptools``,
and then installing ``tfep`` from the source (I plan to add a ``tfep`` pip/conda package in the near future). Here is an
example that creates a separate conda environment with all the dependencies and installs ``tfep``.

.. code-block:: bash

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
