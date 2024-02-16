#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Utilities implementing batch parallelization strategies.

The module currently implements a serial (:class:``.SerialStrategy``) and a batch
parallelization strategy based on the ``multiprocessing.Pool`` class
(:class:``.ProcessPoolStrategy``). This latter can be used to evaluate potential
energies for batches of positions in an embarassingly parallel way.

All strategies must implement a new parallelization strategy compatible with the
``tfep`` library, see the documentation of :class:``.ParallelizationStrategy``.

See Also
--------
:class:``tfep.potentials.psi4.Psi4PotentialEnergyFunc``
    A potential energy function supporting parallelization strategies with usage
    example.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc


# =============================================================================
# PARALLELIZATION STRATEGY INTERFACE
# =============================================================================

class ParallelizationStrategy(abc.ABC):
    """The API of a ParallelizationStrategy object.

    To implement a new parallelization strategy compatible with the ``tfep``
    library, the object must implement the ``run`` method, which encapsulate the
    parallelization of a function over the arguments to distribute.

    """
    @abc.abstractmethod
    def run(self, func, args):
        """Parallelize the function over the arguments.

        Parameters
        ----------
        func : Callable
            The function to parallelize over the arguments.
        args : Iterable
            The arguments to distribute across parallel executions of the function.
            ``args[i]`` are the arguments of the ``i``-th execution, and they are
            passed to the function as ``func(*args[i])``.

        Returns
        -------
        results : List
            ``results[i]`` is the return value of the function called with
            ``args[i]``.

        """
        pass


# =============================================================================
# SERIAL STRATEGY
# =============================================================================

class SerialStrategy(ParallelizationStrategy):
    """A simple serial execution of the function."""

    def run(self, func, args):
        """Serially execute the function over the arguments.

        Parameters
        ----------
        func : Callable
            The function to execute.
        args : Iterable
            ``args[i]`` are the arguments of the ``i``-th execution, and they are
            passed to the function as ``func(*args[i])``.

        Returns
        -------
        results : List
            ``results[i]`` is the return value of the function called with
            ``args[i]``.

        """
        return [func(*arg) for arg in args]


# =============================================================================
# MULTIPROCESSING POOL STRATEGY
# =============================================================================

class ProcessPoolStrategy(ParallelizationStrategy):
    """Parallelization over a pool of processes using ``Pool.starmap``.

    Parameters
    ----------
    pool : multiprocessing.Pool
        The pool of processes to use to map the arguments.

    """

    def __init__(self, pool):
        self.pool = pool

    def run(self, func, args):
        """Parallelize the function over the arguments.

        Parameters
        ----------
        func : Callable
            The function to parallelize over the arguments.
        args : Iterable
            The arguments to distribute across parallel executions of the function.
            ``args[i]`` are the arguments of the ``i``-th execution, and they are
            passed to the function as ``func(*args[i])``.

        Returns
        -------
        results : List
            ``results[i]`` is the return value of the function called with
            ``args[i]``.

        """
        return self.pool.starmap(func, args)
