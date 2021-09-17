#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Miscellanea utility functions."""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import contextlib
import os


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@contextlib.contextmanager
def temporary_cd(dir_path):
    """Context manager that temporarily sets the working directory.

    Parameters
    ----------
    dir_path : str or None
        The path to the temporary working directory. If ``None``, the working
        directory is not changed. This might be useful to avoid branching code.

    """
    if dir_path is None:
        yield
    else:
        old_dir_path = os.getcwd()
        os.chdir(dir_path)
        try:
            yield
        finally:
            os.chdir(old_dir_path)
