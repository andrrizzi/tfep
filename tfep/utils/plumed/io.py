#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Provide utility functions to read output files generated by PLUMED.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import numpy as np


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def read_table_field_names(
        file_path
):
    """
    Return the names provided in the '#! FIELDS ...' of the output file.

    Parameters
    ----------
    file_path : str
        The path to the PLUMED output file.

    Returns
    -------
    field_names : List[str]
        ``field_names[i]`` is the field name for the ``i``-th column of
        the table in the output file.

    """
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#! FIELDS'):
                return line.split()[2:]

    # No FIELDS record could be found.
    raise ValueError(f"No '#! FIELDS' record could be found in file {file_path}")


def read_table_n_rows(
        file_path
):
    """
    Read the number of rows in the table.

    Parameters
    ----------
    file_path : str
        The path to the PLUMED output file.

    Returns
    -------
    n_rows : int
        The number of rows in the table.

    """
    with open(file_path, 'r') as f:
        # Skip comments and blank lines.
        n_rows = sum(1 for line in f if not(line.startswith('#!') or line == ''))
    return n_rows


def read_table(
        file_path,
        col_names=None,
        as_array=False,
        row_filter_func=None,
        dtype=None,
        ordering_col_name=None,
):
    """
    Read one or more columns from a PLUMED output file.

    Parameters
    ----------
    file_path : str
        The path to the PLUMED output file to read.
    col_names : List[str], optional
        A list of column names to read. These names correspond to those
        provided in the initial '#! FIELDS ...' header record. If not
        given, the function reads all the columns.
    as_array : bool, optional
        If ``True``, the data is returned as a single array of shape
        ``(n_rows, n_cols)``. Otherwise, the table is returned in ``dict``
        format with one column for each key. Default is ``False``.
    row_filter_func : Callable[[str], bool], optional
        Rows for which this function returns False will be skipped.
    dtype : type, optional
        The type of the returned array. Otherwise, the type is inferred
        from the data.
    ordering_col_name : str, optional
        If given, rows are re-ordered in increasing order of the value
        of this column.

    Returns
    -------
    data : numpy.ndarray or Dict[str, numpy.ndarray]
        If ``as_array == False``, ``data[col_name][i]`` is the value in
        the ``i``-th record of column ``col_name`` of the table in the
        output file. Otherwise, ``data[col_names.index(col_name)][i]``
        is the corresponding value.

    """
    # First read the field names.
    field_names = read_table_field_names(file_path)

    # Check if we need to read all columns.
    if col_names is None:
        col_names = field_names

    # Make sure we need to read at least one column.
    if len(col_names) == 0:
        raise ValueError("col_names must be either None or contain at least one value")

    # Transform names into column indices, respecting the order of col_names.
    columns_to_read = [field_names.index(name) for name in col_names]
    if len(columns_to_read) != len(col_names):
        raise ValueError(f"Can't find columns {col_names}. Fields in the file are {field_names}")

    # Read the file using numpy.
    with open(file_path, 'r') as fh:
        # Check if we need to apply a filter or not.
        if row_filter_func is not None:
            fh = filter(row_filter_func, fh)

        # Read file.
        data_matrix = np.loadtxt(fh, comments='#!', usecols=columns_to_read, unpack=not as_array)

    # Re-order rows.
    if ordering_col_name is not None:
        col_idx = col_names.index(ordering_col_name)
        sorting_indices = np.argsort(data_matrix[col_idx])
        data_matrix[:] = data_matrix[:, sorting_indices]

    if as_array:
        return data_matrix

    # Convert to dictionary.
    if len(col_names) == 1:
        data = {col_names[0]: data_matrix}
    else:
        data = {col_name: col for col_name, col in zip(col_names, data_matrix)}
    return data


def write_table(data, file_path, col_names=None):
    """
    Take a table in dict format and write it in the format used by PLUMED.

    Currently, files are always overwritten.

    Parameters
    ----------
    data : Dict[str, numpy.ndarray]
        data[col_name][i] is i-th value for column 'col_name'.
    file_path : str
        The path to the file where to write the table.
    col_names : List[str], optional
        Allow writing only a subset of columns. If not given, all columns
        are written.

    See Also
    --------
    read_table

    """
    # Default to writing all columns.
    if col_names is None:
        col_names = list(data.keys())

    # Isolate all arrays that must be saved.
    all_arrays = [data[name] for name in col_names]

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Overwrite existing file.
    with open(file_path, 'w') as f:
        # First write the name of the fields.
        f.write('#! FIELDS ' + ' '.join(col_names) + '\n')

        # Save all columns.
        for row_values in zip(*all_arrays):
            f.write(' '.join([str(v) for v in row_values]) + '\n')