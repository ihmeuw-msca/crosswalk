# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
    `utils` module for the `crosswalk` package, provides utility functions.
"""
import numpy as np


def is_numerical_array(x, shape=None, not_nan=True, not_inf=True):
    """Check if the given variable a numerical array.
    Args:
        x (numpy.ndarray):
            The array need to be checked.
        shape (tuple{int, int} | None, optional)
            The shape of the array
        not_nan (bool, optional):
            Optional variable check if the array contains nan.
        not_inf (bool, optional):
            Optional variable check if the array contains inf.
    Returns:
        bool: if x is a numerical numpy array.
    """
    ok = isinstance(x, np.ndarray)
    if not ok:
        return ok
    ok = ok and np.issubdtype(x.dtype, np.number)
    if not_nan:
        ok = ok and (not np.isnan(x).any())
    if not_inf:
        ok = ok and (not np.isinf(x).any())
    if shape is not None:
        ok = ok and (x.shape == shape)

    return ok


def sizes_to_indices(sizes):
    """Converting sizes to corresponding indices.

    Args:
        sizes (numpy.dnarray):
            An array consist of non-negative number.

    Returns:
        list{range}:
            List the indices.
    """
    indices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        indices.append(range(a, b))
        a += size

    return indices
