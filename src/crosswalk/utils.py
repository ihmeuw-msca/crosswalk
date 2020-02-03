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


def array_structure(x):
    """Return the structure of the array.

    Args:
        x (numpy.ndarray):
            The numpy array need to be studied.

    Returns:
        tuple{int, numpy.ndarray, numpy.ndarray}:
            Return the number of unique elements in the array, counts for each
            unique element and unique element.
    """
    assert isinstance(x, np.ndarray)
    unique_x, x_sizes = np.unique(x, return_counts=True)
    num_x = x_sizes.size

    return num_x, x_sizes, unique_x


def default_input(input, default=None):
    """Process the keyword input in the function.

    Args:
        input:
            Keyword input for the function.

    Return:
        `default` when input is `None`, otherwise `input`.
    """
    if input is None:
        return default
    else:
        return input


def log_to_linear(mean, sd):
    """Transform mean and standard deviation from log space to linear space.
    Using Delta method.

    Args:
        mean (numpy.ndarray):
            Mean in log space.
        sd (numpy.ndarray):
            Standard deviation in log space.

    Returns:
        tuple{numpy.ndarray, numpy.ndarray}:
            Mean and standard deviation in linear space.
    """
    assert mean.size == sd.size
    assert (sd >= 0.0).all()
    linear_mean = np.exp(mean)
    linear_sd = np.exp(mean)*sd

    return linear_mean, linear_sd


def linear_to_log(mean, sd):
    """Transform mean and standard deviation from linear space to log space.
    Using delta method.

    Args:
        mean (numpy.ndarray):
            Mean in linear space.
        sd (numpy.ndarray):
            Standard deviation in linear space.

    Returns:
        tuple{numpy.ndarray, numpy.ndarray}:
            Mean and standard deviation in log space.
    """
    assert mean.size == sd.size
    assert (mean > 0.0).all()
    assert (sd >= 0.0).all()
    log_mean = np.log(mean)
    log_sd = sd/mean

    return log_mean, log_sd


def logit_to_linear(mean, sd):
    """Transform mean and standard deviation from logit space to linear space.
    Using Delta method.

    Args:
        mean (numpy.ndarray):
            Mean in logit space.
        sd (numpy.ndarray):
            Standard deviation in logit space.

    Returns:
        tuple{numpy.ndarray, numpy.ndarray}:
            Mean and standard deviation in linear space.
    """
    assert mean.size == sd.size
    assert (sd >= 0.0).all()
    linear_mean = 1.0/(1.0 + np.exp(-mean))
    linear_sd = (np.exp(mean)/(1.0 + np.exp(mean))**2)*sd

    return linear_mean, linear_sd


def linear_to_logit(mean, sd):
    """Transform mean and standard deviation from linear space to logit space.
    Using delta method.

    Args:
        mean (numpy.ndarray):
            Mean in linear space.
        sd (numpy.ndarray):
            Standard deviation in linear space.

    Returns:
        tuple{numpy.ndarray, numpy.ndarray}:
            Mean and standard deviation in logit space.
    """
    assert mean.size == sd.size
    assert ((mean > 0.0) & (mean < 1.0)).all()
    assert (sd >= 0.0).all()
    logit_mean = np.log(mean/(1.0 - mean))
    logit_sd = sd/(mean*(1.0 - mean))

    return logit_mean, logit_sd
