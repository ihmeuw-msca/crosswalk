# -*- coding: utf-8 -*-
"""
utils
~~~~~
`utils` module for the `crosswalk` package, provides utility functions.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def is_numerical_array(
    x: NDArray,
    shape: tuple[int, int] | None = None,
    not_nan: bool = True,
    not_inf: bool = True,
) -> bool:
    """Check if the given variable is a numeric array.

    Parameters
    ----------
    x : NDArray
        The array being checked.
    shape : tuple[int, int], optional
        The shape of the array. If None, shape validation is skipped and only
        dtype/NaN/inf checks are performed, by default None.
    not_nan : bool, optional
        Optional variable check if the array contains nan, by default True.
    not_inf : bool, optional
        Optional variable check if the array contains inf, by default True.

    Returns
    -------
    bool
        If `x` is a numerical numpy array.
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


def sizes_to_indices(sizes: NDArray) -> list[range]:
    """Converts sizes to corresponding indices.

    Parameters
    ----------
    sizes : NDArray
        An array consisting of non-negative numbers.

    Returns
    -------
    list[range]
        List of indices.
    """
    indices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        indices.append(range(a, b))
        a += size

    return indices


def sizes_to_slices(sizes: NDArray) -> list[slice]:
    """Converts sizes to corresponding slices.

    Parameters
    ----------
    sizes : NDArray
        An array consisting of non-negative numbers.

    Returns
    -------
    list[slice]
        List of slices.
    """
    slices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        slices.append(slice(a, b))
        a += size

    return slices


def array_structure(x: Iterable) -> tuple[int, NDArray, NDArray]:
    """Return the structure of the array.

    Parameters
    ----------
    x : Iterable
        The numpy array need to be studied.

    Returns
    -------
    tuple[int, NDArray, NDArray]
        Return the number of unique elements in the array, counts for each
        unique element and unique element.
    """
    x = flatten_list(list(x))
    unique_x, x_sizes = np.unique(x, return_counts=True)
    num_x = x_sizes.size

    return num_x, x_sizes, unique_x


def default_input(input: type[Any], default: Any = None) -> type[Any] | None:
    """Process the keyword input in the function.

    Parameters
    ----------
    input : type[Any]
        Keyword input for the function.
    default : Any, optional
        Default value to be returned, by default None.

    Returns
    -------
    type[Any] | None
        `default` when `input` is `None`, otherwise `input`.
    """
    if input is None:
        return default
    else:
        return input


def log_to_linear(mean: NDArray, sd: NDArray) -> tuple[NDArray, NDArray]:
    """Transform mean and standard deviation from log space to linear space using delta method.

    Parameters
    ----------
    mean : NDArray
        Mean in log space.
    sd : NDArray
        Standard deviation in log space.

    Returns
    -------
    tuple[NDArray, NDArray]
        Mean and standard deviation in linear space.
    """
    if mean.size != sd.size:
        raise ValueError(
            f"size of mean and sd must be equal. They are {mean.size} and {sd.size} respectively"
        )
    if (sd < 0.0).any():
        raise ValueError("negative sd is forbidden for this operation")
    linear_mean = np.exp(mean)
    linear_sd = np.exp(mean) * sd

    return linear_mean, linear_sd


def linear_to_log(mean: NDArray, sd: NDArray) -> tuple[NDArray, NDArray]:
    """Transform mean and standard deviation from linear space to log space using delta method.

    Parameters
    ----------
    mean : NDArray
        Mean in linear space.
    sd : NDArray
        Standard deviation in linear space.

    Returns
    -------
    tuple[NDArray, NDArray]
        Mean and standard deviation in log space.
    """
    if mean.size != sd.size:
        raise ValueError(
            f"size of mean and sd must be equal. They are {mean.size} and {sd.size} respectively"
        )
    if (mean <= 0).any():
        raise ValueError("mean <= 0 is forbidden for this operation")
    if (sd < 0.0).any():
        raise ValueError("negative sd is forbidden for this operation")
    log_mean = np.log(mean)
    log_sd = sd / mean

    return log_mean, log_sd


def logit_to_linear(mean: NDArray, sd: NDArray) -> tuple[NDArray, NDArray]:
    """Transform mean and standard deviation from logit space to linear space using delta method.

    Parameters
    ----------
    mean : NDArray
        Mean in logit space.
    sd : NDArray
        Standard deviation in logit space.

    Returns
    -------
    tuple[NDArray, NDArray]
        Mean and standard deviation in linear space.
    """
    if mean.size != sd.size:
        raise ValueError(
            f"size of mean and sd must be equal. They are {mean.size} and {sd.size} respectively"
        )
    if (sd < 0).any():
        raise ValueError("negative sd is forbidden for this operation")
    linear_mean = 1.0 / (1.0 + np.exp(-mean))
    linear_sd = (np.exp(mean) / (1.0 + np.exp(mean)) ** 2) * sd

    return linear_mean, linear_sd


def linear_to_logit(mean: NDArray, sd: NDArray) -> tuple[NDArray, NDArray]:
    """Transform mean and standard deviation from linear space to logit space using delta method.

    Parameters
    ----------
    mean : NDArray
        Mean in linear space.
    sd : NDArray
        Standard deviation in linear space.

    Returns
    -------
    tuple[NDArray, NDArray]
        Mean and standard deviation in logit space.
    """
    if mean.size != sd.size:
        raise ValueError(
            f"size of mean and sd must be equal. They are {mean.size} and {sd.size} respectively"
        )
    if ((mean <= 0) | (mean >= 1)).any():
        raise ValueError("mean must be within (0, 1) for this operation")
    if (sd < 0.0).any():
        raise ValueError("negative sd is forbidden for this operation")
    logit_mean = np.log(mean / (1.0 - mean))
    logit_sd = sd / (mean * (1.0 - mean))

    return logit_mean, logit_sd


def flatten_list(my_list: list[list]) -> list:
    """Flatten list so that it will be a list of non-list objects.

    Parameters
    ----------
    my_list : list[list]
        List to be flattened.

    Returns
    -------
    list
        Flattened list.

    Raises
    ------
    ValueError
        If `my_list` is not a list.
    """
    if not isinstance(my_list, list):
        raise ValueError("Input must be a list.")

    result = []
    for element in my_list:
        if isinstance(element, list):
            result.extend(flatten_list(element))
        else:
            result.append(element)

    return result


def process_dorms(
    dorms: str | None = None,
    size: int | None = None,
    default_dorm: str = "Unknown",
    dorm_separator: str | None = None,
) -> list[list[str]]:
    """Process the dorms.

    Parameters
    ----------
    dorms : str | None, optional
        Input definition or methods. If None, returns a list of
        ``[[default_dorm]] * size``, assigning the same default label to all
        observations. Requires ``size`` to be provided, by default None.
    size : int | None, optional
        Size of the dorm array, only used and required when `dorms` is None.
        If None when ``dorms`` is also None, a ValueError is raised,
        by default None.
    default_dorm : str, optional
        Default dorm used when `dorms` is None, by default "Unknown".
    dorm_separator : str | None, optional
        Dorm separator for when multiple definition or methods present. If
        None, dorm strings are split on whitespace (Python's default
        ``str.split(None)`` behavior), by default None.

    Returns
    -------
    list[list[str]]
        List of list of definition or methods. The second
        layer of list is for convenience when there are
        multiple definition or methods.

    Raises
    ------
    ValueError
        When `size` and `dorms` are both None.
    """
    if dorms is None:
        if size is None:
            raise ValueError("Size cannot be None, when dorms is None.")
        return [[default_dorm]] * size
    else:
        return [dorm.split(dorm_separator) for dorm in dorms]


def p_value(mean: NDArray, std: NDArray, one_tailed: bool = False) -> NDArray:
    """Compute the p value from mean and standard deviation.

    Parameters
    ----------
    mean : NDArray
        Mean of the samples.
    std : NDArray
        Standard deviation of the samples.
    one_tailed : bool, optional
        If `True` then use the one tailed test, by default False.

    Returns
    -------
    NDArray
        An array of p-values.
    """
    if (std <= 0.0).any():
        raise ValueError("standard deviation must be greater than 0")
    if hasattr(mean, "__iter__") and hasattr(std, "__iter__"):
        if len(mean) != len(std):
            raise ValueError(
                f"mean and std must have the same size. they are {len(mean)} and {len(std)} respectively"
            )

    prob = norm.cdf(np.array(mean) / np.array(std))
    pval = np.minimum(prob, 1 - prob)
    if not one_tailed:
        pval *= 2
    return pval
