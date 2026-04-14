# -*- coding: utf-8 -*-
"""
utils
~~~~~
`utils` module for the `crosswalk` package, provides utility functions.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def is_numerical_array(
    x: npt.NDArray,
    shape: tuple[int, int] | None = None,
    not_nan: bool = True,
    not_inf: bool = True,
) -> bool:
    """check if the given variable is a numeric array

    Parameters
    ----------
    x : npt.NDArray
        The array being checked.
    shape : tuple[int, int], optional
        The shape of the array, by default None
    not_nan : bool, optional
        Optional variable check if the array contains nan, by default True
    not_inf : bool, optional
        Optional variable check if the array contains inf, by default True

    Returns
    -------
    bool
        if x is a numerical numpy array.
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


def sizes_to_indices(sizes: npt.NDArray) -> list[range]:
    """converts sizes to corresponding indices

    Parameters
    ----------
    sizes : npt.NDArray
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


def sizes_to_slices(sizes: npt.NDArray) -> list[slice]:
    """convertes sizes to corresponding slices

    Parameters
    ----------
    sizes : npt.NDArray
        An array consisting of non-negative numbers

    Returns
    -------
    list[slice]
        list of slices
    """
    slices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        slices.append(slice(a, b))
        a += size

    return slices


def array_structure(x: Iterable) -> tuple[int, npt.NDArray, npt.NDArray]:
    """Return the structure of the array

    Parameters
    ----------
    x : Iterable
        The numpy array need to be studied.

    Returns
    -------
    tuple[int, npt.NDArray, npt.NDArray]
        Return the number of unique elements in the array, counts for each
        unique element and unique element.
    """
    x = flatten_list(list(x))
    unique_x, x_sizes = np.unique(x, return_counts=True)
    num_x = x_sizes.size

    return num_x, x_sizes, unique_x


def default_input(input: type[Any], default: None = None) -> type[Any] | None:
    """process the keyword input in the function

    Parameters
    ----------
    input : type[Any]
        Keyword input for the function.
    default : None, optional
        default value to be returned, by default None

    Returns
    -------
    type[Any] | None
        `default` when input is `None`, otherwise `input`.
    """
    if input is None:
        return default
    else:
        return input


def log_to_linear(
    mean: npt.NDArray, sd: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """transform mean and standard deviation from log space to linear space using delta method

    Parameters
    ----------
    mean : npt.NDArray
        Mean in log space.
    sd : npt.NDArray
        Standard deviation in log space.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
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


def linear_to_log(
    mean: npt.NDArray, sd: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """transform mean and standard deviation from linear space to log space using delta method

    Parameters
    ----------
    mean : npt.NDArray
        Mean in linear space.
    sd : npt.NDArray
        Standard deviation in linear space.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
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


def logit_to_linear(
    mean: npt.NDArray, sd: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """transform mean and standard deviation from logit space to linear space using delta method

    Parameters
    ----------
    mean : npt.NDArray
        Mean in logit space.
    sd : npt.NDArray
        Standard deviation in logit space.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
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


def linear_to_logit(
    mean: npt.NDArray, sd: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """transform mean and standard deviation from linear space to logit space using delta method

    Parameters
    ----------
    mean : npt.NDArray
        Mean in linear space.
    sd : npt.NDArray
        Standard deviation in linear space.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
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
    """flatten list so that it will be a list of non-list objects

    Parameters
    ----------
    my_list : list[list]
        list to be flattened

    Returns
    -------
    list
        flattened list

    Raises
    ------
    ValueError
        if my_list is not a list
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
    """process the dorms

    Parameters
    ----------
    dorms : str | None, optional
        Input definition or methods, by default None
    size : int | None, optional
        Size of the dorm array, only used and required when dorms is None, by default None
    default_dorm : str, optional
        Default dorm used when dorms is None, by default "Unknown"
    dorm_separator : str | None, optional
        Dorm separator for when multiple definition or methods present, by default None

    Returns
    -------
    list[list[str]]
        List of list of definition or methods. The second layer of list is for convenience
            when there are multiple definition or methods.

    Raises
    ------
    ValueError
        when size and dorms are both None
    """
    if dorms is None:
        if size is None:
            raise ValueError("Size cannot be None, when dorms is None.")
        return [[default_dorm]] * size
    else:
        return [dorm.split(dorm_separator) for dorm in dorms]


def p_value(
    mean: npt.NDArray, std: npt.NDArray, one_tailed: bool = False
) -> npt.NDArray:
    """comput the p value from mean and standard deviation

    Parameters
    ----------
    mean : npt.NDArray
        mean of the samples
    std : npt.NDArray
        standard deviation of the samples
    one_tailed : bool, optional
        if `True` then use the one tailed test, by default False

    Returns
    -------
    npt.NDArray
        an array of p-values
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
